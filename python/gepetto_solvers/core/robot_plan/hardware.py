"""The hand-open reference pose, travel limits, and clamping to them.

This is the half of ``robot_plan`` that has to know about the solver and about
the physical hand; the timing half deliberately does not.

SIGN: positive tendon displacement = tendon pulled in = FLEXING, measured from
the hand-open pose.

TWO KINDS OF LIMIT, ONE CLAMP. A tendon hand's stop is its motor's calibrated
flexion travel, from ``HandConfig``; a joint-space hand's is the URDF's joint
limits, from ``hand.joint_limits()``. :func:`travel_limits` normalizes both to
``{digit: (lo, hi)}`` with ``lo``/``hi`` as arrays the width of that hand's
command, so :func:`clamp_to_travel` never has to know which it was handed.
Everything above :func:`travel_limits` in this file is tendon-only and is simply
not reached on the other path.
"""

from dataclasses import replace

import numpy as np

from .types import COMMAND_UNITS, JOINT_POSITION_RAD, TENDON_DISPLACEMENT_M


def _solvers():
    """The solver module, imported on use rather than at import time.

    Everything in this file that BUILDS a plan needs the compiled `gepetto_solvers`
    binding; nothing that TIMES one does. `epfl_hand_control`'s executor node
    imports this module for `plan_schedule` and `sample_at` alone, and it should
    not have to carry a factor-graph solver into a real-time control loop to get
    them -- the whole point of moving that loop out of the visualizer was to stop
    it sharing a process with heavy machinery.
    """
    from .. import solvers
    return solvers


def _hand(hand=None):
    """The hand these hardware numbers belong to.

    Every function here takes ``hand=None`` and lands on the default, so the
    existing no-argument callers are unchanged; a caller working with another
    hand passes it and gets THAT hand's actuator map, open tensions and travel
    limits instead of this one's.
    """
    if hand is not None:
        return hand
    from ..hands import get_hand
    return get_hand()


def _drive_index(hand):
    """The index of the actuated tendon in a digit's length vector.

    Every displacement in this module is read at this index. A hand that drives
    more than one actuator per digit has no single such index, and
    ``drive_value`` says so rather than silently reading the first.
    """
    return hand.actuation.drive_indices[0]


def open_tendon_lengths(params=None, solver=None, hand=None):
    """Per-finger actuated-tendon length with the hand OPEN, from the model itself.

    ``params`` is a ``solvers.HandSolveParams`` and ``solver`` a
    ``solvers.HandFKSolver``; they are unannotated because `solvers` is imported
    on use rather than at module scope (see :func:`_solvers`), and an annotation
    naming a type this module never imports is a forward reference that resolves
    to nothing.

    This is the zero every displacement in a plan is measured from, and taking it
    from the model rather than from ``HandConfig.zero_bend_lengths`` is what lets
    playback work with no absolute solver-to-hardware calibration: both ends of
    the subtraction come from the same kinematics, so only the *change* is
    commanded, and the change is what the integrating servo node consumes.

    "Open" is the tension set the hardware's open pose was CALIBRATED AT --
    ``HandConfig.zero_bend_flexor_tensions`` / ``zero_bend_passive_tension``, see
    :func:`open_pose_tensions` -- not every flexor at zero. A real open hand
    carries its flexors' background pull, and the zero-tension model hyperextends
    ~3.3 mm of tendon past it (5.6 mm on the thumb): measuring from there biases
    every commanded displacement by that much in the FLEXING direction, and sends
    a robot readback of an open hand back to a hyperextended posture the hardware
    cannot reach. At the calibrated tensions the model reproduces
    ``HandConfig.zero_bend_lengths`` to within 0.04 mm, which is what
    :func:`check_open_lengths` then measures.

    The tensions come from the calibration and NOT from ``params``, so this is a
    property of the hand rather than of whatever the GUI's sliders happen to hold
    -- which is what lets a caller cache the answer for the life of the process.

    Solves on a COPY of the params (``replace``), so a caller's live params object
    -- the one the GUI is mutating from another thread -- is never touched. Pass
    ``solver`` to reuse a warm one; its own params are put back before returning,
    since ``HandFKSolver`` reads them on every solve and a caller that handed us
    its live solver would otherwise find its hand had fallen open.
    """
    solvers = _solvers()
    hand = _hand(hand)
    params = params or solvers.HandSolveParams()
    if solver is None:
        solver = solvers.HandFKSolver(replace(params), hand)
    borrowed = solver.params
    solver.params = _open_pose_params(params, solver.finger_names, hand)
    try:
        result = solver.solve()
    finally:
        solver.params = borrowed
    idx = _drive_index(hand)
    return {name: float(lengths[idx])
            for name, lengths in zip(result.finger_names, result.displacements(0))}


def open_pose_tensions(hand=None):
    """``(passive, {solver digit: flexor tension})`` for the calibrated open hand.

    From ``HandConfig`` when epfl_hand_core is importable, else the fallback copy
    below -- same degrade-to-a-note rule as :func:`_hardware_open_lengths`, except
    that this one is load-bearing rather than a cross-check, so the fallback is a
    real copy of the numbers rather than a None.
    """
    hw = _hand(hand).hardware
    config = _hand_config()
    if config is None:
        return hw.open_passive, dict(hw.open_drive)
    flexors = {digit: float(config.zero_bend_flexor_tensions[actuator])
               for digit, actuator in hw.actuator_names.items()
               if actuator in config.zero_bend_flexor_tensions}
    return float(config.zero_bend_passive_tension), flexors


def _open_pose_params(params, finger_names, hand=None):
    """``params`` posed at the calibrated open hand, as a copy.

    ``finger_names`` is the solver's own digit order, since ``flexor_tensions`` is
    positional: keying the calibration by name and re-ordering it here is what
    keeps this correct if the hand is ever built with its fingers in another
    order, or with a digit missing. A digit the calibration says nothing about
    keeps whatever ``params`` holds for it.
    """
    passive, flexors = open_pose_tensions(hand)
    tensions = [float(flexors.get(name, held))
                for name, held in zip(finger_names, params.flexor_tensions)]
    return replace(params, flexor_tensions=tensions, passive_tension=passive)


def check_open_lengths(open_lengths, params=None, hand=None):
    """Cross-check the model's open lengths, and PROVE the flexion sign.

    Two failures this catches, both of which otherwise show up only as a hand
    that moves the wrong way on real hardware:

    * the model's open pose drifting away from the hardware calibration
      (``HandConfig.zero_bend_lengths``), which biases every displacement by the
      difference;
    * the actuated tendon getting LONGER under flexion, which would make
      ``open - current`` negative for a closing hand and send the fingers to the
      extension stop. Nothing in the repo states the polarity, so it is measured
      here (one FK solve at a probe tension) rather than assumed.

    Returns ``(notes, ok)``: human-readable lines for the status/log, and False if
    the sign check failed -- which the caller must treat as fatal, because every
    displacement in the plan would then have the wrong sign.
    """
    notes, ok = [], True

    # -- the sign, measured --
    solvers = _solvers()
    hand = _hand(hand)
    probe_tension = hand.hardware.flexion_probe
    params = params or solvers.HandSolveParams()
    n = len(params.flexor_tensions)
    probe = replace(params, flexor_tensions=[probe_tension] * n)
    flexed = solvers.HandFKSolver(probe, hand).solve()
    idx = _drive_index(hand)
    deltas = {name: open_lengths[name] - float(lengths[idx])
              for name, lengths in zip(flexed.finger_names, flexed.displacements(0))
              if name in open_lengths}
    worst = min(deltas.values()) if deltas else 0.0
    if worst <= 0.0:
        ok = False
        notes.append(
            f"**tendon sign check FAILED**: at {probe_tension:g} N of "
            f"flexor tension the actuated tendon did not shorten on every finger "
            f"(worst {worst * 1e3:+.2f} mm). Playback would drive the hand the "
            f"wrong way; refusing to build a plan.")
    else:
        notes.append(
            f"tendon sign check: flexing at {probe_tension:g} N pulls in "
            f"{min(deltas.values()) * 1e3:.1f}-{max(deltas.values()) * 1e3:.1f} mm "
            f"across the {len(deltas)} digits (positive = pulled in, as commanded)")

    # -- the hardware's own numbers, if they are reachable --
    hardware = _hardware_open_lengths(hand)
    if hardware is None:
        notes.append("_HandConfig unavailable -- open lengths not cross-checked "
                     "against the hardware calibration._")
        return notes, ok

    drift = {name: open_lengths[name] - hardware[name]
             for name in open_lengths if name in hardware}
    if drift:
        worst_name = max(drift, key=lambda k: abs(drift[k]))
        if abs(drift[worst_name]) > hand.hardware.open_length_warn:
            notes.append(
                f"**open-length drift {drift[worst_name] * 1e3:+.1f} mm on "
                f"{worst_name}** (model {open_lengths[worst_name] * 1e3:.1f} mm vs "
                f"HandConfig.zero_bend_lengths {hardware[worst_name] * 1e3:.1f} mm). "
                f"Displacements are differences so playback still moves the right "
                f"way, but the two calibrations have drifted apart.")
        else:
            notes.append(
                f"open lengths agree with HandConfig.zero_bend_lengths to "
                f"{abs(drift[worst_name]) * 1e3:.1f} mm (worst: {worst_name})")
    return notes, ok


def _hand_config():
    """A ``HandConfig``, or None where epfl_hand_core is not installed.

    Optional on purpose: the visualizer runs on machines with no epfl_hand_core
    install, and a missing hardware config must degrade to a note (or to the
    fallback tensions above) rather than stop a plan being built."""
    try:
        from epfl_hand_core.config import HandConfig
    except ImportError:
        return None
    return HandConfig()


def _hardware_open_lengths(hand=None):
    """``HandConfig.zero_bend_lengths`` keyed by SOLVER digit name, or None."""
    config = _hand_config()
    if config is None:
        return None
    out = {}
    for digit, actuator in _hand(hand).hardware.actuator_names.items():
        if actuator in config.finger_names:
            index = config.finger_names.index(actuator)
            out[digit] = float(config.zero_bend_lengths[index])
    return out


def hardware_travel_limits(hand=None):
    """Per solver digit, the usable flexion travel ``(0.0, max_m)`` from
    ``HandConfig``, or None when epfl_hand_core is not installed.

    ``zero_bend - fully_flexed``: about 17.8 mm on index. A solve can easily ask
    for more than that -- the model has no motor -- so whoever publishes the plan
    clamps to this and says it did."""
    config = _hand_config()
    if config is None:
        return None
    out = {}
    for digit, actuator in _hand(hand).hardware.actuator_names.items():
        if actuator not in config.finger_names:
            continue
        index = config.finger_names.index(actuator)
        travel = float(config.zero_bend_lengths[index]
                       - config.fully_flexed_lengths[index])
        if travel > 0.0:
            out[digit] = (0.0, travel)
    return out


def joint_travel_limits(hand=None):
    """Per solver digit, ``(lo, hi)`` joint arrays from the URDF, or None.

    The joint-space analogue of :func:`hardware_travel_limits`, and the ONLY
    joint-limit guard on the wire: ``rigid_urdf`` reads the URDF's limits but does
    not enforce them in the solve (see ``docs/adding_a_hand.md``), so a contact
    solve can perfectly well converge with a joint past its stop. Whoever
    publishes the plan clamps to this and says it did.

    None -- rather than an empty dict -- for a hand that supplies no
    ``joint_limits``, so a caller can tell "this hand has none" from "this hand's
    limits are all zero-width".
    """
    hand = _hand(hand)
    supplier = getattr(hand, "joint_limits", None)
    if supplier is None:
        return None
    out = {}
    for name, pairs in zip(hand.digit_names, supplier()):
        lo = np.array([float(a) for a, _ in pairs], float)
        hi = np.array([float(b) for _, b in pairs], float)
        out[name] = (lo, hi)
    return out or None


def travel_limits(hand=None):
    """``{digit: (lo, hi)}`` as arrays, whichever kind of stop this hand has.

    The one entry point :func:`clamp_to_travel` uses, so the clamp itself is the
    same code for both hands. Returns None when the limits could not be
    determined at all, which the caller must report rather than pass over -- a
    missing safety clamp is never allowed to be silent.
    """
    hand = _hand(hand)
    if "displacement" in hand.features:
        travel = hardware_travel_limits(hand)
        if not travel:
            return None
        # Widened to the command's width so the clamp is shape-agnostic. A tendon
        # hand drives one actuator per digit, so this is width 1 by construction.
        width = len(hand.actuation.drive_indices)
        return {name: (np.full(width, float(lo)), np.full(width, float(hi)))
                for name, (lo, hi) in travel.items()}
    return joint_travel_limits(hand)


# Below this much overreach, IN THE DISPLAY UNITS of each command kind, a clamp
# is not worth a note. The hand-open waypoint a phase-4 close starts from sits at
# a displacement of ~1e-16 m rather than a clean zero -- FK arithmetic, not a real
# command -- and the lower stop rounds it up, which reported as "asked for 0.0 mm
# beyond its 6.4 mm of travel" on every digit that was not even moving. Well under
# the 0.1 mm the note prints at, so nothing a reader could have acted on is hidden.
#
# Per kind rather than one number, because 0.05 of a millimetre and 0.05 of a
# radian are not remotely the same claim: the latter is 2.9 degrees, which is a
# real overreach that must not be swallowed.
_CLAMP_REPORT = {
    TENDON_DISPLACEMENT_M: 0.05,        # mm
    JOINT_POSITION_RAD: 1e-3,           # rad, ~0.06 deg
}


def clamp_to_travel(plan, limits=None, hand=None):
    """Clamp every digit command to the hand's travel, in place-ish.

    Returns ``(plan, notes)`` with a NEW plan (the caller's is untouched) and one
    note per digit that had to be clamped, naming how far past the stop the solve
    had asked for. A solve routinely asks for more than the robot has --
    ``fully_flexed_lengths`` is max flexion at the measured torque limit rather
    than the geometric limit, and ``rigid_urdf`` does not enforce joint limits at
    all -- and the hand node would saturate and warn once per digit anyway;
    clamping here means the interpolation still lands on a reachable target
    instead of spending its last seconds commanding a stop.

    ``limits`` is ``{digit: (lo, hi)}`` of arrays; omit it and this asks
    :func:`travel_limits` for the plan's hand. A hand whose limits are unavailable
    is NOT silently passed over: the plan comes back untouched with a note saying
    so, because a clamp that quietly did nothing reads exactly like one that had
    nothing to do.
    """
    if limits is None:
        limits = travel_limits(hand)
    if limits is None:
        return plan, ["**no travel limits available** -- the plan is NOT clamped "
                      "here. Whatever the hand node enforces is the only stop "
                      "left; fix the environment."]
    if not limits:
        return plan, []

    suffix, scale = COMMAND_UNITS.get(plan.command_kind, ("", 1.0))
    report_above = _CLAMP_REPORT.get(plan.command_kind, 0.0)
    worst: dict[str, float] = {}
    waypoints = []
    for waypoint in plan.waypoints:
        clamped = {}
        for name, value in waypoint.digit_cmd.items():
            bounds = limits.get(name)
            if bounds is None:
                clamped[name] = value
                continue
            bounded = np.clip(value, *bounds)
            excess = float(np.max(np.abs(value - bounded)))
            if excess > 0.0:
                worst[name] = max(worst.get(name, 0.0), excess)
            clamped[name] = bounded
        waypoints.append(replace(waypoint, digit_cmd=clamped))

    notes = [f"**{name} clamped**: the solve asked for {excess * scale:.1f} "
             f"{suffix} beyond its travel"
             for name, excess in sorted(worst.items())
             if excess * scale > report_above]
    return replace(plan, waypoints=waypoints), notes
