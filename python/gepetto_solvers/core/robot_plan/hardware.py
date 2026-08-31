"""The hand-open reference pose, tendon travel limits, and clamping to them.

This is the half of ``robot_plan`` that has to know about the solver and about
the physical hand; the timing half deliberately does not.

SIGN: positive tendon displacement = tendon pulled in = FLEXING, measured from
the hand-open pose.
"""

from dataclasses import replace

from .types import FLEXOR_IDX


def _solvers():
    """The solver module, imported on use rather than at import time.

    Everything in this file that BUILDS a plan needs the compiled `gepetto_solvers`
    binding; nothing that TIMES one does. `gepetto_control`'s executor node
    imports this module for `plan_schedule` and `sample_at` alone, and it should
    not have to carry a factor-graph solver into a real-time control loop to get
    them -- the whole point of moving that loop out of the visualizer was to stop
    it sharing a process with heavy machinery.

    The constant above is checked against the real one here, so the duplication
    cannot silently drift.
    """
    from .. import solvers
    if solvers.FLEXOR_IDX != FLEXOR_IDX:
        raise RuntimeError(
            f"FLEXOR_IDX disagrees: robot_plan says {FLEXOR_IDX}, solvers says "
            f"{solvers.FLEXOR_IDX}. Every tendon displacement in this module is "
            f"read at that index.")
    return solvers


# Solver digit -> the finger name the hardware knows it by (HandConfig.finger_names).
# `thumb_add` has no flexor tendon and is never commanded from a plan -- see
# HandConfig's TODO(all-fingers) and finger_servo_node._range_for.
HARDWARE_FINGER_NAMES = {
    "index": "index_flex",
    "middle": "middle_flex",
    "ring": "ring_flex",
    "pinky": "pinky_flex",
    "thumb": "thumb_flex",
}


# How far the model's own open-hand tendon lengths may sit from the hardware's
# calibrated ``HandConfig.zero_bend_lengths`` before build_plan complains. The two
# are independent derivations of the same quantity (the hardware numbers are
# themselves "calculated from factor graph", but from a possibly older morphology),
# so a few mm of disagreement is expected and a few cm means they have drifted
# apart and the displacements will be biased by the difference.
OPEN_LENGTH_WARN_M = 0.005


# The tension used to prove which way flexion moves the actuated tendon (see
# check_open_lengths). Well inside the GUI's 0-3 N slider range, and large enough
# that the resulting length change dwarfs solver noise.
_FLEXION_PROBE_TENSION = 1.5


# Fallback copy of HandConfig's open-pose tension set (zero_bend_passive_tension /
# zero_bend_flexor_tensions), for a machine with no gepetto_core install. Keyed by
# SOLVER digit. Keep in step with gepetto_core/config.py -- these two agreeing is
# what makes the model's open hand and the hardware's the same hand.
_OPEN_PASSIVE_TENSION = 0.5


_OPEN_FLEXOR_TENSIONS = {
    "index": 0.84, "middle": 0.84, "ring": 0.84, "pinky": 1.03, "thumb": 0.84,
}


def open_tendon_lengths(params=None, solver=None):
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
    params = params or solvers.HandSolveParams()
    if solver is None:
        solver = solvers.HandFKSolver(replace(params))
    borrowed = solver.params
    solver.params = _open_pose_params(params, solver.finger_names)
    try:
        result = solver.solve()
    finally:
        solver.params = borrowed
    return {name: float(lengths[FLEXOR_IDX])
            for name, lengths in zip(result.finger_names, result.tendon_lengths(0))}


def open_pose_tensions():
    """``(passive, {solver digit: flexor tension})`` for the calibrated open hand.

    From ``HandConfig`` when gepetto_core is importable, else the fallback copy
    below -- same degrade-to-a-note rule as :func:`_hardware_open_lengths`, except
    that this one is load-bearing rather than a cross-check, so the fallback is a
    real copy of the numbers rather than a None.
    """
    config = _hand_config()
    if config is None:
        return _OPEN_PASSIVE_TENSION, dict(_OPEN_FLEXOR_TENSIONS)
    flexors = {solver_name: float(config.zero_bend_flexor_tensions[hardware_name])
               for solver_name, hardware_name in HARDWARE_FINGER_NAMES.items()
               if hardware_name in config.zero_bend_flexor_tensions}
    return float(config.zero_bend_passive_tension), flexors


def _open_pose_params(params, finger_names):
    """``params`` posed at the calibrated open hand, as a copy.

    ``finger_names`` is the solver's own digit order, since ``flexor_tensions`` is
    positional: keying the calibration by name and re-ordering it here is what
    keeps this correct if the hand is ever built with its fingers in another
    order, or with a digit missing. A digit the calibration says nothing about
    keeps whatever ``params`` holds for it.
    """
    passive, flexors = open_pose_tensions()
    tensions = [float(flexors.get(name, held))
                for name, held in zip(finger_names, params.flexor_tensions)]
    return replace(params, flexor_tensions=tensions, passive_tension=passive)


def check_open_lengths(open_lengths, params=None):
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
    params = params or solvers.HandSolveParams()
    n = len(params.flexor_tensions)
    probe = replace(params, flexor_tensions=[_FLEXION_PROBE_TENSION] * n)
    flexed = solvers.HandFKSolver(probe).solve()
    deltas = {name: open_lengths[name] - float(lengths[FLEXOR_IDX])
              for name, lengths in zip(flexed.finger_names, flexed.tendon_lengths(0))
              if name in open_lengths}
    worst = min(deltas.values()) if deltas else 0.0
    if worst <= 0.0:
        ok = False
        notes.append(
            f"**tendon sign check FAILED**: at {_FLEXION_PROBE_TENSION:g} N of "
            f"flexor tension the actuated tendon did not shorten on every finger "
            f"(worst {worst * 1e3:+.2f} mm). Playback would drive the hand the "
            f"wrong way; refusing to build a plan.")
    else:
        notes.append(
            f"tendon sign check: flexing at {_FLEXION_PROBE_TENSION:g} N pulls in "
            f"{min(deltas.values()) * 1e3:.1f}-{max(deltas.values()) * 1e3:.1f} mm "
            f"across the five digits (positive = pulled in, as commanded)")

    # -- the hardware's own numbers, if they are reachable --
    hardware = _hardware_open_lengths()
    if hardware is None:
        notes.append("_HandConfig unavailable -- open lengths not cross-checked "
                     "against the hardware calibration._")
        return notes, ok

    drift = {name: open_lengths[name] - hardware[name]
             for name in open_lengths if name in hardware}
    if drift:
        worst_name = max(drift, key=lambda k: abs(drift[k]))
        if abs(drift[worst_name]) > OPEN_LENGTH_WARN_M:
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
    """A ``HandConfig``, or None where gepetto_core is not installed.

    Optional on purpose: the visualizer runs on machines with no gepetto_core
    install, and a missing hardware config must degrade to a note (or to the
    fallback tensions above) rather than stop a plan being built."""
    try:
        try:
            from gepetto_core.config import HandConfig
        except ImportError:      # older installs expose the same package as `gepetto`
            from gepetto.config import HandConfig
    except ImportError:
        return None
    return HandConfig()


def _hardware_open_lengths():
    """``HandConfig.zero_bend_lengths`` keyed by SOLVER digit name, or None."""
    config = _hand_config()
    if config is None:
        return None
    out = {}
    for solver_name, hardware_name in HARDWARE_FINGER_NAMES.items():
        if hardware_name in config.finger_names:
            index = config.finger_names.index(hardware_name)
            out[solver_name] = float(config.zero_bend_lengths[index])
    return out


def hardware_travel_limits():
    """Per solver digit, the usable flexion travel ``(0.0, max_m)`` from
    ``HandConfig``, or None when gepetto_core is not installed.

    ``zero_bend - fully_flexed``: about 17.8 mm on index. A solve can easily ask
    for more than that -- the model has no motor -- so whoever publishes the plan
    clamps to this and says it did."""
    config = _hand_config()
    if config is None:
        return None
    out = {}
    for solver_name, hardware_name in HARDWARE_FINGER_NAMES.items():
        if hardware_name not in config.finger_names:
            continue
        index = config.finger_names.index(hardware_name)
        travel = float(config.zero_bend_lengths[index]
                       - config.fully_flexed_lengths[index])
        if travel > 0.0:
            out[solver_name] = (0.0, travel)
    return out


# Below this much overreach (metres) a clamp is not worth a note. The hand-open
# waypoint a phase-4 close starts from sits at a displacement of ~1e-16 m rather
# than a clean zero -- FK arithmetic, not a real command -- and the lower stop
# rounds it up, which reported as "asked for 0.0 mm beyond its 6.4 mm of travel"
# on every digit that was not even moving. Well under the 0.1 mm the note prints
# at, so nothing a reader could have acted on is being hidden.
_CLAMP_REPORT_M = 5e-5


def clamp_to_travel(plan, limits=None):
    """Clamp every displacement to the hardware's flexion travel, in place-ish.

    Returns ``(plan, notes)`` with a NEW plan (the caller's is untouched) and one
    note per finger that had to be clamped, naming how far past the stop the
    solve had asked for. A solve routinely asks for more travel than the motors
    have -- ``fully_flexed_lengths`` is max flexion at the measured torque limit,
    not the geometric limit -- and the servo node would saturate and warn once per
    finger anyway; clamping here means the interpolation still lands on a
    reachable target instead of spending its last seconds commanding a stop.
    """
    limits = hardware_travel_limits() if limits is None else limits
    if not limits:
        return plan, []

    worst: dict[str, float] = {}
    waypoints = []
    for waypoint in plan.waypoints:
        clamped = {}
        for name, value in waypoint.tendon_disp.items():
            lo, hi = limits.get(name, (None, None))
            if lo is None:
                clamped[name] = value
                continue
            bounded = min(max(value, lo), hi)
            if bounded != value:
                worst[name] = max(worst.get(name, 0.0), abs(value - bounded))
            clamped[name] = bounded
        waypoints.append(replace(waypoint, tendon_disp=clamped))

    notes = [f"**{name} clamped**: the solve asked for {excess * 1e3:.1f} mm "
             f"beyond its {limits[name][1] * 1e3:.1f} mm of travel"
             for name, excess in sorted(worst.items())
             if excess > _CLAMP_REPORT_M]
    return replace(plan, waypoints=waypoints), notes
