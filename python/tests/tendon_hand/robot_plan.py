"""Turn a solve into something a robot can execute: waypoints, then samples.

The visualizer solves in *tensions* and reports *states*; the hardware wants
*tendon displacements* and a *pose stream*. This module is the conversion, and it
is deliberately ROS-free and viser-free -- pure numpy in, pure numpy out -- so it
can be exercised headlessly (``viz_interactive --smoke``) and so ``crest-sparse``
never grows a dependency on rclpy. The ROS side (``gepetto_control``) imports
this; nothing here imports the ROS side.

Two stages, and they are separate on purpose:

**build_plan** samples the solve. One :class:`Waypoint` per recorded Augmented
Lagrangian OUTER ITERATION -- the same snapshots the *Solve steps* scrubber
replays -- carrying the wrist pose the iterate actually reached and the tendon
displacement each finger was holding. Nothing is interpolated and nothing is
timed: a plan is a path through configuration space, and it says nothing about
how fast to walk it.

**interpolate** times it. Each segment gets the duration its slowest channel
needs at the configured speed ceilings, positions lerp, rotations slerp, tendons
lerp. The output is a list of :class:`Sample` at a fixed rate, ready to be fed a
tick at a time to a servo publisher.

THE ITERATES ARE OPTIMIZER ITERATIONS, NOT A PLANNED PATH. They converge to a
grasp; they do not promise to stay collision-free or monotonic on the way, and a
cold start with ``ik_settle_steps = 0`` visibly hyperextends before it recovers
(see ``_IK_SETTLE_TENSION_COV`` in solvers.py). ``source="final"`` exists for
when you want the destination without the journey.

SIGN, everywhere in this module: positive tendon displacement = tendon pulled in
= FLEXING, measured from the hand-open pose. That matches
``finger_servo_node``'s ``~/delta_tendon_cmds`` and its state topics, and
``finger_slider_node``'s mm readout. It is the OPPOSITE of the raw motor counts
(flexing decreases the count on this hardware).
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from .solvers import (FLEXOR_IDX, HandFKSolver, HandSolveParams,
                      solved_wrist_pose)


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
    "index": 0.85, "middle": 0.80, "ring": 0.90, "pinky": 0.95, "thumb": 0.85,
}


@dataclass
class Waypoint:
    """One solve state, in the units the robot is commanded in.

    ``wrist_pose`` is in the VISER WORLD frame -- the plan carries
    ``corner_viz`` so the consumer can register that frame against the physical
    bench, rather than this module guessing at a robot frame it knows nothing
    about.
    """
    wrist_pose: np.ndarray                  # 4x4, viser world frame
    tendon_disp: Dict[str, float]           # solver finger name -> metres, + = flexing
    note: str = ""                          # the iterate's own status line, if any


@dataclass
class Sample:
    """One control tick's worth of command, with the feed-forward rates that
    produced it. A resolved-rate controller wants both: the pose to servo toward
    and the velocity the path itself is moving at."""
    t: float                                # seconds from the start of playback
    wrist_pose: np.ndarray                  # 4x4, viser world frame
    tendon_disp: Dict[str, float]           # metres, + = flexing
    linear_velocity: np.ndarray             # m/s, viser world frame
    angular_velocity: np.ndarray            # rad/s, viser world frame
    waypoint: int = 0                       # which waypoint this tick is heading to


@dataclass
class SolvePlan:
    """A whole solve, ready to be registered against the robot and executed."""
    waypoints: List[Waypoint]
    #: Position of the viser table square's minimum corner, in the viser world
    #: frame. Paired with the physical corner (``lbr_workspace_table_link``) this
    #: is the registration between the two worlds -- see the ROS-side bridge.
    corner_viz: np.ndarray
    #: Solver digit names carrying a displacement, in solver order.
    finger_names: List[str]
    #: Per-finger hand-open reference length (m), the zero of every displacement.
    open_lengths: Dict[str, float]
    #: Human-readable notes from the build (open-length cross-check, sign check).
    notes: List[str] = field(default_factory=list)

    def duration_hint(self):
        """Waypoint count, for a status line. The real duration is not known
        until :func:`interpolate` applies the speed ceilings."""
        return len(self.waypoints)


def open_tendon_lengths(params: Optional[HandSolveParams] = None,
                        solver: Optional[HandFKSolver] = None):
    """Per-finger actuated-tendon length with the hand OPEN, from the model itself.

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
    params = params or HandSolveParams()
    if solver is None:
        solver = HandFKSolver(replace(params))
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
    params = params or HandSolveParams()
    n = len(params.flexor_tensions)
    probe = replace(params, flexor_tensions=[_FLEXION_PROBE_TENSION] * n)
    flexed = HandFKSolver(probe).solve()
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


def build_plan(result, configs, corner_viz, open_lengths, source="history",
               start=0):
    """A :class:`SolvePlan` from a solved :class:`~.solvers.HandResult`.

    ``source``:
      ``"history"``  one waypoint per recorded AL outer iteration, in order.
      ``"final"``    the converged state alone, as a single waypoint.

    ``start`` skips leading iterates (the scrubber's index, for "play from here"),
    and is ignored by ``"final"``. A result with no recorded iterates -- an FK
    pose, or a solve that never stepped -- yields the single final waypoint
    whatever ``source`` says, because there is no history to play.

    ``configs`` is the solver's ``(name, cfg)`` list, needed by
    :func:`~.solvers.solved_wrist_pose` to recover the wrist the solve actually
    reached (the wrist is a variable, and contact moves it off the commanded pose).
    """
    n = result.num_iterates()
    if source == "final" or n <= 1:
        views = [result]
        notes = ["converged state"]
    else:
        start = int(np.clip(start, 0, n - 1))
        views = [result.at_iterate(i) for i in range(start, n)]
        raw = result.iterate_notes
        notes = [(raw[i] if raw is not None and i < len(raw) else f"iterate {i}")
                 for i in range(start, n)]

    waypoints = []
    for view, note in zip(views, notes):
        lengths = view.tendon_lengths(0)
        waypoints.append(Waypoint(
            wrist_pose=np.asarray(solved_wrist_pose(configs, view.frames[0]), float),
            tendon_disp={name: open_lengths[name] - float(length[FLEXOR_IDX])
                         for name, length in zip(view.finger_names, lengths)
                         if name in open_lengths},
            note=note))

    return SolvePlan(waypoints=waypoints,
                     corner_viz=np.asarray(corner_viz, float).reshape(3),
                     finger_names=list(result.finger_names),
                     open_lengths=dict(open_lengths))


# ---------------------------------------------------------------------------
# Timing: waypoints -> a fixed-rate sample stream.
# ---------------------------------------------------------------------------

def _rotation_error(R_to, R_from):
    """The rotation vector (axis * angle, rad) taking ``R_from`` to ``R_to``.

    Hand-rolled rather than pulled from scipy so this module has no dependency
    beyond numpy: the whole point of it being pure is that the ROS node and the
    headless smoke test can both import it. The clip guards the arccos against
    a trace a hair outside [-1, 1] from floating-point error, which is otherwise
    a NaN on the identity rotation -- exactly the case a converged solve hits.
    """
    R = np.asarray(R_to, float) @ np.asarray(R_from, float).T
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


def _rotation_from_vector(rotvec):
    """Rodrigues: a rotation vector back to a 3x3. Inverse of _rotation_error."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(rotvec, float) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _slerp(R_a, R_b, s):
    """Geodesic interpolation between two rotations, ``s`` in [0, 1]. Exact
    (not a normalized lerp): it walks the constant-rate path scipy's Slerp walks,
    via the same log/exp pair used for the pose error above."""
    return _rotation_from_vector(s * _rotation_error(R_b, R_a)) @ np.asarray(R_a, float)


def segment_durations(plan, max_linear, max_angular, max_tendon, min_duration=0.05):
    """How long each waypoint-to-waypoint segment must take, in seconds.

    The slowest channel sets the pace: a segment is only as fast as its linear
    travel, its rotation and its tendon travel each allow. That coupling is the
    point -- running the wrist at full speed while the fingers lag would put the
    hand at the object with the grasp half closed.

    ``min_duration`` keeps a segment that barely moves from collapsing to zero
    ticks; consecutive AL iterates late in a converging solve differ by microns.
    """
    durations = []
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        linear = float(np.linalg.norm(b.wrist_pose[:3, 3] - a.wrist_pose[:3, 3]))
        angular = float(np.linalg.norm(
            _rotation_error(b.wrist_pose[:3, :3], a.wrist_pose[:3, :3])))
        tendon = max((abs(b.tendon_disp[name] - a.tendon_disp.get(name, 0.0))
                      for name in b.tendon_disp), default=0.0)
        durations.append(max(min_duration,
                             linear / max_linear if max_linear > 0 else 0.0,
                             angular / max_angular if max_angular > 0 else 0.0,
                             tendon / max_tendon if max_tendon > 0 else 0.0))
    return durations


def interpolate(plan, hz=100.0, max_linear=0.2, max_angular=0.4, max_tendon=0.0163,
                min_duration=0.05):
    """Time the plan and sample it at ``hz``, with feed-forward rates.

    Speed arguments are CEILINGS, not setpoints -- see :func:`segment_durations`.
    Their defaults are the fractions the visualizer opens on: 50% of MoveIt
    Servo's ``scale.linear``/``scale.rotational`` (0.4 m/s, 0.8 rad/s) and 25% of
    ``HandConfig.max_tendon_speed`` (0.065 m/s).

    A single-waypoint plan yields one sample with zero velocity: "go here", which
    is what a resolved-rate controller needs to servo to a static target.
    """
    if not plan.waypoints:
        return []
    if len(plan.waypoints) == 1:
        w = plan.waypoints[0]
        return [Sample(0.0, np.asarray(w.wrist_pose, float), dict(w.tendon_disp),
                       np.zeros(3), np.zeros(3), 0)]

    period = 1.0 / float(hz)
    durations = segment_durations(plan, max_linear, max_angular, max_tendon,
                                  min_duration)
    samples, t0 = [], 0.0

    for k, (a, b) in enumerate(zip(plan.waypoints, plan.waypoints[1:])):
        duration = durations[k]
        # Rates are constant within a segment (the interpolation is linear in s),
        # so they are computed once per segment rather than per tick.
        p_a, p_b = a.wrist_pose[:3, 3], b.wrist_pose[:3, 3]
        rotvec = _rotation_error(b.wrist_pose[:3, :3], a.wrist_pose[:3, :3])
        linear_velocity = (p_b - p_a) / duration
        angular_velocity = rotvec / duration

        # The last tick of a segment is dropped -- it is the first tick of the
        # next one, and emitting both would stall a tick on every waypoint. The
        # final waypoint is appended once, after the loop.
        ticks = max(1, int(round(duration / period)))
        for i in range(ticks):
            s = i / ticks
            T = np.eye(4)
            T[:3, :3] = _slerp(a.wrist_pose[:3, :3], b.wrist_pose[:3, :3], s)
            T[:3, 3] = p_a + s * (p_b - p_a)
            samples.append(Sample(
                t=t0 + i * period,
                wrist_pose=T,
                tendon_disp={name: (a.tendon_disp.get(name, 0.0)
                                    + s * (value - a.tendon_disp.get(name, 0.0)))
                             for name, value in b.tendon_disp.items()},
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                waypoint=k + 1))
        t0 += ticks * period

    last = plan.waypoints[-1]
    samples.append(Sample(t0, np.asarray(last.wrist_pose, float),
                          dict(last.tendon_disp), np.zeros(3), np.zeros(3),
                          len(plan.waypoints) - 1))
    return samples


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

    worst, waypoints = {}, []
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
             for name, excess in sorted(worst.items())]
    return replace(plan, waypoints=waypoints), notes


def prepend_current(plan, wrist_pose, tendon_disp):
    """Put the robot's CURRENT state on the front of the plan as waypoint 0.

    Without this the first tick is a step change. A plan's own first waypoint is
    not where the robot is: it is the solve's initial guess, which already carries
    whatever flexor tension the sliders were commanding (~2 mm of displacement on
    a default grasp scene) and a wrist at the commanded start pose. Playing it
    cold asks the hand to be somewhere it is not, instantly -- which the tendon
    node absorbs as one saturated ramp and the arm's resolved-rate loop absorbs as
    a burst of maximum twist.

    Prepending the measured state instead makes the approach an ordinary segment,
    so it gets a duration from the same speed ceilings as every other one and the
    hand moves onto the start of the solve at a controlled rate.

    ``tendon_disp`` may name only some of the digits (a finger that failed to read
    is left out of ``measured_state``); anything missing falls back to the plan's
    own first waypoint, i.e. that finger is assumed to be where the plan wants it
    and simply is not moved by the approach segment.
    """
    if not plan.waypoints:
        return plan
    first = plan.waypoints[0]
    current = Waypoint(
        wrist_pose=np.asarray(wrist_pose, float),
        tendon_disp={name: float(tendon_disp.get(name, value))
                     for name, value in first.tendon_disp.items()},
        note="current robot state")
    return replace(plan, waypoints=[current] + list(plan.waypoints))


def summarize(plan, samples=None):
    """A one-paragraph markdown description of a plan, for the GUI status line."""
    lines = [f"**{len(plan.waypoints)} waypoint(s)**"]
    if len(plan.waypoints) >= 2:
        first, last = plan.waypoints[0], plan.waypoints[-1]
        travel = float(np.linalg.norm(last.wrist_pose[:3, 3] - first.wrist_pose[:3, 3]))
        rotation = float(np.linalg.norm(
            _rotation_error(last.wrist_pose[:3, :3], first.wrist_pose[:3, :3])))
        tendon = max((abs(last.tendon_disp[name] - first.tendon_disp.get(name, 0.0))
                      for name in last.tendon_disp), default=0.0)
        lines.append(f"wrist {travel * 1e3:.0f} mm / {np.degrees(rotation):.0f}°, "
                     f"tendon up to {tendon * 1e3:.1f} mm")
    if samples:
        lines.append(f"{len(samples)} ticks, {samples[-1].t:.1f} s")
    return " &nbsp; ".join(lines)
