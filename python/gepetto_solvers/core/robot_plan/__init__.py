"""Turn a solve into something a robot can execute: waypoints, then samples.

The visualizer solves in *tensions* and reports *states*; the hardware wants
*tendon displacements* and a *pose stream*. This package is the conversion, and it
is deliberately ROS-free and viser-free -- pure numpy in, pure numpy out -- so it
can be exercised headlessly (``viz_interactive --smoke``) and so ``gepetto_solvers``
never grows a dependency on rclpy. The ROS side (``epfl_hand_control``) imports
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

SIGN, everywhere in this package: positive tendon displacement = tendon pulled in
= FLEXING, measured from the hand-open pose. That matches
``finger_servo_node``'s ``~/delta_tendon_cmds`` and its state topics, and
``finger_slider_node``'s mm readout. It is the OPPOSITE of the raw motor counts
(flexing decreases the count on this hardware).

Layout, split out of what used to be one 879-line module:

===============  =======================================================
:mod:`types`     ``Waypoint`` / ``Sample`` / ``SolvePlan`` / ``PathSchedule``
:mod:`se3`       SO(3) and se(3) on bare numpy
:mod:`hardware`  the hand-open pose, travel limits, clamping
:mod:`pacing`    how long each segment must take
:mod:`schedule`  waypoints + ceilings -> a fixed-rate sample stream
:mod:`building`  a solve -> a plan, and describing one
===============  =======================================================

The names below are the module's public surface and are re-exported here, so
``from gepetto_solvers.core import robot_plan`` keeps working exactly as before.
"""

from .building import build_plan, prepend_current, summarize
from .hardware import (
    check_open_lengths,
    clamp_to_travel,
    hardware_travel_limits,
    open_pose_tensions,
    open_tendon_lengths,
)
from .pacing import describe_pacing, pacing_summary, segment_durations
from .schedule import interpolate, plan_schedule, sample_at

# The two underscore-prefixed rotation helpers are re-exported on purpose: the
# timing tests and viz_interactive both reach for them by their original names,
# which were module-level when this was one file.
from .se3 import (
    _rotation_error as _rotation_error,
)
from .se3 import (
    _rotation_from_vector as _rotation_from_vector,
)
from .se3 import (
    se3_adjoint,
    se3_exp,
    se3_log,
)
from .types import PathSchedule, Sample, SolvePlan, Waypoint

__all__ = [
    # types
    "PathSchedule",
    "Sample",
    "SolvePlan",
    "Waypoint",
    # se(3)
    "se3_adjoint",
    "se3_exp",
    "se3_log",
    # hardware
    "check_open_lengths",
    "clamp_to_travel",
    "hardware_travel_limits",
    "open_pose_tensions",
    "open_tendon_lengths",
    # pacing / scheduling
    "describe_pacing",
    "interpolate",
    "pacing_summary",
    "plan_schedule",
    "sample_at",
    "segment_durations",
    # building
    "build_plan",
    "prepend_current",
    "summarize",
]
