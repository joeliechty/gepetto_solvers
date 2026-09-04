"""Turn a solve into something a robot can execute: waypoints, then samples.

The visualizer solves in whatever poses a hand -- *tensions* for the tendon hand,
*joint targets* for a joint-space one -- and reports *states*; the robot wants a
stream of *commands* to its driven actuators plus a *pose stream*. This package is
the conversion, and it is deliberately ROS-free and viser-free -- pure numpy in,
pure numpy out -- so it can be exercised headlessly (``viz_interactive --smoke``)
and so ``gepetto_solvers`` never grows a dependency on rclpy. The ROS side
(``gepetto_ros``) imports this; nothing here imports the ROS side.

ONE HAND-DEPENDENT DECISION, MADE ONCE, IN :mod:`building`. A waypoint carries one
vector of ``len(hand.actuation.drive_indices)`` numbers per digit --  width 1 in
metres of tendon displacement for the tendon hand, width 4 in radians of joint
position for the Allegro -- and ``SolvePlan.command_kind`` says which. Everything
after that point (pacing, scheduling, interpolation, the wire format) treats a
digit command as a point in R^K and never asks what the numbers mean.

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

SIGN, for a tendon plan: positive displacement = tendon pulled in = FLEXING,
measured from the hand-open pose. That matches ``finger_servo_node``'s
``~/delta_tendon_cmds`` and its state topics, and ``finger_slider_node``'s mm
readout. It is the OPPOSITE of the raw motor counts (flexing decreases the count
on this hardware). A joint plan has no such convention to get wrong: it commands
absolute positions in the URDF's own sense.

Layout, split out of what used to be one 879-line module:

===============  =======================================================
:mod:`types`     ``Waypoint`` / ``Sample`` / ``SolvePlan`` / ``PathSchedule``
:mod:`se3`       SO(3) and se(3) on bare numpy
:mod:`hardware`  the hand-open pose, travel limits, clamping
:mod:`pacing`    how long each segment must take
:mod:`schedule`  waypoints + ceilings -> a fixed-rate sample stream
:mod:`building`  a solve -> a plan, and describing one
:mod:`wire`      a plan <-> flat arrays, for a transport to carry
===============  =======================================================

The names below are the module's public surface and are re-exported here, so
``from gepetto_solvers.core import robot_plan`` keeps working exactly as before.
"""

from .building import build_plan, command_kind, prepend_current, summarize
from .hardware import (
    check_open_lengths,
    clamp_to_travel,
    hardware_travel_limits,
    joint_travel_limits,
    open_pose_tensions,
    open_tendon_lengths,
    travel_limits,
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
from .types import (
    COMMAND_UNITS,
    JOINT_POSITION_RAD,
    TENDON_DISPLACEMENT_M,
    PathSchedule,
    Sample,
    SolvePlan,
    Waypoint,
    as_command,
)
from .wire import FIELDS, flatten_plan, unflatten_plan

__all__ = [
    # types
    "COMMAND_UNITS",
    "JOINT_POSITION_RAD",
    "PathSchedule",
    "Sample",
    "SolvePlan",
    "TENDON_DISPLACEMENT_M",
    "Waypoint",
    "as_command",
    # se(3)
    "se3_adjoint",
    "se3_exp",
    "se3_log",
    # hardware
    "check_open_lengths",
    "clamp_to_travel",
    "hardware_travel_limits",
    "joint_travel_limits",
    "open_pose_tensions",
    "open_tendon_lengths",
    "travel_limits",
    # pacing / scheduling
    "describe_pacing",
    "interpolate",
    "pacing_summary",
    "plan_schedule",
    "sample_at",
    "segment_durations",
    # building
    "build_plan",
    "command_kind",
    "prepend_current",
    "summarize",
    # wire
    "FIELDS",
    "flatten_plan",
    "unflatten_plan",
]
