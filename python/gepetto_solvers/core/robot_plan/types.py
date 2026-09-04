"""The four data structures a plan is made of.

Deliberately dependency-free: a plan is numpy and dataclasses, nothing more, so
the ROS side can import these without dragging in a solver.

ONE VECTOR PER DIGIT, WHATEVER DRIVES IT. A waypoint carries the wrist pose plus
``len(hand.actuation.drive_indices)`` numbers per digit -- the DRIVEN actuators
and nothing else. The tendon hand drives one (the flexor), so its vectors are
width 1 in metres of tendon displacement; the Allegro drives four, so its vectors
are width 4 in radians of joint position. ``SolvePlan.command_kind`` says which,
and it is the only thing downstream has to branch on.

That width is what makes the timing half of this package hand-agnostic: pacing,
scheduling and interpolation all treat a digit command as a point in R^K and never
ask what the numbers mean. The actuated INDEX that used to be duplicated here is
gone for the same reason -- it is a property of the hand
(``hand.actuation.drive_indices``), read only by the half of this package that
already imports a solver.
"""

from dataclasses import dataclass, field

import numpy as np

#: ``command_kind`` values. A plan says which units its digit vectors are in, so
#: a consumer can label a plot axis or a log line without knowing the hand.
TENDON_DISPLACEMENT_M = "tendon_displacement_m"
JOINT_POSITION_RAD = "joint_position_rad"

#: Human units per kind, for status lines and plot titles: (suffix, scale).
#: Multiply a stored value by the scale to get the displayed number.
COMMAND_UNITS = {
    TENDON_DISPLACEMENT_M: ("mm", 1e3),
    JOINT_POSITION_RAD: ("rad", 1.0),
}


def as_command(value) -> np.ndarray:
    """Coerce one digit's command to a 1-D float array.

    Accepts a bare scalar so a width-1 caller can keep writing ``0.0`` -- which
    ``prepend_current``'s callers and every tendon-era test do -- without every
    one of them having to learn that a command is a vector now.
    """
    return np.atleast_1d(np.asarray(value, float)).ravel()


def _scalar_view(digit_cmd) -> dict[str, float]:
    """``{digit: scalar}`` for a width-1 command map.

    Kept because on a tendon hand "the tendon displacement" is a single number
    and the readouts that print it say so. Raises on a wider command rather than
    silently returning the first joint, which is the rule
    ``Actuation.drive_value`` already follows.
    """
    out = {}
    for name, value in digit_cmd.items():
        if value.size != 1:
            raise ValueError(
                f"tendon_disp: digit {name!r} carries {value.size} commanded "
                f"values, so there is no single displacement to read; use "
                f"digit_cmd.")
        out[name] = float(value[0])
    return out


@dataclass
class Waypoint:
    """One solve state, in the units the robot is commanded in.

    ``wrist_pose`` is in the VISER WORLD frame -- the plan carries
    ``corner_viz`` so the consumer can register that frame against the physical
    bench, rather than this module guessing at a robot frame it knows nothing
    about.
    """
    wrist_pose: np.ndarray                  # 4x4, viser world frame
    #: solver digit name -> (K,) driven-actuator command. Tendon: 1 value, metres,
    #: + = pulled in = flexing. Joint-space: one value per joint, radians.
    digit_cmd: dict[str, np.ndarray]
    note: str = ""                          # the iterate's own status line, if any

    def __post_init__(self):
        self.digit_cmd = {name: as_command(value)
                          for name, value in self.digit_cmd.items()}

    @property
    def tendon_disp(self) -> dict[str, float]:
        """The width-1 view: ``{digit: scalar}``. Read-only; see
        :func:`_scalar_view`."""
        return _scalar_view(self.digit_cmd)


@dataclass
class Sample:
    """One control tick's worth of command, with the feed-forward that produced
    it. A resolved-rate controller wants both: the pose to servo toward and the
    velocity the reference itself is moving at."""
    t: float                                # seconds from the start of playback
    wrist_pose: np.ndarray                  # 4x4, viser world frame
    digit_cmd: dict[str, np.ndarray]        # as on Waypoint
    #: Feed-forward as a BODY twist [v(3) m/s, w(3) rad/s], in the wrist's own
    #: frame -- so it needs no rotation when the reference pose is mapped into the
    #: robot base frame. Zero once the path has ended. See :class:`PathSchedule`.
    body_twist: np.ndarray
    waypoint: int = 0                       # which waypoint this tick is heading to

    def __post_init__(self):
        self.digit_cmd = {name: as_command(value)
                          for name, value in self.digit_cmd.items()}

    @property
    def tendon_disp(self) -> dict[str, float]:
        """The width-1 view; see :func:`_scalar_view`."""
        return _scalar_view(self.digit_cmd)


@dataclass
class SolvePlan:
    """A whole solve, ready to be registered against the robot and executed."""
    waypoints: list[Waypoint]
    #: Position of the viser table square's minimum corner, in the viser world
    #: frame. Paired with the physical corner (``lbr_workspace_table_link``) this
    #: is the registration between the two worlds -- see the ROS-side bridge.
    corner_viz: np.ndarray
    #: Solver digit names carrying a command, in solver order.
    digit_names: list[str]
    #: Per-digit hand-open reference length (m), the zero of every displacement.
    #: Tendon hands only; empty on a joint-space plan, which has no such zero.
    open_lengths: dict[str, float]
    #: What the digit vectors MEAN: one of :data:`TENDON_DISPLACEMENT_M` or
    #: :data:`JOINT_POSITION_RAD`. The consumer's only branch.
    command_kind: str = TENDON_DISPLACEMENT_M
    #: Human-readable notes from the build (open-length cross-check, sign check).
    notes: list[str] = field(default_factory=list)

    @property
    def dof_per_digit(self) -> int:
        """Width of every digit vector in this plan.

        Derived rather than stored: a stored copy is one more thing that can
        disagree with the arrays it describes, and a ragged plan must fail here
        rather than on the wire.
        """
        widths = {int(value.size)
                  for waypoint in self.waypoints
                  for value in waypoint.digit_cmd.values()}
        if not widths:
            return 0
        if len(widths) > 1:
            raise ValueError(
                f"ragged plan: digit commands of widths {sorted(widths)}. Every "
                f"digit of every waypoint must carry the same number of driven "
                f"actuators.")
        return widths.pop()

    def duration_hint(self):
        """Waypoint count, for a status line. The real duration is not known
        until :func:`interpolate` applies the speed ceilings."""
        return len(self.waypoints)


@dataclass
class PathSchedule:
    """A plan's timing, precomputed once so the path can be sampled at any ``t``.

    The executor advances its own clock and asks "where should the wrist be at
    time t" each tick, so the schedule has to be samplable at arbitrary ``t``
    rather than only on a fixed grid; that is what this plus :func:`sample_at`
    answer. :func:`interpolate` is then just a walk of the same pair over a grid.

    Durations are QUANTIZED to whole control periods. Two reasons: it makes
    :func:`interpolate` exactly a walk of :func:`sample_at` over the grid, so the
    two can never drift apart, and it makes the per-segment feed-forward twist the
    rate the target actually moves at rather than the rate it was asked to move
    at -- which is the number a resolved-rate controller is fed.

    The feed-forward is ONE BODY TWIST per segment, not a separated linear and
    angular pair in the plan's frame. That is what makes it the exact derivative
    of the reference :func:`sample_at` walks -- a constant body twist integrates
    to ``T_k @ se3_exp(V * t)``, which is the reference -- rather than merely
    agreeing with it at the segment edges. It also needs no frame rotation
    downstream: a body twist is expressed in the wrist's own frame, which is the
    same frame whether the plan is written in viser or robot-base coordinates.
    """
    durations: list[float]              # per segment, seconds, whole periods
    edges: np.ndarray                   # segment start times, len = n_seg + 1
    total: float                        # seconds
    #: Per segment, the constant body twist [v(3) m/s, w(3) rad/s] whose flow for
    #: `durations[k]` carries waypoint k exactly onto waypoint k+1.
    body_twist: list[np.ndarray]
