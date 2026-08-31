"""The four data structures a plan is made of, and the flexor index.

Deliberately dependency-free: a plan is numpy and dataclasses, nothing more, so
the ROS side can import these without dragging in a solver.
"""

from dataclasses import dataclass, field

import numpy as np

#: Index of the actuated flexor in a finger's tendon-length vector. Duplicated
#: from `solvers` rather than imported so that the TIMING half of this module --
#: `plan_schedule`, `sample_at`, `interpolate` -- needs nothing but numpy. See
#: `_solvers` below, which checks the two still agree.
FLEXOR_IDX = 5


@dataclass
class Waypoint:
    """One solve state, in the units the robot is commanded in.

    ``wrist_pose`` is in the VISER WORLD frame -- the plan carries
    ``corner_viz`` so the consumer can register that frame against the physical
    bench, rather than this module guessing at a robot frame it knows nothing
    about.
    """
    wrist_pose: np.ndarray                  # 4x4, viser world frame
    tendon_disp: dict[str, float]           # solver finger name -> metres, + = flexing
    note: str = ""                          # the iterate's own status line, if any


@dataclass
class Sample:
    """One control tick's worth of command, with the feed-forward that produced
    it. A resolved-rate controller wants both: the pose to servo toward and the
    velocity the reference itself is moving at."""
    t: float                                # seconds from the start of playback
    wrist_pose: np.ndarray                  # 4x4, viser world frame
    tendon_disp: dict[str, float]           # metres, + = flexing
    #: Feed-forward as a BODY twist [v(3) m/s, w(3) rad/s], in the wrist's own
    #: frame -- so it needs no rotation when the reference pose is mapped into the
    #: robot base frame. Zero once the path has ended. See :class:`PathSchedule`.
    body_twist: np.ndarray
    waypoint: int = 0                       # which waypoint this tick is heading to


@dataclass
class SolvePlan:
    """A whole solve, ready to be registered against the robot and executed."""
    waypoints: list[Waypoint]
    #: Position of the viser table square's minimum corner, in the viser world
    #: frame. Paired with the physical corner (``lbr_workspace_table_link``) this
    #: is the registration between the two worlds -- see the ROS-side bridge.
    corner_viz: np.ndarray
    #: Solver digit names carrying a displacement, in solver order.
    finger_names: list[str]
    #: Per-finger hand-open reference length (m), the zero of every displacement.
    open_lengths: dict[str, float]
    #: Human-readable notes from the build (open-length cross-check, sign check).
    notes: list[str] = field(default_factory=list)

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
