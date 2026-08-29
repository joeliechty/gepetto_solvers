"""Reusable FK / IK / trajectory-planner solver classes for the tendon hand.

This factors the shared *build -> solve -> extract* skeleton that the demo
scripts each re-implement inline into four classes behind one common base
(:class:`HandSolverBase`), driven by a single :class:`HandSolveParams` and
returning a uniform :class:`HandResult`.

The point is a *unified* way to call the three C++ solvers, so the interactive
visualizer -- and any future code -- can flip between FK, IK and the trajectory
planner without duplicating the setup boilerplate.

===========================  ==============================================
:class:`HandFKSolver`        pure kinematics, no contact
:class:`HandIKSolver`        single terminal grasp, one shot
:class:`HandIKStepper`       the same, one AL outer iteration per call
:class:`HandPlannerSolver`   a K+1-step trajectory with GP temporal priors
===========================  ==============================================

Layout, split out of what used to be one 3294-line module:

===================  ====================================================
:mod:`capabilities`  what the installed extension can do; ``_set_if``
:mod:`frames`        pose conventions and frame arithmetic
:mod:`scene_resolve` object placement, table seating, opposition sign
:mod:`witness`       residuals recomputed from the solved poses
:mod:`params`        :class:`HandSolveParams`
:mod:`result`        :class:`HandResult`
:mod:`presets`       the staged-pipeline phase presets
:mod:`motion`        synchronized close and lift (not solves)
:mod:`base`          :class:`HandSolverBase`
:mod:`fk` :mod:`ik` :mod:`stepper` :mod:`planner`   the four solvers
===================  ====================================================

**Independent readouts.** Every constraint family has a function in
:mod:`witness` that recomputes its residual from the *solved poses* rather than
reading the solver's own number, which is what makes a stall diagnosable.
:func:`orient_opposition_axis` is the one that is *not* a readout -- see its
docstring; every caller of :func:`default_half_space_axis` must apply it.

Every public name is re-exported here, so ``from ...core import solvers``
and ``from ...core.solvers import X`` behave exactly as before the split.
"""

# `object_extent_along` is pure spec geometry and lives in geometry.scene so the
# in-plane width sweep there can reach it, but callers have always said
# `solvers.object_extent_along`, so it is re-exported here.
from ..geometry.scene import object_extent_along  # noqa: E402  (kept last for clarity)
from .base import HandSolverBase
# `_set_if` is internal, but several docstrings point at `solvers._set_if` as the
# name of the stale-binding pattern, so it stays resolvable at that path.
from .capabilities import (
    _set_if as _set_if,
)
from .capabilities import (
    FLEXOR_IDX,
    NUM_FINGERS,
    capabilities,
    tendon_diag,
)
from .fk import HandFKSolver
from .frames import (
    DEFAULT_WRIST_RPY,
    DEFAULT_WRIST_XYZ,
    R_to_euler,
    default_wrist_pose,
    disc_frame_error,
    disc_pose,
    euler_to_R,
    solved_wrist_pose,
    wrist_pose_for_disc_target,
    wrist_pose_from_xyzrpy,
    wrist_to_disc,
)
from .ik import HandIKSolver
from .motion import (
    CLOSE_FRACTION,
    CLOSE_PROBE_STEP,
    CLOSE_REFINE,
    CLOSE_STEPS,
    CLOSE_TOL_M,
    LIFT_HEIGHT_M,
    LIFT_STEPS,
    lift_wrist,
    synchronized_close,
)
from .params import HandSolveParams
from .planner import HandPlannerSolver
from .presets import PHASE_PRESETS, PhasePreset, apply_phase_preset
from .result import HandResult
from .scene_resolve import (
    auto_table_origin,
    default_half_space_axis,
    default_object_center,
    orient_opposition_axis,
    resolve_constraint_plane_origin,
    resolve_scene,
    resolve_table_origin,
)
from .stepper import HandIKStepper, StepStatus
from .witness import (
    PlanarGap,
    finger_plane_witness,
    free_sphere_plane_witness,
    half_space_witness,
    planar_gap_witness,
    plane_witness,
    pregrasp_axis_witness,
    pregrasp_center_witness,
    pregrasp_centroid_witness,
    tip_gap_matrix,
)

#: The three one-shot solver classes, by the name the CLIs select them with.
SOLVERS = {
    "FK": HandFKSolver,
    "IK": HandIKSolver,
    "Planner": HandPlannerSolver,
}

__all__ = [
    # solver classes
    "HandFKSolver",
    "HandIKSolver",
    "HandIKStepper",
    "HandPlannerSolver",
    "HandSolverBase",
    "SOLVERS",
    "StepStatus",
    # params / result / presets
    "HandSolveParams",
    "HandResult",
    "PHASE_PRESETS",
    "PhasePreset",
    "apply_phase_preset",
    # capabilities
    "FLEXOR_IDX",
    "NUM_FINGERS",
    "capabilities",
    "tendon_diag",
    # frames
    "DEFAULT_WRIST_RPY",
    "DEFAULT_WRIST_XYZ",
    "R_to_euler",
    "default_wrist_pose",
    "disc_frame_error",
    "disc_pose",
    "euler_to_R",
    "solved_wrist_pose",
    "wrist_pose_for_disc_target",
    "wrist_pose_from_xyzrpy",
    "wrist_to_disc",
    # scene resolution
    "auto_table_origin",
    "default_half_space_axis",
    "default_object_center",
    "object_extent_along",
    "orient_opposition_axis",
    "resolve_constraint_plane_origin",
    "resolve_scene",
    "resolve_table_origin",
    # witnesses
    "PlanarGap",
    "finger_plane_witness",
    "free_sphere_plane_witness",
    "half_space_witness",
    "planar_gap_witness",
    "plane_witness",
    "pregrasp_axis_witness",
    "pregrasp_center_witness",
    "pregrasp_centroid_witness",
    "tip_gap_matrix",
    # motion
    "CLOSE_FRACTION",
    "CLOSE_PROBE_STEP",
    "CLOSE_REFINE",
    "CLOSE_STEPS",
    "CLOSE_TOL_M",
    "LIFT_HEIGHT_M",
    "LIFT_STEPS",
    "lift_wrist",
    "synchronized_close",
]
