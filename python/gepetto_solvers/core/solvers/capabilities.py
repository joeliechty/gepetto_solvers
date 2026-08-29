"""What the installed extension can do, and the two constants everything shares.

The installed ``.so`` routinely lags the C++ source, so every newer config field
is set through :func:`_set_if` and :func:`capabilities` tells a caller -- chiefly
the visualizers -- which controls to grey out instead of crashing. A control that
silently does nothing is almost always a False here.
"""

import numpy as np

import gepetto_solvers

from ..objects import OBJECTS_DIR

#: The ``objects/`` directory holding the baked .vdb SDF grids.
_OBJECTS_DIR = OBJECTS_DIR

# Anatomical hand digit count / config order: index, middle, ring, pinky, thumb.
NUM_FINGERS = 5


# The flexor tendon is index 5 in the 6-tendon anatomical routing (scene.TENDON_NAMES).
FLEXOR_IDX = 5


def tendon_diag(passive, active, n=6, idx=FLEXOR_IDX):
    """Diagonal tendon covariance with a distinct entry for the ACTUATED tendon.

    Every tendon prior in this hand is anisotropic in the same way: five
    spring-backed passives that behave one way and one motor-driven flexor that
    behaves another. Writing that as one helper keeps the split from drifting
    apart between the priors that have to agree about it.
    """
    d = np.full(int(n), float(passive))
    d[idx] = float(active)
    return np.diag(d)


def _set_if(obj, name, value):
    """Set ``obj.name = value`` only if the binding exposes that field.

    The installed ``_gepetto_solvers`` extension can lag the C++ source: newer config
    fields (inexact-AL tolerances, slide-grasp ``k_touch``, the analytic-ellipsoid
    / table env fields) may be absent until the module is rebuilt. Guarding keeps
    the solvers working on the current binary and picks the fields up automatically
    once it is rebuilt."""
    if hasattr(obj, name):
        setattr(obj, name, value)
        return True
    return False


def capabilities():
    """What the *installed* binding supports, so callers (the visualizer) can gate
    unsupported controls instead of crashing on a stale build."""
    env = gepetto_solvers.EnvironmentConfig()
    pc = gepetto_solvers.TendonHandTrajectoryPlannerConfig()
    fc = gepetto_solvers.TendonFingerSolverConfig()
    return {
        "ellipsoid": hasattr(env, "ellipsoid_semi_axes"),
        # Section 1.2 ellipsoid SETS (the YCB objects). Gated separately from
        # "ellipsoid" because the set factor landed much later than the single
        # one, so a binding can easily have one and not the other.
        "ellipsoid_set": hasattr(env, "ellipsoid_set"),
        # Narrowing an ellipsoid set's CONTACT targets to the authored grasp
        # subset. Gated apart from "ellipsoid_set" again: a binding that can load
        # a YCB object cannot necessarily narrow it, and offering the choice on
        # one that cannot would contact every shell while claiming otherwise.
        "grasp_subset": hasattr(env, "contact_ellipsoid_subset"),
        "table": hasattr(env, "plane_normal"),
        "collision_cull": hasattr(env, "collision_cull_margin"),
        "k_touch": hasattr(pc, "k_touch"),
        # Per-iteration solve snapshots off the single-shot solver, so the
        # visualizer can scrub an IK solve's convergence. The trajectory planner
        # has had this for longer; TendonHandSolver only gained it with the
        # iterate scrubber, so a stale binding must not offer the control.
        "solve_iterates": hasattr(gepetto_solvers.TendonHandSolver,
                                  "get_intermediate_solutions"),
        # Driving the AL outer loop one iteration at a time (HandIKStepper).
        # Probed via reset_al_duals because the other half of that build --
        # TendonHandSolver honoring skip_marginals -- is a behavior change no
        # hasattr can see, and setting the flag on a binding without it would
        # read an empty marginals object. Both ship together.
        "ik_stepping": hasattr(gepetto_solvers.TendonHandSolver, "reset_al_duals"),
        # Seeding a single-shot solve / stepper with a posture from an earlier
        # solve (HandSolveParams.initial_state), the way the controller has
        # always been seedable. Without it a rebuilt stepper can only cold-start.
        "solver_seed": hasattr(gepetto_solvers.TendonHandSolverConfig(),
                               "initial_state"),
        # Carrying the AL multipliers across a REBUILD, matched by constraint
        # identity (TendonHandSolver.set_initial_duals). Without it a rebuilt
        # solver restarts the penalty schedule from scratch, which is visible as
        # the hand drifting off constraints it had already satisfied.
        "dual_transfer": hasattr(gepetto_solvers.TendonHandSolver,
                                 "set_initial_duals"),
        # Eq 2.18-2.19 pre-grasp hand-centering. Needs a rebuilt binding with
        # EnvironmentConfig.pregrasp_center_node.
        "pregrasp_center": hasattr(env, "pregrasp_center_node"),
        # Eq 2.16-2.17 opposition half-space and Eq 2.12-2.15's drop-normal-row
        # SDF contact form. Both fields have been on EnvironmentConfig for a
        # while (the §1.8 controller used them), so these are almost always
        # True; gated for consistency with every other control here and as a
        # defensive check against a stale binding.
        "opposition": hasattr(env, "half_space_enabled"),
        # The opposition half-space's MINIMUM STANDOFF (HalfSpaceGapFactor's
        # d_min): a newer field than the half-space itself, so a binding can
        # have the constraint and not the standoff -- in which case the
        # constraint still builds, just with no minimum distance.
        "half_space_margin": hasattr(env, "half_space_margin"),
        # The half-space standing on its own field (half_space_node) instead of
        # riding on table_contact_node. Without it the constraint is only built
        # for fingers that are ALSO driven onto the table, so checking it alone
        # builds nothing.
        "half_space_standalone": hasattr(env, "half_space_node"),
        # Finger-finger avoidance as its own switch. Without it, self-collision
        # is whatever the object/table collision toggles imply and cannot be
        # turned off.
        "self_collision": hasattr(env, "self_collision"),
        # Planar-bending approximation: PlanarBendFactor per rod segment, keeping
        # each finger in its own flexion plane. Probed on the FINGER config
        # because that is where the switch rides -- the sigmas are per-finger rod
        # physics, like sigma_twist_rot, not a hand-level environment setting.
        "planar_bending": hasattr(fc, "planar_bending"),
        "drop_normal_row": hasattr(env, "contact_drop_normal_row"),
        # Pre-grasp short-axis alignment (companion to Eq 2.16-2.17). Needs a
        # rebuilt binding with EnvironmentConfig.pregrasp_align_node.
        "pregrasp_axis_align": hasattr(env, "pregrasp_align_node"),
        # Pre-grasp PINCH-CENTROID centering: drive the measured hand-frame
        # meeting point of the checked digits (config.HAND_PINCH_POSES) onto
        # the object. Needs a rebuilt binding with
        # EnvironmentConfig.pregrasp_centroid_point.
        "pregrasp_centroid": hasattr(env, "pregrasp_centroid_point"),
        # Eq 13 in-plane object CONTACT -- the constraint, as opposed to
        # "planar_gap" below, which is only the query. Separate probes because
        # they landed separately: a binding can measure the in-plane distance
        # without being able to solve against it.
        "planar_contact": hasattr(env, "object_contact_in_plane"),
        # Eq 11/13 tendon-aligned planar distance, as a QUERY -- the visualizer's
        # in-plane overlay. A module-level free function rather than a config
        # field, because nothing in the graph builds this factor yet: it is
        # measurement only, so a binding without it loses the overlay and
        # nothing else.
        "planar_gap": hasattr(gepetto_solvers, "ellipsoid_set_planar_gap"),
        # Whether TendonHandSolver.solve releases the GIL while the C++ solve
        # runs. Without it the whole interpreter is frozen for the duration of
        # every AL outer iteration (~1.4 s measured), so the visualizer's E-STOP
        # cannot even be RECEIVED: viser dispatches button callbacks on a thread
        # pool, and those threads are not queued behind the solve, they are
        # unschedulable. The stop then lands only once the iteration ends, which
        # is exactly when it has stopped being useful. Probed via a module
        # attribute because a py::call_guard is invisible to hasattr.
        "gil_release": getattr(gepetto_solvers, "solve_releases_gil", False),
    }
