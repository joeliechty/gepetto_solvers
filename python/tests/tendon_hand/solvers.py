"""Reusable FK / IK / trajectory-planner solver classes for the tendon hand.

This factors the shared *build -> solve -> extract* skeleton that the demo
scripts (``ik_5f_contact.py``, ``fk_5f_sweep.py``, ``traj_5f_contact.py``)
each re-implement inline into three classes behind one common base
(:class:`HandSolverBase`), driven by a single :class:`HandSolveParams` struct
and returning a uniform :class:`HandResult`.

The point is a *unified* way to call the three solvers, so the interactive viser
visualizer (``viz_interactive.py``) -- and any future code -- can flip between
FK, IK and the trajectory planner without duplicating the setup boilerplate. The
existing demo scripts are left untouched; this module reuses their helpers
(``config.py`` / ``scene.py``) rather than replacing them.

Three flavours, matching the demos:

* :class:`HandFKSolver`      -- pure kinematics, no contact (``fk_5f_sweep.py``).
  Uses a *uniform* tension prior (a tight-passive/loose-flexor prior is
  underdetermined without contact) and keeps its solver so repeated solves
  warm-start via ``set_wrist_pose``.
* :class:`HandIKSolver`      -- single terminal grasp with per-finger SDF/analytic
  contact (``ik_5f_contact.py``); the C++ side routes to the Augmented Lagrangian.
* :class:`HandPlannerSolver` -- a K+1-step grasp trajectory with GP temporal priors
  (``traj_5f_contact.py`` / ``traj_5f_slide_grasp.py``).

Collision avoidance (Section 1.5) and the support-plane "table" (Section 1.6)
are opt-in via the params and applied to the IK/planner solves; FK stays a pure
kinematics solve (the renderer can still draw the spheres/table for reference).
"""

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, NamedTuple, Optional

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    disc_node_indices, attach_contact, attach_collision, attach_table,
    tip_node_index, opposition_directions, opposition_axis_from_object,
    attach_half_space, attach_pregrasp_center, attach_pregrasp_axis_alignment,
    attach_pregrasp_centroid, pinch_pose_for_mask)
from .scene import (
    OBJECT_CENTER, GRASP_SPHERE_CENTER, GRASP_FLEXOR_TENSION, TABLE_NORMAL,
    get_primitive_specs, primitive_surface_witness, object_principal_inplane_axis,
    # Re-exported: pure spec geometry, and it lives in scene.py so the in-plane
    # width sweep there can reach it. Callers still say solvers.object_extent_along.
    object_extent_along)


# The _objects/ directory holding the baked .vdb SDF grids (relative to this file).
_OBJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "_objects")

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

    The installed ``_crest_sparse`` extension can lag the C++ source: newer config
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
    env = crest_sparse.EnvironmentConfig()
    pc = crest_sparse.TendonHandTrajectoryPlannerConfig()
    return {
        "ellipsoid": hasattr(env, "ellipsoid_semi_axes"),
        # Section 1.2 ellipsoid SETS (the YCB objects). Gated separately from
        # "ellipsoid" because the set factor landed much later than the single
        # one, so a binding can easily have one and not the other.
        "ellipsoid_set": hasattr(env, "ellipsoid_set"),
        "table": hasattr(env, "plane_normal"),
        "collision_cull": hasattr(env, "collision_cull_margin"),
        "k_touch": hasattr(pc, "k_touch"),
        # Per-iteration solve snapshots off the single-shot solver, so the
        # visualizer can scrub an IK solve's convergence. The trajectory planner
        # has had this for longer; TendonHandSolver only gained it with the
        # iterate scrubber, so a stale binding must not offer the control.
        "solve_iterates": hasattr(crest_sparse.TendonHandSolver,
                                  "get_intermediate_solutions"),
        # Driving the AL outer loop one iteration at a time (HandIKStepper).
        # Probed via reset_al_duals because the other half of that build --
        # TendonHandSolver honoring skip_marginals -- is a behavior change no
        # hasattr can see, and setting the flag on a binding without it would
        # read an empty marginals object. Both ship together.
        "ik_stepping": hasattr(crest_sparse.TendonHandSolver, "reset_al_duals"),
        # Seeding a single-shot solve / stepper with a posture from an earlier
        # solve (HandSolveParams.initial_state), the way the controller has
        # always been seedable. Without it a rebuilt stepper can only cold-start.
        "solver_seed": hasattr(crest_sparse.TendonHandSolverConfig(),
                               "initial_state"),
        # Carrying the AL multipliers across a REBUILD, matched by constraint
        # identity (TendonHandSolver.set_initial_duals). Without it a rebuilt
        # solver restarts the penalty schedule from scratch, which is visible as
        # the hand drifting off constraints it had already satisfied.
        "dual_transfer": hasattr(crest_sparse.TendonHandSolver,
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
        "drop_normal_row": hasattr(env, "contact_drop_normal_row"),
        # Pre-grasp short-axis alignment (companion to Eq 2.16-2.17). Needs a
        # rebuilt binding with EnvironmentConfig.pregrasp_align_node.
        "pregrasp_axis_align": hasattr(env, "pregrasp_align_node"),
        # Pre-grasp PINCH-CENTROID centering: drive the measured hand-frame
        # meeting point of the checked digits (config.HAND_PINCH_POSES) onto
        # the object. Needs a rebuilt binding with
        # EnvironmentConfig.pregrasp_centroid_point.
        "pregrasp_centroid": hasattr(env, "pregrasp_centroid_point"),
        # Eq 11/13 tendon-aligned planar distance, as a QUERY -- the visualizer's
        # in-plane overlay. A module-level free function rather than a config
        # field, because nothing in the graph builds this factor yet: it is
        # measurement only, so a binding without it loses the overlay and
        # nothing else.
        "planar_gap": hasattr(crest_sparse, "ellipsoid_set_planar_gap"),
        # Whether TendonHandSolver.solve releases the GIL while the C++ solve
        # runs. Without it the whole interpreter is frozen for the duration of
        # every AL outer iteration (~1.4 s measured), so the visualizer's E-STOP
        # cannot even be RECEIVED: viser dispatches button callbacks on a thread
        # pool, and those threads are not queued behind the solve, they are
        # unschedulable. The stop then lands only once the iteration ends, which
        # is exactly when it has stopped being useful. Probed via a module
        # attribute because a py::call_guard is invisible to hasattr.
        "gil_release": getattr(crest_sparse, "solve_releases_gil", False),
    }


# ---------------------------------------------------------------------------
# Hand base / wrist start pose.
# ---------------------------------------------------------------------------

def euler_to_R(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) rotation matrix from radians.

    The convention the wrist-pose sliders and the headless harnesses all quote
    poses in, defined once here so a pose typed into a CLI, a slider and a test
    all mean the same rotation."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    """(roll, pitch, yaw) in radians from a ZYX rotation matrix. Inverse of
    :func:`euler_to_R`, so a pose can round-trip through the sliders."""
    R = np.asarray(R, float)
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) > 1.0 - 1e-9:      # gimbal lock: fold roll into yaw
        return 0.0, float(pitch), float(np.arctan2(-R[0, 1], R[1, 1]))
    return (float(np.arctan2(R[2, 1], R[2, 2])), float(pitch),
            float(np.arctan2(R[1, 0], R[0, 0])))


def wrist_pose_from_xyzrpy(xyz, rpy):
    """4x4 base pose from a translation (m) and ZYX euler angles (rad)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(*rpy)
    T[:3, 3] = np.asarray(xyz, float)
    return T


def solved_wrist_pose(configs, frame):
    """The wrist pose a solved frame actually ended at, as a 4x4.

    The wrist is a VARIABLE, not a fixed input: its prior is soft (sigma_wrist_*)
    and contact pulls against it, so a solve that presses the hand onto a table
    moves the base tens of millimetres away from the commanded pose. Nothing in
    the result reports it directly, but each finger's node-0 pose is
    ``T_0 = T_wrist o T_offset``, so inverting finger 0's offset recovers it (all
    fingers agree to machine precision -- they share the one variable)."""
    name, cfg = configs[0]
    T0 = np.asarray(frame[name].marginals.rod.states[0].pose.mean, float)
    return T0 @ np.linalg.inv(np.asarray(cfg.hand_base_offset, float))


# The default hand base pose: lifted 75 mm along the support normal and pitched
# -1.22 rad about +Y. The mount puts the palm along the base frame's -x, so that
# pitch swings the palm to face roughly -z -- i.e. the hand hovers palm-down over
# the object at the default grasp locus, fingers already aimed at it, instead of
# standing at the identity pose with the palm pointing sideways and the fingers
# through the scene.
#
# This is the posing the interactive visualizer opens on. Keep the two in sync:
# the visualizer seeds its sliders from these numbers rather than repeating them.
DEFAULT_WRIST_XYZ = (0.0, 0.0, 0.075)
DEFAULT_WRIST_RPY = (0.0, -1.22, 0.0)


def default_wrist_pose():
    """The default hand base pose as a 4x4 (fresh array per call, since it is a
    dataclass field default and callers mutate poses in place)."""
    return wrist_pose_from_xyzrpy(DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY)


# ---------------------------------------------------------------------------
# Scene helpers (shared object placement, mirroring the demo scripts).
# ---------------------------------------------------------------------------

def default_object_center(primitive, spec):
    """Default world center for a primitive, matching the demo scripts: the big
    grasp sphere, capsule and analytic ellipsoids sit at the flexed-fingertip
    locus (``GRASP_SPHERE_CENTER``); the smaller primitives stay at ``OBJECT_CENTER``."""
    if (primitive in ("big_sphere", "capsule")
            or spec["type"] in ("ellipsoid", "ellipsoid_set")):
        return np.array(GRASP_SPHERE_CENTER, dtype=float)
    return np.array(OBJECT_CENTER, dtype=float)


def resolve_scene(params):
    """Resolve (spec, center, rotation, 4x4 pose) for the object from the params,
    filling center/rotation from the primitive when left unset."""
    spec = get_primitive_specs()[params.primitive]
    center = (np.asarray(params.object_center, float)
              if params.object_center is not None
              else default_object_center(params.primitive, spec))
    rotation = (np.asarray(params.object_rotation, float)
                if params.object_rotation is not None
                else np.asarray(spec.get("rotation", np.eye(3)), float))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    return spec, center, rotation, pose


def auto_table_origin(params, spec, object_center):
    """The support-plane origin implied by the scene alone: the object seated on
    ``params.plane_normal`` at a burial fraction of ``params.table_burial``.

    ``table_burial`` is the fraction of the object's FULL along-normal extent
    lying below the plane, so the origin is

        c - (1 - 2 * burial) * half_extent * n_hat

    which is tangent to the underside at 0.0 (the object rests on the table) and
    through the centroid at 0.5 (half-buried). See
    :attr:`HandSolveParams.table_burial` for why half-buried is the default.

    Deliberately ignores ``params.plane_origin``. A GUI offering an ABSOLUTE
    plane height has to seed and re-seat its control from the scene's own answer;
    reading :func:`resolve_table_origin` for that would feed the control's own
    output back into itself, and any offset applied on top would compound on
    every call. Split out so both readings have exactly one definition of the
    seating rule.
    """
    n = np.asarray(params.plane_normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    # getattr: params-like objects predating table_burial keep the old geometry.
    burial = float(getattr(params, "table_burial", 0.0))
    # The object's world orientation, so a rotated object is seated on the
    # profile it actually presents to the plane. Falls back to the primitive's
    # own baked rotation when the caller has not overridden it.
    rotation = (params.object_rotation if params.object_rotation is not None
                else spec.get("rotation"))
    depth = (1.0 - 2.0 * burial) * object_extent_along(spec, n, rotation)
    return np.asarray(object_center, float) - depth * n


def resolve_table_origin(params, spec, object_center):
    """Resolve the support-plane origin: explicit ``params.plane_origin`` if set,
    else the scene's own seating rule (see :func:`auto_table_origin`)."""
    if params.plane_origin is not None:
        return np.asarray(params.plane_origin, float)
    return auto_table_origin(params, spec, object_center)



def _sphere_nodes(fm):
    """``[(node_index, is_tip)]`` for one finger's collision spheres, taken from
    the marginals rather than the configs -- the same ``disc_pose_idx`` walk the
    renderer draws, so an overlay can never mark a sphere the picture does not
    show. The tip is the last rod state, matching ``contact_witness``."""
    tip = len(fm.rod.states) - 1
    return [(int(i), int(i) == tip) for i in fm.tendon_config.disc_pose_idx]


def _plane_measure(center, radius, origin, n_hat):
    """``(point_on_sphere, foot_on_plane, signed_gap)`` for one sphere against the
    support plane: Eq 1.104's ``c_support = (c - p).n_hat - r``, zero at contact
    and negative when the sphere is through the plane."""
    d = float((np.asarray(center, float) - origin) @ n_hat)
    return (center - radius * n_hat, center - d * n_hat, d - radius)


def plane_witness(params, result, k=0, names=None):
    """Per-contact-finger ``{name: (sphere_pt, foot_pt, signed_gap)}`` against the
    support plane at frame ``k``.

    Same 3-tuple shape as :meth:`HandResult.contact_witness`, so a renderer draws
    a table clearance exactly the way it draws an object gap. These are the
    spheres phase 1/2 hold at ``c_support = 0`` (Eq 1.104/1.110); phase 3 relaxes
    them back to the ``<= 0`` inequality, where the same number is a clearance.

    ``names`` overrides which fingers are reported. The default (the fingers
    designated for OBJECT contact) is what the §1.8 overlays want, since a
    controller phase designates one contact set for both surfaces; a solve that
    targets the two surfaces separately passes ``result.table_contact_names()``.
    """
    frame = result.frames[k]
    origin = np.asarray(resolve_table_origin(params, result.spec,
                                             result.object_center), float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    wanted = set(result.contact_names() if names is None else names)

    out = {}
    for name, radius in zip(result.finger_names, result.tip_radii):
        if name not in wanted:
            continue
        fm = frame[name].marginals
        c = np.asarray(fm.rod.states[-1].pose.mean, float)[:3, 3]
        out[name] = _plane_measure(c, float(radius), origin, n_hat)
    return out


def free_sphere_plane_witness(params, result, k=0, names=None):
    """The same measurement for every sphere NOT designated for support contact,
    keyed ``"{finger}/{node}"``.

    These carry the Eq 1.106 avoidance inequality rather than an equality, and
    they are the ones that silently stall a phase: a single sphere driven through
    the table dominates the merit function and the inner LM rejects every step.

    ``names`` (as on :func:`plane_witness`) is the designated support set whose
    tips are excluded here; it must be the same list, or a fingertip is either
    reported twice or not at all.
    """
    frame = result.frames[k]
    origin = np.asarray(resolve_table_origin(params, result.spec,
                                             result.object_center), float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    contact = set(result.contact_names() if names is None else names)

    out = {}
    for name, tip_radius in zip(result.finger_names, result.tip_radii):
        fm = frame[name].marginals
        poses = fm.rod.states
        for node, is_tip in _sphere_nodes(fm):
            # A contact finger's tip is the designated support sphere and is
            # reported by plane_witness; every other sphere lands here.
            if is_tip and name in contact:
                continue
            c = np.asarray(poses[node].pose.mean, float)[:3, 3]
            r = float(tip_radius) if is_tip else float(params.collision_radius)
            out[f"{name}/{node}"] = _plane_measure(c, r, origin, n_hat)
    return out


def tip_gap_matrix(tips, radii):
    """Pairwise fingertip SURFACE gaps (m) for a set of contact spheres.

    ``tips`` is an ``(n, 3)`` array of sphere centers and ``radii`` the matching
    ``(n,)`` radii, so entry ``(i, j)`` is
    ``||c_i - c_j|| - (r_i + r_j)`` -- zero when the two tip spheres just touch
    and negative when they interpenetrate. The diagonal is set to ``+inf`` so
    ``.min()`` over the matrix is the closest DISTINCT pair without having to
    mask it out.

    Every other witness helper in this module measures a fingertip against a
    *surface* (the object, the support plane, a half-space); this is the
    finger-to-finger counterpart, which a pinch has to be judged on -- two
    fingertips can each sit far from any object yet be touching each other.
    """
    c = np.asarray(tips, dtype=float).reshape(-1, 3)
    r = np.asarray(radii, dtype=float).reshape(-1)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    gaps = d - (r[:, None] + r[None, :])
    np.fill_diagonal(gaps, np.inf)
    return gaps


def default_half_space_axis(spec, rotation, plane_normal):
    """The opposition split axis (Eq 2.16-2.17's ``m_hat``) derived from the
    object's own geometry, for when ``HandSolveParams.half_space_axis`` is
    unset: perpendicular, within the support plane, to the object's longest
    in-plane axis (:func:`scene.object_principal_inplane_axis`), via
    :func:`config.opposition_axis_from_object`.

    This is what makes the split LINE run along the object's length (e.g.
    lengthwise along a pen, thumb on one side and fingers on the other across
    its diameter) instead of across it. The length is measured off the object's
    silhouette on the support plane, so it finds a direction that is not one of
    the object's own frame axes -- a YCB screwdriver lying at 27 degrees to its
    export frame gets 27 degrees, not the nearest axis. Falls back to world +Y
    (giving ``m_hat = -X``, thumb on the -X side) only when the object is
    in-plane isotropic (below the degeneracy ratio) --
    :func:`scene.object_principal_inplane_axis`'s own fallback.

    Returns the LINE, not the side assignment: the sign is inherited from the
    object's principal-axis direction, which is an arbitrary convention (which
    end of the sweep the widest direction landed on, or the +Y fallback). The
    fallback's sign is chosen to agree with the hand rather than fight it, but
    the measured one cannot be. Which half the THUMB is asked to
    stay on is a statement about the hand, not the object -- see
    :func:`orient_opposition_axis`, which every caller must apply before
    building the constraint."""
    e_long, _ratio = object_principal_inplane_axis(spec, rotation, plane_normal)
    return opposition_axis_from_object(plane_normal, e_long)


def orient_opposition_axis(axis, thumb_pt, finger_pts, flip=None):
    """``(oriented_axis, flipped)`` -- ``axis`` signed so that ``+m_hat`` points
    from the opposing fingers TOWARD the thumb at the posture given.

    :func:`config.opposition_directions` hands the thumb ``+m_hat`` and every
    other finger ``-m_hat``, so the sign of ``m_hat`` IS the side assignment.
    Deriving it from the object alone (:func:`default_half_space_axis`) picks
    that assignment by a coin flip, and when it lands the wrong way up the
    constraint asks the thumb and the fingers to TRADE sides -- a ~180 degree
    roll of the whole hand about the object. Measured on the phase-0 pen scene
    that is a 32 mm demand on the thumb and 70-75 mm on the fingers, from a
    start pose that already satisfies the constraint in the other orientation:
    the AL stalls at 3 outer iterations with a violation of 1.09e3 and the hand
    never moves. Orienting by the hand instead turns the same scene into a
    solve that runs to a 3e-7 violation.

    ``flip`` overrides the measurement: None (default) picks the nearer
    orientation as described, False keeps the derived sign, True inverts it --
    the way to ask for the opposition the hand is NOT already in, which is a
    genuine (large) repositioning move rather than a mislabeling.
    """
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    if flip is not None:
        return (-axis if flip else axis), bool(flip)
    pts = np.asarray(finger_pts, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return axis, False
    reach = float((np.asarray(thumb_pt, float).reshape(3) - pts.mean(axis=0))
                  @ axis)
    return (-axis, True) if reach < 0.0 else (axis, False)


def half_space_witness(params, result, k=0, names=None):
    """Per-contact-finger ``{name: (sphere_pt, foot_pt, signed_margin_m)}``
    against the Eq 2.16-2.17 opposition half-space at frame ``k``.

    ``signed_margin`` is ``-c_half`` (the C++ factor's own sign convention,
    negated back to something readable): positive means the finger is on its
    designated side (constraint satisfied, with that much room to spare),
    negative means it has crossed to the wrong side by that many meters.
    ``foot_pt`` is the fingertip projected onto the splitting plane along the
    finger's own ``m_hat`` (``+axis`` for the thumb, ``-axis`` for everyone
    else -- :func:`config.opposition_directions`'s convention).

    Measured against the finger's OWN boundary -- the split plane offset by
    ``params.half_space_margin``, which is exactly what the C++ residual's
    ``+ d_min`` term does -- so zero stays the constraint's zero set (and the
    overlay's green/red flip stays the constraint's own) once a minimum standoff
    is asked for, rather than reporting slack against a boundary the solver is
    no longer using.

    ``names`` defaults to ``params.contact_fingers`` (None = every finger) --
    the mask :func:`config.attach_half_space` is written with, so the overlay
    covers exactly the fingers the constraint was built for. It used to default
    to the TABLE-contact set, back when the C++ layer only built the constraint
    for a finger that was also driven onto the table.
    """
    frame = result.frames[k]
    p_split = (np.asarray(params.half_space_split, dtype=float).reshape(3)
              if params.half_space_split is not None else result.object_center)
    if names is None:
        mask = params.contact_fingers
        names = (list(result.finger_names) if mask is None else
                 [n for n, on in zip(result.finger_names, mask) if on])
    wanted = set(names)

    def tip(name):
        return np.asarray(frame[name].marginals.rod.states[-1].pose.mean,
                          float)[:3, 3]

    if params.half_space_axis is not None:
        # Already oriented -- either the caller's own axis, or the one
        # _attach_opposition resolved and wrote back after building the graph.
        axis = np.asarray(params.half_space_axis, dtype=float).reshape(3)
        axis = axis / (np.linalg.norm(axis) or 1.0)
    else:
        # No solve has resolved it (an FK pose, say). Orient it here off THIS
        # frame's own fingertips, by the same rule _attach_opposition uses, so
        # the overlay never shows the mirror image of the constraint an IK solve
        # would build.
        axis, _flipped = orient_opposition_axis(
            default_half_space_axis(result.spec, result.object_rotation,
                                    params.plane_normal),
            tip("thumb") if "thumb" in result.finger_names else np.zeros(3),
            [tip(n) for n in wanted if n != "thumb"],
            flip=params.half_space_flip)

    out = {}
    for name in result.finger_names:
        if name not in wanted:
            continue
        c = tip(name)
        m_hat = axis if name == "thumb" else -axis
        # The boundary this finger is actually held to: the split pushed out by
        # the standoff along its own m_hat (0 => the plain splitting plane).
        p_bound = p_split + float(params.half_space_margin) * m_hat
        margin = float((c - p_bound) @ m_hat)   # >= 0 => correct side
        foot = c - margin * m_hat
        out[name] = (c, foot, margin)
    return out


def pregrasp_center_witness(params, result, k=0):
    """``(hand_centroid_pt, target_pt, gap_m)`` for the Eq 2.18-2.19 pre-grasp
    hand-centering constraint at frame ``k``, or None if the thumb or the
    opposing set has no fingers designated (:meth:`HandResult.contact_names`).

    ``hand_centroid_pt`` is the midpoint of the thumb's and the opposing
    (non-thumb, contact-designated) fingers' contact-sphere centers --
    ``c_hand`` in the paper's notation. ``target_pt`` is the object centroid
    raised by ``h_clear`` along ``plane_normal``. ``gap_m`` is their Euclidean
    separation (0 at the constraint's zero set); unlike the other witness
    functions this is a single HAND-level tuple, not one per finger.
    """
    names = result.contact_names()
    if "thumb" not in names:
        return None
    others = [n for n in names if n != "thumb"]
    if not others:
        return None

    frame = result.frames[k]

    def tip(name):
        fm = frame[name].marginals
        return np.asarray(fm.rod.states[-1].pose.mean, float)[:3, 3]

    c_thumb = tip("thumb")
    c_others = np.mean([tip(n) for n in others], axis=0)
    hand_centroid = 0.5 * (c_thumb + c_others)

    n_hat = np.asarray(params.plane_normal, dtype=float).reshape(3)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    h_clear = params.h_clear if params.h_clear is not None else 0.02
    target = np.asarray(result.object_center, dtype=float) + h_clear * n_hat

    gap = float(np.linalg.norm(hand_centroid - target))
    return (hand_centroid, target, gap)


def pregrasp_centroid_witness(params, result, k=0):
    """``(pinch_pt, target_pt, gap_m)`` for the pre-grasp PINCH-CENTROID
    constraint at frame ``k``, or None if the checked digits have no measured
    pinch pose (:func:`config.pinch_pose`).

    Same 3-tuple shape as :func:`pregrasp_center_witness`, so the renderer
    draws it identically -- but it measures something different.
    ``pregrasp_center_witness`` reads where the fingertips ACTUALLY are;
    this pushes the hand-frame constant through the SOLVED WRIST POSE:

        pinch_pt = T_wrist * c_local

    which is an independent re-derivation of exactly what the C++
    ``PreGraspCentroidFactor`` computes, from the solved values rather than
    from the factor. So a disagreement between this readout and the solver's
    own residual is a real signal, not a display artifact.

    Needs ``configs`` to recover the wrist pose from the solved frame, which
    it takes from the result's own finger list via :func:`solved_wrist_pose`.
    """
    from .config import pinch_pose

    pose = pinch_pose(result.contact_names())
    if pose is None:
        return None

    # solved_wrist_pose needs (name, cfg) pairs; rebuild the same hand the
    # result came from. Cheap (no solve) and keeps this a free function.
    configs = get_default_hand_configs()
    T = solved_wrist_pose(configs, result.frames[k])
    c_local = np.asarray(pose.centroid, dtype=float).reshape(3)
    pinch_pt = T[:3, :3] @ c_local + T[:3, 3]

    n_hat = np.asarray(params.plane_normal, dtype=float).reshape(3)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    h_clear = params.h_clear if params.h_clear is not None else 0.02
    target = np.asarray(result.object_center, dtype=float) + h_clear * n_hat

    return (pinch_pt, target, float(np.linalg.norm(pinch_pt - target)))


def pregrasp_axis_witness(params, result, k=0):
    """``(c_thumb, c_others_mean, angle_deg)`` for the pre-grasp short-axis
    alignment constraint at frame ``k``, or None if the thumb or the opposing
    set has no fingers designated (:meth:`HandResult.contact_names`).

    ``angle_deg`` is the acute angle between the achieved thumb-vs-opposing
    connecting vector and ``default_half_space_axis`` -- 0 at the constraint's
    zero set (either parallel or antiparallel), up to 90 at worst. Recomputes
    the SAME axis :meth:`HandSolverBase._attach_pregrasp_axis_alignment` uses,
    from ``result.spec``/``result.object_rotation`` rather than a stored
    value, so the overlay always matches the axis the last-attached
    constraint actually used.
    """
    names = result.contact_names()
    if "thumb" not in names:
        return None
    others = [n for n in names if n != "thumb"]
    if not others:
        return None

    frame = result.frames[k]

    def tip(name):
        fm = frame[name].marginals
        return np.asarray(fm.rod.states[-1].pose.mean, float)[:3, 3]

    c_thumb = tip("thumb")
    c_others = np.mean([tip(n) for n in others], axis=0)
    v = c_thumb - c_others
    vn = np.linalg.norm(v)
    if vn < 1e-9:
        return None
    v_hat = v / vn

    axis = default_half_space_axis(result.spec, result.object_rotation,
                                   params.plane_normal)
    cos_a = abs(float(v_hat @ axis))
    cos_a = min(1.0, max(-1.0, cos_a))
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    return (c_thumb, c_others, angle_deg)


def finger_plane_witness(result, k=0):
    """Per-finger ``{name: (base_pt, tip_pt, pinch_pt)}`` at frame ``k`` -- the
    three world points that span that finger's *pinch plane* -- or None if the
    checked digits have no measured pinch pose (:func:`config.pinch_pose`).

    The three points are the finger's METACARPAL BASE (rod node 0, where the
    finger meets the palm), its FINGERTIP (rod node -1, the contact node), and
    the pinch centroid the checked digits close on. The first two move with the
    posture; the third is the hand-frame constant from
    :data:`config.HAND_PINCH_POSES` carried through the solved wrist pose,
    exactly as :func:`pregrasp_centroid_witness` carries it -- so all fingers
    share one pinch point and their planes form a fan through it.

    Read it as the plane that finger has to sweep *in* to reach the meeting
    point: the finger's own curl plane is base + tip plus the flexor's pull
    direction, and how far this plane tilts out of it is how much of the closure
    is happening sideways, where the tendons have no authority.

    None rather than a default when the combination was never measured -- the
    same contract :func:`config.pinch_pose` documents. A per-finger entry is
    still returned when the three points are COLLINEAR (a finger whose tip
    happens to point at the centroid); the plane is undefined there, and the
    renderer -- not this function -- decides what to do about it, since only it
    knows what it was going to draw.

    Rendering only: nothing here is a constraint the solver saw.
    """
    from .config import pinch_pose

    pose = pinch_pose(result.contact_names())
    if pose is None:
        return None

    frame = result.frames[k]
    # Same reconstruction pregrasp_centroid_witness makes: the centroid is in the
    # WRIST frame, so it only becomes a world point through the SOLVED wrist.
    T = solved_wrist_pose(get_default_hand_configs(), frame)
    c_local = np.asarray(pose.centroid, dtype=float).reshape(3)
    pinch_pt = T[:3, :3] @ c_local + T[:3, 3]

    out = {}
    for name in result.finger_names:
        if name not in frame:
            continue
        states = frame[name].marginals.rod.states
        base = np.asarray(states[0].pose.mean, float)[:3, 3]
        tip = np.asarray(states[-1].pose.mean, float)[:3, 3]
        out[name] = (base, tip, pinch_pt)
    return out


class PlanarGap(NamedTuple):
    """One finger's tendon-aligned in-plane distance readout (Eq 11 / Eq 13).

    ``sphere_pt``  world point on the contact sphere's surface, aimed at ``foot``
                   -- the same convention :meth:`HandResult.contact_witness` uses,
                   so the two overlays start from the same place.
    ``foot``       the EXACT nearest point on the drawn cross-section (or, in
                   fallback, the exact 3D surface witness).
    ``gap``        the FACTOR's number, as a surface gap: ``d_planar - r``, zero at
                   Eq 13's zero set, negative once the sphere is through the
                   cross-section.
    ``fallback``   True when no member is being measured in-plane -- the plane
                   missed everything, or it is degenerate -- so ``gap`` is the
                   ordinary 3D distance and the plane is telling you nothing.
    ``sections``   one ``(N, 3)`` world polyline per member the plane actually
                   cuts; empty in fallback.

    Note the deliberate split: the LINE is exact geometry, the LABEL is the
    factor's first-order approximation of it. Where they visibly disagree, that
    disagreement is the approximation error the solver would be working with,
    which is the thing worth being able to see.
    """
    sphere_pt: np.ndarray
    foot: np.ndarray
    gap: float
    fallback: bool
    sections: list


def planar_gap_witness(params, result, k=0):
    """Per-finger :class:`PlanarGap` at frame ``k``, or None when there is nothing
    to measure.

    None (rather than a partial answer) when any of the three requirements is
    missing, each of which is a real "this cannot be drawn" rather than a failure:

      * the installed binding has no ``ellipsoid_set_planar_gap``
        (``capabilities()["planar_gap"]``),
      * the object has no analytic ellipsoid form (:func:`scene.ellipsoid_members`
        returns None for the baked-SDF and box/cylinder primitives),
      * the designated digits have no measured pinch pose, so Eq 11 has no
        centroid and therefore no plane -- the same contract
        :func:`config.pinch_pose` documents.

    The distance itself comes from the C++ factor, not from a NumPy re-derivation:
    the whole point of the overlay is to show what that factor would report.
    """
    import crest_sparse

    from .config import pinch_pose
    from .scene import (ellipsoid_members, plane_ellipse_section,
                        ELLIPSOID_SET_BETA)

    if not hasattr(crest_sparse, "ellipsoid_set_planar_gap"):
        return None
    members = ellipsoid_members(result.spec)
    if members is None:
        return None
    pose = pinch_pose(result.contact_names())
    if pose is None:
        return None

    configs = get_default_hand_configs()
    by_name = dict(configs)
    frame = result.frames[k]
    T_wrist = solved_wrist_pose(configs, frame)

    center = np.asarray(result.object_center, dtype=float)
    R_obj = np.asarray(result.object_rotation, dtype=float)
    T_obj = np.eye(4)
    T_obj[:3, :3] = R_obj
    T_obj[:3, 3] = center

    # The same members, twice over: as EllipsoidPrimitives for the C++ number, and
    # posed into the world for the cross-section outlines.
    prims = []
    for semi_axes, R_m, c_m in members:
        p = crest_sparse.EllipsoidPrimitive()
        p.semi_axes = semi_axes
        local = np.eye(4)
        local[:3, :3] = R_m
        local[:3, 3] = c_m
        p.local_pose = local
        prims.append(p)
    beta = float(params.ellipsoid_set_beta
                 if params.ellipsoid_set_beta is not None
                 else result.spec.get("beta", ELLIPSOID_SET_BETA))
    c_local = np.asarray(pose.centroid, dtype=float).reshape(3)

    out = {}
    for name, radius in zip(result.finger_names, result.tip_radii):
        cfg = by_name.get(name)
        if name not in frame or cfg is None:
            continue
        T_tip = np.asarray(frame[name].marginals.rod.states[-1].pose.mean, float)
        tip = T_tip[:3, 3]
        # Eq 11's p_base, in the WRIST frame -- the finger's mounting offset, not a
        # solved node pose (node 0 has no key of its own under root reparameterization).
        base_local = np.asarray(cfg.hand_base_offset, dtype=float)[:3, 3]

        rep = crest_sparse.ellipsoid_set_planar_gap(
            T_tip, T_obj, T_wrist, float(radius), prims, beta, base_local, c_local)
        gap = float(rep["distance"]) - float(radius)
        fallback = not any(w > 0.0 for w in rep["weight"])

        sections = []
        n_hat = np.asarray(rep["normal"], dtype=float).reshape(3)
        if np.linalg.norm(n_hat) > 0.0:
            for (semi_axes, R_m, c_m), w in zip(members, rep["weight"]):
                if w <= 0.0:
                    continue        # the plane misses this member: no section to draw
                curve = plane_ellipse_section(semi_axes, R_obj @ R_m,
                                              center + R_obj @ c_m, tip, n_hat)
                if curve is not None:
                    sections.append(curve)

        if sections:
            stacked = np.vstack(sections)
            foot = stacked[int(np.argmin(np.linalg.norm(stacked - tip, axis=1)))]
        else:
            # Nothing in-plane to aim at, so land the line on the real 3D surface --
            # which is also what the reported number has fallen back to.
            _d, foot_local, _n = primitive_surface_witness(
                R_obj.T @ (tip - center), result.spec)
            foot = center + R_obj @ foot_local

        direction = foot - tip
        norm = np.linalg.norm(direction)
        sphere_pt = tip + (radius * direction / norm) if norm > 1e-12 else tip
        out[name] = PlanarGap(sphere_pt, foot, gap, fallback, sections)
    return out


# ---------------------------------------------------------------------------
# Params / results.
# ---------------------------------------------------------------------------

@dataclass
class HandSolveParams:
    """Every knob the three solvers expose, with the demo-script defaults.

    Shared by FK / IK / planner; each solver reads only the fields it needs. The
    interactive visualizer mutates one instance of this from its GUI controls.
    """
    # --- Scene / object ---
    # The Section 1.8 default scene: a 35 mm-radius analytic sphere, half-buried
    # in the support plane (see table_burial). Resting ON the table its crown
    # would sit 70 mm up, outside the ~50 mm the fingertips can reach off their
    # ~55 mm shell; half-buried the exposed dome is 35 mm, which is both
    # reachable and the low-profile-object case Section 1.8 is about.
    primitive: str = "mid_sphere_ellipsoid"
    object_center: Optional[np.ndarray] = None      # None => derive from primitive
    object_rotation: Optional[np.ndarray] = None     # None => primitive's rotation
    # LogSumExp sharpness for an `ellipsoid_set` (ycb:) object; None keeps the
    # spec's own value. Only the smooth-min standoff moves with it -- the
    # constraint surface sits up to ln(K)/beta outside the true union. Inert for
    # every other primitive type.
    ellipsoid_set_beta: Optional[float] = None

    # --- Wrist start pose + prior ---
    # A palm-down HOVER above the object (see DEFAULT_WRIST_XYZ /
    # DEFAULT_WRIST_RPY), not the identity pose. Identity puts the hand at the
    # grasp locus itself, which for the bigger primitives means starting INSIDE
    # them -- on big_sphere the collision spheres begin ~31 mm through the
    # surface, the merit function starts at ~3e6 and the AL solve stalls at
    # iters=1 with nothing able to move. From the hover pose the same scene
    # starts clear of everything at a cost of ~58.
    #
    # What it costs, measured: this is a start to APPROACH from, so a single-shot
    # IK with the default (tight, sigma 1e-4) wrist prior cannot close it -- the
    # base is pinned ~0.11 m off big_sphere and the contact violation freezes.
    # Loosen sigma_wrist_pos/rot, or let something that is allowed to move the
    # base do the positioning (the planner's GP-linked wrist states).
    #
    # Callers that derive their own start overwrite this and are unaffected.
    wrist_pose: np.ndarray = field(default_factory=default_wrist_pose)
    sigma_wrist_pos: float = 1e-4
    sigma_wrist_rot: float = 1e-3

    # --- Tensions (per-finger flexor + shared passive background) ---
    passive_tension: float = 0.5
    flexor_tensions: List[float] = field(
        default_factory=lambda: [GRASP_FLEXOR_TENSION] * NUM_FINGERS)
    tip_wrench_sigma: float = 1e-3
    # How loose the ACTUATED (flexor) tendon's tension prior is once contact
    # is expected to move it away from its commanded value -- squared into
    # the tension covariance's flexor entry by _flexor_tension_cov(), the
    # same sigma-squared-into-covariance pattern tip_wrench_sigma uses above.
    # Default reproduces the historical hardcoded flexor variance (1e-1)
    # exactly.
    flexor_tension_sigma: float = 0.1 ** 0.5
    # Same, for the five PASSIVE tendons -- their physics is a spring holding
    # roughly constant tension, so this is normally left tight. Default
    # reproduces the historical hardcoded passive variance (1e-6) exactly.
    # Dropping much below that (mixed with a much looser flexor scale) risks
    # an IndeterminantLinearSystem, so treat 1e-3 as close to a floor.
    passive_tension_sigma: float = 1e-3

    # --- Which fingertips are solved for contact (IK / planner; FK ignores it) ---
    # One flag per finger, in ``configs`` order. A False finger contributes no
    # contact constraint -- to any surface -- but keeps its collision spheres and
    # plane avoidance, so it is still kept out of the object and (wherever
    # avoidance is active) off the table. All-True is the legacy behavior.
    # WHICH surfaces the True fingers are driven onto is object_contact /
    # table_contact below.
    contact_fingers: List[bool] = field(
        default_factory=lambda: [True] * NUM_FINGERS)

    # --- WHICH SURFACE those fingers are driven onto (IK / planner) ---
    # Orthogonal to contact_fingers, which stays the FINGER selection: the
    # effective per-surface mask is (contact_fingers AND the flag below). So a
    # solve can chase the object only (the legacy behavior), the table only, or
    # both -- which is what makes a stalled grasp bisectable, since the two
    # constraint families can be switched on one at a time.
    #
    # table_contact additionally needs `table` on; without a configured plane
    # there is nothing to touch and it is silently inert.
    object_contact: bool = True
    table_contact: bool = False
    # Eq 2.12-2.15: use the 4-row [c_R, c_O, c_T1, c_T2] SDF witness contact
    # form (c_N dropped) instead of the default 5-row form. Only affects a
    # non-ellipsoid (SDF) object's witness contact -- inert for the analytic
    # ellipsoid's center-direct form, which has no normal row.
    contact_drop_normal_row: bool = False
    # Eq 2.18-2.19: center the midpoint of the thumb's and the opposing
    # fingers' contact points over the object, raised by h_clear along
    # plane_normal. Needs the thumb AND at least one other finger checked in
    # contact_fingers, or the C++ layer silently skips it.
    pregrasp_center: bool = False
    # Companion to Eq 2.16-2.17: align the vector between the thumb's and the
    # opposing fingers' contact centroids with the SAME axis the opposition
    # half-space split uses (perpendicular to the object's longest in-plane
    # axis), direction-agnostically. Independent of half_space/pregrasp_center
    # -- it computes its own copy of that axis via default_half_space_axis()
    # regardless of whether the opposition constraint itself is on. Needs the
    # thumb AND at least one other finger checked in contact_fingers.
    pregrasp_axis_align: bool = False
    # Pre-grasp PINCH-CENTROID centering: drive the point where the checked
    # digits are MEASURED to meet (config.HAND_PINCH_POSES, in the wrist
    # frame) onto the object centroid, raised by h_clear along plane_normal.
    #
    # The hardcoded-point sibling of pregrasp_center above. That one averages
    # the fingertips' achieved positions, so it only says something once the
    # fingers are already near the grasp; this one constrains the WRIST alone,
    # so it positions the hand such that closing those digits would close them
    # on the object -- true whatever the fingers are currently doing.
    #
    # Silently inert for a digit set with no measured pose (fewer than two
    # digits, or any set without the thumb) -- see config.pinch_pose.
    pregrasp_centroid: bool = False

    # --- Augmented Lagrangian (IK / planner) ---
    al_mu: float = 1.0
    al_rate: float = 2.0
    al_iters: int = 40
    # How many leading stepper steps run with the flexor prior PINNED as tightly
    # as the passives (see HandIKStepper.step). Settles the cold start before the
    # flexor is released to flexor_tension_sigma; 0 restores the old behaviour.
    ik_settle_steps: int = 1

    # --- Planner-only ---
    K: int = 10
    dt: float = 0.1
    gp_wrist: float = 1e-2
    gp_tense: float = 1.0
    gp_len: float = 0.0
    start_flexor: float = 0.5
    al_inner_tol: float = 1e-2
    al_abs_cost_tol: float = 1e12

    # Warm-start posture for the IK stepper: the ``marginals`` of any solve on
    # the same finger configs (``HandResult.state(k)``), or None for the
    # straight-rod, zero-tension cold start. Carries a converged grasp across a
    # rebuild -- which is the only way to change the CONSTRAINT SET and continue
    # from where the solve got to, since a new constraint set needs a new solver
    # and a new solver otherwise cold-starts. Needs a binding with
    # ``TendonHandSolverConfig.initial_state`` (capabilities()["solver_seed"]).
    initial_state: Optional[object] = None

    # The other half of a warm start: the Augmented Lagrangian multipliers and
    # penalty weight of a previous solve (``HandResult.duals``), matched onto
    # this solve's constraints by identity. ``initial_state`` carries where the
    # hand IS; this carries how hard each constraint was being held there. Both
    # are needed to change a constraint and continue -- with the posture alone
    # the rebuilt solve restarts at mu = al_mu with every multiplier at zero and
    # visibly drifts off the constraints it had already satisfied before being
    # dragged back. Needs capabilities()["dual_transfer"].
    initial_duals: Optional[object] = None

    # Ceiling on the penalty weight a transfer may carry in. mu is global, so a
    # rebuilt problem inherits it for constraints it has never seen: too high and
    # the new constraint is pinned as rigidly as the old ones and cannot recruit
    # any motion, too low and the old ones are held only weakly.
    al_transfer_mu_max: float = 1e4

    # --- Diagnostics (opt-in; off by default so normal solves are unchanged) ---
    # When True the C++ side records the per-outer-iteration AL trace
    # (al_iteration_mus / _costs / _violations on the result meta) plus
    # step-by-step Values snapshots. Used by debug_al_trace.py; left off for the
    # visualizer since it adds per-iteration bookkeeping.
    record_iterations: bool = False

    # --- Collision avoidance (Section 1.5, opt-in; IK / planner) ---
    # OBJECT collision: keep every non-contact sphere out of the object surface.
    # Independent of the table's own avoidance (plane_avoidance below) and of
    # finger-finger (self_collision): the three families share one set of
    # collision spheres but each is gated on its own field, so any combination
    # of them is available. The sphere set is attached whenever any of the three
    # wants it (_attach_environment).
    collision: bool = False
    # FINGER-FINGER collision: keep the fingers out of each other. Default True,
    # unlike the other two -- self-intersection is never wanted, and it needs no
    # object and no table. Costs the most factors of the three (every
    # cross-finger sphere pair), which is what cull_margin exists to trim.
    self_collision: bool = True
    collision_radius: float = 0.003
    collision_sigma: float = 1e-4
    num_proximal_discs: int = 2
    cull_margin: Optional[float] = None

    # --- Support plane / "table" (Section 1.6, opt-in; IK / planner) ---
    table: bool = False
    plane_origin: Optional[np.ndarray] = None       # None => seat from the scene
    plane_normal: np.ndarray = field(
        default_factory=lambda: np.array(TABLE_NORMAL, float))
    # TABLE collision: keep every non-contact sphere out of the half-space. Needs
    # only `table`, not `collision` -- the solvers attach the collision sphere set
    # whenever any of the three avoidance consumers wants it.
    plane_avoidance: bool = True
    k_touch: Optional[int] = None                    # planner slide-grasp schedule
    # Fraction of the object's FULL along-normal extent sitting BELOW the plane.
    # 0.0 = tangent to the underside, i.e. the object rests on the table (the
    # Section 1.6 slide-and-grasp geometry); 0.5 = plane through the centroid,
    # i.e. half-buried. Consumed by auto_table_origin(); ignored entirely when
    # plane_origin is set explicitly.
    #
    # Default 0.5 because a whole object resting on the table puts its crown out
    # of the hand's reach envelope (see the `primitive` note above), and because
    # a half-buried proxy is how a genuinely low-profile object presents itself:
    # a shallow dome above the surface with no undercut to reach around.
    #
    # NOTE this does NOT feed h_clear, which stays measured from the object
    # CENTROID. Half-buried, the centroid lies on the
    # plane, so the hand hovers pregrasp_margin + half_extent above the table
    # over a half_extent dome -- a larger effective gap than pregrasp_margin
    # nominally promises. That is known and accepted, not an oversight.
    table_burial: float = 0.5

    # --- Phased controller (Section 1.8, Controller mode only) ---
    # Which constraint set is active: 0 = pre-grasp positioning, 1 = support
    # contact, 2 = object approach, 3 = on-object servoing. The controller never
    # advances itself — the policy stays here (or in the GUI) so it can be
    # iterated on without a rebuild.
    phase: int = 1
    # What anchors a tick to the measured state: "tension" (Eq 1.95, the
    # simulation default), "length" (the Eq 1.13 analogue, hardware-faithful) or
    # "both" (diagnostic; over-constrains a real tick).
    step_anchor: str = "tension"
    # Step-prior covariances (Eq 1.94/1.95 and the length analogue). These are the
    # per-tick trust region: how far the hand base, the tendon tensions and the
    # tendon lengths may move in one control step.
    #
    # The base sigmas are LOOSE on purpose. A trust region tight enough to pin
    # the base (the old 1e-3 / 1e-2) makes the prior stiffer than anything that
    # can push against it -- the recorded AL penalty ceiling is mu ~ 8e3, versus
    # 1/sigma^2 = 1e6 at sigma = 1e-3 -- so neither phase 1's support equality
    # nor phase 0's pre-grasp target can move the hand at all, and the controller
    # silently solves with a frozen base. Measured: phase 1 descends only above
    # sigma_pos ~ 1.1e-2, exactly where 1/sigma^2 drops below that mu ceiling.
    #
    # Loosened again to 1e-1 / 1.0: the wrist is arm-mounted, so a control tick
    # may legitimately command a macro repositioning, and 1e-2 was still an order
    # of magnitude short of letting phase 1 use it. Measured over 40 phase-1 ticks
    # (small sphere, index+thumb contact, plane through the object midpoint):
    #
    #   sigma_pos / sigma_rot | base travel | support violation
    #   1e-2  / 1e-1          |     1.9 mm  | 0.085 -> 0.083 m
    #   3.2e-2 / 3.2e-1       |     7.3 mm  | 0.085 -> 0.075 m
    #   1e-1  / 1.0           |    32.2 mm  | 0.085 -> 0.048 m
    #
    # NOTE what this does NOT buy: the base still barely ROTATES (~3 deg at every
    # setting above, since a loose prior only removes resistance -- it supplies no
    # torque), so a phase whose residual needs the palm tilted is not fixed by
    # loosening these. That is an al_mu / ctrl_al_iters question; see the AL
    # penalty budget below.
    #
    # And what it COSTS: freeing the base also frees it to push the fingers into
    # things, because the collision inequality resisting that is only as strong as
    # the same weak per-tick penalty. On ctrl_5f_phases (which already failed its
    # penetration check at the old default, in phase 1) the worst finger-object
    # clearance goes -6.3 mm -> -6.5 mm in phase 1 and +3.5 mm -> -9.4 mm in phase
    # 2. Raising al_mu to ~1e2 more than recovers it (+39 mm / +46 mm) but freezes
    # the servo -- the inner LM then reports iters=1 from the second tick. There
    # is no setting of this pair that does both, which points at the real problem
    # being graph SCALING rather than the trust region: the passive-tension step
    # priors (1e-6 variance, 150 of them) carry ~3.1e6 of error against ~3e-2 in
    # the constraints, i.e. 99.9% of the graph, and that is what the inner LM is
    # actually solving.
    #
    # Both are per-tick trust regions, so a caller that wants a slower hand should
    # rate-limit the COMMAND (as phase 0 does with pregrasp_slew_*) rather than
    # tighten these -- see the frozen-base note above.
    sigma_wrist_pos_step: float = 1e-1
    sigma_wrist_rot_step: float = 1.0

    # --- Tendon step priors: ACTIVE and PASSIVE are different machines -------
    # The controller has no BetweenFactor GP (that is the trajectory planner);
    # p_step(Q | Q_curr) and p_step(L | L_curr) ARE its entire step-to-step
    # regularization, so how they are split across the six tendons is what
    # decides which parts of the hand can move in a tick.
    #
    # Only tendon 5 (FLEXOR_IDX) is actuated. The other five are spring-backed,
    # and the two facts that follow are not symmetric:
    #
    #   TENSION  A spring holds roughly CONSTANT tension as it takes up slack,
    #            so the passive tensions are pinned hard (1e-6) and stay pinned
    #            under BOTH anchors -- that is their physics, not a modelling
    #            convenience, and it does not depend on what the motor is doing.
    #            Only the flexor's tension is free, at sigma_q_step.
    #   LENGTH   The motor commands the ACTIVE tendon's length, so that is the
    #            real measurement to anchor on (tight). A passive tendon's length
    #            changes freely as the finger moves -- pinning it would be
    #            pinning the joint angles, freezing the hand.
    #
    # sigma_l_step_active is a per-tick trust region on commanded tendon travel:
    # 1e-3 allows ~1 mm of 1-sigma motion. It has to stay loose enough that the
    # hand can actually get from one state to the next; if ticks stall with the
    # flexor barely moving, this is the first thing to check against the real
    # flexor excursion between an open and a closed hand.
    sigma_q_step: float = 1e-1
    sigma_l_step_passive: float = 1e-1
    sigma_l_step_active: float = 1e-3
    # Opposition half-space (Eq 2.16-2.17 / Eq 1.92), read by
    # HandSolverBase._attach_opposition(): the splitting point (None => the
    # object centroid) and the in-plane axis the split runs along (None =>
    # solvers.default_half_space_axis, derived from the object's own longest
    # in-plane axis so the split runs along an elongated object's length
    # rather than a fixed world direction). Needs table_contact fingers to act
    # on. Default False (this field used to be read only by the deleted §1.8
    # controller, which gated it by phase rather than this flag -- defaulting
    # it True here, now that it is live, would silently add a constraint to
    # every existing caller of HandSolveParams() that never touches this
    # field).
    half_space: bool = False
    half_space_split: Optional[np.ndarray] = None
    half_space_axis: Optional[np.ndarray] = None
    # Which SIDE of the split the thumb is asked to stay on. The derived axis
    # only fixes the split LINE; its sign is an arbitrary object-frame
    # convention, and getting it backwards asks the thumb and fingers to trade
    # sides (see orient_opposition_axis -- it stalls the solve outright).
    # None (default) = orient by the hand's current posture, False = keep the
    # derived sign, True = invert it. Ignored when half_space_axis is given
    # explicitly, which is taken as already oriented.
    half_space_flip: Optional[bool] = None
    # Minimum standoff (m) each contact finger must keep from the splitting
    # line, along its own m_hat: HalfSpaceGapFactor's d_min, so the constraint
    # is -(c - p_split) . m_hat + half_space_margin <= 0. 0.0 (the default) is
    # the original "anywhere on my own side" form, which a fingertip sitting
    # exactly ON the split already satisfies -- so the thumb and the opposing
    # fingers can be driven arbitrarily close together while both are "legal".
    # A positive value holds them 2 * margin apart, which is what makes this
    # useful as a PRE-grasp opening. Needs a binding carrying
    # EnvironmentConfig.half_space_margin (capabilities()["half_space_margin"]).
    half_space_margin: float = 0.0
    # Optional per-finger phase-3 witness targets (Eq 1.111); None entries mean
    # "contact anywhere on the surface" for that finger.
    witness_targets: Optional[List[Optional[np.ndarray]]] = None
    # A control tick's AL budget: outer iterations per tick. Small on purpose,
    # because with ctrl_al_warm_duals below the outer loop genuinely IS amortized
    # across ticks -- mu and the multipliers pick up where the last tick left off,
    # so a tick only has to advance the homotopy a little.
    #
    # HISTORICAL NOTE, kept because the conclusion inverted. This used to be
    # documented as amortized when it was not: SolverBase::optimize() built a
    # fresh AugmentedLagrangianOptimizer every call, so mu restarted at al_mu and
    # the duals at zero, capping a tick's penalty at
    # al_mu * al_rate^(ctrl_al_iters - 1) -- mu ~ 8 at the defaults, against the
    # mu ~ 8e3 an offline solve reaches. Measured on phase 1 then (small sphere,
    # index+thumb, 30 ticks), raising the budget bought nothing:
    #
    #   iters/tick |  mu cap  | support viol | base rotation | AL iters RUN
    #        4     |      8   |   0.0529 m   |    2.9 deg    |     2
    #       20     |   5.2e5  |   0.0478 m   |    2.9 deg    |    1-5
    #       40     |   5.5e11 |   0.0480 m   |    2.9 deg    |    1-5
    #
    # The outer loop never spent the budget it had: it exits on the stagnation
    # test (|d violation| < al_rel_violation_tol && |d cost| < al_rel_cost_tol),
    # which fires as soon as the inner LM rejects every step and both deltas are
    # exactly zero. That is still true -- a bigger budget WITHIN a tick still
    # buys mostly no-op outer iterations. What changed is that the progress a
    # tick does make now survives into the next one, so the ladder is climbed
    # across ticks instead of being rebuilt and abandoned on each.
    ctrl_al_iters: int = 4
    # Carry mu and the Lagrange multipliers from tick to tick (see above). This
    # is what makes the phased controller an Augmented Lagrangian method rather
    # than a weak penalty method restarted 30 times.
    ctrl_al_warm_duals: bool = True
    # Ceiling on the carried mu. mu compounds across ticks by design, and this
    # is what stops it running away -- but the value is NOT just a safety guard,
    # it is the balance point between the two constraint families and it is
    # sharp. Measured on ctrl_5f_phases (mid sphere, half-buried, all five
    # fingers), sweeping only this:
    #
    #   mu_max |  2   4   8  | 16
    #   result | PASS PASS PASS | FAIL (phase 1 penetrates)
    #
    # Above ~8 the support EQUALITY out-muscles the plane-avoidance
    # INEQUALITIES: the contact tips are driven onto the plane hard enough to
    # rotate the finger until a proximal sphere dips through it. Both families
    # carry multipliers, but the equality is always active while an inequality
    # only accumulates once violated, so a big shared mu favours the equality.
    #
    # Keeping mu small is the right shape for AL anyway -- lambda is supposed to
    # do the feasibility work, and mu only has to be large enough to keep the
    # subproblem convex. This is what a penalty method gets wrong, and why the
    # fix here was carrying lambda rather than raising mu.
    ctrl_al_mu_max: float = 8.0
    # Skip the Marginals factorization (a tick only consumes the means).
    ctrl_skip_marginals: bool = True

    # --- Phase 0: pre-grasp positioning (Section 1.8, Eq 1.92-1.98) ---
    # Explicit 4x4 target T_base,pre. None => derive it from the hand's own
    # forward kinematics via :func:`pregrasp_wrist_pose`.
    pregrasp_wrist_pose: Optional[np.ndarray] = None
    # Hover height of the CONTACT-SPHERE CENTROID above the object centroid along
    # the support normal. None => object_extent_along(spec, n_hat) +
    # pregrasp_margin, i.e. scaled to the object rather than an absolute number
    # (the capsule and cylinder stand their long axis along +Z, so a value tuned
    # on a sphere would not clear them).
    #
    # Also read live by HandSolverBase._attach_pregrasp_center() (Eq 2.19's
    # h_clear) when pregrasp_center is on -- same physical quantity, a
    # clearance offset along the support normal. None there falls back to a
    # flat 0.02 m rather than the object_extent_along derivation above (that
    # helper belonged to the deleted §1.8 phase-0 code).
    h_clear: Optional[float] = None
    pregrasp_margin: float = 0.04
    # Eq 1.92: Q_pre = [c]*5 + [c + pregrasp_flexor_offset], the "slightly curled"
    # pre-grasp posture. pregrasp_flexor_absolute overrides the offset form.
    # NOTE the default puts the flexor at 0.75 N, MORE curled than
    # GRASP_FLEXOR_TENSION (0.6), so phase 1 extends the fingers to reach the
    # table rather than curling further.
    pregrasp_flexor_offset: float = 0.25
    pregrasp_flexor_absolute: Optional[float] = None
    # Per-tick cap on how far the phase-0 TARGET may advance toward T_base,pre
    # (m and rad). This -- not the sigma ratio -- is the real rate limiter, and
    # it is what makes phase 0 work at all.
    #
    # The pre-grasp pose is roughly a 172 deg rotation away from an identity base
    # pose, and handing a stiff Pose3 prior a target that far off drives the merit
    # function to ~3e6 and the inner LM rejects every step (iters=1, nothing
    # moves, forever). Slewing a waypoint toward the target keeps the commanded
    # pose close to the achieved one, so the prior stays well-scaled and the
    # linearization stays valid the whole way. Expressed in m/tick and rad/tick
    # because that is what a caller actually wants to reason about.
    pregrasp_slew_pos: float = 0.02
    pregrasp_slew_rot: float = 0.25
    # Eq 1.94 Sigma_pre,base -- how hard the hand tracks the slewed waypoint. The
    # tracking lag is the sigma RATIO against sigma_wrist_*_step: the target and
    # step priors multiply, so a fraction
    #   rho = sigma_pre^2 / (sigma_pre^2 + sigma_step^2)
    # of the remaining error survives each tick.
    #
    # But the ABSOLUTE stiffness matters more than the ratio, and in the opposite
    # direction to what you might expect. A prior tight enough to whiten its own
    # residual to ~80 (e.g. sigma_rot = 3e-3 against a 0.25 rad waypoint) leaves
    # the linear system too badly scaled against the rod-physics factors for the
    # inner LM to take any step at all: it quits at iters=1 and the hand never
    # moves. Measured, sigma_rot >= ~1e-2 is navigable and reproduces the
    # predicted rho exactly; below that the servo is dead. Keep these loose and
    # let pregrasp_slew_* set the speed.
    sigma_pregrasp_pos: float = 3e-3
    sigma_pregrasp_rot: float = 3e-2
    # Eq 1.95's SECOND tension prior (on top of the step prior). Off by default
    # because it is inert in simulation: _tension_priors' mean is the COMMANDED
    # tension, not a measurement, so phase 0 simply commands Q_pre directly and a
    # second Gaussian at the same target adds nothing. Turn it on to exercise the
    # spec-faithful two-prior form, which is what hardware (a genuinely measured
    # Q_curr) will need.
    pregrasp_tension_prior: bool = False
    # Split passive/active for consistency with every other tendon prior, but
    # equal by default: unlike the step priors, Q_pre names a target for all six
    # tendons and there is no reason to pull on them with different authority.
    sigma_pregrasp_q_passive: float = 1e-1
    sigma_pregrasp_q_active: float = 1e-1
    # SUPERSEDED / unused: this used to gate deriving the Eq 1.92 half-space
    # axis from the object's longest in-plane axis as an opt-in, off by
    # default (world +X as m_hat). That derivation (m_hat = n_hat x e_long) is now
    # UNCONDITIONAL whenever half_space_axis is None -- see
    # HandSolverBase._attach_opposition() / default_half_space_axis() -- since
    # world +X turned out to be actively wrong for elongated objects (it
    # bisects a pen across its short axis instead of splitting along its
    # length). Kept only so an old caller that set this doesn't hit an
    # AttributeError; it is read nowhere.
    derive_half_space_axis: bool = False
    # Close the Eq 1.93 Theta_curr loop: after each tick, write the SOLVED base
    # pose back so the step prior is anchored to the achieved state. Without this
    # the prior mean stays at the construction-time pose forever and the base is
    # effectively pinned there — phase 0 cannot servo at all, and phases 1-3
    # cannot reposition the hand to reach the object.
    wrist_feedback: bool = True


class _FingerSol:
    """Duck-typed per-finger solution the viser/pyvista renderers consume:
    exposes ``.marginals`` (a ``TendonFingerMarginals``) and ``.meta``. Same shim
    the demo scripts build inline (e.g. ``ik_5f_contact.py``)."""

    __slots__ = ("marginals", "meta")

    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta


def _make_frame(finger_names, hand_marginals, meta):
    """One render frame: ``{finger_name: _FingerSol}`` for a single hand state."""
    return {name: _FingerSol(fm, meta)
            for name, fm in zip(finger_names, hand_marginals.fingers)}


@dataclass
class HandResult:
    """Uniform result for all three solvers. ``frames`` has length 1 for FK/IK and
    K+1 for the planner, so a step-scrubber can index it the same way regardless."""
    frames: List[dict]
    meta: object
    spec: dict
    object_center: np.ndarray
    object_rotation: np.ndarray
    finger_names: List[str]
    tip_radii: List[float]
    # Which fingers are DESIGNATED for contact (None = all). The mask, not the
    # set a solve happened to constrain: FK constrains none of them but still
    # carries it, so the §1.8 goal overlays -- p_bar, the opposition split, the
    # support-plane equalities -- describe the same finger set in the FK posing
    # state that the controller will enforce once a phase is picked.
    contact_fingers: Optional[List[bool]] = None
    # The raw ``TendonHandMarginals`` behind each frame, same indexing as
    # ``frames``. ``frames`` splits a solve up per finger for rendering, which
    # loses the bundle the C++ side wants back: this is the form
    # ``HandSolveParams.initial_state`` takes to warm-start a solver from a
    # posture instead of a straight hand.
    states: Optional[List[object]] = None
    # Solver-convergence snapshots: one entry per recorded iteration, each a
    # full ``frames``-shaped list (so an entry is indexed by trajectory step
    # exactly like ``frames`` is). Populated only when the solve ran with
    # ``HandSolveParams.record_iterations`` on a binding that exposes the
    # snapshots; None otherwise. ``iterate_states`` is the raw-marginals
    # parallel, the same relationship ``states`` has to ``frames``.
    iterates: Optional[List[List[dict]]] = None
    iterate_states: Optional[List[List[object]]] = None
    # One short markdown line per iterate, supplied by whoever produced the
    # snapshots. A stepped solve knows the cost/violation/mu behind each of its
    # entries directly; a one-shot recorded solve leaves this None and the
    # caller falls back to indexing ``meta``'s AL trace.
    iterate_notes: Optional[List[str]] = None
    # Which fingers were driven onto the SUPPORT PLANE, the table counterpart of
    # ``contact_fingers`` (which stays the object set). None = none of them, i.e.
    # the object-only solves every caller ran before the two were separable.
    # Appended last on purpose: several call sites build a result positionally.
    table_contact_fingers: Optional[List[bool]] = None
    # The solve's Augmented Lagrangian state (``crest_sparse.ALDuals``): the
    # multipliers and penalty weight, tagged with the identity of the constraint
    # each belongs to. Feed to ``HandSolveParams.initial_duals`` to continue this
    # solve after a rebuild. Only the stepper fills it; None everywhere else.
    duals: Optional[object] = None
    # ``crest_sparse.ALTransferReport`` for the transfer INTO this solve, i.e.
    # how many of its constraints inherited a multiplier. None when nothing was
    # carried in.
    dual_transfer: Optional[object] = None

    def state(self, k=0):
        """The solved hand state at frame ``k``, for seeding another solver.
        None on a result built before this field existed."""
        return None if self.states is None else self.states[k]

    def num_iterates(self):
        """How many solver-convergence snapshots this result carries (0 when the
        solve did not record any)."""
        return 0 if self.iterates is None else len(self.iterates)

    def at_iterate(self, i):
        """This result as it stood at recorded iteration ``i`` -- the same object
        with ``frames``/``states`` swapped for that snapshot.

        Everything downstream (the gap readouts, ``worst_gap``, the renderer)
        works off ``frames``, so a swapped-frames view makes all of it describe
        the intermediate state with no further plumbing. The view drops its own
        ``iterates`` so it cannot be re-scrubbed recursively."""
        return replace(self, frames=self.iterates[i],
                       states=None if self.iterate_states is None
                       else self.iterate_states[i],
                       iterates=None, iterate_states=None, iterate_notes=None)

    def contact_names(self):
        """The fingers designated to touch the object -- everything the gap
        readouts should be judged on. All of them when unmasked."""
        if self.contact_fingers is None:
            return list(self.finger_names)
        return [name for name, on in zip(self.finger_names, self.contact_fingers)
                if on]

    def table_contact_names(self):
        """The fingers designated to touch the SUPPORT PLANE. Empty unless the
        solve targeted the table (unlike :meth:`contact_names`, whose None case
        means "all of them" -- there the mask is an optional restriction, here it
        is the whole opt-in)."""
        if self.table_contact_fingers is None:
            return []
        return [name for name, on in zip(self.finger_names,
                                         self.table_contact_fingers) if on]

    def contact_witness(self, k=0):
        """Per-finger ``{name: (sphere_surface_pt, object_surface_pt, gap_m)}`` in
        world coordinates at frame ``k``: the shortest segment from each fingertip
        contact sphere to the object surface, and its signed length (~0 at contact,
        negative if the sphere interpenetrates).

        Uses the analytic ``primitive_surface_witness``, so for the baked-SDF
        primitives this measures against the analytic look-alike rather than the
        .vdb grid -- the same approximation :meth:`surface_gaps` has always made,
        differing only within the ``edge_radius`` fillets."""
        frame = self.frames[k]
        out = {}
        R = self.object_rotation
        for name, radius in zip(self.finger_names, self.tip_radii):
            fm = frame[name].marginals
            # Same node the renderer draws the contact sphere on (tip_node_index).
            tip = np.asarray(fm.rod.states[-1].pose.mean)[:3, 3]
            dist, foot_local, n_local = primitive_surface_witness(
                R.T @ (tip - self.object_center), self.spec)
            surface_pt = self.object_center + R @ foot_local
            sphere_pt = tip - radius * (R @ n_local)
            out[name] = (sphere_pt, surface_pt, dist - radius)
        return out

    def surface_gaps(self, k=0):
        """Per-finger fingertip surface gap (m, ~0 at contact) at frame ``k``,
        reusing the analytic surface distance the demos report with."""
        return {name: gap for name, (_, _, gap) in self.contact_witness(k).items()}

    def tendon_lengths(self, k=0):
        """Per-finger tendon lengths at frame ``k``, in ``finger_names`` order --
        the L component of Theta_curr a Section 1.8 control tick anchors on."""
        frame = self.frames[k]
        return [np.asarray(frame[name].marginals.tendon_lengths, float)
                for name in self.finger_names]

    def worst_gap(self, k=0):
        """Largest |gap| to the OBJECT over the fingers that were *asked* to touch
        it, so a masked subset grasp isn't scored on fingers left free."""
        gaps = self.surface_gaps(k)
        names = self.contact_names()
        return max((abs(gaps[n]) for n in names if n in gaps), default=0.0)

    def worst_table_gap(self, params, k=0):
        """The same score against the SUPPORT PLANE, over the fingers driven onto
        it. 0.0 when the solve targeted no table contact.

        Takes ``params`` because the plane is not part of a result: its origin is
        re-resolved from the scene the same way the solve resolved it."""
        names = self.table_contact_names()
        if not names:
            return 0.0
        gaps = plane_witness(params, self, k, names=names)
        return max((abs(g) for _p, _f, g in gaps.values()), default=0.0)


# ---------------------------------------------------------------------------
# Phase presets.
# ---------------------------------------------------------------------------

@dataclass
class PhasePreset:
    """A named group of ``HandSolveParams`` overrides for one phase of the
    §1.8-style pipeline (0: pre-grasp positioning, 1: support contact, 2:
    object approach, 3: on-object servoing -- only phase 0 is populated so
    far). Only the fields listed in ``overrides`` are touched when applied --
    wrist pose, per-finger flexor tensions, AL/collision tuning sliders, table
    height offset etc. are left at whatever the caller already has, since
    those are generic solver knobs rather than part of what DEFINES a phase."""
    label: str
    overrides: Dict[str, object]


PHASE_PRESETS: Dict[str, PhasePreset] = {
    "phase0": PhasePreset(
        label="Phase 0: pre-grasp positioning",
        overrides=dict(
            object_contact=False,
            table_contact=False,
            collision=True,
            table=True,
            plane_avoidance=True,
            # Centering is done by the PINCH CENTROID, not by the achieved
            # fingertip midpoint: the measured hand-frame pinch point is a
            # constraint on the wrist alone, so it positions the hand for a
            # grasp whatever the fingers are doing now, while pregrasp_center
            # only says something once they are nearly closed. The two impose
            # different targets, so exactly one of them runs.
            pregrasp_center=False,
            pregrasp_centroid=True,
            # Off with it: the opposition half-space keeps the thumb and the
            # opposing fingers apart around the split, which the pinch centroid
            # already implies (it places the hand so closing those digits closes
            # them ON the object) -- and it is the constraint most prone to
            # stalling the solve on a bad side assignment.
            half_space=False,
            # The one term here that actually rotates the wrist: the centroid
            # constraint is satisfiable by translation alone.
            pregrasp_axis_align=True,
            # Standoff above the object centroid for the pinch point.
            h_clear=0.07,
            contact_drop_normal_row=False,
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Loose wrist prior: phase 0 is a big repositioning move, so the
            # wrist must be free to get there rather than held near its start.
            sigma_wrist_pos=1.0,
            sigma_wrist_rot=1.0,
            # Explicit even though it equals the field's own default -- states
            # plainly that phase 0 uses the standard flexor looseness rather
            # than leaving it at "whatever the slider happened to be at."
            flexor_tension_sigma=0.1 ** 0.5,
        ),
    ),
    "phase1": PhasePreset(
        label="Phase 1: support contact",
        overrides=dict(
            object_contact=False,
            table_contact=True,
            collision=True,
            table=True,
            # Table COLLISION off (a deliberate departure from the paper, which
            # keeps it on): phase 1 drives the fingers deliberately onto the
            # plane, so the avoidance half-space is pushing against the very
            # contact this phase exists to make. `table` stays on -- the plane
            # itself is still needed, table_contact is built against it.
            plane_avoidance=False,
            # The three pre-grasp-only constraints did their job getting the
            # hand into position in phase 0; phase 1 slides the fingers onto
            # the table and doesn't need them anymore.
            half_space=False,
            pregrasp_center=False,
            pregrasp_axis_align=False,
            pregrasp_centroid=False,
            contact_drop_normal_row=False,
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Tighter than phase 0's 1.0: the big repositioning move is done,
            # so the wrist is held closer to where phase 0 left it -- but not
            # fully rigid (phase 0's own sigma_wrist_pos/rot default is much
            # smaller still), since settling into contact needs some give.
            sigma_wrist_pos=0.01,
            sigma_wrist_rot=0.01,
            flexor_tension_sigma=0.1 ** 0.5,
            # h_clear intentionally omitted -- pregrasp_center is off here, so
            # a clearance value would be inert and misleading to state.
        ),
    ),
    "phase2": PhasePreset(
        label="Phase 2: object approach",
        overrides=dict(
            # The only real change from phase 1: the object is now ALSO a
            # contact target, approached while table contact is maintained.
            object_contact=True,
            table_contact=True,
            collision=True,
            table=True,
            # Off for the same reason as phase 1: table contact is maintained
            # here, so the avoidance half-space would fight it. Object
            # collision (`collision`) stays on -- only the PLANE's avoidance
            # is dropped.
            plane_avoidance=False,
            half_space=False,
            pregrasp_center=False,
            pregrasp_axis_align=False,
            pregrasp_centroid=False,
            contact_drop_normal_row=False,
            contact_fingers=[True, True, False, False, True],  # index, middle, thumb
            # Loose again, like phase 0 -- object approach (sliding across
            # the table toward the object while keeping table contact) is
            # another significant motion, not the small settle phase 1's
            # 0.01 assumes.
            sigma_wrist_pos=1.0,
            sigma_wrist_rot=1.0,
            flexor_tension_sigma=0.1 ** 0.5,
            # h_clear intentionally omitted, as in phase1 -- pregrasp_center
            # is off, so a clearance value would be inert and misleading.
        ),
    ),
    # phase3 lands here later, same shape.
}


def apply_phase_preset(params: HandSolveParams, name: str) -> HandSolveParams:
    """Apply ``PHASE_PRESETS[name]``'s overrides onto ``params`` IN PLACE
    (``setattr`` per field), returning it for chaining. An override naming a
    field that doesn't exist on ``HandSolveParams`` raises -- a typo in a
    preset should fail loudly, not silently no-op."""
    preset = PHASE_PRESETS[name]
    for field_name, value in preset.overrides.items():
        if not hasattr(params, field_name):
            raise AttributeError(
                f"phase preset {name!r} sets unknown HandSolveParams field "
                f"{field_name!r}")
        setattr(params, field_name, value)
    return params


# ---------------------------------------------------------------------------
# Solver base + the three flavours.
# ---------------------------------------------------------------------------

class HandSolverBase:
    """Shared setup for the tendon-hand solvers: builds the anatomical hand from
    ``gepetto_core`` dims and holds the resolved scene. Subclasses implement
    :meth:`solve`."""

    def __init__(self, params: Optional[HandSolveParams] = None):
        self.params = params or HandSolveParams()
        self.dims = load_hand_dimensions()
        self.configs = get_default_hand_configs(self.dims)
        self.tip_radii = default_hand_tip_radii(self.dims)
        self.finger_names = [name for name, _ in self.configs]
        self.spec, self.object_center, self.object_rotation, self.object_pose = \
            resolve_scene(self.params)

    # -- contact masks --
    #
    # One finger selection (params.contact_fingers) times one per-surface switch.
    # Both surfaces read the same finger list, so "index and thumb, on the table
    # only" is a two-flag change rather than two masks to keep in sync.

    def _object_contact_mask(self):
        """Fingers driven onto the OBJECT surface."""
        return [bool(b) and self.params.object_contact
                for b in self.params.contact_fingers]

    def _table_contact_mask(self):
        """Fingers driven onto the SUPPORT PLANE. Empty without ``params.table``:
        there is no plane configured to touch."""
        on = self.params.table_contact and self.params.table
        return [bool(b) and on for b in self.params.contact_fingers]

    # -- environment attachment (mutates self.configs in place) --

    def _attach_contact(self):
        """Per-finger contact env: shared object surface + this finger's tip node
        as the terminal contact (``ik_5f_contact.py`` block). Fingers masked off
        get a collision-only env instead -- which is also what every finger gets
        with ``params.object_contact`` off, leaving the object present as
        collision geometry but with nothing driven onto it."""
        attach_contact(self.configs, self.spec, _OBJECTS_DIR,
                       self.params.primitive, self.object_pose,
                       tip_radii=self.tip_radii,
                       contact_fingers=self._object_contact_mask(),
                       drop_normal_row=self.params.contact_drop_normal_row,
                       ellipsoid_set_beta=self.params.ellipsoid_set_beta)

    def _attach_collision(self, avoidance=True):
        """Add Section 1.5 collision spheres onto each finger's (already attached)
        env. Reuses the contact env, so it works for SDF and ellipsoid objects
        alike (the vdb path is only used if a finger has no env yet).

        ``avoidance`` selects whether the finger-OBJECT inequalities are built
        and ``params.self_collision`` whether the finger-finger ones are; the
        spheres themselves are declared either way, because the support plane
        builds its own inequalities on the same set."""
        vdb = (None if self.spec["type"] in ("ellipsoid", "ellipsoid_set")
               else os.path.normpath(os.path.join(_OBJECTS_DIR, self.spec["vdb"])))
        attach_collision(self.configs, vdb, self.object_pose,
                         radius=self.params.collision_radius,
                         sigma=self.params.collision_sigma,
                         num_proximal_discs=self.params.num_proximal_discs,
                         cull_margin=self.params.cull_margin,
                         avoidance=avoidance,
                         self_collision=self.params.self_collision)

    def _attach_table(self):
        """Attach the Section 1.6 support plane to every finger's env."""
        origin = resolve_table_origin(self.params, self.spec, self.object_center)
        attach_table(self.configs, origin, self.params.plane_normal,
                     avoidance=self.params.plane_avoidance,
                     tip_radii=self.tip_radii,
                     contact_fingers=self._table_contact_mask())

    def _attach_opposition(self):
        """Attach the Eq 2.16-2.17 opposition half-space to every finger's env.

        Masked by the shared ``contact_fingers``, like every other constraint
        in the set: the C++ layer builds this one off its own
        ``half_space_node``, so it no longer needs -- or silently waits for --
        table contact on the same finger.
        The thumb (identified by name, the hand-wide convention) gets ``+axis``
        and every other checked finger gets ``-axis``
        (:func:`opposition_directions`). ``half_space_split`` defaults to the
        object center; ``half_space_axis`` defaults to
        :func:`default_half_space_axis` -- derived from the object's own
        longest in-plane axis, so the split runs along an elongated object's
        length (e.g. a pen) rather than a fixed world direction that is only
        right by coincidence.

        That derived axis fixes the split LINE only. Its SIGN -- which half the
        thumb is sent to -- is oriented against the hand's current posture by
        :func:`orient_opposition_axis`, because the object-frame sign is
        arbitrary and the wrong one asks the hand to turn itself inside out.
        The resolved axis is written back onto ``params.half_space_axis`` so the
        witness overlay and any later rebuild describe the constraint that was
        actually built, rather than re-deriving and disagreeing with it."""
        explicit = self.params.half_space_axis is not None
        axis = (self.params.half_space_axis if explicit
               else default_half_space_axis(self.spec, self.object_rotation,
                                            self.params.plane_normal))
        if not explicit:
            thumb, others = self._opposition_tips()
            axis, _flipped = orient_opposition_axis(
                axis, thumb, others, flip=self.params.half_space_flip)
            self.params.half_space_axis = np.asarray(axis, float)
        directions = opposition_directions(self.configs, axis=axis)
        split = (self.params.half_space_split if self.params.half_space_split is not None
                else self.object_center)
        attach_half_space(self.configs, split, directions,
                          contact_fingers=self.params.contact_fingers,
                          margin=self.params.half_space_margin)

    def _opposition_tips(self):
        """``(thumb_tip, [other checked fingertips])`` at the posture this solve
        starts from, for orienting the opposition axis.

        Measured with a throwaway FK solve (~180 ms, cached for the life of this
        solver) rather than read off the configs: the fingertips are where the
        TENSIONS put them, and the thumb-vs-fingers direction swings by more
        than a right angle across the flexor range -- the finger BASES, which
        are free to read, sit only ~5 mm apart along the opposition axis and get
        the sign wrong for 3 of 7 sampled wrist poses. Only the sign of one dot
        product is taken from this, so the ~100 mm the tips move between the FK
        pose and a warm-started posture cannot change the answer."""
        if getattr(self, "_fk_probe_tips", None) is None:
            frame = HandFKSolver(self.params).solve().frames[0]
            self._fk_probe_tips = {
                name: np.asarray(frame[name].marginals.rod.states[-1].pose.mean,
                                 float)[:3, 3]
                for name in self.finger_names}
        tips = self._fk_probe_tips
        mask = self.params.contact_fingers
        others = [tips[n] for n, on in zip(self.finger_names, mask)
                  if on and n != "thumb"]
        return tips.get("thumb"), others

    def _attach_pregrasp_center(self):
        """Attach the Eq 2.18-2.19 pre-grasp hand-centering constraint, using
        the shared ``contact_fingers`` mask to pick which fingers (thumb +
        opposing set) participate, and ``plane_normal`` as the clearance axis."""
        h_clear = self.params.h_clear if self.params.h_clear is not None else 0.02
        attach_pregrasp_center(self.configs, clearance_height=h_clear,
                               clearance_normal=self.params.plane_normal,
                               contact_fingers=self.params.contact_fingers)

    def _attach_pregrasp_axis_alignment(self):
        """Attach the pre-grasp short-axis alignment constraint (companion to
        Eq 2.16-2.17), using the shared ``contact_fingers`` mask for the thumb
        + opposing set. Computes its own copy of the opposition axis via
        :func:`default_half_space_axis` -- independent of whether
        ``_attach_opposition()`` itself runs, so this stays toggleable on its
        own."""
        axis = default_half_space_axis(self.spec, self.object_rotation,
                                       self.params.plane_normal)
        attach_pregrasp_axis_alignment(self.configs, axis,
                                       contact_fingers=self.params.contact_fingers)

    def _attach_pregrasp_centroid(self):
        """Attach the pre-grasp pinch-centroid constraint for the CHECKED
        digits, and report whether it went on.

        Returns the :class:`config.PinchPose` used, or None when the checked
        set has no measured pose. The return value exists so a caller can say
        so out loud: the C++ layer skips an unconfigured constraint silently,
        and a constraint that quietly does nothing is the trap this whole
        family of toggles keeps setting.
        """
        pose = pinch_pose_for_mask(self.configs, self.params.contact_fingers)
        if pose is None:
            return None
        h_clear = self.params.h_clear if self.params.h_clear is not None else 0.02
        attach_pregrasp_centroid(self.configs, pose.centroid,
                                 clearance_height=h_clear,
                                 clearance_normal=self.params.plane_normal)
        return pose

    def _attach_environment(self):
        """The whole constraint environment for one solve, per the independent
        toggles (object contact, table contact, object collision, table
        collision, opposition half-space, pre-grasp centering, pre-grasp
        short-axis alignment).

        Every constraint family is gated on its own toggle alone -- checking one
        builds it, full stop. The collision sphere SET is shared, so it is
        attached whenever ANY of its three consumers (object, finger-finger,
        plane) wants it, and each family's own field then decides what gets
        built on it.

        Shared by the IK solver, the IK stepper and the planner so the three
        cannot drift into building different environments from the same params."""
        self._attach_contact()
        if (self.params.collision or self.params.self_collision
                or (self.params.table and self.params.plane_avoidance)):
            self._attach_collision(avoidance=self.params.collision)
        if self.params.table:
            self._attach_table()
        if self.params.half_space:
            self._attach_opposition()
        if self.params.pregrasp_center:
            self._attach_pregrasp_center()
        if self.params.pregrasp_axis_align:
            self._attach_pregrasp_axis_alignment()
        if self.params.pregrasp_centroid:
            self._attach_pregrasp_centroid()

    # -- prior builders --

    def _tension_priors(self, cov, means=None):
        """One ``VectorXGaussian`` per finger: passive tendons at the background
        hold, flexor (index 5) at that finger's commanded tension.

        ``means`` overrides the per-finger mean vectors wholesale -- the Section
        1.8 phase-0 pre-grasp posture commands ``Q_pre`` that way.
        """
        priors = []
        for i, (_, cfg) in enumerate(self.configs):
            if means is not None:
                mean = np.asarray(means[i], float)
            else:
                mean = np.full(cfg.num_tendons, self.params.passive_tension)
                mean[FLEXOR_IDX] = self.params.flexor_tensions[i]
            priors.append(crest_sparse.VectorXGaussian(mean, cov))
        return priors

    def _length_priors(self, means, cov):
        """One ``VectorXGaussian`` per finger pinning that finger's tendon lengths
        near ``means[i]`` — the Eq 1.13 / Eq 1.95 length step prior the Section 1.8
        controller uses to anchor a tick to the measured motor positions."""
        return [crest_sparse.VectorXGaussian(np.asarray(m, float), cov)
                for m in means]

    def _tip_wrenches(self):
        cov = self.params.tip_wrench_sigma ** 2 * np.eye(6)
        return [crest_sparse.Vector6Gaussian(np.zeros(6), cov) for _ in self.configs]

    def _flexor_tension_cov(self):
        """The "tight-passive / loose-flexor" tension-prior covariance used
        outside the leading settle steps: the five passives at
        ``params.passive_tension_sigma ** 2`` (their physics -- a spring holds
        roughly constant tension, so this is normally left tight), the
        actuated flexor (index 5) at ``params.flexor_tension_sigma ** 2`` so
        contact can drive it away from its commanded value. Read live every
        call, like ``_tip_wrenches()``, so a mid-solve slider drag takes
        effect on the next step with no stepper rebuild."""
        cov = np.diag([self.params.passive_tension_sigma ** 2] * 6)
        cov[FLEXOR_IDX, FLEXOR_IDX] = self.params.flexor_tension_sigma ** 2
        return cov

    def _result(self, frames, meta, contact_fingers=None, states=None,
                iterates=None, iterate_states=None, iterate_notes=None,
                table_contact_fingers=None, duals=None, dual_transfer=None):
        # The table mask defaults to the one this solve actually built, so every
        # result carries it without each call site restating it; a caller that
        # means "no table" (the controller) passes an explicit all-False list.
        if table_contact_fingers is None:
            table_contact_fingers = self._table_contact_mask()
        return HandResult(frames, meta, self.spec, self.object_center,
                          self.object_rotation, self.finger_names, self.tip_radii,
                          contact_fingers, states, iterates, iterate_states,
                          iterate_notes, table_contact_fingers, duals,
                          dual_transfer)

    def solve(self) -> HandResult:  # pragma: no cover - abstract
        raise NotImplementedError


class HandFKSolver(HandSolverBase):
    """Pure-kinematics hand solve driven by tensions (no contact). Builds its
    ``TendonHandSolver`` once and re-commands the wrist each solve, so repeated
    calls warm-start from the previous solution (``fk_5f_sweep.py``)."""

    def __init__(self, params: Optional[HandSolveParams] = None):
        super().__init__(params)
        cfg = crest_sparse.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.max_iterations = 500
        self._solver = crest_sparse.TendonHandSolver(self.configs, cfg)

    def solve(self) -> HandResult:
        # Re-aim the shared wrist prior (warm start; no rebuild).
        self._solver.set_wrist_pose(self.params.wrist_pose)
        # Uniform prior on every tendon: a tight-passive/loose-flexor prior is
        # underdetermined without contact (IndeterminantLinearSystem on the
        # tension variable) -- see fk_5f_sweep.py.
        cov = (1e-2) ** 2 * np.eye(6)
        sol = self._solver.solve(self._tension_priors(cov), self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        # FK constrains no finger, but the mask still rides along: it is read
        # live off params (not baked in at construction), and the goal overlays
        # drawn over an FK pose -- p_bar, the opposition split, the support-plane
        # equalities -- are all statements about the DESIGNATED contact set.
        return self._result([frame], sol.meta, self.params.contact_fingers,
                            [sol.marginals])


# The tight-passive/loose-flexor prior itself now lives on
# HandSolverBase._flexor_tension_cov() (params.flexor_tension_sigma,
# params.passive_tension_sigma), shared
# by the one-shot IK solve, the stepper and the planner so the three cannot
# drift into solving subtly different problems -- the whole point of the
# stepper is that it advances *this* solve.

# The same prior with the FLEXOR pinned as hard as the passives, used only to
# settle a cold start (HandIKStepper.step, params.ik_settle_steps).
#
# The C++ initial values seed every tension at ZERO on a rod already at its rest
# shape (TendonRobotModel::get_initial_values) -- a guess that looks right but is
# statically inconsistent. Against a commanded 0.5 N passive at sigma 1e-3 that
# is 0.5 * (0.5/1e-3)^2 * 25 = 3.125e6 units of prior cost, which is exactly the
# cost the AL trace reports at its iteration 0. The first inner LM solve spends
# its whole budget hauling the 25 passive tensions home, and the flexor -- whose
# prior is 1e5x weaker in weight -- is the cheap direction that absorbs the
# leftover inconsistency: it swings to about -0.9 N. Negative tension is
# HYPEREXTENSION, so the hand visibly bends backwards, and since the inner LM is
# capped at 100 iterations per outer step it then takes ~13 steps to crawl back
# to the FK pose the GUI was already showing. Measured: it is one continuous LM
# descent chopped into 100-iteration slices, not the constraints pulling (a 400-
# iteration budget reaches step 4's state in one step, and the excursion is
# identical for al_mu from 1e-6 to 10, and for contact-only, collision-only or
# both).
#
# Pinning the flexor for the settling step removes the soft direction, so the
# transient is absorbed by the passives alone: step 1 lands exactly on the FK
# pose and the solve reaches the same converged grasp in the same number of steps
# and less wall time.
_IK_SETTLE_TENSION_COV = np.diag([1e-6] * 6)


class HandIKSolver(HandSolverBase):
    """Single terminal grasp: each fingertip driven onto the shared object surface
    by a hard contact constraint (Augmented Lagrangian). ``ik_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_environment()

        cfg = crest_sparse.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.al_initial_mu = self.params.al_mu
        cfg.base.al_mu_increase_rate = self.params.al_rate
        cfg.base.al_max_iterations = self.params.al_iters
        _set_if(cfg.base, "record_iterations", self.params.record_iterations)
        if self.params.record_iterations:
            # On the AL path an interval of 0 already means "snapshot every outer
            # iteration", but the plain LM/Dogleg path (a contact-free solve)
            # stores nothing unless the interval is > 0. Setting 1 records every
            # iteration either way.
            _set_if(cfg.base, "iteration_sample_interval", 1)

        solver = crest_sparse.TendonHandSolver(self.configs, cfg)
        sol = solver.solve(self._tension_priors(self._flexor_tension_cov()),
                           self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        iterates, iterate_states = self._collect_iterates(solver, sol)
        return self._result([frame], sol.meta, self.params.contact_fingers,
                            [sol.marginals], iterates, iterate_states)

    def _collect_iterates(self, solver, sol):
        """The solve's convergence snapshots as ``(iterates, iterate_states)``,
        or ``(None, None)`` when nothing was recorded.

        The sequence is initial guess, then one entry per recorded iteration,
        then the final solution. That last entry is what ``sol.marginals``
        already holds, so it may repeat the last AL iterate -- it is appended
        anyway so the end of the scrubber always shows exactly the state the
        rest of the result reports on."""
        if not (self.params.record_iterations
                and hasattr(solver, "get_intermediate_solutions")):
            return None, None
        states = ([solver.get_initial_solution()]
                  + list(solver.get_intermediate_solutions())
                  + [sol.marginals])
        iterates = [[_make_frame(self.finger_names, hm, sol.meta)] for hm in states]
        return iterates, [[hm] for hm in states]


class StepStatus(NamedTuple):
    """Where a stepped IK solve stands after the last outer iteration."""
    state: str        # "running" | "converged" | "stalled"
    violation: float  # worst constraint violation
    cost: float       # objective (constraint penalty excluded)
    mu: float         # current AL penalty weight
    steps: int        # outer iterations taken so far

    @property
    def done(self):
        return self.state != "running"


class HandIKStepper(HandSolverBase):
    """The IK solve of :class:`HandIKSolver`, advanced one Augmented Lagrangian
    outer iteration per :meth:`step` call instead of run to convergence in one go.

    Every step solves the *identical* graph the one-shot solve builds -- same
    contact constraints, same priors (``_flexor_tension_cov()``), same tolerances. The
    only difference is that the outer loop is told to stop after one iteration
    and resume on the next call: ``al_max_iterations = 1`` with
    ``al_warm_start_duals`` carries mu, the Lagrange multipliers and the values
    across calls, so N steps are the N iterations the one-shot solve would have
    run internally. Holding the loop counter is the whole trick.

    The one deliberate departure is the leading ``params.ik_settle_steps`` steps,
    which pin the flexor prior to settle the cold start before releasing it (see
    ``_IK_SETTLE_TENSION_COV``). Without it the first steps are spent watching the
    hand hyperextend and crawl back, which is a solver transient rather than
    anything the solve is being asked to do.

    Like :class:`HandFKSolver` this owns its ``crest_sparse.TendonHandSolver`` for
    its lifetime -- that is what lets anything carry at all, since a solver
    rebuilt per call cold-starts even its values. Tensions and the wrist pose are
    passed per step, so they stay live between steps; anything that changes the
    CONSTRAINT SET (object, contact mask, collision, table) needs :meth:`reset`,
    because the carried duals describe the old constraints.
    """

    def __init__(self, params: Optional[HandSolveParams] = None):
        super().__init__(params)
        self._build()

    # -- construction / restart --

    def _build(self):
        self._attach_environment()

        cfg = crest_sparse.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.al_initial_mu = self.params.al_mu
        cfg.base.al_mu_increase_rate = self.params.al_rate
        # One outer iteration per solve() call, resumed from the last one.
        cfg.base.al_max_iterations = 1
        _set_if(cfg.base, "al_warm_start_duals", True)
        # Effectively uncapped mu, and it has to be. al_warm_mu_max exists for
        # the Section 1.8 controller, where mu compounds forever across ticks of
        # a MOVING problem and would eventually swamp the rod physics -- but the
        # one-shot solve this stepper reproduces runs optimize() with no clamp at
        # all, so its mu reaches ~2^28 by the time the contact closes. Leaving
        # the 1e4 default in place stalls the stepped solve at a ~60 mm gap while
        # the one-shot reaches 0.1 mm: the cap, not the method, is the difference.
        _set_if(cfg.base, "al_warm_mu_max", 1e12)
        # ...but a mu carried in from ANOTHER solver is clamped: see
        # HandSolveParams.al_transfer_mu_max.
        _set_if(cfg.base, "al_transfer_mu_max", self.params.al_transfer_mu_max)
        # The al_iteration_* arrays are the only readout of what a step did, and
        # they are populated only when recording is on -- so this is not optional
        # here the way it is for the one-shot solve.
        _set_if(cfg.base, "record_iterations", True)
        # Means-only extraction: the covariance factorization is the most
        # expensive step after the optimizer itself, and nothing downstream of a
        # step reads a covariance. Gated, because a binding without the
        # means-only branch would read an empty marginals object.
        if capabilities()["ik_stepping"]:
            _set_if(cfg.base, "skip_marginals", True)
        # Warm-start posture, when the caller committed one. Read here rather
        # than in __init__ so a reset() picks up whatever params carries now:
        # seeding IS the mechanism for carrying a solve across the rebuild that
        # a changed constraint set forces.
        if self.params.initial_state is not None:
            _set_if(cfg, "initial_state", self.params.initial_state)

        # Read the stopping tolerances back off the config rather than keeping a
        # second copy: status() has to mirror the C++ convergence test (which
        # reports no stop reason of its own), and reading the same object it was
        # configured from is the only way the two cannot drift.
        self._tols = (cfg.base.al_abs_violation_tol, cfg.base.al_abs_cost_tol,
                      cfg.base.al_rel_violation_tol, cfg.base.al_rel_cost_tol)

        self._solver = crest_sparse.TendonHandSolver(self.configs, cfg)
        # Multipliers carried in from a previous solver, re-seated onto THIS
        # solver's constraints by identity on its first solve. Set on the solver
        # rather than the config because the remap needs this graph, which does
        # not exist until that solve runs.
        if (self.params.initial_duals is not None
                and hasattr(self._solver, "set_initial_duals")):
            self._solver.set_initial_duals(self.params.initial_duals)
        self._history = []      # TendonHandMarginals per step (initial guess first)
        self._frames = []       # the same states as render frames
        self._notes = []        # one readout line per history entry
        self._steps = 0
        self._status = StepStatus("running", float("inf"), float("inf"),
                                  self.params.al_mu, 0)
        self._prev = None       # (violation, cost) of the previous step

    def reset(self):
        """Full cold start: straight-hand values, zero duals, mu back to
        ``al_mu``, and the scene/envs re-derived from the current params.

        Rebuilding is what makes it a *cold* start -- ``get_initial_values()``
        runs only in the C++ constructor, so nothing short of a new solver
        restores the initial posture. Re-running the base init also gives
        :meth:`_build` clean finger configs, since ``_attach_*`` mutate them in
        place and would otherwise stack a second environment on the first."""
        super().__init__(self.params)
        self._build()

    def restart_al(self):
        """Re-run the penalty schedule from the CURRENT posture: drops the duals
        and mu but keeps the pose reached so far. The weaker sibling of
        :meth:`reset`; no-op on a binding without the accessor."""
        if hasattr(self._solver, "reset_al_duals"):
            self._solver.reset_al_duals()
            self._prev = None
            self._status = self._status._replace(state="running",
                                                 mu=self.params.al_mu)
            return True
        return False

    # -- stepping --

    def status(self) -> StepStatus:
        return self._status

    def al_duals(self):
        """This solve's AL multipliers + penalty weight, tagged by constraint
        identity -- what a differently-constrained rebuild takes to continue
        holding the constraints the two problems share. None on a binding
        without the accessor."""
        if not hasattr(self._solver, "get_al_duals"):
            return None
        return self._solver.get_al_duals()

    def dual_transfer(self):
        """How much of the incoming transfer matched (or None). A 0/N here on a
        rebuild that should have matched means the constraint TAGS drifted, not
        that the problem genuinely changed."""
        if not hasattr(self._solver, "al_transfer_report"):
            return None
        rep = self._solver.al_transfer_report()
        return rep if rep.total else None

    def factor_errors(self):
        """``[(factor_family, count, total_error)]`` at the CURRENT values.

        Read-only; the one readout that says which part of the graph the inner LM
        is spending its budget on, which is what decides whether a stalled step is
        a constraint problem or a scaling problem. Empty on a binding without the
        accessor."""
        if not hasattr(self._solver, "get_factor_error_summary"):
            return []
        return [(str(name), int(count), float(err))
                for name, count, err in self._solver.get_factor_error_summary()]

    def _settling(self):
        """True while this step should pin the flexor to settle the cold start.

        Counted off ``self._steps`` (steps already taken), not off a flag, so a
        :meth:`reset` re-settles and :meth:`restart_al` -- which keeps the posture
        -- correctly does not.

        NEVER when the solve was seeded (``params.initial_state``). Settling pins
        every tendon at its COMMANDED mean, which is the right way to absorb a
        cold start's statically inconsistent Q = 0 guess -- and exactly the wrong
        thing to do to a warm one: a contact solve drives the flexor well away
        from its commanded value (1.28 N against a commanded 0.6 N is typical),
        so pinning it back hauls the fingers open by ~57 mm on step 1 and throws
        away the posture the seed existed to preserve. A seeded start is already
        consistent, which is the only thing settling was ever for."""
        if self.params.initial_state is not None:
            return False
        return self._steps < max(int(self.params.ik_settle_steps), 0)

    def step(self) -> HandResult:
        """Advance the AL outer loop by exactly one iteration."""
        # Re-aimed every step so the wrist slider stays live mid-solve; the
        # tension priors are rebuilt from params for the same reason.
        self._solver.set_wrist_pose(self.params.wrist_pose)
        settling = self._settling()
        cov = _IK_SETTLE_TENSION_COV if settling else self._flexor_tension_cov()
        sol = self._solver.solve(self._tension_priors(cov), self._tip_wrenches())

        if not self._history:
            # The pre-step values of the first step: the true initial guess, and
            # the only frame the history cannot get from a solve result.
            self._append(self._solver.get_initial_solution(), sol.meta,
                         "initial guess")
        self._steps += 1
        self._update_status(sol.meta, settling=settling)
        s = self._status
        self._append(sol.marginals, sol.meta,
                     f"step {s.steps} &nbsp; violation={s.violation:.3e} "
                     f"&nbsp; cost={s.cost:.4g} &nbsp; mu={s.mu:.3g}"
                     + (" &nbsp; *(settling: flexor pinned)*" if settling else ""))

        return self._result(
            [self._frames[-1]], sol.meta, self.params.contact_fingers,
            [self._history[-1]],
            [[f] for f in self._frames], [[hm] for hm in self._history],
            list(self._notes), duals=self.al_duals(),
            dual_transfer=self.dual_transfer())

    def _append(self, hand_marginals, meta, note):
        self._history.append(hand_marginals)
        self._frames.append(_make_frame(self.finger_names, hand_marginals, meta))
        self._notes.append(note)

    def _update_status(self, meta, settling=False):
        """Mirror ``ConstrainedOptimizer::checkConvergence`` on this step's trace.

        With ``al_max_iterations = 1`` the C++ loop always exits on its iteration
        test before evaluating the tolerances, and it records no stop reason, so
        the caller has to apply the same two tests itself. Each call logs a seed
        state plus the one iterate; the last entry is the state we just reached.

        A ``settling`` step is exempt from both verdicts and leaves no baseline
        behind: it minimizes a DIFFERENT objective (the flexor is pinned), so its
        cost is not comparable to the released steps on either side of it, and a
        settled cold start that stops moving is the step doing its job rather than
        the solve giving up."""
        def last(name, default=float("nan")):
            arr = list(getattr(meta, name, []) or [])
            return float(arr[-1]) if arr else default

        viol, cost, mu = (last("al_iteration_violations"),
                          last("al_iteration_costs"),
                          last("al_iteration_mus", self._status.mu))
        abs_v, abs_c, rel_v, rel_c = self._tols
        if settling:
            state = "running"
        elif viol < abs_v and cost < abs_c:
            state = "converged"
        elif (self._prev is not None
              and abs(viol - self._prev[0]) < rel_v
              and abs(cost - self._prev[1]) < rel_c):
            state = "stalled"
        else:
            state = "running"
        self._prev = None if settling else (viol, cost)
        self._status = StepStatus(state, viol, cost, mu, self._steps)

    def run(self, max_steps=200, on_step=None, should_stop=None) -> StepStatus:
        """Step until the solve converges, stalls, is stopped, or hits the cap.

        ``on_step(result, status)`` fires after every iteration (that is what
        lets a caller animate the convergence) and ``should_stop()`` is polled
        between iterations, so an interactive caller can break out without
        waiting for the whole run."""
        for _ in range(max_steps):
            if should_stop is not None and should_stop():
                break
            result = self.step()
            if on_step is not None:
                on_step(result, self._status)
            if self._status.done:
                break
        return self._status

    # A single "solve" of a stepper is one iteration, mirroring the controller's
    # solve == one tick, so it can stand in wherever a solver is expected.
    solve = step


class HandPlannerSolver(HandSolverBase):
    """A K+1-step grasp trajectory tied by GP temporal priors on the wrist pose and
    finger tensions, with terminal contact constraints. ``traj_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_environment()

        n = self.configs[0][1].num_tendons
        pc = crest_sparse.TendonHandTrajectoryPlannerConfig()
        pc.K = self.params.K
        pc.dt = self.params.dt
        pc.wrist_pose = self.params.wrist_pose
        pc.sigma_wrist_pos = self.params.sigma_wrist_pos
        pc.sigma_wrist_rot = self.params.sigma_wrist_rot
        pc.gp_wrist_Qc = self.params.gp_wrist * np.eye(6)
        pc.gp_tense_Qc = self.params.gp_tense * np.eye(n)
        pc.gp_len_Qc = (self.params.gp_len * np.eye(n)
                        if self.params.gp_len > 0.0 else np.zeros((0, 0)))
        pc.base.linear_solver_type = "MULTIFRONTAL_QR"
        pc.base.al_initial_mu = self.params.al_mu
        pc.base.al_mu_increase_rate = self.params.al_rate
        pc.base.al_max_iterations = self.params.al_iters
        # Inexact-AL tuning and slide-grasp scheduling exist only on newer builds.
        _set_if(pc.base, "al_inner_rel_tol_initial", self.params.al_inner_tol)
        _set_if(pc.base, "al_abs_cost_tol", self.params.al_abs_cost_tol)
        _set_if(pc.base, "record_iterations", self.params.record_iterations)
        if self.params.table and self.params.k_touch is not None:
            _set_if(pc, "k_touch", self.params.k_touch)

        planner = crest_sparse.TendonHandTrajectoryPlanner(self.configs, pc)

        # Target tensions at k>=1 (tight passive / loose flexor), plus the measured
        # k=0 start (open hand at start_flexor, all pinned) that the trajectory
        # closes from.
        cov = self._flexor_tension_cov()
        start_cov = np.diag([1e-6] * n)
        starts = []
        for _, cfg in self.configs:
            sm = np.full(cfg.num_tendons, self.params.passive_tension)
            sm[FLEXOR_IDX] = self.params.start_flexor
            starts.append(crest_sparse.VectorXGaussian(sm, start_cov))

        result = planner.plan(self._tension_priors(cov), self._tip_wrenches(),
                              start_tensions=starts)
        frames = [_make_frame(self.finger_names, hm, result.meta)
                  for hm in result.trajectory]
        return self._result(frames, result.meta, self.params.contact_fingers,
                            list(result.trajectory))


# Convenience registry the visualizer uses to switch modes.
SOLVERS = {
    "FK": HandFKSolver,
    "IK": HandIKSolver,
    "Planner": HandPlannerSolver,
}
