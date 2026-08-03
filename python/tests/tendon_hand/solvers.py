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
from typing import List, NamedTuple, Optional

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    disc_node_indices, attach_contact, attach_collision, attach_table,
    attach_half_space, attach_witness_targets, opposition_directions,
    opposition_axis_from_object, rotation_from_two_axes, tip_node_index)
from .scene import (
    OBJECT_CENTER, GRASP_SPHERE_CENTER, GRASP_FLEXOR_TENSION, TABLE_NORMAL,
    get_primitive_specs, primitive_surface_witness, primitive_surface_gap,
    object_principal_inplane_axis, proxy_semi_axes)


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
        "table": hasattr(env, "plane_normal"),
        "collision_cull": hasattr(env, "collision_cull_margin"),
        "k_touch": hasattr(pc, "k_touch"),
        # Section 1.8 phased controller. Needs both the solver class and the env
        # fields its phase schedule reads, so a partially-rebuilt binding can't
        # report itself as capable.
        "controller": (hasattr(crest_sparse, "TendonHandController")
                       and hasattr(env, "support_contact_node")),
        # Phase 0 pre-grasp positioning. Needs the enum value, the config's
        # target fields AND the accessor that closes the Theta_curr loop -- with
        # any one missing the phase either builds no target prior or cannot
        # servo, so a partial rebuild must not report itself as capable.
        "pregrasp": (
            hasattr(getattr(crest_sparse, "ControllerPhase", None), "PreGrasp")
            and hasattr(crest_sparse.TendonHandControllerConfig(),
                        "pregrasp_wrist_pose")
            and hasattr(crest_sparse.TendonHandController, "current_wrist_pose")
        ) if hasattr(crest_sparse, "TendonHandController") else False,
        # The solved phase-3 witness points p_c,obj. Without it a caller can only
        # show the analytic surface projection, which is a look-alike rather than
        # the variable the Eq 1.114-1.117 residuals act on.
        "witness": hasattr(getattr(crest_sparse, "TendonHandController", None),
                           "current_witness_points"),
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


def wrist_pose_from_xyzrpy(xyz, rpy):
    """4x4 base pose from a translation (m) and ZYX euler angles (rad)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(*rpy)
    T[:3, 3] = np.asarray(xyz, float)
    return T


# The default hand base pose: lifted 75 mm along the support normal and pitched
# -1.22 rad about +Y. The mount puts the palm along the base frame's -x (see
# pregrasp_local_geometry), so that pitch swings the palm to face roughly -z --
# i.e. the hand hovers palm-down over the object at the default grasp locus,
# fingers already aimed at it, instead of standing at the identity pose with the
# palm pointing sideways and the fingers through the scene.
#
# This is the posing the interactive visualizer opens on and the start every
# headless harness that does not compute its own pose (free_space_start_pose,
# pregrasp_wrist_pose) inherits. Keep the two in sync: the visualizer seeds its
# sliders from these numbers rather than repeating them.
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
    if primitive in ("big_sphere", "capsule") or spec["type"] == "ellipsoid":
        return np.array(GRASP_SPHERE_CENTER, dtype=float)
    return np.array(OBJECT_CENTER, dtype=float)


def object_extent_along(spec, normal):
    """Approximate object half-size along ``normal`` (m) -- used to seat a default
    support plane tangent to the object's underside. Only a default; the plane
    height is user-adjustable."""
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    t = spec["type"]
    if t == "sphere":
        return float(spec["radius"])
    if t in ("cylinder", "capsule"):
        # These primitives are rotated (Rx 90 deg) to stand their local Y axis
        # along world +Z: half-height along Z, radius laterally.
        cap = spec["radius"] if t == "capsule" else 0.0
        along_z = spec["height"] / 2.0 + cap
        return float(along_z if abs(n[2]) >= 0.5 else spec["radius"])
    if t == "cube":
        return float(np.abs(np.asarray(spec["half_extents"], float) * n).sum())
    if t == "ellipsoid":
        return float(np.abs(np.asarray(spec["semi_axes"], float) * n).sum())
    return 0.05


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
    depth = (1.0 - 2.0 * burial) * object_extent_along(spec, n)
    return np.asarray(object_center, float) - depth * n


def resolve_table_origin(params, spec, object_center):
    """Resolve the support-plane origin: explicit ``params.plane_origin`` if set,
    else the scene's own seating rule (see :func:`auto_table_origin`)."""
    if params.plane_origin is not None:
        return np.asarray(params.plane_origin, float)
    return auto_table_origin(params, spec, object_center)


def split_point(params, spec, object_center):
    """The Eq 1.99 splitting point ``p_split``: explicit ``params.half_space_split``
    if given, else the object centroid projected onto the support plane -- §1.8's
    heuristic.

    Module-level so the pre-controller (FK posing) state can draw the same point
    the controller will enforce, rather than a second, drift-prone copy of the
    rule.
    """
    if params.half_space_split is not None:
        return np.asarray(params.half_space_split, float)
    origin = resolve_table_origin(params, spec, object_center)
    n = np.asarray(params.plane_normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    c = np.asarray(object_center, float)
    return c - np.dot(c - np.asarray(origin, float), n) * n


def opposition_axis(params, spec, object_rotation):
    """``(m_hat, e_long_ratio)``: the Eq 1.99 half-space normal shared by phases 0
    and 1, and how anisotropic the object was in-plane (None when not derived).

    Either the object-derived ``n_hat x e_long`` (§1.8's "split along the object's
    longest in-plane axis") or the explicit ``half_space_axis``, defaulting to
    world +X. Derivation is opt-in because it changes that legacy default, and
    most demo primitives are in-plane isotropic anyway.
    """
    if params.derive_half_space_axis:
        e_long, ratio = object_principal_inplane_axis(
            spec, object_rotation, params.plane_normal,
            fallback=params.half_space_axis)
        return opposition_axis_from_object(params.plane_normal, e_long), ratio
    return (np.asarray(params.half_space_axis, float)
            if params.half_space_axis is not None
            else np.array([1.0, 0.0, 0.0])), None


# ---------------------------------------------------------------------------
# Phase 0 pre-grasp posture (Section 1.8, Eq 1.92-1.93).
# ---------------------------------------------------------------------------

def pregrasp_tension_means(params, configs):
    """Per-finger ``Q_pre`` (Eq 1.92): passive tendons at the background hold,
    flexor at ``passive + pregrasp_flexor_offset`` (or the absolute override).

    Single definition shared by the forward-kinematics evaluation of
    ``p_bar_local_contact`` and by the Eq 1.95 prior, so the posture the pose was
    computed for and the posture the solver is asked to hold cannot drift apart.
    """
    flexor = (params.pregrasp_flexor_absolute
              if params.pregrasp_flexor_absolute is not None
              else params.passive_tension + params.pregrasp_flexor_offset)
    means = []
    for _, cfg in configs:
        m = np.full(cfg.num_tendons, float(params.passive_tension))
        m[FLEXOR_IDX] = float(flexor)
        means.append(m)
    return means


# Cache for pregrasp_local_geometry: the triad and tip centroid depend only on
# the hand and the pre-grasp tension, never on the object or the support plane,
# so a GUI sweeping the clearance height pays the two FK solves once.
_PREGRASP_GEOMETRY_CACHE = {}


def pregrasp_local_geometry(params):
    """The hand's own pre-grasp geometry in the BASE frame, as a dict with keys
    ``p_bar`` (contact-sphere centroid at ``Q_pre``), ``a_hat`` (palm-facing),
    ``g_hat`` (base -> fingers) and ``s_hat`` (thumb-ward lateral).

    Measured, not assumed. §1.8 says the wrist's approach axis is its local +z,
    but that is not this hand's mounting: on the default mount the palm faces the
    base frame's -x, the fingers grow along +y and the thumb-ward lateral axis is
    +z (all ~15 deg off the nominal world axes). Hard-coding the notes' rule would
    mis-orient the hand about 90 deg, and the mount convention has already changed
    once (the CAD tendon-routing flip), so the axes are derived from forward
    kinematics instead:

    * ``a_hat`` is the CURL direction — where the fingertips move as flexor
      tension rises, i.e. where the object ends up. Taken as the displacement of
      the tip centroid between the fully-extended background tension and
      ``Q_pre``, which is the palm-facing direction by construction.
    * ``g_hat`` is the tip centroid's direction, orthogonalized against
      ``a_hat``.
    * ``s_hat = g_hat x a_hat``, sign-checked against the thumb's offset from the
      four fingers. The thumb offset is NOT used to define ``s_hat`` directly: the
      thumb is shorter, so its tip sits behind the fingertips and that vector
      tilts ~9 deg out of the lateral plane.

    Both solves run with an identity base pose so the world frame *is* the base
    frame, but the achieved base pose is still divided out — the FK base prior is
    only sigma 1e-4/1e-3, so it lands near identity rather than at it.
    """
    flexor = float(params.pregrasp_flexor_absolute
                   if params.pregrasp_flexor_absolute is not None
                   else params.passive_tension + params.pregrasp_flexor_offset)
    key = (float(params.passive_tension), flexor,
           tuple(bool(b) for b in params.contact_fingers))
    if key in _PREGRASP_GEOMETRY_CACHE:
        return _PREGRASP_GEOMETRY_CACHE[key]

    def _tips(flexor_tension):
        # Q_pre is uniform across fingers, so commanding it through
        # flexor_tensions reuses HandFKSolver's existing tension-prior path.
        p = replace(params, wrist_pose=np.eye(4),
                    flexor_tensions=[flexor_tension] * NUM_FINGERS)
        fk = HandFKSolver(p)
        frame = fk.solve().frames[0]
        tips = np.array([
            np.asarray(frame[name].marginals.rod.states[tip_node_index(cfg)]
                       .pose.mean, float)[:3, 3]
            for name, cfg in fk.configs])
        # Divide out the achieved base pose: T_0 = T_base o hand_base_offset.
        node0 = np.asarray(frame[fk.finger_names[0]].marginals.rod.states[0]
                           .pose.mean, float)
        base_inv = np.linalg.inv(
            node0 @ np.linalg.inv(
                np.asarray(fk.configs[0][1].hand_base_offset, float)))
        return (base_inv[:3, :3] @ tips.T).T + base_inv[:3, 3]

    tips_pre = _tips(flexor)
    tips_ext = _tips(float(params.passive_tension))

    mask = [bool(b) for b in params.contact_fingers]
    if not any(mask):
        raise ValueError("pregrasp_local_geometry: no contact fingers enabled")
    sel = np.array([i for i, m in enumerate(mask) if m])
    p_bar = tips_pre[sel].mean(axis=0)

    a = tips_pre.mean(axis=0) - tips_ext.mean(axis=0)
    na = np.linalg.norm(a)
    if na < 1e-9:
        raise ValueError(
            "pregrasp_local_geometry: the fingertips did not move between the "
            "background tension and Q_pre, so the curl (palm-facing) direction "
            "is undefined; raise pregrasp_flexor_offset")
    a = a / na

    g = p_bar - (p_bar @ a) * a
    g = g / np.linalg.norm(g)
    s = np.cross(g, a)
    # Sign: s_hat must point to the thumb side, matching opposition_directions'
    # "+axis for the thumb". The thumb is the last config.
    if len(mask) > 1 and mask[-1]:
        others = [i for i in sel if i != len(mask) - 1]
        if others:
            thumb_offset = tips_pre[-1] - tips_pre[others].mean(axis=0)
            if thumb_offset @ s < 0.0:
                s = -s

    out = {"p_bar": p_bar, "a_hat": a, "g_hat": g, "s_hat": s,
           "curl_distance": float(na)}
    _PREGRASP_GEOMETRY_CACHE[key] = out
    return out


def _so3_log(R):
    """Rotation-vector log of a 3x3 rotation, branch-safe near +-pi."""
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    angle = np.arccos(c)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # Near pi the skew part vanishes; take the axis from R + I, whose columns
        # are all parallel to the rotation axis, and fix the sign from the skew.
        A = R + np.eye(3)
        axis = A[:, int(np.argmax(np.linalg.norm(A, axis=0)))]
        axis = axis / np.linalg.norm(axis)
        if (R[2, 1] - R[1, 2]) * axis[0] + (R[0, 2] - R[2, 0]) * axis[1] + \
           (R[1, 0] - R[0, 1]) * axis[2] < 0.0:
            axis = -axis
        return angle * axis
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle * w / (2.0 * np.sin(angle))


def _so3_exp(w):
    """Rodrigues exponential of a rotation vector."""
    t = np.linalg.norm(w)
    if t < 1e-12:
        return np.eye(3)
    k = w / t
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def slew_toward(T_curr, T_goal, max_pos, max_rot):
    """A waypoint from ``T_curr`` toward ``T_goal``, capped at ``max_pos`` metres
    and ``max_rot`` radians.

    Rotation and translation advance by the SAME fraction (the smaller of the two
    caps' allowances), so the waypoint stays on the straight path to the goal
    rather than arriving in orientation long before position or vice versa.
    Returns ``T_goal`` itself once it is within both caps.
    """
    T_curr = np.asarray(T_curr, float)
    T_goal = np.asarray(T_goal, float)
    w = _so3_log(T_curr[:3, :3].T @ T_goal[:3, :3])
    d = T_goal[:3, 3] - T_curr[:3, 3]

    angle, dist = np.linalg.norm(w), np.linalg.norm(d)
    s = 1.0
    if max_rot > 0.0 and angle > max_rot:
        s = min(s, max_rot / angle)
    if max_pos > 0.0 and dist > max_pos:
        s = min(s, max_pos / dist)
    if s >= 1.0:
        return T_goal.copy()

    T = np.eye(4)
    T[:3, :3] = T_curr[:3, :3] @ _so3_exp(w * s)
    T[:3, 3] = T_curr[:3, 3] + d * s
    return T


def free_space_start_pose(params, *, margin=0.01, orientation=None,
                          center_on_object=False, spec=None, object_center=None,
                          object_rotation=None):
    """A collision-FREE starting base pose: the given orientation, lifted along
    the support normal until every collision sphere clears the table and the
    object by ``margin``.

    Phase 0 servos the hand from wherever it starts, and it cannot dig itself out
    of a deep initial penetration: the collision inequalities dominate the merit
    function, and the inner LM rejects every step (``iters=1``, nothing moves).
    An identity base pose is exactly that bad case for the demo scene -- the
    fingers start ~37 mm through the table -- so callers should begin in free
    space and let phase 0 do the positioning, which is what it is for.

    ``center_on_object`` additionally slides the base IN THE PLANE so the
    contact-sphere centroid sits over the object, the way Eq 1.93 places it.
    Without it the base sits on the world origin and only the orientation is
    chosen -- but the centroid is ~0.14 m out along the hand's own axes, so the
    orientation swings it a long way: rolling the hand 180 deg about the palm
    axis moves the fingertips ~0.27 m across, and a start that happened to sit
    near the object no longer does. Opt-in, because it changes the pose the
    existing demos are calibrated against.

    Returns ``(T 4x4, info)``; ``info`` carries the lift distance, the in-plane
    offset and the achieved clearances.
    """
    if spec is None or object_center is None or object_rotation is None:
        spec, object_center, object_rotation, _ = resolve_scene(params)

    n = np.asarray(params.plane_normal, float)
    n = n / np.linalg.norm(n)
    origin = resolve_table_origin(params, spec, object_center)

    R = (np.eye(3) if orientation is None
         else np.asarray(orientation, float).reshape(3, 3))
    T0 = np.eye(4)
    T0[:3, :3] = R

    # In-plane placement, chosen before the lift so the clearance scan below runs
    # at the position the hand will actually occupy -- centred over the object it
    # has to clear, which needs a bigger lift than sitting beside it.
    offset = np.zeros(3)
    if center_on_object:
        want = np.asarray(object_center, float) - R @ pregrasp_local_geometry(
            params)["p_bar"]
        offset = want - (want @ n) * n      # in-plane only; the lift owns n_hat
    T0[:3, 3] = offset

    spheres = _collision_spheres_at(params, T0)

    def clearances(d):
        table = min(float((x + d * n - origin) @ n) - r for x, r in spheres)
        obj = min(primitive_surface_gap(
            object_rotation.T @ (x + d * n - object_center), spec) - r
            for x, r in spheres)
        return table, obj

    # Lifting along +n_hat moves every sphere away from the plane and, once clear
    # of it, away from an object resting on it -- so the clearance is monotone in
    # d and a coarse-then-fine scan is enough. Starts from the exact lift the
    # (linear) table constraint needs, which is usually already the answer.
    d = max(0.0, margin - min(float((x - origin) @ n) - r for x, r in spheres))
    for _ in range(400):
        table, obj = clearances(d)
        if table >= margin and obj >= margin:
            break
        d += 0.005

    T0[:3, 3] = offset + d * n
    table, obj = clearances(d)
    return T0, {"lift": d, "offset": offset, "table_clearance": table,
                "object_clearance": obj, "margin": margin}


def _collision_spheres_at(params, wrist_pose):
    """``[(world_position, radius)]`` for every collision sphere, from one FK
    solve at ``Q_pre`` with the given base pose."""
    flexor = float(params.pregrasp_flexor_absolute
                   if params.pregrasp_flexor_absolute is not None
                   else params.passive_tension + params.pregrasp_flexor_offset)
    p = replace(params, wrist_pose=np.asarray(wrist_pose, float),
                flexor_tensions=[flexor] * NUM_FINGERS)
    fk = HandFKSolver(p)
    frame = fk.solve().frames[0]
    out = []
    for i, (name, cfg) in enumerate(fk.configs):
        tip = tip_node_index(cfg)
        for node in disc_node_indices(cfg):
            x = np.asarray(frame[name].marginals.rod.states[node].pose.mean,
                           float)[:3, 3]
            out.append((x, fk.tip_radii[i] if node == tip
                        else params.collision_radius))
    return out


def pregrasp_wrist_pose(params, *, spec=None, object_center=None,
                        object_rotation=None, m_hat=None):
    """The Eq 1.93 pre-grasp target pose ``T_base,pre``, as ``(T 4x4, info)``.

    Orientation: the palm-facing axis is aimed anti-parallel to the support
    normal (palm down onto the surface) and the thumb-ward lateral axis along the
    opposition half-space normal ``m_hat`` (fingers primed for opposition). Those
    two consume all three rotational degrees of freedom, so the base -> fingers
    axis is a consequence, and the palm-down and opposition rules cannot conflict.

    Position: the contact-sphere centroid is placed a clearance height above the
    object centroid along the surface normal, so the base sits wherever puts it
    there --

        t_pre = p_obj + h_clear * n_hat - R_pre @ p_bar_local_contact

    ``info`` carries the derived axes, the resolved clearance and ``m_hat``, for
    the demos to print and the visualizer to render: a wrong ``R_pre`` is the most
    likely failure here and is otherwise silent.
    """
    if spec is None or object_center is None or object_rotation is None:
        spec, object_center, object_rotation, _ = resolve_scene(params)

    n = np.asarray(params.plane_normal, float)
    n = n / np.linalg.norm(n)

    if m_hat is None:
        m_hat = (params.half_space_axis if params.half_space_axis is not None
                 else np.array([1.0, 0.0, 0.0]))
    m = np.asarray(m_hat, float).reshape(3)
    m = m - (m @ n) * n
    if np.linalg.norm(m) < 1e-9:
        raise ValueError(
            "pregrasp_wrist_pose: the opposition axis is parallel to the support "
            "normal, so it defines no in-plane direction to align the thumb with")
    m = m / np.linalg.norm(m)

    # Measured from the object CENTROID, deliberately independent of where the
    # support plane sits: this is "hover clear of the object", and the guard
    # below is a pure object-frame check. With params.table_burial > 0 the
    # centroid is at or below the plane, so the clearance ABOVE THE TABLE comes
    # out larger than pregrasp_margin alone -- see the table_burial note on
    # HandSolveParams. Known and accepted; do not re-base this on the plane
    # without re-tuning pregrasp_margin.
    h_clear = params.h_clear
    if h_clear is None:
        h_clear = object_extent_along(spec, n) + params.pregrasp_margin
    h_clear = float(h_clear)
    extent = object_extent_along(spec, n)
    if h_clear <= extent:
        raise ValueError(
            f"pregrasp_wrist_pose: h_clear ({h_clear:.4f} m) is inside the object, "
            f"whose half-size along the support normal is {extent:.4f} m; the "
            f"pre-grasp posture must hover clear of it")

    if params.pregrasp_wrist_pose is not None:
        T = np.asarray(params.pregrasp_wrist_pose, float).reshape(4, 4)
        return T, {"source": "explicit", "h_clear": h_clear, "m_hat": m,
                   "n_hat": n}

    geom = pregrasp_local_geometry(params)
    R = rotation_from_two_axes(geom["a_hat"], geom["s_hat"], -n, m)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(object_center, float) + h_clear * n - R @ geom["p_bar"]

    info = dict(geom)
    info.update({"source": "derived", "h_clear": h_clear, "m_hat": m, "n_hat": n,
                 "hover_point": np.asarray(object_center, float) + h_clear * n})
    return T, info


# ---------------------------------------------------------------------------
# Section 1.8 goal geometry.
#
# The world-frame quantities each phase's constraint set is WRITTEN IN TERMS OF,
# as plain numpy: the support-plane distances, the opposition split, the
# pre-grasp target and its alignment axes, the object proxy, the witness points.
# `phase_violations()` reports one scalar per family, which says a phase is
# unhappy but not where; these say where.
#
# Nothing here feeds a solver -- it is read-only reporting, for the visualizer's
# overlays and for headless checks. Kept beside the definitions it mirrors so the
# two cannot drift: the same `p_split`, the same `m_hat`, the same tangent rule.
# ---------------------------------------------------------------------------

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


def contact_centroid(result, k=0):
    """World-frame centroid of the contact spheres at frame ``k`` -- the live
    counterpart of the pre-grasp ``p_bar``, and the point Eq 1.93 places at the
    hover height. ``None`` when no finger is masked in."""
    frame = result.frames[k]
    names = [n for n in result.contact_names() if n in frame]
    if not names:
        return None
    return np.mean([np.asarray(frame[n].marginals.rod.states[-1].pose.mean,
                               float)[:3, 3] for n in names], axis=0)


def half_space_residuals(result, p_split, m_hat, k=0):
    """Per-contact-finger ``{name: (contact_point, m_hat_i, c_half)}`` for the
    Eq 1.99 opposition split, where ``c_half = -(c - p_split).m_hat_i <= 0``.

    ``m_hat_i`` is that finger's own half-space normal: ``+m_hat`` for the thumb
    and ``-m_hat`` for the rest, straight from :func:`opposition_directions` so
    the drawn side and the enforced side are the same call.
    """
    frame = result.frames[k]
    p_split = np.asarray(p_split, float)
    # opposition_directions reads only len(configs), so the name list stands in
    # for the configs here -- the visualizer has no configs of its own.
    dirs = opposition_directions(result.finger_names, axis=m_hat)
    contact = set(result.contact_names())

    out = {}
    for name, m_i in zip(result.finger_names, dirs):
        if name not in contact or name not in frame:
            continue
        c = np.asarray(frame[name].marginals.rod.states[-1].pose.mean,
                       float)[:3, 3]
        out[name] = (c, np.asarray(m_i, float), float(-(c - p_split) @ m_i))
    return out


def contact_frames(result, k=0):
    """Per-finger ``{name: (witness_pt, N_hat, t1, t2)}``: the phase-3 C-frame the
    Eq 1.116/1.117 tangent-slip residuals are written in, built at the ANALYTIC
    surface witness.

    ``N_hat`` is the object's outward surface normal there and ``t1``/``t2`` span
    its tangent plane. The choice of tangent basis is arbitrary (any two spanning
    vectors give the same constraint), so this is the frame's orientation, not
    the solver's own basis vectors.
    """
    out = {}
    for name, (sphere_pt, surface_pt, _gap) in result.contact_witness(k).items():
        d = np.asarray(surface_pt, float) - np.asarray(sphere_pt, float)
        nrm = np.linalg.norm(d)
        # At contact the segment collapses; fall back to the outward radial
        # direction so the frame stays defined rather than blowing up.
        if nrm < 1e-9:
            d = np.asarray(surface_pt, float) - np.asarray(result.object_center,
                                                           float)
            nrm = np.linalg.norm(d) or 1.0
        n_hat = d / nrm
        seed = (np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9
                else np.array([0.0, 1.0, 0.0]))
        t1 = np.cross(n_hat, seed)
        t1 = t1 / (np.linalg.norm(t1) or 1.0)
        out[name] = (np.asarray(surface_pt, float), n_hat, t1,
                     np.cross(n_hat, t1))
    return out


def pregrasp_goal_geometry(params, *, spec=None, object_center=None,
                           object_rotation=None, m_hat=None):
    """The phase-0 goal geometry that needs no controller: ``T_pre``, the hover
    point, the clearance height and the alignment axes.

    Available in the FK-posing state, before any controller exists -- which is
    exactly when you want to see where phase 0 is going to take the hand.
    Returns ``None`` if the target cannot be resolved for this scene (e.g. a
    clearance height inside the object), so a visualizer degrades to drawing
    nothing rather than dying on a slider drag.
    """
    if spec is None or object_center is None or object_rotation is None:
        spec, object_center, object_rotation, _ = resolve_scene(params)
    if m_hat is None:
        m_hat, _ = opposition_axis(params, spec, object_rotation)
    try:
        T_pre, info = pregrasp_wrist_pose(
            params, spec=spec, object_center=object_center,
            object_rotation=object_rotation, m_hat=m_hat)
    except ValueError:
        return None
    return {"T_pre": T_pre, "info": info}


def goal_geometry(params, result, *, k=0, phase=None, T_base=None,
                  waypoint=None, pregrasp=None, m_hat=None,
                  witness_points=None, witness_available=False):
    """Every §1.8 phase goal for one solved frame, in world coordinates.

    One dict, so a renderer makes a single call and picks what it draws. Works
    off a plain :class:`HandResult`, so an FK pose (no controller) reports the
    same geometry as a control tick -- the goals are properties of the scene and
    the posture, not of whether a solver is running.

    The controller-only pieces are passed in rather than recomputed:
    ``T_base``/``waypoint`` (its retained state), and ``witness_points`` (its
    solved ``Symbol('Y', i)``). Left out, those entries are ``None`` and the
    caller falls back to the analytic witness in ``object_witness``.
    """
    spec = result.spec
    center = np.asarray(result.object_center, float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    if m_hat is None:
        m_hat, _ = opposition_axis(params, spec, result.object_rotation)
    m_hat = np.asarray(m_hat, float)

    if pregrasp is None:
        pregrasp = pregrasp_goal_geometry(
            params, spec=spec, object_center=center,
            object_rotation=result.object_rotation, m_hat=m_hat)
    info = (pregrasp or {}).get("info", {})

    geom = {
        "phase": params.phase if phase is None else phase,
        "n_hat": n_hat,
        "plane_origin": np.asarray(
            resolve_table_origin(params, spec, center), float),
        "object_center": center,
        "object_rotation": np.asarray(result.object_rotation, float),
        "contact_names": result.contact_names(),
        "centroid": contact_centroid(result, k),

        # Phase 0 (Eq 1.92-1.98).
        "T_pre": (pregrasp or {}).get("T_pre"),
        "T_base": None if T_base is None else np.asarray(T_base, float),
        "waypoint": None if waypoint is None else np.asarray(waypoint, float),
        "hover_point": info.get("hover_point"),
        "h_clear": info.get("h_clear"),
        "a_hat": info.get("a_hat"),
        "g_hat": info.get("g_hat"),
        "s_hat": info.get("s_hat"),

        # Phase 1 (Eq 1.99-1.107).
        "p_split": split_point(params, spec, center),
        "m_hat": m_hat,
        "plane_gaps": plane_witness(params, result, k),
        "free_plane_gaps": free_sphere_plane_witness(params, result, k),
        "half_gaps": None,

        # Phase 2 (Eq 1.108-1.113): the surface phase 2 actually enforces, which
        # for a cube/cylinder/capsule is NOT the drawn object mesh.
        "proxy_semi_axes": np.asarray(proxy_semi_axes(spec), float),

        # Phase 3 (Eq 1.114-1.118).
        "object_witness": result.contact_witness(k),
        "contact_frames": contact_frames(result, k),
        # Per-finger solved witnesses, None where the phase instantiates none.
        "witness_points": witness_points,
        # Whether the BINDING can report solved witnesses at all -- not whether
        # any exist here. Phases 0-2 legitimately have none, so an all-None
        # `witness_points` on a capable binding is a real answer, while the same
        # list from an incapable one means "unknown, use the analytic witness".
        "witness_available": bool(witness_available),
        "witness_targets": params.witness_targets,
    }
    geom["half_gaps"] = half_space_residuals(result, geom["p_split"], m_hat, k)
    return geom


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
    # base do the positioning (phase 0, or the planner's GP-linked wrist states).
    #
    # Callers that derive their own start (ctrl_5f_phases via
    # free_space_start_pose, viz_controller) overwrite this and are unaffected.
    wrist_pose: np.ndarray = field(default_factory=default_wrist_pose)
    sigma_wrist_pos: float = 1e-4
    sigma_wrist_rot: float = 1e-3

    # --- Tensions (per-finger flexor + shared passive background) ---
    passive_tension: float = 0.5
    flexor_tensions: List[float] = field(
        default_factory=lambda: [GRASP_FLEXOR_TENSION] * NUM_FINGERS)
    tip_wrench_sigma: float = 1e-3

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

    # --- Augmented Lagrangian (IK / planner) ---
    al_mu: float = 1.0
    al_rate: float = 2.0
    al_iters: int = 40
    # How many leading stepper steps run with the flexor prior PINNED as tightly
    # as the passives (see HandIKStepper.step). Settles the cold start before the
    # flexor is released to _IK_TENSION_COV; 0 restores the old behaviour.
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

    # --- Diagnostics (opt-in; off by default so normal solves are unchanged) ---
    # When True the C++ side records the per-outer-iteration AL trace
    # (al_iteration_mus / _costs / _violations on the result meta) plus
    # step-by-step Values snapshots. Used by debug_al_trace.py; left off for the
    # visualizer since it adds per-iteration bookkeeping.
    record_iterations: bool = False

    # --- Collision avoidance (Section 1.5, opt-in; IK / planner) ---
    # OBJECT collision: keep every non-contact sphere out of the object surface.
    # Independent of the table's own avoidance (plane_avoidance below); the two
    # share one set of collision spheres, and finger-finger avoidance comes along
    # whenever EITHER is on.
    collision: bool = False
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
    # whenever either avoidance consumer wants it.
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
    # CENTROID (see pregrasp_wrist_pose). Half-buried, the centroid lies on the
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
    # Opposition half-space (Eq 1.92): the splitting point (None => the object
    # centroid projected onto the support plane) and the in-plane axis the split
    # runs along (None => world +X).
    half_space: bool = True
    half_space_split: Optional[np.ndarray] = None
    half_space_axis: Optional[np.ndarray] = None
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
    # Derive the Eq 1.92 half-space axis from the object's longest in-plane axis
    # (m_hat = n_hat x e_long) instead of using half_space_axis. Off by default:
    # it changes the legacy world-+X default (to +Y for an in-plane-isotropic
    # object), and most demo primitives are degenerate anyway.
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
    # ``HandControllerSolver(initial_state=...)`` and ``set_theta_curr(state=...)``
    # take to start a controller from a posture instead of a straight hand.
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
                       contact_fingers=self._object_contact_mask())

    def _attach_collision(self, avoidance=True):
        """Add Section 1.5 collision spheres onto each finger's (already attached)
        env. Reuses the contact env, so it works for SDF and ellipsoid objects
        alike (the vdb path is only used if a finger has no env yet).

        ``avoidance`` selects whether the finger-OBJECT inequalities are built;
        the spheres themselves are declared either way, because the support plane
        and the finger-finger pairs are built on the same set."""
        vdb = (None if self.spec["type"] == "ellipsoid"
               else os.path.normpath(os.path.join(_OBJECTS_DIR, self.spec["vdb"])))
        attach_collision(self.configs, vdb, self.object_pose,
                         radius=self.params.collision_radius,
                         sigma=self.params.collision_sigma,
                         num_proximal_discs=self.params.num_proximal_discs,
                         cull_margin=self.params.cull_margin,
                         avoidance=avoidance)

    def _attach_table(self):
        """Attach the Section 1.6 support plane to every finger's env."""
        origin = resolve_table_origin(self.params, self.spec, self.object_center)
        attach_table(self.configs, origin, self.params.plane_normal,
                     avoidance=self.params.plane_avoidance,
                     tip_radii=self.tip_radii,
                     contact_fingers=self._table_contact_mask())

    def _attach_environment(self):
        """The whole constraint environment for one solve, per the four
        independent toggles (object contact, table contact, object collision,
        table collision).

        The collision sphere SET is attached whenever either avoidance consumer
        wants it; ``env.collision_avoidance`` then selects only whether the
        finger-OBJECT inequalities are built. Finger-finger avoidance rides on the
        set, so it is active whenever either collision toggle is.

        Shared by the IK solver, the IK stepper and the planner so the three
        cannot drift into building different environments from the same params.
        The Section 1.8 controller deliberately does NOT route through here -- it
        derives its per-phase envs in C++ from one pristine base env."""
        self._attach_contact()
        if self.params.collision or (self.params.table and self.params.plane_avoidance):
            self._attach_collision(avoidance=self.params.collision)
        if self.params.table:
            self._attach_table()

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

    def _result(self, frames, meta, contact_fingers=None, states=None,
                iterates=None, iterate_states=None, iterate_notes=None,
                table_contact_fingers=None):
        # The table mask defaults to the one this solve actually built, so every
        # result carries it without each call site restating it; a caller that
        # means "no table" (the controller) passes an explicit all-False list.
        if table_contact_fingers is None:
            table_contact_fingers = self._table_contact_mask()
        return HandResult(frames, meta, self.spec, self.object_center,
                          self.object_rotation, self.finger_names, self.tip_radii,
                          contact_fingers, states, iterates, iterate_states,
                          iterate_notes, table_contact_fingers)

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


# Tight passive / loose flexor: the optimizer drives contact through the flexor
# while the passives stay pinned. Shared by the one-shot IK solve and the stepper
# so the two cannot drift into solving subtly different problems -- the whole
# point of the stepper is that it advances *this* solve.
_IK_TENSION_COV = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])

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
        sol = solver.solve(self._tension_priors(_IK_TENSION_COV),
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
    contact constraints, same priors (``_IK_TENSION_COV``), same tolerances. The
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

        # Read the stopping tolerances back off the config rather than keeping a
        # second copy: status() has to mirror the C++ convergence test (which
        # reports no stop reason of its own), and reading the same object it was
        # configured from is the only way the two cannot drift.
        self._tols = (cfg.base.al_abs_violation_tol, cfg.base.al_abs_cost_tol,
                      cfg.base.al_rel_violation_tol, cfg.base.al_rel_cost_tol)

        self._solver = crest_sparse.TendonHandSolver(self.configs, cfg)
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
        -- correctly does not."""
        return self._steps < max(int(self.params.ik_settle_steps), 0)

    def step(self) -> HandResult:
        """Advance the AL outer loop by exactly one iteration."""
        # Re-aimed every step so the wrist slider stays live mid-solve; the
        # tension priors are rebuilt from params for the same reason.
        self._solver.set_wrist_pose(self.params.wrist_pose)
        settling = self._settling()
        cov = _IK_SETTLE_TENSION_COV if settling else _IK_TENSION_COV
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
            list(self._notes))

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
        cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
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


class HandControllerSolver(HandSolverBase):
    """Section 1.8 phased real-time controller.

    Where :class:`HandPlannerSolver` solves one K+1-step trajectory offline, this
    re-solves a SINGLE-state constrained IK problem per control tick, anchored to
    the measured robot state by the Eq 1.93-1.95 step priors. The phases are
    different constraint sets over the same graph, not time windows:

      0. ``PreGrasp``        servo to a collision-free hover posture (no equalities)
      1. ``SupportContact``  fingertips onto the table, in opposed half-spaces
      2. ``ObjectApproach``  slide along the table onto the ellipsoid proxy
      3. ``ObjectServo``     lift off as needed and servo on the exact geometry

    The controller is built ONCE in ``__init__`` (like :class:`HandFKSolver`), so
    every ``step()`` warm-starts from the previous tick and ``set_phase()``
    preserves the converged state. Rebuild the object — do not mutate
    ``params`` — when the scene, the contact mask or the collision set changes:
    those alter the constraint set, which a warm-started tick cannot absorb.
    """

    def __init__(self, params: Optional[HandSolveParams] = None,
                 initial_lengths=None, initial_state=None):
        """``initial_state`` is Theta_curr's robot POSTURE -- the ``marginals`` of
        any solve on the same finger configs (an FK pose, or another
        controller's tick). Without it the controller cold-starts from a straight
        hand with Q = 0, and tick 1 is spent travelling back to wherever the
        robot actually is: the fingers visibly extend before they curl.
        ``initial_lengths`` is the separate L_curr the length anchor needs."""
        super().__init__(params)
        p = self.params

        # One pristine env per finger. The C++ phase schedule derives every
        # phase's constraint set from these, so they carry the proxy AND the
        # exact surface, the collision spheres, the plane and the half-space --
        # but none of the per-phase flags.
        self._attach_contact(proxy_and_exact=True)
        self._attach_collision()
        self._attach_table()

        # Resolve the opposition axis ONCE and share it: phase 0 aligns the
        # hand's thumb-ward lateral axis to it and phase 1 splits the support
        # surface across it, so a single source keeps the pre-grasp posture and
        # the half-space consistent by construction.
        self._m_hat = self._opposition_axis()
        if p.half_space:
            attach_half_space(
                self.configs, self._split_point(),
                opposition_directions(self.configs, axis=self._m_hat),
                contact_fingers=p.contact_fingers)
        if p.witness_targets is not None:
            attach_witness_targets(self.configs, p.witness_targets,
                                   contact_fingers=p.contact_fingers)

        cfg = crest_sparse.TendonHandControllerConfig()
        cfg.wrist_pose = p.wrist_pose
        cfg.sigma_wrist_pos = p.sigma_wrist_pos_step
        cfg.sigma_wrist_rot = p.sigma_wrist_rot_step
        cfg.phase = _CONTROLLER_PHASES[p.phase]
        cfg.step_anchor = _STEP_ANCHORS[p.step_anchor.lower()]
        cfg.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
        cfg.base.al_initial_mu = p.al_mu
        cfg.base.al_mu_increase_rate = p.al_rate
        cfg.base.al_max_iterations = p.ctrl_al_iters
        cfg.base.al_abs_cost_tol = p.al_abs_cost_tol
        cfg.base.skip_marginals = p.ctrl_skip_marginals
        _set_if(cfg.base, "record_iterations", p.record_iterations)
        # Carry mu and the multipliers across ticks. Not a tuning knob: without
        # it a tick's penalty resets to al_mu every time and can only reach
        # al_mu * al_rate^(ctrl_al_iters - 1) before the graph is thrown away,
        # which is far too small to enforce the hard constraints.
        _set_if(cfg.base, "al_warm_start_duals", p.ctrl_al_warm_duals)
        _set_if(cfg.base, "al_warm_mu_max", p.ctrl_al_mu_max)
        # Theta_curr's posture (see the constructor docstring).
        if initial_state is not None:
            _set_if(cfg, "initial_state", initial_state)

        self._controller = crest_sparse.TendonHandController(self.configs, cfg)
        # Measured tendon lengths L_curr. From a caller-supplied Theta_curr when
        # given -- the visualizer hands over an FK-posed state that way -- else
        # bootstrapped on the first tick from the controller's own retained
        # state (there is no motor to read in simulation).
        self._lengths = (None if initial_lengths is None
                         else [np.asarray(v, float) for v in initial_lengths])
        # The controller's own commanded base pose (Theta_curr's T_base). Kept
        # here rather than read from params every tick so the feedback loop is
        # not clobbered by a caller that rewrites params.wrist_pose each tick --
        # which the visualizer does, from its pose sliders.
        self._base_pose = np.asarray(p.wrist_pose, float).copy()
        # Phase 0 target, built on first use: it costs two FK solves and is
        # wasted on a controller that never enters phase 0.
        self._T_pre = None
        self._pregrasp_info = None
        self._pregrasp_waypoint = None
        if p.phase == 0:
            self._ensure_pregrasp_target()

    def _opposition_axis(self):
        """The Eq 1.99 half-space normal ``m_hat``, shared by phases 0 and 1."""
        m, self._e_long_ratio = opposition_axis(
            self.params, self.spec, self.object_rotation)
        return m

    def _ensure_pregrasp_target(self):
        """Resolve ``T_base,pre`` (Eq 1.93) once, sharing the opposition axis."""
        if self._T_pre is None:
            self._T_pre, self._pregrasp_info = pregrasp_wrist_pose(
                self.params, spec=self.spec, object_center=self.object_center,
                object_rotation=self.object_rotation, m_hat=self._m_hat)
        return self._T_pre

    @property
    def pregrasp_target(self):
        """``(T_base,pre 4x4, info)`` -- the phase-0 target and how it was derived."""
        self._ensure_pregrasp_target()
        return self._T_pre, self._pregrasp_info

    def _solved_base_pose(self, sol):
        """The solved ``T_base``, for the Eq 1.93 Theta_curr feedback loop.

        Prefers the controller's own accessor; falls back to recovering it from
        the first finger's node-0 pose as ``T_0 o hand_base_offset^-1`` (exact --
        the reparameterization defines T_0 that way), so the feedback loop works
        on a binding predating ``current_wrist_pose``.
        """
        if hasattr(self._controller, "current_wrist_pose"):
            return np.asarray(self._controller.current_wrist_pose(), float)
        node0 = np.asarray(
            sol.marginals.fingers[0].rod.states[0].pose.mean, float)
        return node0 @ np.linalg.inv(
            np.asarray(self.configs[0][1].hand_base_offset, float))

    def _split_point(self):
        """Eq 1.99 splitting point for this scene."""
        return split_point(self.params, self.spec, self.object_center)

    # The controller reads params.contact_fingers directly, NOT the object /
    # table masks the §1.3-1.6 solvers split it into: a phase designates one
    # contact set and C++ decides per phase which surface it acts on (phase_env
    # clears table_contact_node itself). Routing these through the split masks
    # would let a GUI's "table contact off" silently empty the base env the phase
    # schedule is derived from.
    def _attach_contact(self, proxy_and_exact=False):
        attach_contact(self.configs, self.spec, _OBJECTS_DIR,
                       self.params.primitive, self.object_pose,
                       tip_radii=self.tip_radii,
                       contact_fingers=self.params.contact_fingers,
                       proxy_and_exact=proxy_and_exact)

    def _attach_table(self):
        origin = resolve_table_origin(self.params, self.spec, self.object_center)
        attach_table(self.configs, origin, self.params.plane_normal,
                     avoidance=self.params.plane_avoidance,
                     tip_radii=self.tip_radii,
                     contact_fingers=self.params.contact_fingers)

    def set_phase(self, phase: int):
        """Switch the active constraint set (0, 1, 2 or 3), keeping the converged
        robot state."""
        if phase == 0:
            self._ensure_pregrasp_target()
        self._controller.set_phase(_CONTROLLER_PHASES[phase])
        self.params.phase = phase

    def set_theta_curr(self, *, wrist_pose=None, lengths=None, state=None):
        """Overwrite the measured state Theta_curr (Eq 1.93) the next tick anchors
        on: the base pose, the per-finger tendon lengths, and/or the posture.

        ``wrist_pose`` and ``lengths`` only move the step priors' MEANS -- they
        say "the robot is over there now" and the next tick slews toward that
        inside the trust region, leaving the retained values alone.

        ``state`` (a solve's ``marginals``) is the stronger form: it replaces the
        retained posture outright and drops the accumulated AL duals, for a jump
        no trust region could absorb in one tick -- a re-posed hand in a GUI, or
        a resync against the hardware after an unmodelled disturbance. Pair it
        with ``wrist_pose`` so the base prior's mean agrees with the posture.
        Returns False if the binding predates the accessor, in which case rebuild
        the solver with ``initial_state`` instead.
        """
        if wrist_pose is not None:
            self._base_pose = np.asarray(wrist_pose, float).copy()
            self.params.wrist_pose = self._base_pose
        if state is not None:
            if not hasattr(self._controller, "set_state"):
                return False
            self._controller.set_state(state)
        if lengths is not None:
            self._lengths = [np.asarray(v, float) for v in lengths]
        return True

    def phase_violations(self):
        """``[(family, max_abs_violation), ...]`` at the current solution -- what a
        phase-advance policy reads.

        The phase-0 pose families are re-measured here against the FINAL target
        rather than the slewed waypoint. C++ only knows the waypoint it was handed
        (correctly -- it reports the prior it actually built), but "am I there
        yet?" means distance to ``T_base,pre``; against the waypoint it would read
        near-zero from the first tick and an advance policy would fire instantly.
        """
        viol = list(self._controller.phase_violations())
        if self.params.phase != 0 or self._T_pre is None:
            return viol
        T = self._base_pose
        pos = float(np.linalg.norm(T[:3, 3] - self._T_pre[:3, 3]))
        rot = float(np.linalg.norm(
            _so3_log(self._T_pre[:3, :3].T @ T[:3, :3])))
        out = [(n, v) for n, v in viol
               if n not in ("pregrasp_pos", "pregrasp_rot")]
        return [("pregrasp_pos", pos), ("pregrasp_rot", rot)] + out

    def witness_points(self):
        """The solved phase-3 witness points ``p_c,obj``, one per finger with
        ``None`` where the finger has none -- or ``None`` wholesale on a binding
        predating the accessor.

        Only phase 3 instantiates them, so an all-``None`` list is the normal
        answer in phases 0-2 and is NOT the same thing as the unsupported case:
        the caller falls back to the analytic surface witness only for the
        latter."""
        if not hasattr(self._controller, "current_witness_points"):
            return None
        return [None if p is None else np.asarray(p, float).reshape(3)
                for p in self._controller.current_witness_points()]

    def goal_geometry(self, result, k=0):
        """Every §1.8 goal for ``result``, with this controller's own retained
        state filled in: the achieved base pose, the slewed phase-0 waypoint, the
        shared ``m_hat`` and the solved witness points."""
        pts = self.witness_points()
        return goal_geometry(
            self.params, result, k=k, phase=self.params.phase,
            T_base=self._base_pose,
            # Only phase 0 commands a slewed waypoint. The last one survives on
            # the solver after a phase switch, so report it by PHASE rather than
            # by presence -- otherwise a stale target frame hangs in the scene
            # through phases 1-3, looking like something is still being servoed.
            waypoint=self._pregrasp_waypoint if self.params.phase == 0 else None,
            pregrasp=({"T_pre": self._T_pre, "info": self._pregrasp_info}
                      if self._T_pre is not None else None),
            m_hat=self._m_hat, witness_points=pts,
            witness_available=pts is not None)

    def step(self) -> HandResult:
        """One control tick."""
        p = self.params
        self._controller.set_wrist_pose(self._base_pose)

        # Phase 0: re-aim the Eq 1.94/1.95 targets. Live setters, so a GUI can
        # move the clearance height or the servo rate between ticks without a
        # rebuild. Guarded: on a binding predating the phase-0 config fields the
        # solve degrades to a plain step-prior hold rather than crashing.
        if p.phase == 0 and hasattr(self._controller, "set_pregrasp_target"):
            # Slew a waypoint rather than commanding T_pre outright: the target
            # can be a ~172 deg rotation away from an identity base pose, and a
            # stiff Pose3 prior at that distance drives the merit function to
            # ~3e6 and the inner LM rejects every step. The waypoint keeps the
            # commanded pose near the achieved one all the way there.
            self._pregrasp_waypoint = slew_toward(
                self._base_pose, self._ensure_pregrasp_target(),
                p.pregrasp_slew_pos, p.pregrasp_slew_rot)
            self._controller.set_pregrasp_target(
                self._pregrasp_waypoint,
                p.sigma_pregrasp_pos, p.sigma_pregrasp_rot,
                self._pregrasp_tension_priors() if p.pregrasp_tension_prior else [])

        # Tension step prior. Tight passives / loose flexor: the optimizer drives
        # contact through the flexor while the passives stay pinned. Under the
        # length anchor the FLEXOR's tension relaxes further still, since length
        # is then what carries Theta_curr and tension must stay free to respond
        # to a disturbance contact.
        #
        # The passives stay at 1e-6 in BOTH anchor modes, deliberately. They are
        # held by springs, not motors, so near-constant tension is their physical
        # behaviour whatever the actuated tendon is being commanded to do --
        # this is not a leftover from the tension anchor.
        q_sigma = p.sigma_q_step if p.step_anchor.lower() == "tension" else 1.0
        cov = tendon_diag(1e-6, q_sigma ** 2, n=self.configs[0][1].num_tendons)

        # Phase 0 COMMANDS Q_pre rather than relying on the Eq 1.95 target prior:
        # this mean is the commanded tension, not a measurement, so a second
        # Gaussian at the same target would only park the flexor at a blend of
        # the two. See HandSolveParams.pregrasp_tension_prior.
        means = pregrasp_tension_means(p, self.configs) if p.phase == 0 else None

        lengths = []
        if p.step_anchor.lower() in ("length", "both"):
            if self._lengths is None:
                # First tick: no measurement yet (and in simulation, no motor to
                # read). Take L_curr straight from the controller's retained
                # state, so tick 0 is anchored exactly like every later one --
                # solving first to obtain it would need the anchor we do not yet
                # have, which is a chicken-and-egg the C++ side rejects outright.
                self._lengths = [np.asarray(v, float)
                                 for v in self._controller.current_tendon_lengths()]
            # Loose passives / tight flexor -- the MIRROR of the tension prior
            # above, not a copy of it. The motor commands the actuated tendon's
            # length, so that is the measurement worth anchoring on; a passive
            # tendon's length is free to change as the finger moves (the spring
            # takes up the slack), and pinning it would pin the joint angles and
            # freeze the hand. See HandSolveParams.sigma_l_step_*.
            lengths = self._length_priors(
                self._lengths,
                tendon_diag(p.sigma_l_step_passive ** 2,
                            p.sigma_l_step_active ** 2,
                            n=self.configs[0][1].num_tendons))

        sol = self._controller.step(self._tension_priors(cov, means),
                                    self._tip_wrenches(), lengths)
        # Carry the achieved lengths forward as the next tick's L_curr.
        self._lengths = [np.asarray(f.tendon_lengths, float)
                         for f in sol.marginals.fingers]

        # Close the Eq 1.93 Theta_curr loop on the base pose: the step prior's
        # mean becomes the achieved pose, so the next tick's trust region is
        # centered where the hand actually is. Without this the mean stays at the
        # construction-time pose and the base is pinned there -- phase 0 could not
        # servo, and phases 1-3 could not reposition to reach the object. Also
        # mirrored into params so a later phase (and the visualizer) sees it.
        if p.wrist_feedback:
            self._base_pose = self._solved_base_pose(sol)
            p.wrist_pose = self._base_pose

        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        return self._result([frame], sol.meta, p.contact_fingers,
                            [sol.marginals])

    def _pregrasp_tension_priors(self):
        """Eq 1.95's target prior on Q, one ``VectorXGaussian`` per finger.

        Split passive/active like every other tendon prior, though both halves
        default to the same ``sigma_pregrasp_q``: Q_pre names a value for all six
        tendons, so there is no a-priori reason to pull on them with different
        authority. The split is here so retuning one does not silently retune
        the other."""
        p = self.params
        cov = tendon_diag(p.sigma_pregrasp_q_passive ** 2,
                          p.sigma_pregrasp_q_active ** 2,
                          n=self.configs[0][1].num_tendons)
        return [crest_sparse.VectorXGaussian(m, cov)
                for m in pregrasp_tension_means(self.params, self.configs)]

    # A single solve() is one tick, so the registry below can drive this class
    # exactly like the other three.
    solve = step


# Phase / anchor name -> binding enum. Built lazily-ish at import: a binding
# without the controller simply has no entries, and capabilities()["controller"]
# gates every path that would index these.
_CONTROLLER_PHASES = (
    {1: crest_sparse.ControllerPhase.SupportContact,
     2: crest_sparse.ControllerPhase.ObjectApproach,
     3: crest_sparse.ControllerPhase.ObjectServo}
    if hasattr(crest_sparse, "ControllerPhase") else {})

# Phase 0 arrived after the other three, so a binding that predates it still
# drives phases 1-3 rather than failing to import.
if _CONTROLLER_PHASES and hasattr(crest_sparse.ControllerPhase, "PreGrasp"):
    _CONTROLLER_PHASES[0] = crest_sparse.ControllerPhase.PreGrasp

_STEP_ANCHORS = (
    {"tension": crest_sparse.StepAnchor.Tension,
     "length": crest_sparse.StepAnchor.Length,
     "both": crest_sparse.StepAnchor.Both}
    if hasattr(crest_sparse, "StepAnchor") else {})


# Convenience registry the visualizer uses to switch modes.
SOLVERS = {
    "FK": HandFKSolver,
    "IK": HandIKSolver,
    "Planner": HandPlannerSolver,
    "Controller": HandControllerSolver,
}
