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
from typing import List, Optional

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
    }


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
    """The support-plane origin implied by the scene alone: tangent to the
    object's underside along ``params.plane_normal``, so the object rests on it.

    Deliberately ignores ``params.plane_origin``. A GUI offering an ABSOLUTE
    plane height has to seed and re-seat its control from the scene's own answer;
    reading :func:`resolve_table_origin` for that would feed the control's own
    output back into itself, and any offset applied on top would compound on
    every call. Split out so both readings have exactly one definition of the
    tangent rule.
    """
    n = np.asarray(params.plane_normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    return np.asarray(object_center, float) - object_extent_along(spec, n) * n


def resolve_table_origin(params, spec, object_center):
    """Resolve the support-plane origin: explicit ``params.plane_origin`` if set,
    else tangent to the object's underside along ``params.plane_normal``."""
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


def plane_witness(params, result, k=0):
    """Per-contact-finger ``{name: (sphere_pt, foot_pt, signed_gap)}`` against the
    support plane at frame ``k``.

    Same 3-tuple shape as :meth:`HandResult.contact_witness`, so a renderer draws
    a table clearance exactly the way it draws an object gap. These are the
    spheres phase 1/2 hold at ``c_support = 0`` (Eq 1.104/1.110); phase 3 relaxes
    them back to the ``<= 0`` inequality, where the same number is a clearance.
    """
    frame = result.frames[k]
    origin = np.asarray(resolve_table_origin(params, result.spec,
                                             result.object_center), float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)

    out = {}
    for name, radius in zip(result.finger_names, result.tip_radii):
        if name not in result.contact_names():
            continue
        fm = frame[name].marginals
        c = np.asarray(fm.rod.states[-1].pose.mean, float)[:3, 3]
        out[name] = _plane_measure(c, float(radius), origin, n_hat)
    return out


def free_sphere_plane_witness(params, result, k=0):
    """The same measurement for every sphere NOT designated for support contact,
    keyed ``"{finger}/{node}"``.

    These carry the Eq 1.106 avoidance inequality rather than an equality, and
    they are the ones that silently stall a phase: a single sphere driven through
    the table dominates the merit function and the inner LM rejects every step.
    """
    frame = result.frames[k]
    origin = np.asarray(resolve_table_origin(params, result.spec,
                                             result.object_center), float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    contact = set(result.contact_names())

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
    primitive: str = "big_sphere"
    object_center: Optional[np.ndarray] = None      # None => derive from primitive
    object_rotation: Optional[np.ndarray] = None     # None => primitive's rotation

    # --- Wrist start pose + prior ---
    wrist_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    sigma_wrist_pos: float = 1e-4
    sigma_wrist_rot: float = 1e-3

    # --- Tensions (per-finger flexor + shared passive background) ---
    passive_tension: float = 0.5
    flexor_tensions: List[float] = field(
        default_factory=lambda: [GRASP_FLEXOR_TENSION] * NUM_FINGERS)
    tip_wrench_sigma: float = 1e-3

    # --- Which fingertips are solved for contact (IK / planner; FK ignores it) ---
    # One flag per finger, in ``configs`` order. A False finger contributes no
    # contact constraint -- neither to the object nor to the table -- but keeps
    # its collision spheres and plane avoidance, so it is still kept out of the
    # object and (wherever avoidance is active) off the table. All-True is the
    # legacy behavior: every fingertip driven onto the object.
    contact_fingers: List[bool] = field(
        default_factory=lambda: [True] * NUM_FINGERS)

    # --- Augmented Lagrangian (IK / planner) ---
    al_mu: float = 1.0
    al_rate: float = 2.0
    al_iters: int = 40

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
    collision: bool = False
    collision_radius: float = 0.003
    collision_sigma: float = 1e-4
    num_proximal_discs: int = 2
    cull_margin: Optional[float] = None

    # --- Support plane / "table" (Section 1.6, opt-in; IK / planner) ---
    table: bool = False
    plane_origin: Optional[np.ndarray] = None       # None => under the object
    plane_normal: np.ndarray = field(
        default_factory=lambda: np.array(TABLE_NORMAL, float))
    plane_avoidance: bool = True
    k_touch: Optional[int] = None                    # planner slide-grasp schedule

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
    sigma_wrist_pos_step: float = 1e-2
    sigma_wrist_rot_step: float = 1e-1
    sigma_q_step: float = 1e-1
    sigma_l_step: float = 1e-3
    # Opposition half-space (Eq 1.92): the splitting point (None => the object
    # centroid projected onto the support plane) and the in-plane axis the split
    # runs along (None => world +X).
    half_space: bool = True
    half_space_split: Optional[np.ndarray] = None
    half_space_axis: Optional[np.ndarray] = None
    # Optional per-finger phase-3 witness targets (Eq 1.111); None entries mean
    # "contact anywhere on the surface" for that finger.
    witness_targets: Optional[List[Optional[np.ndarray]]] = None
    # A control tick's AL budget. Small on purpose: the outer loop is amortized
    # across ticks, since the constraint set is unchanged between them and each
    # tick warm-starts from the last.
    ctrl_al_iters: int = 4
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
    sigma_pregrasp_q: float = 1e-1
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

    def contact_names(self):
        """The fingers designated to touch the object -- everything the gap
        readouts should be judged on. All of them when unmasked."""
        if self.contact_fingers is None:
            return list(self.finger_names)
        return [name for name, on in zip(self.finger_names, self.contact_fingers)
                if on]

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
        """Largest |gap| over the fingers that were *asked* to touch, so a masked
        subset grasp isn't scored on fingers left free."""
        gaps = self.surface_gaps(k)
        names = self.contact_names()
        return max((abs(gaps[n]) for n in names if n in gaps), default=0.0)


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

    # -- environment attachment (mutates self.configs in place) --

    def _attach_contact(self):
        """Per-finger contact env: shared object surface + this finger's tip node
        as the terminal contact (``ik_5f_contact.py`` block). Fingers masked off
        by ``params.contact_fingers`` get a collision-only env instead."""
        attach_contact(self.configs, self.spec, _OBJECTS_DIR,
                       self.params.primitive, self.object_pose,
                       tip_radii=self.tip_radii,
                       contact_fingers=self.params.contact_fingers)

    def _attach_collision(self):
        """Add Section 1.5 collision spheres onto each finger's (already attached)
        env. Reuses the contact env, so it works for SDF and ellipsoid objects
        alike (the vdb path is only used if a finger has no env yet)."""
        vdb = (None if self.spec["type"] == "ellipsoid"
               else os.path.normpath(os.path.join(_OBJECTS_DIR, self.spec["vdb"])))
        attach_collision(self.configs, vdb, self.object_pose,
                         radius=self.params.collision_radius,
                         sigma=self.params.collision_sigma,
                         num_proximal_discs=self.params.num_proximal_discs,
                         cull_margin=self.params.cull_margin)

    def _attach_table(self):
        """Attach the Section 1.6 support plane to every finger's env."""
        origin = resolve_table_origin(self.params, self.spec, self.object_center)
        attach_table(self.configs, origin, self.params.plane_normal,
                     avoidance=self.params.plane_avoidance,
                     tip_radii=self.tip_radii,
                     contact_fingers=self.params.contact_fingers)

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

    def _result(self, frames, meta, contact_fingers=None):
        return HandResult(frames, meta, self.spec, self.object_center,
                          self.object_rotation, self.finger_names, self.tip_radii,
                          contact_fingers)

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
        return self._result([frame], sol.meta, self.params.contact_fingers)


class HandIKSolver(HandSolverBase):
    """Single terminal grasp: each fingertip driven onto the shared object surface
    by a hard contact constraint (Augmented Lagrangian). ``ik_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_contact()
        if self.params.collision:
            self._attach_collision()
        if self.params.table:
            self._attach_table()

        cfg = crest_sparse.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.al_initial_mu = self.params.al_mu
        cfg.base.al_mu_increase_rate = self.params.al_rate
        cfg.base.al_max_iterations = self.params.al_iters
        _set_if(cfg.base, "record_iterations", self.params.record_iterations)

        solver = crest_sparse.TendonHandSolver(self.configs, cfg)
        # Tight passive / loose flexor: the optimizer drives contact through the
        # flexor while the passives stay pinned.
        cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
        sol = solver.solve(self._tension_priors(cov), self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        return self._result([frame], sol.meta, self.params.contact_fingers)


class HandPlannerSolver(HandSolverBase):
    """A K+1-step grasp trajectory tied by GP temporal priors on the wrist pose and
    finger tensions, with terminal contact constraints. ``traj_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_contact()
        if self.params.collision:
            self._attach_collision()
        if self.params.table:
            self._attach_table()

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
        return self._result(frames, result.meta, self.params.contact_fingers)


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
                 initial_lengths=None):
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

    def _attach_contact(self, proxy_and_exact=False):
        attach_contact(self.configs, self.spec, _OBJECTS_DIR,
                       self.params.primitive, self.object_pose,
                       tip_radii=self.tip_radii,
                       contact_fingers=self.params.contact_fingers,
                       proxy_and_exact=proxy_and_exact)

    def set_phase(self, phase: int):
        """Switch the active constraint set (0, 1, 2 or 3), keeping the converged
        robot state."""
        if phase == 0:
            self._ensure_pregrasp_target()
        self._controller.set_phase(_CONTROLLER_PHASES[phase])
        self.params.phase = phase

    def set_theta_curr(self, *, wrist_pose=None, lengths=None):
        """Overwrite the measured state Theta_curr (Eq 1.93) the next tick anchors
        on: the base pose and/or the per-finger tendon lengths.

        The retained ``values_`` are NOT touched -- this says "the robot is over
        there now", and the next tick slews toward it inside the step-prior trust
        region. For a wholesale teleport, build a new solver instead (passing
        ``initial_lengths``): a warm controller cannot absorb an arbitrary jump in
        one tick.
        """
        if wrist_pose is not None:
            self._base_pose = np.asarray(wrist_pose, float).copy()
            self.params.wrist_pose = self._base_pose
        if lengths is not None:
            self._lengths = [np.asarray(v, float) for v in lengths]

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
        # length anchor the tensions relax further still, since length is then
        # what carries Theta_curr and tension must stay free to respond to a
        # disturbance contact.
        q_sigma = p.sigma_q_step if p.step_anchor.lower() == "tension" else 1.0
        cov = np.diag([1e-6] * FLEXOR_IDX + [q_sigma ** 2])

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
            lengths = self._length_priors(
                self._lengths, (p.sigma_l_step ** 2) * np.eye(self.configs[0][1].num_tendons))

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
        return self._result([frame], sol.meta, p.contact_fingers)

    def _pregrasp_tension_priors(self):
        """Eq 1.95's target prior on Q, one ``VectorXGaussian`` per finger."""
        cov = (self.params.sigma_pregrasp_q ** 2) * np.eye(
            self.configs[0][1].num_tendons)
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
