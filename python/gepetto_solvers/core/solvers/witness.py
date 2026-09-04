"""Independent readouts: every constraint family's residual, recomputed from the
SOLVED POSES rather than read off the solver.

This is what makes a stall diagnosable. If the solver says a constraint is
satisfied and the witness here disagrees, the disagreement is the bug -- and
because these use the analytic surface, they cannot inherit the solver's own
error.
"""

from typing import NamedTuple

import numpy as np

from ..geometry.scene import (
    primitive_surface_gap,
    primitive_surface_normal,
    primitive_surface_witness,
    subset_spec,
)
from ..hands.tendon_5f import (
    get_default_hand_configs,
)
from .frames import solved_wrist_pose
from .scene_resolve import (
    default_half_space_axis,
    orient_opposition_axis,
    resolve_constraint_plane_origin,
)


def _sphere_nodes(fm):
    """``[(node_index, is_tip)]`` for one digit's collision spheres, taken from
    the marginals rather than the configs -- ``collision_sites`` is the same list
    the renderer walks, so an overlay can never mark a sphere the picture does
    not show. The tip is the last site, matching ``contact_witness``.

    Off ``DigitState.collision_sites``, which every mechanism fills, rather than
    off the tendon extras: a rigid digit has no ``tendon_config``, and reaching
    for one made every witness here tendon-only.
    """
    tip = len(fm.sites) - 1
    return [(int(i), int(i) == tip) for i in fm.collision_sites]


def _sphere_radius(node_is_tip, tip_radius, collision_radius):
    """The radius the solve gave one collision sphere: the digit's own tip radius
    at the contact site, the shared ``collision_radius`` everywhere else. Same
    rule the graph builds the inequalities with, so a drawn gap is the gap the
    constraint sees."""
    return float(tip_radius) if node_is_tip else float(collision_radius)


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
    origin = np.asarray(resolve_constraint_plane_origin(
        params, result.spec, result.object_center), float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    wanted = set(result.contact_names() if names is None else names)

    out = {}
    for name, radius in zip(result.finger_names, result.tip_radii):
        if name not in wanted:
            continue
        fm = frame[name].marginals
        c = np.asarray(fm.sites[-1].pose.mean, float)[:3, 3]
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
    origin = np.asarray(resolve_constraint_plane_origin(
        params, result.spec, result.object_center), float)
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    contact = set(result.contact_names() if names is None else names)

    out = {}
    for name, tip_radius in zip(result.finger_names, result.tip_radii):
        fm = frame[name].marginals
        poses = fm.sites
        for node, is_tip in _sphere_nodes(fm):
            # A contact finger's tip is the designated support sphere and is
            # reported by plane_witness; every other sphere lands here.
            if is_tip and name in contact:
                continue
            c = np.asarray(poses[node].pose.mean, float)[:3, 3]
            r = _sphere_radius(is_tip, tip_radius, params.collision_radius)
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
    opposing = result.opposing_digit

    def tip(name):
        return np.asarray(frame[name].marginals.sites[-1].pose.mean,
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
            (tip(opposing) if opposing in result.finger_names else np.zeros(3)),
            [tip(n) for n in wanted if n != opposing],
            flip=params.half_space_flip)

    out = {}
    for name in result.finger_names:
        if name not in wanted:
            continue
        c = tip(name)
        m_hat = axis if name == opposing else -axis
        # The boundary this finger is actually held to: the split pushed out by
        # the standoff along its own m_hat (0 => the plain splitting plane).
        p_bound = p_split + float(params.half_space_margin) * m_hat
        margin = float((c - p_bound) @ m_hat)   # >= 0 => correct side
        foot = c - margin * m_hat
        out[name] = (c, foot, margin)
    return out


def pregrasp_center_witness(params, result, k=0):
    """``(hand_centroid_pt, target_pt, gap_m)`` for the Eq 2.18-2.19 pre-grasp
    hand-centering constraint at frame ``k``, or None if the OPPOSING digit
    (``result.opposing_digit``) or the opposed set has no fingers designated
    (:meth:`HandResult.contact_names`).

    ``hand_centroid_pt`` is the midpoint of the OPPOSING digit's and the
    opposed (contact-designated) digits' contact-sphere centers --
    ``c_hand`` in the paper's notation. ``target_pt`` is the object centroid
    raised by ``h_clear`` along ``plane_normal``. ``gap_m`` is their Euclidean
    separation (0 at the constraint's zero set); unlike the other witness
    functions this is a single HAND-level tuple, not one per finger.
    """
    names = result.contact_names()
    opposing = result.opposing_digit
    if opposing is None or opposing not in names:
        return None
    others = [n for n in names if n != opposing]
    if not others:
        return None

    frame = result.frames[k]

    def tip(name):
        fm = frame[name].marginals
        return np.asarray(fm.sites[-1].pose.mean, float)[:3, 3]

    c_opposing = tip(opposing)
    c_others = np.mean([tip(n) for n in others], axis=0)
    hand_centroid = 0.5 * (c_opposing + c_others)

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
    from ..hands.tendon_5f import (
    pinch_pose,
)

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
    """``(c_opposing, c_others_mean, angle_deg)`` for the pre-grasp short-axis
    alignment constraint at frame ``k``, or None if the OPPOSING digit
    (``result.opposing_digit``) or the opposed set has no fingers designated
    (:meth:`HandResult.contact_names`).

    ``angle_deg`` is the acute angle between the achieved opposing-vs-opposed
    connecting vector and ``default_half_space_axis`` -- 0 at the constraint's
    zero set (either parallel or antiparallel), up to 90 at worst. Recomputes
    the SAME axis :meth:`HandSolverBase._attach_pregrasp_axis_alignment` uses,
    from ``result.spec``/``result.object_rotation`` rather than a stored
    value, so the overlay always matches the axis the last-attached
    constraint actually used.
    """
    names = result.contact_names()
    opposing = result.opposing_digit
    if opposing is None or opposing not in names:
        return None
    others = [n for n in names if n != opposing]
    if not others:
        return None

    frame = result.frames[k]

    def tip(name):
        fm = frame[name].marginals
        return np.asarray(fm.sites[-1].pose.mean, float)[:3, 3]

    c_opposing = tip(opposing)
    c_others = np.mean([tip(n) for n in others], axis=0)
    v = c_opposing - c_others
    vn = np.linalg.norm(v)
    if vn < 1e-9:
        return None
    v_hat = v / vn

    axis = default_half_space_axis(result.spec, result.object_rotation,
                                   params.plane_normal)
    cos_a = abs(float(v_hat @ axis))
    cos_a = min(1.0, max(-1.0, cos_a))
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    return (c_opposing, c_others, angle_deg)


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
    from ..hands.tendon_5f import (
    pinch_pose,
)

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
        states = frame[name].marginals.sites
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
    import gepetto_solvers

    from ..geometry.scene import (
        ELLIPSOID_SET_BETA,
        ellipsoid_members,
        plane_ellipse_section,
    )
    from ..hands.tendon_5f import (
    pinch_pose,
)

    if not hasattr(gepetto_solvers, "ellipsoid_set_planar_gap"):
        return None
    # The CONTACT members, narrowed the same way the factor's were -- an overlay
    # drawing cross-sections of shells the constraint never saw is exactly the
    # drift this function exists to avoid.
    members = ellipsoid_members(subset_spec(result.spec, result.contact_subset))
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
        p = gepetto_solvers.EllipsoidPrimitive()
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
        T_tip = np.asarray(frame[name].marginals.sites[-1].pose.mean, float)
        tip = T_tip[:3, 3]
        # Eq 11's p_base, in the WRIST frame -- the finger's mounting offset, not a
        # solved node pose (node 0 has no key of its own under root reparameterization).
        base_local = np.asarray(cfg.hand_base_offset, dtype=float)[:3, 3]

        # The metric goes across too: an overlay drawing a Taubin distance beside
        # a solve that used the exact one is the same drift as drawing a plane the
        # solve never used.
        rep = gepetto_solvers.ellipsoid_set_planar_gap(
            T_tip, T_obj, T_wrist, float(radius), prims, beta, base_local, c_local,
            taubin=bool(getattr(params, "ellipsoid_taubin", False)))
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


class GraspWrench(NamedTuple):
    """The h_grasp residual, recomputed from a solve's own contact points.

    ``force`` and ``torque`` are the two halves of the 6-vector; ``norm`` is the
    length of the whole thing, and ``digits`` names what went into it.

    ``force`` is a sum of UNIT vectors, so its magnitude is bounded by the number
    of contacts and is directly readable: 0 is balanced, and |C| means every
    contact pushes the same way. That interpretability is the point of reporting
    the raw residual rather than the solver's number.
    """
    force: np.ndarray       # sum of -n_i                (dimensionless)
    torque: np.ndarray      # sum of -(p_i - t_obj) x n_i (metres)
    norm: float
    digits: list
    # The terms the two sums are OF, in ``digits`` order: the contact point p_i
    # and the OUTWARD unit normal n_i there. Carried because the residual alone
    # cannot be drawn -- a net wrench of zero looks identical to one nobody
    # measured, and what makes the constraint readable is the arrangement of the
    # individual -n_i arrows that cancelled to produce it.
    points: tuple = ()
    normals: tuple = ()


def grasp_wrench_witness(result, k=0, names=None):
    """The net virtual wrench of the contacts at frame ``k`` -- h_grasp, measured
    rather than read off the solver.

    Uses :meth:`HandResult.contact_witness`, so the surface point and its normal
    come from the ANALYTIC surface, where the constraint the solver built reads
    the baked grid. The two agree to within the grid's fillet, and for the
    phases this constraint runs in that is the right trade: the whole reason for
    an independent readout is that it cannot inherit the solver's own error, and
    sampling the same grid through the same interpolation would forfeit exactly
    that.

    WHY THE RAW RESIDUAL. The violation the AL reports is WHITENED, so it scales
    as ``1 / sigma_grasp_*`` -- loosening those sigmas divides the reported
    number without moving a single fingertip. Measured on the Allegro hand
    reaching for the 35 mm sphere, sweeping ``sigma_grasp_force`` alone:

        sigma_force |    1    |   10    |   100   |  1000   |  1e4
        AL violation| 1.1e+0  | 1.3e-1  | 1.4e-2  | 1.7e-3  | 1.9e-4
        raw |wrench| |   1.1   |   1.3   |   1.4   |   1.7   |  1.9

    The reported violation falls by four orders of magnitude while the grasp gets
    no better at all. This function is what says so.
    """
    # The CONTACT digits, not every digit. h_grasp is built over exactly the
    # witness points the graph created, so summing an uncommanded finger's
    # nearest-surface direction into the total would report a wrench the
    # constraint never saw -- and it is not a small effect: including the
    # Allegro's idle ring finger adds a whole unit vector to a residual whose
    # satisfied value is zero.
    names = names if names is not None else result.contact_names()
    witness = result.contact_witness(k)
    t_obj = np.asarray(result.object_center, float)

    force = np.zeros(3)
    torque = np.zeros(3)
    used, points, normals = [], [], []
    for name in names:
        if name not in witness:
            continue
        sphere_pt, surface_pt, _gap = witness[name]
        # The outward surface normal at the contact, from the segment the witness
        # already measured: it runs from the sphere's surface to the object's, so
        # its direction IS the inward normal.
        direction = np.asarray(surface_pt, float) - np.asarray(sphere_pt, float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue        # coincident points: no direction to credit
        n_hat = -direction / norm          # outward, matching the factor's n_i
        p_i = np.asarray(surface_pt, float)
        force += -n_hat
        torque += -np.cross(p_i - t_obj, n_hat)
        used.append(name)
        points.append(p_i)
        normals.append(n_hat)

    return GraspWrench(force, torque,
                       float(np.linalg.norm(np.concatenate([force, torque]))),
                       used, tuple(points), tuple(normals))


def ellipsoid_grasp_wrench_witness(params, result, k=0, names=None):
    """The net virtual wrench of the contacts at frame ``k`` measured on the
    sphere CENTERS against the analytic ellipsoid -- h_grasp,E, recomputed rather
    than read off the solver.

    Same :class:`GraspWrench` shape as :func:`grasp_wrench_witness`, so a
    renderer draws either one the same way, and the same warning applies: the
    violation the AL reports is WHITENED and falls as ``1 / sigma_grasp_ell_*``
    without a fingertip moving. This function is the number to judge the
    constraint by.

    WHY IT IS NOT :func:`grasp_wrench_witness` WITH A DIFFERENT SURFACE. That one
    takes its normal from the contact-witness segment -- the direction from the
    sphere's surface to the object's -- which is a WITNESS-point quantity and
    degenerates exactly when the two points coincide, i.e. at contact. The
    constraint this measures has no witness point at all: it is evaluated at the
    sphere center ``c_i``, where the normal field is perfectly well defined
    whether the sphere is 10 cm away, touching, or interpenetrating. That is what
    makes it readable during the approach, which is the phase it exists for.

      force  += -n_i
      torque += -(c_i - t_obj) x n_i

    The RADIUS IS ABSENT, and not by omission: the virtual force acts at
    ``p_i = c_i - r_i n_i`` but is collinear with the radius vector, so
    ``r_i (n_i x n_i) = 0`` drops out of the torque exactly. The contact point
    and the center give the identical wrench, which is what lets the constraint
    key off the center. A drawn arrow can therefore be rooted at either; this
    reports the centers, matching where the factor evaluates.

    Two deliberate choices about WHICH geometry, both matching the C++ factor:

    * the FULL ``result.spec``, never ``contact_subset``. The normal field is a
      property of the object's shape, not of which shells a finger was cleared to
      touch -- narrowing it would report a normal pointing into a housing the
      object really has. Same choice :func:`object_collision_witness` makes.
    * the analytic proxy, never the baked SDF. Running on the smooth proxy is the
      whole point of this constraint, so measuring it against the grid would
      score it on a surface it was never built against.

    The normal here is the central-difference gradient of ``primitive_surface_gap``,
    which for an ``ellipsoid_set`` is the SAME LogSumExp smooth min the factor
    blends -- so this agrees with the constraint's own field, including its
    ``ln(K)/beta`` seam standoff, while still being an independent computation
    (NumPy differences of the fused distance; C++ softmin-weights the members'
    analytic gradients). It reads the default EXACT metric, so a solve run under
    ``ellipsoid_taubin`` will diverge from it in the far field.

    ``names`` overrides which digits are summed; defaults to the object-contact
    set. It must be the set the constraint was actually built over -- summing an
    uncommanded finger's nearest-surface direction adds a whole unit vector to a
    residual whose satisfied value is zero.
    """
    names = names if names is not None else result.contact_names()
    frame = result.frames[k]
    center = np.asarray(result.object_center, float)
    R = np.asarray(result.object_rotation, float)
    t_obj = center

    force = np.zeros(3)
    torque = np.zeros(3)
    used, points, normals = [], [], []
    for name in names:
        if name not in frame:
            continue
        # The contact sphere's CENTER -- the same site the renderer draws the
        # contact sphere on, and the one whose site pose the factor keys off.
        c_i = np.asarray(frame[name].tip_point(), float)
        n_hat = R @ primitive_surface_normal(R.T @ (c_i - center), result.spec)
        force += -n_hat
        torque += -np.cross(c_i - t_obj, n_hat)
        used.append(name)
        points.append(c_i)
        normals.append(n_hat)

    return GraspWrench(force, torque,
                       float(np.linalg.norm(np.concatenate([force, torque]))),
                       used, tuple(points), tuple(normals))


def object_collision_witness(params, result, k=0, names=None):
    """Per free sphere ``{"{finger}/{node}": (sphere_pt, surface_pt, signed_gap)}``
    against the OBJECT at frame ``k`` -- the ``h_pen`` inequality's own distance.

    The object counterpart of :func:`free_sphere_plane_witness`, and it reports
    the same three things for the same reason: ``h_pen = (r + eps) - d <= 0`` is
    satisfied exactly when ``signed_gap = d - r`` clears the margin, so a
    renderer draws it as a line and a number without knowing which surface it is
    measuring against.

    Two deliberate choices about WHICH geometry:

    * the FULL spec, never ``contact_subset``. Contact may be narrowed to an
      object's grasp shells, but collision is not -- every shell is still there
      to be avoided, and measuring against the narrowed set would report
      clearance from a housing the sphere is inside of.
    * the analytic proxy, not the baked SDF, matching what the collision
      inequalities are built against even when contact has been pointed at the
      exact form (``object_contact_exact``). That split is the whole point of
      the two-surface formulation, so the overlay has to keep it.

    ``names`` is the object-contact set whose TIPS are excluded (they carry the
    equality and are reported by :meth:`HandResult.contact_witness`); defaults to
    ``result.contact_names()``, and must be the same list the contact overlay
    used or a fingertip is drawn twice or not at all.
    """
    frame = result.frames[k]
    center = np.asarray(result.object_center, float)
    R = np.asarray(result.object_rotation, float)
    contact = set(result.contact_names() if names is None else names)

    out = {}
    for name, tip_radius in zip(result.finger_names, result.tip_radii):
        if name not in frame:
            continue
        fm = frame[name].marginals
        poses = fm.sites
        for node, is_tip in _sphere_nodes(fm):
            if is_tip and name in contact:
                continue
            c = np.asarray(poses[node].pose.mean, float)[:3, 3]
            r = _sphere_radius(is_tip, tip_radius, params.collision_radius)
            dist, foot_local, n_local = primitive_surface_witness(
                R.T @ (c - center), result.spec)
            surface_pt = center + R @ foot_local
            out[f"{name}/{node}"] = (c - r * (R @ n_local), surface_pt,
                                     float(dist) - r)
    return out


def self_collision_witness(params, result, k=0, max_gap=0.02):
    """Cross-digit sphere pairs at frame ``k``, keyed
    ``"{finger_a}/{node_a}/{finger_b}/{node_b}"`` and valued
    ``(point_on_a, point_on_b, signed_gap)``.

    ``signed_gap`` is ``||c_a - c_b|| - (r_a + r_b)`` -- zero when the two
    spheres just touch, negative once they interpenetrate, which is the zero set
    of the finger-finger inequality (``params.self_collision``). The two points
    are on each sphere's SURFACE facing the other, so the drawn segment is the
    gap itself rather than a centre-to-centre line that would read as
    penetrating whenever the fingers are merely close.

    SAME-DIGIT pairs are excluded, matching the constraint: consecutive spheres
    on one finger overlap by construction, and an inequality between them would
    be violated in every posture the hand can hold.

    ``max_gap`` is a DISPLAY window, not a constraint: the pair count is
    quadratic (five digits of four spheres is 160 pairs), and a line drawn
    between two spheres a hand-width apart says nothing. Only pairs at or under
    it are returned, so what comes back is the set actually near collision. Pass
    ``None`` for every pair.
    """
    frame = result.frames[k]

    spheres = []
    for name, tip_radius in zip(result.finger_names, result.tip_radii):
        if name not in frame:
            continue
        fm = frame[name].marginals
        for node, is_tip in _sphere_nodes(fm):
            spheres.append((
                name, node,
                np.asarray(fm.sites[node].pose.mean, float)[:3, 3],
                _sphere_radius(is_tip, tip_radius, params.collision_radius)))

    out = {}
    for i, (name_a, node_a, c_a, r_a) in enumerate(spheres):
        for name_b, node_b, c_b, r_b in spheres[i + 1:]:
            if name_a == name_b:
                continue
            d = c_b - c_a
            dist = float(np.linalg.norm(d))
            gap = dist - (r_a + r_b)
            if max_gap is not None and gap > max_gap:
                continue
            # Coincident centres have no direction to put the surface points on;
            # report the centres themselves rather than dividing by zero.
            u = d / dist if dist > 1e-12 else np.zeros(3)
            out[f"{name_a}/{node_a}/{name_b}/{node_b}"] = (
                c_a + r_a * u, c_b - r_b * u, gap)
    return out


class EllipsoidMetric(NamedTuple):
    """One fingertip's distance to an ellipsoid object under BOTH metrics.

    ``exact`` is the Eberly orthogonal distance the solver uses by default;
    ``taubin`` is the first-order gradient-weighted algebraic approximation
    (``EnvironmentConfig::ellipsoid_taubin``), which the paper reports as
    ill-conditioned for eccentric shells. Both are surface gaps -- the sphere
    radius already subtracted -- so each is zero at the contact equality's own
    zero set and they are directly comparable.

    ``in_use`` says which one the SOLVE was configured with
    (``params.ellipsoid_taubin``), so the overlay can mark the number the
    residual was actually built from rather than leaving a reader to guess.

    Drawn side by side because the difference IS the diagnostic: the two agree
    near the surface and diverge in the far field (a true 15 mm gap from the
    flat ``coin`` reads ~8 mm under Taubin), and seeing the divergence grow with
    eccentricity is what the exact metric was adopted for.
    """
    sphere_pt: np.ndarray
    surface_pt: np.ndarray
    exact: float
    taubin: float
    in_use: str


def ellipsoid_metric_witness(params, result, k=0, names=None):
    """Per contact finger ``{name: EllipsoidMetric}`` at frame ``k``, or None when
    the object has no ellipsoid form.

    None -- rather than a table of identical numbers -- for the sphere, cylinder,
    capsule and cube primitives: their distances are exact SDFs under either
    setting, so ``ellipsoid_taubin`` is inert there and a comparison overlay
    would be two copies of one number claiming to be a measurement.

    Both metrics come from :func:`primitive_surface_gap`, which mirrors
    ``EllipsoidDistance`` including the ellipsoid SET's LogSumExp smooth min --
    so the pair reported here is the pair the two builds of the factor would
    report, not a re-derivation free to drift from either.
    """
    if result.spec.get("type") not in ("ellipsoid", "ellipsoid_set"):
        return None

    frame = result.frames[k]
    center = np.asarray(result.object_center, float)
    R = np.asarray(result.object_rotation, float)
    # The CONTACT surface, narrowed exactly as contact_witness narrows it: this
    # sits beside the contact gap line and has to be measuring the same shells.
    spec = subset_spec(result.spec, result.contact_subset)
    wanted = set(result.contact_names() if names is None else names)
    in_use = "taubin" if getattr(params, "ellipsoid_taubin", False) else "exact"

    out = {}
    for name, radius in zip(result.finger_names, result.tip_radii):
        if name not in wanted or name not in frame:
            continue
        tip = frame[name].tip_point()
        local = R.T @ (tip - center)
        dist, foot_local, n_local = primitive_surface_witness(local, spec)
        out[name] = EllipsoidMetric(
            tip - radius * (R @ n_local), center + R @ foot_local,
            float(dist) - float(radius),
            float(primitive_surface_gap(local, spec, taubin=True)) - float(radius),
            in_use)
    return out
