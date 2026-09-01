"""The support plane, and the opposition half-space that splits the hand across it.

Part of the ``attach_*`` family: every function here MUTATES the per-finger
configs in place and returns them for chaining, and every one accepts
``contact_fingers``, a per-finger bool mask. A masked-off finger still gets the
environment -- so collision and plane avoidance keep protecting it -- but no
``target_contact_node``, which the C++ layer reads as a collision-only env.
"""

import numpy as np

from ..hands.tendon_5f import (
    _resolve_contact_mask,
    tip_node_index,
)


def attach_table(configs, plane_origin, plane_normal, *,
                 avoidance=True, contact_node=None, radius=None,
                 tip_radii=None, dims=None, contact_fingers=None):
    """Attach a Section 1.6 world-fixed analytic support plane ("table") to every
    finger of a hand config list, in place. Returns ``configs`` for chaining.

    The table is a half-space with origin ``plane_origin`` and OUTWARD unit normal
    ``plane_normal`` (``SDF_table(p) = (p - origin) . normal``). The plane fields
    are written onto each finger's existing ``sdf_contact`` env (created if absent)
    so contact/collision/table share one env. Per finger this sets:
      * ``plane_origin`` / ``plane_normal`` — the support surface,
      * ``plane_avoidance`` = ``avoidance`` — the free-space approach collision
        (Eq 1.59): every non-tip collision sphere is kept out of the half-space,
      * ``table_contact_node`` — the fingertip node that slides on the plane
        (defaults to ``tip_node_index(cfg)``, and see ``contact_fingers`` below).
        That node gets a SINGLE-residual equality on its sphere CENTER,
        ``Dist_plane(c) = 0`` (``PlaneCollisionGapFactor`` as a
        ``ZeroCostConstraint``) — not the original §1.6 five-residual witness
        form, which introduced a free contact point whose gauge four of its rows
        existed only to pin. The C++ planner *schedules* this field per step
        around ``k_touch`` (cleared during the approach phase, kept during the
        slide phase), so it is safe to set it for every step here,
      * ``table_contact_radius`` — that tip's contact sphere radius.

    ``contact_fingers`` (None = all, the legacy behavior) is the same per-finger
    bool mask :func:`attach_contact` takes: a finger that is not solving for
    contact gets the plane and its avoidance inequality but *no*
    ``table_contact_node``, so nothing asks it to touch the table either. Where
    ``avoidance`` is active that fingertip is then held *above* the plane rather
    than pinned to it — the C++ layer exempts the table contact node from plane
    avoidance (its collision would fight the sliding equality it is pinned by),
    and a masked-off finger no longer has one. During the planner's slide phase
    (k >= ``k_touch``) plane avoidance is off for every finger by design, so
    there a masked-off fingertip is simply unconstrained by the plane.

    ``radius`` overrides the per-finger tip radius for all fingers; otherwise
    ``tip_radii[i]`` (if given) or the env's existing ``contact_node_radius`` is
    used. The plane is treated as absent by the C++ layer whenever the normal has
    zero norm, so this is a no-op-safe opt-in that leaves plane-free runs unchanged.
    """
    import gepetto_solvers

    mask = _resolve_contact_mask(configs, contact_fingers)
    origin = np.asarray(plane_origin, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            env = gepetto_solvers.EnvironmentConfig()
        env.plane_origin = origin
        env.plane_normal = normal
        env.plane_avoidance = avoidance
        if not mask[i]:
            # No sliding equality for this finger; clear rather than skip, in case
            # the env already carried a contact node from an earlier attach.
            env.table_contact_node = None
        else:
            env.table_contact_node = (contact_node if contact_node is not None
                                      else tip_node_index(cfg))
            if radius is not None:
                env.table_contact_radius = radius
            elif tip_radii is not None:
                env.table_contact_radius = tip_radii[i]
            else:
                env.table_contact_radius = env.contact_node_radius
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def opposition_axis_from_object(plane_normal, e_long):
    """``m_hat = n_hat x e_long``: the opposition axis implied by splitting the
    support surface ALONG the object's longest in-plane axis.

    Putting the split *line* along ``e_long`` (e.g. lengthwise along a pen)
    means the half-space normal -- the direction that actually separates thumb
    from fingers -- is perpendicular to it within the plane. That is exactly
    ``n_hat x e_long``: thumb and fingers end up opposed ACROSS the object's
    width, not split along its length. Get ``e_long`` from
    :func:`scene.object_principal_inplane_axis`, which already handles
    degenerate (in-plane-isotropic) objects with a documented fallback.

    NOTE this generally differs from :func:`opposition_directions`'s legacy
    default of world +X, which is only correct by coincidence when it happens
    to already be perpendicular to the object's long axis -- for an elongated
    object (a pen) oriented so its length runs along world +X, using world +X
    as ``m_hat`` directly splits the two groups ACROSS the object's length
    (bisecting its short axis) instead of along it, putting the thumb near one
    end and the fingers near the other.

    For an object with no long axis at all (a ball) ``e_long`` falls back to
    world +Y, so ``m_hat`` comes out -X: thumb on the -X side of the object,
    the opposing fingers on +X.
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    e = np.asarray(e_long, dtype=float).reshape(3)
    e = e - (e @ n) * n
    ne = np.linalg.norm(e)
    if ne < 1e-9:
        raise ValueError(
            "opposition_axis_from_object: e_long is parallel to the plane "
            "normal, so it defines no in-plane split direction")
    m = np.cross(n, e / ne)
    return m / np.linalg.norm(m)


def opposition_directions(configs, *, thumb_index=-1, axis=None):
    """Per-finger in-plane unit vectors ``m_hat`` for the Eq 2.16-2.17 (Eq 1.92)
    half-space split.

    Divide the support surface in half along ``axis`` and put the thumb on one
    half, the grasping fingers on the other. This returns ``+axis`` for the
    thumb and ``-axis`` for every other finger, so the two groups are driven to
    opposite halves.

    ``axis`` (default world +X, which is thumb-on-+X and so the MIRROR of what
    the derived path now produces for a shapeless object -- see
    :func:`opposition_axis_from_object`; every live caller passes an explicit
    axis from ``solvers.default_half_space_axis``, and new ones should too)
    must lie IN the support plane -- the
    constraint is only radius-independent when ``n_table . m_hat = 0``, which
    is what makes its Jacobian constant. ``thumb_index`` defaults to the last
    config, matching :func:`get_default_hand_configs` (four fingers, then the
    thumb).
    """
    if axis is None:
        axis = np.array([1.0, 0.0, 0.0])
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    n = len(configs)
    thumb = thumb_index % n
    return [axis if i == thumb else -axis for i in range(n)]


def attach_half_space(configs, split_point, directions, *, contact_fingers=None,
                      margin=0.0, contact_node=None):
    """Attach the Eq 2.16-2.17 (Eq 1.92) opposition half-space to every masked-in
    finger's env, in place. Returns ``configs`` for chaining.

    ``split_point`` is a point on the splitting line (e.g. the object centroid
    projected onto the support surface); ``directions`` is one in-plane unit
    vector per finger, as produced by :func:`opposition_directions`. A finger
    masked off by ``contact_fingers`` gets no half-space.

    ``contact_node`` is the node whose sphere center is constrained (default
    ``tip_node_index(cfg)``, the same fingertip :func:`attach_table` slides),
    written onto this constraint's OWN field ``env.half_space_node``. Standing
    on its own field is the point: the constraint used to be built off
    ``table_contact_node``, so it silently did nothing without table contact.
    It needs no support plane and no contact of any kind, and can be attached
    before or after anything else -- it does need an env to write onto, so call
    it after :func:`attach_contact` or :func:`attach_collision` has made one.

    ``margin`` (m, >= 0) is the MINIMUM STANDOFF each finger must keep from the
    splitting line, written onto ``env.half_space_margin`` -- the constraint the
    C++ ``HalfSpaceGapFactor`` builds is then

        -(c - p_split) . m_hat + margin <= 0 ,

    so the thumb's side and the opposing fingers' side are each held ``margin``
    off the split (a corridor of width ``2 * margin`` between them). At 0 -- the
    default, and the original constraint -- a fingertip sitting exactly ON the
    split is already legal, so opposition alone does not stop the digits closing
    onto each other. Raises on a binding too old to carry the field rather than
    silently dropping the standoff; call
    :func:`solvers.capabilities`'s ``half_space_margin`` to gate on it.
    """
    mask = _resolve_contact_mask(configs, contact_fingers)
    if len(directions) != len(configs):
        raise ValueError(
            f"directions has {len(directions)} entries but there are "
            f"{len(configs)} fingers; pass one m_hat per finger.")
    p_split = np.asarray(split_point, dtype=float).reshape(3)
    margin = float(margin)

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        if mask[i]:
            m = np.asarray(directions[i], dtype=float).reshape(3)
            env.half_space_enabled = True
            env.half_space_split_point = p_split
            env.half_space_normal = m / np.linalg.norm(m)
            if hasattr(env, "half_space_node"):
                env.half_space_node = (contact_node if contact_node is not None
                                       else tip_node_index(cfg))
            if margin != 0.0 and not hasattr(env, "half_space_margin"):
                raise AttributeError(
                    "this gepetto_solvers build has no "
                    "EnvironmentConfig.half_space_margin -- rebuild it "
                    "(pip install .) to use an opposition standoff")
            if hasattr(env, "half_space_margin"):
                env.half_space_margin = margin
        else:
            env.half_space_enabled = False
            # Clear rather than skip, in case the env already carried a node
            # from an earlier attach -- same reason attach_table clears
            # table_contact_node for a masked-off finger.
            if hasattr(env, "half_space_node"):
                env.half_space_node = None
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs
