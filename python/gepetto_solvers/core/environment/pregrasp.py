"""Pre-grasp positioning: place the WRIST before anything closes.

Part of the ``attach_*`` family: every function here MUTATES the per-finger
configs in place and returns them for chaining, and every one accepts
``contact_fingers``, a per-finger bool mask. A masked-off finger still gets the
environment -- so collision and plane avoidance keep protecting it -- but no
``target_contact_node``, which the C++ layer reads as a collision-only env.
"""

import numpy as np

from ..hand.config.discs import _resolve_contact_mask
from ..hand.config.morphology import tip_node_index


def attach_pregrasp_center(configs, *, clearance_height=0.0, clearance_normal=None,
                           contact_fingers=None, contact_node=None):
    """Attach the pre-grasp hand-centering constraint (Eq 2.18-2.19) to every
    PARTICIPATING finger's env, in place. Returns ``configs`` for chaining.

    A HAND-LEVEL constraint: the C++ layer collects every finger with
    ``pregrasp_center_node`` set, groups the one named "thumb" against the
    rest, and adds ONE Vector3 equality centering their sphere-center midpoint
    over the object (raised by ``clearance_height`` along ``clearance_normal``).
    Requires the thumb AND at least one other finger to participate, and a
    nonzero ``clearance_normal``, or the C++ layer silently skips the
    constraint.

    ``contact_fingers`` (None = all) selects which fingers participate, the
    same per-finger bool mask :func:`attach_contact`/:func:`attach_table` take.
    Call AFTER attach_contact (needs an existing env with ``object_pose_mean``/
    ``object_pose_cov`` set, so this constraint can anchor the object pose even
    when no other block does).
    """
    mask = _resolve_contact_mask(configs, contact_fingers)
    normal = (np.asarray(clearance_normal, dtype=float).reshape(3)
             if clearance_normal is not None else np.zeros(3))

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        env.pregrasp_clearance_height = clearance_height
        env.pregrasp_clearance_normal = normal
        env.pregrasp_center_node = (
            (contact_node if contact_node is not None else tip_node_index(cfg))
            if mask[i] else None)
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_pregrasp_axis_alignment(configs, axis, *, contact_fingers=None, contact_node=None):
    """Attach the pre-grasp short-axis alignment constraint (companion to
    Eq 2.16-2.17) to every PARTICIPATING finger's env, in place. Returns
    ``configs`` for chaining.

    A HAND-LEVEL constraint, same shape as :func:`attach_pregrasp_center`: the
    C++ layer collects every finger with ``pregrasp_align_node`` set, groups
    the one named "thumb" against the rest, and adds ONE scalar equality
    aligning the vector between their sphere-center centroids with ``axis``,
    direction-agnostically (squared cosine). Requires the thumb AND at least
    one other finger to participate, and a nonzero ``axis``, or the C++ layer
    silently skips the constraint.

    ``axis`` is a caller-supplied world-frame direction -- typically
    ``solvers.default_half_space_axis(...)``, the SAME axis the opposition
    half-space uses (perpendicular to the object's longest in-plane axis).
    Passed in rather than derived here so this stays a pure env-mutation
    helper, matching :func:`attach_half_space`/:func:`attach_pregrasp_center`.

    ``contact_fingers`` (None = all) selects which fingers participate, the
    same per-finger bool mask every other ``attach_*`` helper here takes.
    Call AFTER attach_contact (needs an existing ``cfg.sdf_contact`` env).
    """
    mask = _resolve_contact_mask(configs, contact_fingers)
    m_hat = np.asarray(axis, dtype=float).reshape(3)
    if np.linalg.norm(m_hat) > 0:
        m_hat = m_hat / np.linalg.norm(m_hat)

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        env.pregrasp_align_axis = m_hat
        env.pregrasp_align_node = (
            (contact_node if contact_node is not None else tip_node_index(cfg))
            if mask[i] else None)
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_pregrasp_centroid(configs, centroid, *, clearance_height=0.0,
                             clearance_normal=None):
    """Attach the pre-grasp PINCH-CENTROID centering constraint to every
    finger's env, in place. Returns ``configs`` for chaining.

    A HAND-LEVEL constraint like :func:`attach_pregrasp_center`, but with one
    structural difference that shows up in this signature: there is no
    per-finger mask and no ``contact_node``. ``centroid`` is a point FIXED in
    the wrist frame (from :data:`HAND_PINCH_POSES`, chosen by the caller from
    whichever digits are participating), so no finger opts in and no fingertip
    pose enters the residual -- the C++ layer keys the factor off the shared
    wrist variable and the object, and reads these fields off whichever env it
    finds them on first. They are written to every finger's env anyway, the
    same way ``plane_origin``/``plane_normal`` are, so the envs stay uniform.

    The C++ layer silently skips the constraint when ``clearance_normal`` has
    zero norm, so leaving it unset is a safe no-op. Call AFTER attach_contact
    (needs an existing ``cfg.sdf_contact`` env carrying the object pose, which
    this constraint can end up anchoring).
    """
    c = np.asarray(centroid, dtype=float).reshape(3)
    normal = (np.asarray(clearance_normal, dtype=float).reshape(3)
             if clearance_normal is not None else np.zeros(3))

    for _, cfg in configs:
        env = cfg.sdf_contact
        if env is None:
            continue
        env.pregrasp_centroid_point = c
        env.pregrasp_centroid_clearance = float(clearance_height)
        env.pregrasp_centroid_normal = normal
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs
