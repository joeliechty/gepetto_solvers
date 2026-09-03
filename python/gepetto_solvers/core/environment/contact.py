"""Object contact: drive fingertips onto the object surface.

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


def attach_contact(configs, spec, objects_dir, primitive, object_pose, *,
                   contact_nodes=None,
                   tip_radii=None, radius=None, contact_fingers=None,
                   object_pose_cov=None, proxy_and_exact=False,
                   drop_normal_row=False, ellipsoid_set_beta=None,
                   in_plane=False, pinch_centroid=None, contact_subset=None,
                   object_contact_exact=False):
    """Attach the shared object surface + a terminal tip contact to every finger
    of a hand config list, in place. Returns ``configs`` for chaining.

    This is the block the contact demo scripts (``ik_5f_contact.py``,
    ``traj_5f_contact.py``, ...) write inline, factored out so the solver classes
    and any caller share one definition of "the fingertip touches the object".

    ``contact_fingers`` (None = all, the legacy behavior) is a per-finger bool
    mask: a finger whose flag is False still gets the env — so
    :func:`attach_collision` / :func:`attach_table` can hang off it and keep that
    finger out of the object — but *without* ``target_contact_node``, so the C++
    layer treats it as a collision-only env: ``HandModel::build_graph``
    adds no witness contact factor for it and ``get_initial_values`` seeds no
    witness point. That is the same shape the trajectory planner already builds
    for every step before k=K, so it is a well-trodden path. Use it to solve for
    a pinch/subset grasp instead of forcing all five fingertips onto the object.

    ``radius`` overrides the contact sphere radius for all fingers; otherwise
    ``tip_radii[i]`` is used. The radius is written even for a masked-off finger
    (it is inert without a contact node, and keeps the env self-describing).

    ``proxy_and_exact`` attaches the bounding-ellipsoid proxy *and* the baked SDF,
    so a solve can slide against the proxy in phase 2 and servo on the exact
    geometry in phase 3. Default False keeps the single-surface behavior every
    existing caller relies on.

    ``object_contact_exact`` is the other half of that pair, and the two are
    normally set together: it points the CONTACT equality at the baked SDF while
    leaving the collision inequalities on the proxy attached alongside it. That
    split is the phase-3/4 formulation -- h_rad/h_sdf/h_tan on the true surface,
    h_pen on E_obj -- and it is the one place in this module where the two
    families deliberately read different geometry. Nothing else changes: the
    finger mask, the radius and ``drop_normal_row`` all apply exactly as they do
    to any other witness contact.

    Written for every finger regardless of the mask, like ``drop_normal_row``,
    because it names the contact FORM rather than who is contacting. The C++
    layer raises for it without a grid, or alongside ``in_plane`` (a different
    form on a different surface) -- see ``HandModel::uses_center_direct_contact``.

    ``ellipsoid_set_beta`` overrides the LogSumExp sharpness for an
    ``ellipsoid_set`` object (None = the spec's own value). Only the smooth-min
    STANDOFF changes with it, not the geometry: the constraint surface sits up to
    ln(K)/beta outside the true union. Inert for every other surface kind.

    ``contact_subset`` (``scene.grasp_subset_indices``) restricts which members
    of an ``ellipsoid_set`` the fingertips may be driven onto -- the authored
    "these shells are handles, those only bound the shape" choice that travels
    with a YCB fit. None = every member, the pre-existing behavior, and inert for
    a surface with no members to choose between.

    It narrows CONTACT ONLY. :func:`attach_collision` shares this very env, and
    the whole set stays on it, so the excluded shells keep pushing the fingers
    out while nothing is sent to touch them. That is the point: they are the
    drill's housing, not a handle, and a hand allowed to pass through them would
    be planning against an object that is not there.

    ``drop_normal_row`` (Eq 2.12-2.15) selects the 4-row witness contact form
    [c_R, c_O, c_T1, c_T2] (c_N dropped) instead of the default 5-row form.
    Written for every finger regardless of ``contact_fingers`` -- it is a
    property of the contact FORM, like ``radius``, not gated by the mask. Only
    affects the witness-point contact factor; inert for a center-direct
    ellipsoid contact, which has no normal row to begin with.

    ``in_plane`` (Eq 13) swaps the object contact equality from the full 3D
    distance to the distance measured inside each finger's pulling plane (Eq 11).
    It needs ``pinch_centroid``: the wrist-frame point where the participating
    digits meet (:func:`pinch_pose_for_mask` off the SAME mask), which is the
    plane's third point. Also a property of the contact FORM, so written to every
    finger's env.

    Three ways to ask for something that cannot be built, all of which RAISE
    rather than quietly falling back to the 3D form -- the same reasoning
    :func:`attach_ellipsoid_set` documents. Degrading silently here would leave
    the caller believing a constraint is in the graph that is not, and the
    resulting grasp would look like a solver failure rather than a mis-request:

      * a binding with no ``object_contact_in_plane`` field,
      * an object with no ellipsoid form (cube/cylinder/capsule, and a baked SDF
        with no analytic look-alike): no cross-section for the plane to cut,
      * no ``pinch_centroid``: Eq 11 has no plane without it, which is what a
        thumbless digit set gives you.
    """
    import gepetto_solvers

    from ..geometry.scene import (
        configure_object_proxy_and_exact,
        configure_object_surface,
        ellipsoid_members,
    )

    mask = _resolve_contact_mask(configs, contact_fingers)
    centroid = None
    if object_contact_exact:
        probe = gepetto_solvers.EnvironmentConfig()
        if not hasattr(probe, "object_contact_exact"):
            raise AttributeError(
                "this gepetto_solvers build has no EnvironmentConfig."
                "object_contact_exact, so the contact cannot be pointed at the "
                "baked SDF while the proxy stays on collision -- rebuild it "
                "(pip install . from the repo root)")
        if "vdb" not in spec:
            raise ValueError(
                f"exact object contact needs a baked SDF, but the "
                f"{spec['type']!r} object {primitive!r} names no grid -- bake "
                f"one (python scripts/objects/setup_objects.py), or contact the "
                f"ellipsoid proxy instead")
        if not proxy_and_exact:
            raise ValueError(
                "exact object contact reads the baked SDF while the collision "
                "inequalities read the ellipsoid proxy, so BOTH surfaces have to "
                "be attached -- pass proxy_and_exact=True. Without it only one "
                "surface is configured and there is no proxy left for h_pen")
    if in_plane:
        probe = gepetto_solvers.EnvironmentConfig()
        if not hasattr(probe, "object_contact_in_plane"):
            raise AttributeError(
                "this gepetto_solvers build has no EnvironmentConfig."
                "object_contact_in_plane, so the Eq 13 in-plane contact cannot be "
                "built -- rebuild it (pip install . from the crest-sparse root)")
        if ellipsoid_members(spec) is None:
            raise ValueError(
                f"in-plane contact (Eq 13) needs an ellipsoid surface to cut, but "
                f"the {spec['type']!r} object {primitive!r} has none -- use a "
                f"sphere, an ellipsoid or a ycb: set, or contact it in 3D")
        if pinch_centroid is None:
            raise ValueError(
                "in-plane contact (Eq 13) needs pinch_centroid, the wrist-frame "
                "point Eq 11 spans the pulling plane with; the checked digits have "
                "no measured pinch pose (only combinations INCLUDING THE THUMB "
                "were measured -- see HAND_PINCH_POSES)")
        centroid = np.asarray(pinch_centroid, dtype=float).reshape(3)
    if object_pose_cov is None:
        object_pose_cov = 1e-8 * np.eye(6)
    setup_surface = (configure_object_proxy_and_exact if proxy_and_exact
                     else configure_object_surface)

    for i, (_, cfg) in enumerate(configs):
        env = gepetto_solvers.EnvironmentConfig()
        setup_surface(env, spec, objects_dir, primitive,
                      contact_subset=contact_subset)
        if ellipsoid_set_beta is not None and hasattr(env, "ellipsoid_set_beta"):
            env.ellipsoid_set_beta = float(ellipsoid_set_beta)
        env.object_pose_mean = object_pose
        env.object_pose_cov = object_pose_cov
        env.object_pose_per_step = False
        if radius is not None:
            env.contact_node_radius = radius
        elif tip_radii is not None:
            env.contact_node_radius = tip_radii[i]
        env.contact_drop_normal_row = drop_normal_row
        if object_contact_exact:
            env.object_contact_exact = True
        if centroid is not None:
            env.object_contact_in_plane = True
            env.contact_plane_centroid = centroid
        if mask[i]:
            env.target_contact_node = (tip_node_index(cfg)
                                       if contact_nodes is None
                                       else contact_nodes[i])
        cfg.sdf_contact = env
    return configs
