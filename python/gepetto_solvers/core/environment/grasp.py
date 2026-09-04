"""Geometric grasp alignment: make the contacts SURROUND the object.

Part of the ``attach_*`` family: mutates the per-finger configs in place and
returns them for chaining, and takes ``contact_fingers``, the same per-finger
bool mask every other member of the family takes.

The one constraint here answers a question no per-contact factor can. Each
witness contact sees exactly one fingertip and says only "this one touches";
satisfying all of them is equally well achieved by three fingers landing side by
side on the same face. :func:`attach_grasp_alignment` adds the statement that
distinguishes a grasp from three independent touches -- the contact normals must
geometrically oppose one another -- as a single Vector6 equality over every
contacting digit at once:

    h_grasp({p_i}, T_obj) = sum_i [ -n_i ; -(p_i - t_obj) x n_i ] = 0

Purely KINEMATIC. Every contact is credited with one unit "virtual force" along
its inward surface normal; the constraint says those unit forces, and the torques
they generate about the object origin, cancel. There is no mass, no friction
cone and no force magnitude anywhere in it, so this is a statement about WHERE
the contacts sit, not about whether the hand could actually hold the thing.
"""

import gepetto_solvers

from ..hands.tendon_5f import (
    _resolve_contact_mask,
)


def attach_grasp_alignment(configs, *, contact_fingers=None,
                           sigma_force=1.0, sigma_torque=1.0,
                           curvature_step=None, gradient_step=None):
    """Attach the grasp wrench-equilibrium constraint to every PARTICIPATING
    finger's env, in place. Returns ``configs`` for chaining.

    A HAND-LEVEL constraint, the same shape as the three ``attach_pregrasp_*``
    functions: the C++ layer collects every digit whose env carries
    ``grasp_alignment_enabled`` alongside a ``target_contact_node``, and adds ONE
    Vector6 equality over their witness points. The remaining fields are shared
    constants duplicated onto every participating env, first-one-found-wins, as
    ``pregrasp_clearance_*`` already are.

    Call AFTER :func:`attach_contact`, which is what creates the env and sets the
    contact node this reads.

    ``sigma_force`` / ``sigma_torque`` scale the two halves of the residual
    SEPARATELY, and that is not a nicety. The top three rows are a sum of unit
    normals and so dimensionless; the bottom three are a sum of moment arms and
    carry units of length. A single isotropic sigma therefore weights the two
    against each other by whatever the object's size happens to be -- fine at
    centimetre scale, badly lopsided on anything larger, where the torque rows
    want loosening.

    ``curvature_step`` / ``gradient_step`` (None = the factor's own default of
    half a grid voxel) are the two finite-difference stencils it measures the
    normal field and its shape operator with. The default is the measured sweet
    spot: a sub-voxel stencil resolves the trilinear interpolant's own linear fit
    rather than the geometry. Raise them on a noisy grid; the cost is a smoothed
    normal, which for a constraint Jacobian is the safe direction to err in.

    Two ways to ask for something that cannot be built, both of which RAISE
    rather than quietly no-op -- the same reasoning :func:`attach_contact`
    documents. This constraint is invisible in a pose (a hand can look perfectly
    grasp-like while violating it), so a silent skip would be indistinguishable
    from a satisfied constraint:

      * a binding with no ``grasp_alignment_enabled`` field,
      * fewer than two participating digits, which makes the equality
        unsatisfiable rather than merely hard: one unit force cannot cancel.
    """
    probe = gepetto_solvers.EnvironmentConfig()
    if not hasattr(probe, "grasp_alignment_enabled"):
        raise AttributeError(
            "this gepetto_solvers build has no EnvironmentConfig."
            "grasp_alignment_enabled, so the grasp wrench alignment (h_grasp) "
            "cannot be built -- rebuild it (pip install . from the repo root)")

    mask = _resolve_contact_mask(configs, contact_fingers)
    if sum(1 for on in mask if on) < 2:
        raise ValueError(
            "grasp alignment (h_grasp) balances the virtual forces of two or "
            "more contacts against each other; "
            f"{sum(1 for on in mask if on)} digit(s) are checked. A single unit "
            "force cannot sum to zero, so the constraint would be infeasible "
            "rather than merely tight -- check more digits, or turn it off")

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        env.grasp_alignment_enabled = bool(mask[i])
        env.grasp_alignment_sigma_force = float(sigma_force)
        env.grasp_alignment_sigma_torque = float(sigma_torque)
        if curvature_step is not None:
            env.grasp_alignment_curvature_step = float(curvature_step)
        if gradient_step is not None:
            env.grasp_alignment_gradient_step = float(gradient_step)
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_ellipsoid_grasp_alignment(configs, *, contact_fingers=None,
                                     sigma_force=1.0, sigma_torque=1.0,
                                     curvature_step=None):
    """Attach the APPROXIMATION-PHASE grasp wrench-equilibrium constraint
    (h_grasp,E) to every PARTICIPATING finger's env, in place. Returns
    ``configs`` for chaining.

    The same hand-level Vector6 equality :func:`attach_grasp_alignment` builds,
    and an independent constraint rather than a mode of it -- both may be on at
    once. What differs is where the residual is MEASURED:

        h_grasp,E({c_i}) = sum_i [ -n_i ; -(c_i - t_obj) x n_i ] = 0

    on the contact sphere CENTERS ``c_i``, with ``n_i`` the outward normal of the
    analytic ellipsoid set rather than of a baked SDF grid. The sphere radius
    cancels out of the torque exactly, because the virtual force is collinear
    with the radius vector, so nothing here needs a witness variable.

    THAT IS THE WHOLE POINT. :func:`attach_grasp_alignment` keys off witness
    points, and the C++ layer raises on any digit using the center-direct contact
    form -- so the wrench balance is unavailable in exactly the phase where the
    hand is still being steered by the smooth ellipsoid proxy, before any exact
    contact exists. The contacts then get arranged around the object only at the
    last phase, by which point the approach has already committed to a posture
    where they cannot be. This is the form that runs during the approximation.

    Reads the FULL ellipsoid set, not the grasp subset: the normal field is a
    property of the object's geometry, not of which shells a finger was cleared
    to touch. Same choice the collision gap factor makes.

    Call AFTER :func:`attach_contact`, which is what creates the env and sets the
    contact node this reads.

    ``sigma_force`` / ``sigma_torque`` scale the two halves of the residual
    separately, for the reason :func:`attach_grasp_alignment` documents at
    length: dimensionless force rows against metre torque rows.

    ``curvature_step`` (None = the factor's own default of 1e-5 m) is the
    central-difference stencil for the shape operator dn/dc. Three orders tighter
    than the SDF sibling's half-a-voxel because this field is a closed form, with
    no interpolation floor for the stencil to clear.

    Three ways to ask for something that cannot be built, all of which RAISE
    rather than quietly no-op -- the same reasoning :func:`attach_contact`
    documents, and it applies with extra force here because this constraint is
    invisible in a pose:

      * a binding with no ``ellipsoid_grasp_alignment_enabled`` field,
      * fewer than two participating digits, which makes the equality
        unsatisfiable rather than merely hard: one unit force cannot cancel,
      * an object with no ellipsoid surface, which the C++ layer raises on --
        there is no fall back to a baked SDF, since running on the smooth proxy
        is what this constraint is for.
    """
    probe = gepetto_solvers.EnvironmentConfig()
    if not hasattr(probe, "ellipsoid_grasp_alignment_enabled"):
        raise AttributeError(
            "this gepetto_solvers build has no EnvironmentConfig."
            "ellipsoid_grasp_alignment_enabled, so the approximate grasp "
            "alignment (h_grasp,E) cannot be built -- rebuild it "
            "(pip install . from the repo root)")

    mask = _resolve_contact_mask(configs, contact_fingers)
    if sum(1 for on in mask if on) < 2:
        raise ValueError(
            "approximate grasp alignment (h_grasp,E) balances the virtual "
            "forces of two or more contacts against each other; "
            f"{sum(1 for on in mask if on)} digit(s) are checked. A single unit "
            "force cannot sum to zero, so the constraint would be infeasible "
            "rather than merely tight -- check more digits, or turn it off")

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        env.ellipsoid_grasp_alignment_enabled = bool(mask[i])
        env.ellipsoid_grasp_alignment_sigma_force = float(sigma_force)
        env.ellipsoid_grasp_alignment_sigma_torque = float(sigma_torque)
        if curvature_step is not None:
            env.ellipsoid_grasp_alignment_curvature_step = float(curvature_step)
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs
