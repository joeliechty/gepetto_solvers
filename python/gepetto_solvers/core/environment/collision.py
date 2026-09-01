"""Collision avoidance: the sphere set, and the inequalities built on it.

Part of the ``attach_*`` family: every function here MUTATES the per-finger
configs in place and returns them for chaining, and every one accepts
``contact_fingers``, a per-finger bool mask. A masked-off finger still gets the
environment -- so collision and plane avoidance keep protecting it -- but no
``target_contact_node``, which the C++ layer reads as a collision-only env.
"""

import numpy as np

from ..hands.tendon_5f import (
    disc_node_indices,
    proximal_disc_flags,
)


def attach_collision(configs, vdb_path, object_pose, *,
                     radius=0.003, sigma=1e-4, num_proximal_discs=2,
                     object_pose_cov=None, cull_margin=None, avoidance=True,
                     self_collision=True):
    """Declare the Section 1.5 collision spheres on every finger of a hand config
    list, in place. Returns ``configs`` for chaining.

    Each finger gets collision spheres on its disc nodes (radius ``radius``),
    with the metacarpal discs flagged proximal. If a finger already has an
    ``sdf_contact`` env (e.g. a terminal tip contact), the collision fields are
    added to that same env so contact and collision share one object; otherwise
    a fresh collision-only env is created and the SDF loaded.

    The C++ hand builder then adds, at every trajectory step, sphere-to-SDF
    inequalities keeping each finger out of the object and sphere-to-sphere
    inequalities keeping distinct fingers apart (skipping proximal-proximal
    pairs).

    The sphere SET is one thing; the three constraint families built on it are
    three others, each with its own switch -- ``avoidance`` (finger-OBJECT,
    ``env.collision_avoidance``), ``self_collision`` (FINGER-FINGER,
    ``env.self_collision``) and :func:`attach_table`'s ``avoidance``
    (finger-PLANE, ``env.plane_avoidance``). Declaring the spheres builds
    nothing on its own; every family is gated on its own field alone, so any
    combination of the three is available. ``avoidance=False`` with a support
    plane is how a caller turns table collision on with object collision off;
    ``self_collision`` defaults True because keeping the fingers out of each
    other is wanted in nearly every solve.

    ``cull_margin`` (m, None = keep all pairs): drop finger-finger sphere pairs
    whose gap at the initial values exceeds this margin. Heuristic speedup —
    roughly half the 5-finger trajectory graph is inequality constraints that
    never activate — but a culled pair is unprotected, so rely on the tests'
    all-pairs penetration report to validate the chosen margin. Finger-object
    constraints are never culled.
    """
    import gepetto_solvers

    if object_pose_cov is None:
        object_pose_cov = 1e-8 * np.eye(6)

    for _, cfg in configs:
        env = cfg.sdf_contact            # copy (or None) via the optional binding
        if env is None:
            env = gepetto_solvers.EnvironmentConfig()
            env.load_sdf(vdb_path)
            env.object_pose_mean = object_pose
            env.object_pose_cov = object_pose_cov
            env.object_pose_per_step = False

        nodes = disc_node_indices(cfg)
        env.collision_avoidance = bool(avoidance)
        if not hasattr(env, "self_collision"):
            if not self_collision:
                raise AttributeError(
                    "this gepetto_solvers build has no "
                    "EnvironmentConfig.self_collision, so finger-finger "
                    "avoidance cannot be turned off -- rebuild it "
                    "(pip install .)")
        else:
            env.self_collision = bool(self_collision)
        env.collision_sigma = sigma
        env.collision_node_indices = nodes
        env.collision_node_radii = [radius] * len(nodes)
        env.collision_node_is_proximal = proximal_disc_flags(cfg, num_proximal_discs)
        if cull_margin is not None:
            env.collision_cull_margin = cull_margin
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs
