"""A five-finger tendon *hand* — four fingers plus an opposable thumb, all on
one shared floating wrist — closing every fingertip onto a single shared object
(a full-hand grasp).

This is the five-finger generalization of ``ik_2f_contact.py``:
all fingers are part of ONE factor graph, share ONE wrist variable ``T_base``
(with different ``hand_base_offset`` transforms), and each fingertip is driven
onto the shared SDF object surface by its own hard SDF contact constraint.

The hand is the *anatomical* hand used by ``fk_5f_sweep.py`` —
``config.get_default_hand_configs()`` — whose per-digit bone/joint lengths, palm
origins and base angles come from ``gepetto_core``'s ``parameters.scad`` (with a
hard-coded fallback). Each finger's contact sphere uses that digit's
CAD-derived ``cfg.tip_radius`` (from the distal tip width), not a single
hard-coded radius.

Run (from the ``python/`` directory):
    python scripts/ik_5f_contact.py sphere
"""

import os
import argparse
import time

import numpy as np

import gepetto_solvers

from gepetto_solvers.core.objects import OBJECTS_DIR
from gepetto_solvers.core.hand.config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index)
# Shared primitive geometry + grasp scene constants (see scene.py).
from gepetto_solvers.core.geometry.scene import (
    OBJECT_CENTER, get_primitive_specs, primitive_surface_gap,
    configure_object_surface, GRASP_FLEXOR_TENSION, GRASP_SPHERE_CENTER)


def _add_object_mesh(pv_plotter, spec, center):
    """Render the shared contact object into the pyvista window (best-effort)."""
    import pyvista as pv
    t = spec["type"]
    if t == "sphere":
        mesh = pv.Sphere(radius=spec["radius"], center=center)
    elif t == "cylinder":
        mesh = pv.Cylinder(center=center, direction=(0, 0, 1),
                           radius=spec["radius"], height=spec["height"])
    elif t == "capsule":
        mesh = pv.Capsule(center=center, direction=(0, 0, 1),
                          radius=spec["radius"], cylinder_length=spec["height"])
    elif t == "cube":
        hx, hy, hz = spec["half_extents"]
        mesh = pv.Cube(center=center, x_length=2 * hx, y_length=2 * hy, z_length=2 * hz)
    elif t == "ellipsoid":
        a, b, c = (float(v) for v in spec["semi_axes"])
        mesh = pv.ParametricEllipsoid(a, b, c).translate(center, inplace=False)
    else:
        return
    pv_plotter.add_mesh(mesh, color="goldenrod", opacity=0.35)


def main():
    parser = argparse.ArgumentParser(
        description="Solve a five-finger tendon hand (four fingers + opposable "
                    "thumb, shared floating wrist) grasping a primitive object.")
    parser.add_argument("primitive", nargs="?", default="big_sphere",
                        choices=["big_sphere", "capsule", "sphere", "cylinder", "cube",
                                 "coin", "credit_card", "pen"])
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip the interactive 3D view (headless / tuning).")
    parser.add_argument("--al-mu", type=float, default=1.0, help="AL initial penalty mu")
    parser.add_argument("--al-rate", type=float, default=2.0, help="AL mu increase rate")
    parser.add_argument("--al-iters", type=int, default=40, help="AL max outer iterations")
    parser.add_argument("--sigma-wrist-pos", type=float, default=1e1,
                        help="Wrist position prior std (m); tight by default so the "
                             "wrist stays anchored at identity (object is placed to "
                             "match the fixed-wrist fingertip locus).")
    parser.add_argument("--sigma-wrist-rot", type=float, default=1e1,
                        help="Wrist rotation prior std (rad); tight by default.")
    args = parser.parse_args()

    spec = get_primitive_specs()[args.primitive]
    # The big grasp sphere, the capsule, and the analytic ellipsoids sit at the
    # flexed-fingertip locus; the other (single-finger-scale) primitives stay at
    # OBJECT_CENTER.
    object_center = (GRASP_SPHERE_CENTER
                     if args.primitive in ("big_sphere", "capsule")
                     or spec["type"] == "ellipsoid"
                     else OBJECT_CENTER)
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[0:3, 0:3] = object_rotation
    object_pose[0:3, 3] = object_center

    # --- Shared object surface (one asset, referenced by every finger) ---
    # An analytic hyper-ellipsoid (Section 1.6.3) or a baked SDF grid.
    objects_dir = OBJECTS_DIR

    # --- Five fingers (four + opposable thumb) on the shared wrist ---
    # Anatomical hand from gepetto_core / parameters.scad (same builder as
    # kinematics_test.py); tip_radii holds each digit's CAD-derived contact radius
    # (same digit order as configs). Load dims once so both agree and we don't
    # re-parse the SCAD.
    dims = load_hand_dimensions()
    configs = get_default_hand_configs(dims)
    tip_radii = default_hand_tip_radii(dims)
    finger_names = [name for name, _ in configs]

    for (_, cfg), tip_radius in zip(configs, tip_radii):
        # Each finger's tip lands tangent to the shared SDF surface. env carries
        # the (shared) object + this finger's tip node/radius; the C++ side keys
        # the contact to each finger's own tip pose, so one env per finger is fine.
        env_i = gepetto_solvers.EnvironmentConfig()
        configure_object_surface(env_i, spec, objects_dir, args.primitive)
        env_i.object_pose_mean = object_pose
        env_i.object_pose_cov = 1e-8 * np.eye(6)
        env_i.object_pose_per_step = False
        env_i.contact_node_radius = tip_radius
        env_i.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env_i

    # --- Hand solver config (shared wrist tightly anchored at identity) ---
    # The object is placed at the fixed-wrist fingertip locus, so the wrist is
    # pinned (tight prior) rather than floated; contact is reachable without
    # moving the hand.
    hand_config = gepetto_solvers.TendonHandSolverConfig()
    hand_config.wrist_pose = np.eye(4)
    hand_config.sigma_wrist_pos = args.sigma_wrist_pos
    hand_config.sigma_wrist_rot = args.sigma_wrist_rot
    hand_config.base.linear_solver_type = "MULTIFRONTAL_QR"
    hand_config.base.al_initial_mu = args.al_mu
    hand_config.base.al_mu_increase_rate = args.al_rate
    hand_config.base.al_max_iterations = args.al_iters

    solver = gepetto_solvers.TendonHandSolver(configs, hand_config)
    print(f"Built hand solver with {solver.num_fingers()} fingers.")

    # Same tension pattern as the single-finger test: passive tendons pinned,
    # flexor (index 5) loose so the optimizer drives contact. Its prior mean is
    # GRASP_FLEXOR_TENSION (2 N) — the flexion the big sphere was sized/placed
    # for — so prior and contact agree and the flexor tension Q stays
    # well-conditioned. Tight per-finger tip-wrench prior keeps the AL system
    # well-conditioned (see memory hand-wrench-prior-conditioning).
    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, GRASP_FLEXOR_TENSION])
    tensions_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    all_tensions = [gepetto_solvers.VectorXGaussian(tensions_mean, tensions_cov)
                    for _ in configs]
    all_tip_wrenches = [gepetto_solvers.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                        for _ in configs]

    t0 = time.time()
    solution = solver.solve(all_tensions, all_tip_wrenches)
    dt_ms = (time.time() - t0) * 1000.0

    print(f"Solved in {dt_ms:.1f} ms | iters={solution.meta.iterations} | "
          f"error={solution.meta.error:.4g}")
    for (name, _), tip_radius, fm in zip(configs, tip_radii,
                                         solution.marginals.fingers):
        tip_pose = np.array(fm.rod.states[-1].pose.mean)
        tip_pos = tip_pose[:3, 3]
        tip_local = object_rotation.T @ (tip_pos - object_center)
        gap = primitive_surface_gap(tip_local, spec) - tip_radius
        print(f"  [{name:>6}] tip {tip_pos}  |  surface gap {gap:+.5f} m  "
              f"(target ~0, r={tip_radius:.4f})")

    if args.no_viz:
        return

    from gepetto_solvers.core.plotting.tendon_hand_plotter import TendonHandPlotter

    class _FingerSol:
        pass

    solutions = {}
    for name, fm in zip(finger_names, solution.marginals.fingers):
        s = _FingerSol()
        s.marginals = fm
        s.meta = solution.meta
        solutions[name] = s

    # Four windows, each with a distinct camera angle, so the object's placement
    # relative to the hand is legible from several viewpoints at once
    # (azimuth/elevation in degrees about the object center).
    camera_views = [
        ("front",   165, 20),   # the default oblique-front view
        ("side",     90,  5),   # looking along the palm from the side
        ("top",     165, 85),   # nearly straight down onto the grasp
        ("iso",      45, 35),   # opposite-corner isometric
    ]

    plotters = []
    for name, azimuth, elevation in camera_views:
        plotter = TendonHandPlotter(
            finger_names,
            plot_backbone_ellipsoids=False,
            camera_azimuth=azimuth,
            camera_elevation=elevation,
            camera_focal_point=list(object_center),
            camera_distance=0.5,
        )
        plotter.plotter.plotter.add_text(name, position="upper_left", font_size=12)
        _add_object_mesh(plotter.plotter.plotter, spec, object_center)
        plotter.update(solutions)
        plotters.append(plotter)

    input("Press Enter to close...")


if __name__ == "__main__":
    main()
