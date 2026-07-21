"""Two opposed fingers sharing one floating wrist base, each landing its tip on a
shared object.

This is the multi-finger counterpart of ``ik_1f_contact.py``:
both fingers are part of ONE factor graph, share ONE wrist variable ``T_base``
(with different ``hand_base_offset`` transforms), and each fingertip is driven onto
the object surface by its own hard SDF contact constraint (an opposition grasp).

Run (from the ``python/`` directory):
    python -m tests.tendon_hand.ik_2f_contact sphere
"""

import os
import argparse
import time

import numpy as np

import crest_sparse

from .config import get_two_finger_opposition_configs, tip_node_index
# Shared primitive geometry + analytic surface-gap helper (see scene.py).
from .scene import OBJECT_CENTER, get_primitive_specs, primitive_surface_gap


def _add_object_mesh(pv_plotter, spec, center):
    """Render the contact object into the shared pyvista window (best-effort)."""
    import pyvista as pv
    t = spec["type"]
    if t == "sphere":
        mesh = pv.Sphere(radius=spec["radius"], center=center)
    elif t == "cylinder":
        mesh = pv.Cylinder(center=center, direction=(0, 0, 1),
                           radius=spec["radius"], height=spec["height"])
    elif t == "cube":
        hx, hy, hz = spec["half_extents"]
        mesh = pv.Cube(center=center, x_length=2 * hx, y_length=2 * hy, z_length=2 * hz)
    else:
        return
    pv_plotter.add_mesh(mesh, color="goldenrod", opacity=0.35)


def main():
    parser = argparse.ArgumentParser(
        description="Solve two opposed tendon fingers (shared floating wrist) into "
                    "contact with a primitive object.")
    parser.add_argument("primitive", nargs="?", default="sphere",
                        choices=["sphere", "cylinder", "cube"])
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip the interactive 3D view (headless / tuning).")
    parser.add_argument("--al-mu", type=float, default=1.0, help="AL initial penalty mu")
    parser.add_argument("--al-rate", type=float, default=2.0, help="AL mu increase rate")
    parser.add_argument("--al-iters", type=int, default=40, help="AL max outer iterations")
    args = parser.parse_args()

    spec = get_primitive_specs()[args.primitive]
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[0:3, 0:3] = object_rotation
    object_pose[0:3, 3] = OBJECT_CENTER

    tip_radius = 0.003

    # --- Shared SDF object (one env, referenced by both fingers) ---
    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{args.primitive} (run from the python/ dir).")

    # --- Two opposed fingers on the shared wrist ---
    configs = get_two_finger_opposition_configs(OBJECT_CENTER)
    finger_names = [name for name, _ in configs]

    for _, cfg in configs:
        # Each finger's tip lands tangent to the shared SDF surface. env carries
        # the (shared) object + this finger's tip node/radius; the C++ side keys
        # the contact to each finger's own tip pose, so one env is fine.
        env_i = crest_sparse.EnvironmentConfig()
        env_i.load_sdf(vdb_path)
        env_i.object_pose_mean = object_pose
        env_i.object_pose_cov = 1e-8 * np.eye(6)
        env_i.object_pose_per_step = False
        env_i.contact_node_radius = tip_radius
        env_i.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env_i

    # --- Hand solver config (shared wrist rigidly anchored at identity) ---
    hand_config = crest_sparse.TendonHandSolverConfig()
    hand_config.wrist_pose = np.eye(4)
    hand_config.base.linear_solver_type = "MULTIFRONTAL_QR"
    hand_config.base.al_initial_mu = args.al_mu
    hand_config.base.al_mu_increase_rate = args.al_rate
    hand_config.base.al_max_iterations = args.al_iters

    solver = crest_sparse.TendonHandSolver(configs, hand_config)
    print(f"Built hand solver with {solver.num_fingers()} fingers.")

    # Same tension pattern as the single-finger test: passive tendons pinned,
    # flexor (index 5) loose so the optimizer drives contact.
    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    tensions_cov = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    all_tensions = [crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
                    for _ in configs]
    all_tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                        for _ in configs]

    t0 = time.time()
    solution = solver.solve(all_tensions, all_tip_wrenches)
    dt_ms = (time.time() - t0) * 1000.0

    print(f"Solved in {dt_ms:.1f} ms | iters={solution.meta.iterations} | "
          f"error={solution.meta.error:.4g}")
    for name, fm in zip(finger_names, solution.marginals.fingers):
        tip_pose = np.array(fm.rod.states[-1].pose.mean)
        tip_pos = tip_pose[:3, 3]
        tip_local = object_rotation.T @ (tip_pos - OBJECT_CENTER)
        gap = primitive_surface_gap(tip_local, spec) - tip_radius
        print(f"  [{name}] tip {tip_pos}  |  surface gap {gap:+.5f} m  (target ~0)")

    if args.no_viz:
        return

    from .._plotting.tendon_hand_plotter import TendonHandPlotter

    plotter = TendonHandPlotter(
        finger_names,
        plot_backbone_ellipsoids=False,
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0.03, 0.05, 0],
        camera_distance=0.5,
    )
    _add_object_mesh(plotter.plotter.plotter, spec, OBJECT_CENTER)

    class _FingerSol:
        pass

    solutions = {}
    for name, fm in zip(finger_names, solution.marginals.fingers):
        s = _FingerSol()
        s.marginals = fm
        s.meta = solution.meta
        solutions[name] = s

    plotter.update(solutions)
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
