"""Five-finger tendon *hand* grasp WITH Section 1.5 collision avoidance enabled.

This is ``five_finger_hand_grasp_test.py`` plus ``config.attach_collision``: every
finger gets sphere-to-SDF inequality constraints keeping its disc spheres out of
the object, and sphere-to-sphere inequality constraints keeping distinct fingers
apart. Both are hard inequality constraints (``c_pen <= 0``) handled by the
Augmented Lagrangian optimizer.

Pairs where BOTH spheres are proximal (the metacarpal bones, rigidly attached to
the shared wrist) are never checked -- they cannot move relative to one another.
Base-disc (node 0) spheres are excluded from finger-finger checks for the same
reason. Self-collision within a finger is not modeled (a finger cannot bend far
enough to reach itself).

The script reports, from the converged solution:
  * each fingertip's contact surface gap (should be ~0 -- contact still holds)
  * the worst finger-object penetration over all collision spheres (should be >= 0)
  * the worst cross-finger sphere gap over all checked pairs (should be >= 0)

Run (from the ``python/`` directory):
    python -m tests.tendon_hand.five_finger_hand_collision_test big_sphere --no-viz
"""

import os
import argparse
import time
import itertools

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index, attach_collision, disc_node_indices, proximal_disc_flags)
from .sdf_3dof_contact_kinematics_test import (
    OBJECT_CENTER, get_primitive_specs, primitive_surface_gap)
from .five_finger_hand_grasp_test import (
    GRASP_FLEXOR_TENSION, GRASP_SPHERE_CENTER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primitive", nargs="?", default="big_sphere",
                        help="Object primitive (default: big_sphere).")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--collision-radius", type=float, default=0.003,
                        help="Collision sphere radius (m) on each disc node.")
    # 1.0 = collision constraint rows whitened the same as the contact rows.
    # 1e-4 makes collision 1e4x stronger in the AL merit; the collision
    # inequalities then dominate the terminal contact equalities and the solve
    # "converges" with the fingertips held 1-2 cm off the surface.
    parser.add_argument("--collision-sigma", type=float, default=1.0)
    parser.add_argument("--al-mu", type=float, default=1e4)
    parser.add_argument("--al-rate", type=float, default=2.0)
    # Inequality violation halves per outer iteration; 15 leaves ~0.1 mm of
    # residual penetration, 25 converges both contact and collision cleanly.
    parser.add_argument("--al-iters", type=int, default=25)
    parser.add_argument("--sigma-wrist-pos", type=float, default=1e-4)
    parser.add_argument("--sigma-wrist-rot", type=float, default=1e1)
    args = parser.parse_args()

    spec = get_primitive_specs()[args.primitive]
    object_center = (GRASP_SPHERE_CENTER
                     if args.primitive in ("big_sphere", "capsule")
                     else OBJECT_CENTER)
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[0:3, 0:3] = object_rotation
    object_pose[0:3, 3] = object_center

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{args.primitive} (run from python/).")

    dims = load_hand_dimensions()
    configs = get_default_hand_configs(dims)
    tip_radii = default_hand_tip_radii(dims)
    finger_names = [name for name, _ in configs]

    # Terminal tip contact (same as five_finger_hand_grasp_test).
    for (_, cfg), tip_radius in zip(configs, tip_radii):
        env_i = crest_sparse.EnvironmentConfig()
        env_i.load_sdf(vdb_path)
        env_i.object_pose_mean = object_pose
        env_i.object_pose_cov = 1e-8 * np.eye(6)
        env_i.object_pose_per_step = False
        env_i.contact_node_radius = tip_radius
        env_i.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env_i

    # Section 1.5 collision avoidance, layered onto the same per-finger envs.
    attach_collision(configs, vdb_path, object_pose,
                     radius=args.collision_radius, sigma=args.collision_sigma)

    hand_config = crest_sparse.TendonHandSolverConfig()
    hand_config.wrist_pose = np.eye(4)
    hand_config.sigma_wrist_pos = args.sigma_wrist_pos
    hand_config.sigma_wrist_rot = args.sigma_wrist_rot
    hand_config.base.linear_solver_type = "MULTIFRONTAL_QR"
    hand_config.base.al_initial_mu = args.al_mu
    hand_config.base.al_mu_increase_rate = args.al_rate
    hand_config.base.al_max_iterations = args.al_iters

    solver = crest_sparse.TendonHandSolver(configs, hand_config)
    print(f"Built hand solver with {solver.num_fingers()} fingers "
          f"(collision avoidance ON).")

    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, GRASP_FLEXOR_TENSION])
    tensions_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
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

    # --- Contact still holds: each tip tangent to the surface ---
    print("\nTip contact gaps (target ~0):")
    for (name, _), tip_radius, fm in zip(configs, tip_radii,
                                         solution.marginals.fingers):
        tip_pos = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
        tip_local = object_rotation.T @ (tip_pos - object_center)
        gap = primitive_surface_gap(tip_local, spec) - tip_radius
        print(f"  [{name:>6}] surface gap {gap:+.5f} m (r={tip_radius:.4f})")

    # --- Collision sphere world positions per finger ---
    r_col = args.collision_radius
    spheres = []          # per finger: list of (node_idx, pos, proximal)
    for (_, cfg), fm in zip(configs, solution.marginals.fingers):
        nodes = disc_node_indices(cfg)
        prox = proximal_disc_flags(cfg)
        tip_idx = tip_node_index(cfg)
        entries = []
        for n, p in zip(nodes, prox):
            pos = np.array(fm.rod.states[n].pose.mean)[:3, 3]
            entries.append((n, pos, bool(p), n == tip_idx))
        spheres.append(entries)

    # --- Finger-object: c_pen = r - sdf <= 0  =>  sdf - r >= 0 ---
    # The terminal contact node is excluded (its contact factor owns it).
    print("\nFinger-object collision (sdf - r, want >= 0; contact tip excluded):")
    worst_obj = (None, np.inf)
    for name, entries in zip(finger_names, spheres):
        f_worst = np.inf
        for n, pos, _p, is_tip in entries:
            if is_tip:
                continue
            local = object_rotation.T @ (pos - object_center)
            clearance = primitive_surface_gap(local, spec) - r_col
            f_worst = min(f_worst, clearance)
        print(f"  [{name:>6}] worst clearance {f_worst:+.5f} m")
        if f_worst < worst_obj[1]:
            worst_obj = (name, f_worst)

    # --- Finger-finger: gap = ||pa-pb|| - (ra+rb) >= 0 ---
    # Skip pairs where BOTH spheres are proximal, and skip node-0 (root) spheres.
    print("\nFinger-finger collision (gap, want >= 0; "
          "proximal-proximal and node-0 pairs excluded):")
    worst_ff = (None, np.inf)
    n_pairs = 0
    for (ia, ib) in itertools.combinations(range(len(configs)), 2):
        pair_worst = np.inf
        for (na, pa, proxa, _ta) in spheres[ia]:
            if na == 0:
                continue
            for (nb, pb, proxb, _tb) in spheres[ib]:
                if nb == 0:
                    continue
                if proxa and proxb:
                    continue
                n_pairs += 1
                gap = np.linalg.norm(pa - pb) - 2.0 * r_col
                pair_worst = min(pair_worst, gap)
        label = f"{finger_names[ia]}/{finger_names[ib]}"
        print(f"  [{label:>15}] worst gap {pair_worst:+.5f} m")
        if pair_worst < worst_ff[1]:
            worst_ff = (label, pair_worst)

    print(f"\nChecked {n_pairs} cross-finger sphere pairs.")
    print(f"WORST finger-object clearance: {worst_obj[1]:+.5f} m  ({worst_obj[0]})")
    print(f"WORST finger-finger gap:       {worst_ff[1]:+.5f} m  ({worst_ff[0]})")
    ok = worst_obj[1] >= -1e-4 and worst_ff[1] >= -1e-4
    print("RESULT:", "PASS (no penetration)" if ok else "FAIL (penetration present)")

    if args.no_viz:
        return

    from .._plotting.tendon_hand_plotter import TendonHandMultiViewPlotter

    class _FingerSol:
        pass

    solutions = {}
    for name, fm in zip(finger_names, solution.marginals.fingers):
        s = _FingerSol()
        s.marginals = fm
        s.meta = solution.meta
        solutions[name] = s

    # Four windows around the hand (three azimuths 90 deg apart + near-top-down)
    # with the grasp object rendered, so contact and clearance are visible from
    # any side.
    plotter = TendonHandMultiViewPlotter(
        finger_names,
        plot_backbone_ellipsoids=False,
        camera_focal_point=list(object_center),
        camera_distance=0.5,
        primitives=[dict(spec["plot"](object_center),
                         color="goldenrod", opacity=0.35)],
    )
    plotter.update(solutions)
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
