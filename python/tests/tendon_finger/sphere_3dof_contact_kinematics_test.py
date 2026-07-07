import numpy as np
import time

import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter

from .config import get_6tendon_config


def main():
    config = get_6tendon_config()
    num_tendons = config.num_tendons  # 6

    # Tip node index (last rod node). The C++ side also accepts -1 as a tip
    # alias via CosseratRodModel::clamp_node_idx; we compute the explicit
    # index here for use as the plotter's contact_node_index too.
    num_nodes = config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    tip_node_index = num_nodes - 1

    # Sphere primitive at the p2p goal position used in point_to_point_planning.py,
    # mirrored across x=0 (X negated): the 180-deg CAD tendon-routing flip curls the
    # flexor toward world -X, so the target sphere moves to the -X side too.
    sphere_center = np.array([-6.02088876e-02, 3.77734425e-02, 0.0])
    sphere_radius = 0.025
    tip_radius = 0.003

    sc = crest_sparse.SpherePrimitiveContactConfig()
    sc.finger_node_index = -1
    sc.finger_node_radius = tip_radius
    sc.sphere_center = sphere_center
    sc.sphere_radius = sphere_radius
    sc.sphere_pose_cov = 1e-8 * np.eye(6)
    # Use the 3-residual SphereSphereWitnessContactFactor ([c_R, c_O, c_N]) with
    # an explicit dummy witness point, instead of the 1-residual analytic
    # SphereSphereContactFactor. This is the analytic counterpart of the
    # SDF-backed witness contact and lets us cross-check the witness formulation
    # against the closed-form gap on a pure sphere-sphere problem.
    sc.witness = True
    config.sphere_contact = sc

    # Contact is a hard equality constraint solved with GTSAM's Augmented
    # Lagrangian optimizer (auto-enabled whenever sphere_contact is set): it
    # drives the signed surface gap to ~0 exactly, instead of approximating it
    # with a tight covariance. Convergence is governed by the AL params on
    # config.base. A gradual penalty schedule (al_mu_increase_rate=2) run for
    # enough outer iterations converges to a ~micron gap; growing the penalty
    # too fast (rate >= 5) trips the relative-convergence check early and stalls
    # the finger short of the surface.
    config.base.al_initial_mu = 1.0
    config.base.al_mu_increase_rate = 2.0
    config.base.al_max_iterations = 40

    solver = crest_sparse.TendonFingerSolver(config)

    # Pin passive tendons tight, leave the flexor (index 5) loose so the
    # optimizer can adjust it freely to satisfy the contact factor. Mirrors
    # the goal-tension covariance used in point_to_contact_planning.py.
    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    tensions_cov = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
    tip_wrench = crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)

    t0 = time.time()
    solution = solver.solve(tensions, tip_wrench, None)
    dt_ms = (time.time() - t0) * 1000.0

    tip_pose = np.array(solution.marginals.rod.states[-1].pose.mean)
    tip_pos = tip_pose[:3, 3]
    gap = np.linalg.norm(tip_pos - sphere_center) - (tip_radius + sphere_radius)
    print(f"Solved in {dt_ms:.1f} ms | iters={solution.meta.iterations} | error={solution.meta.error:.4g}")
    print(f"  tip position: {tip_pos}")
    print(f"  signed surface gap (||tip - center|| - (r_a + r_b)): {gap:+.5f} m  (target ~0)")
    print(f"  active tendon (5) tension: {solution.marginals.tensions.mean[5]:.3f}")

    plotter = TendonFingerPlotter(
        plot_backbone_frames=True,
        contact_node_index=tip_node_index,
        contact_node_radius=tip_radius,
        sphere_primitives=[{"center": sphere_center, "radius": sphere_radius,
                            "color": "goldenrod", "opacity": 0.35}],
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.1, 0],
    )
    plotter.update(solution)
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
