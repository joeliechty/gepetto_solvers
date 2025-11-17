import time
import numpy as np

import crest_sparse
from plotting import CosseratRodPlotter


def main():
    k_bending = 0.1
    k_torsion = 0.1
    k_shear = 1e2
    k_extension = 1e2

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    config = crest_sparse.CosseratRodDynamicsConfig()
    config.rod_config.rod_length = 0.5
    config.rod_config.num_nodes = 15
    config.rod_config.K_inv = K_inv
    config.rod_config.sigma_twist_pos = 1.0e-4
    config.rod_config.sigma_twist_rot = 1.0e-4
    config.rod_config.sigma_small_force = 1.0e-4
    config.rod_config.sigma_small_moment = 1.0e-4
    config.rod_config.sigma_base_pose_pos = 1.0e-5
    config.rod_config.sigma_base_pose_rot = 1.0e-5

    config.num_time_steps = 50
    config.dt = 0.1
    config.linear_damping = 1
    config.rotational_damping = 1e-2
    config.linear_inertia = 1
    config.rotational_inertia = 1e-2
    config.initial_tip_wrench = np.array([0, 0.3, 0, -0.7, 0, 0])

    solver = crest_sparse.CosseratRodDynamicsSolver(config)
    solution = solver.solve()

    plotter = CosseratRodPlotter(
        plot_wrenches=False,
        plot_backbone_frames=True,
        plot_internal_wrenches=True,
        camera_azimuth=60, 
        camera_distance=1.5, 
        camera_focal_point=np.array([0, 0, 0.25]))

    print(f"iter:  {solution.meta.iterations}\n"
          f"err:   {solution.meta.error:.3e}\n"
          f"build: {solution.meta.build_time_ms:.2f}\n"
          f"opt:   {solution.meta.optimize_time_ms:.2f}\n"
          f"total: {solution.meta.total_time_ms:.2f}\n")

    for ii in range(len(solution.marginals)):
        sol = solution.marginals[ii]

        plotter.update(sol)
        time.sleep(0.1)

if __name__ == "__main__":
    main()