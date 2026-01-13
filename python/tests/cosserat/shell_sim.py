import time
import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from .._plotting.cosserat_shell_plotter import CosseratShellPlotter


def main():
    k_bending = 1
    k_torsion = 1
    k_shear = 100
    k_extension = 100

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    config = crest_sparse.CosseratShellSolverConfig()
    config.num_nodes_x = 8
    config.num_nodes_y = 30
    config.element_size = 0.1
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-3
    config.sigma_stress_force = 1.0e-3
    config.sigma_stress_moment = 1.0e-3

    solver = crest_sparse.CosseratShellSolver(config)
    
    plotter = CosseratShellPlotter(
        single_plot_mode=True,
        # save_frames_dir_name="cosserat_shell",
        camera_distance=9,
        camera_focal_point=(0, 2.1, 0)
    )

    frame_rate = 30.0
    dt = 1.0 / frame_rate
    t_final = 30.0
    num_steps = int(t_final / dt)

    for step in range(num_steps + 1):
        t = step * dt

        # displacement_mean = np.eye(4)
        # displacement_mean[:3,:3] = Rotation.from_rotvec([
        #     1 / 2 * np.pi * np.sin(0.3 * t), 
        #     2 * np.pi * np.sin(0.35 * t),
        #     1 / 2 * np.pi * np.sin(0.4 * t),
        # ]).as_matrix()

        # displacement_mean[0,3] = 1.5 * np.sin(0.23 * t)
        # displacement_mean[1,3] = 1.2 * np.sin(0.24 * t)
        # displacement_mean[2,3] = 1.5 * np.sin(0.25 * t)

        displacement_mean = np.eye(4)
        displacement_mean[:3,:3] = Rotation.from_rotvec([0, np.pi/2, 0]).as_matrix()

        displacement_cov = (1e-2) ** 2 * np.eye(6)
        displacement_cov[1,1] = (0.15) ** 2
        


        solution = solver.solve(displacement_mean, displacement_cov)
        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")
    

if __name__ == "__main__":
    main()