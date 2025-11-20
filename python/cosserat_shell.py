import time
import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from plotting import CosseratShellPlotter


def main():
    k_bending = 1
    k_torsion = 1
    k_shear = 10
    k_extension = 10

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    config = crest_sparse.CosseratShellSolverConfig()
    config.num_nodes_x = 7
    config.num_nodes_y = 28
    config.element_size = 0.1
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-3
    config.sigma_stress_force = 1.0e-3
    config.sigma_stress_moment = 1.0e-3

    solver = crest_sparse.CosseratShellSolver(config)
    
    plotter = CosseratShellPlotter(
        # single_plot_mode=True,
        camera_distance=5,
        camera_focal_point=(0, 1.5, 0)
    )

    frame_rate = 5.0
    dt = 1.0 / frame_rate
    t_final = 1200.0
    num_steps = int(t_final / dt)

    for step in range(num_steps + 1):
        t = step * dt

        top_displacement = np.eye(4)
        top_displacement[:3,:3] = Rotation.from_rotvec([
            np.sin(0.21 * t), 
            3 * np.sin(0.2 * t), 
            np.sin(0.22 * t)
        ]).as_matrix()

        top_displacement[0,3] = 1.0 * np.sin(0.23 * t)
        top_displacement[0,3] = 2.0 * np.sin(0.24 * t)
        top_displacement[0,3] = 1.0 * np.sin(0.25 * t)

        # top_displacement = np.eye(4)
        # top_displacement[:3,:3] = Rotation.from_rotvec([
        #     0, 
        #     np.pi / 2, 
        #     0
        # ]).as_matrix()


        solution = solver.solve(top_displacement)
        plotter.update(solution)
    

if __name__ == "__main__":
    main()