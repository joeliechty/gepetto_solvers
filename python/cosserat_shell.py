import time
import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from plotting import CosseratShellPlotter


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

    config = crest_sparse.CosseratShellSolverConfig()
    config.num_nodes_x = 5
    config.num_nodes_y = 15
    config.element_size = 0.1
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-3
    config.sigma_twist_rot = 1.0e-3
    config.sigma_stress_force = 1.0e-3
    config.sigma_stress_moment = 1.0e-3

    solver = crest_sparse.CosseratShellSolver(config)
    
    plotter = CosseratShellPlotter(
        single_plot_mode=True)

    solution = solver.solve()
    plotter.update(solution)
    # frame_rate = 5.0
    # dt = 1.0 / frame_rate
    # t_final = 1200.0
    # num_steps = int(t_final / dt)

    # for step in range(num_steps + 1):
    #     t = step * dt

    #     solution = solver.solve(*prior_getter(t), None)
    #     plotter.update(solution)

    #     progress = 100.0 * step / num_steps
    #     print(f"Progress: {progress:5.1f}%", end="\r")

if __name__ == "__main__":
    main()