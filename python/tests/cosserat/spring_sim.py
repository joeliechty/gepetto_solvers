import time
import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from .._plotting.cosserat_rod_plotter import CosseratRodPlotter


def get_tip_wrench_prior(t):
    dir = np.array([np.cos(0.10 * t), np.sin(0.10 * t), np.sin(0.15 * t)])
    dir = dir / np.linalg.norm(dir)

    mag = np.array([0.5, 0.2, 0.2]) * np.sin(t) + [0.3, 0.0, 0.0]

    force = mag * dir

    tip_wrench_mean = np.hstack((np.zeros(3), force))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.01
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_wrench_cov[3:,3:] = sigma ** 2 * np.eye(3)
    
    return crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov), None


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

    config = crest_sparse.CosseratRodSolverConfig()
    config.rod_length = 2
    config.num_nodes = 50
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-3
    config.sigma_twist_rot = 1.0e-3
    config.sigma_small_force = 1.0e-4
    config.sigma_small_moment = 1.0e-4
    config.sigma_base_pose_pos = 1.0e-4
    config.sigma_base_pose_rot = 1.0e-3

    solver = crest_sparse.CosseratRodSolver(config)
    plotter = CosseratRodPlotter(
        plot_base_plate=True,
        base_plate_size=0.05,
        plot_wrenches=True, 
        force_scale=0.2,
        camera_azimuth=-60, 
        camera_distance=1.0, 
        camera_focal_point=np.array([0.3, 0, 0]))

    frame_rate = 5.0
    dt = 1.0 / frame_rate
    t_final = 1200.0
    num_steps = int(t_final / dt)

    nominal_strain = np.zeros(6)
    nominal_strain[5] = 1.0
    nominal_strain[3] = 0.2
    nominal_strain[0] = 25.0

    for step in range(num_steps + 1):
        t = step * dt

        solution = solver.solve(*get_tip_wrench_prior(t), nominal_strain)
        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")

if __name__ == "__main__":
    main()