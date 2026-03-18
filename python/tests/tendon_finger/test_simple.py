import numpy as np
import time

import crest_sparse
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_6tendon_config
from .benchmark import TendonRobotSolver

def main():
    config = get_6tendon_config()
    num_tendons = config.num_tendons  # 6

    solver = crest_sparse.TendonRobotSolver(config)

    dummy_solution = solver.solve(
        crest_sparse.VectorXGaussian(np.zeros(num_tendons), np.eye(num_tendons)),
        crest_sparse.Vector6Gaussian(np.zeros(6), np.eye(6)),
        None)

    plotter = TendonRobotPlotter(
        single_plot_mode=False, 
        # plot_backbone_frames=True,
        camera_azimuth=165,  # Rotated by 90 degrees
        camera_elevation=20,  # Keep default or adjust as needed
        camera_focal_point=[0, 0.1, 0]
    )

    tensions_cov = (1e-2) ** 2 * np.eye(num_tendons)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    # Background tension for passive tendons (Newtons)
    background_tension = 0.5

    for i in range(1000):
        tensions_mean = np.zeros(num_tendons)
        # Tendons 0-4 (passive): constant background tension
        tensions_mean[0] = background_tension
        tensions_mean[1] = background_tension
        tensions_mean[2] = background_tension
        tensions_mean[3] = background_tension
        tensions_mean[4] = background_tension
        # Tendon 5 (active flexor at 180 deg): varies
        tensions_mean[5] = background_tension + 1.0 * (np.cos(0.01 * i - np.pi) + 1)

        tip_wrench_mean = np.zeros(6)

        tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
        tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)



if __name__ == "__main__":
    main()
