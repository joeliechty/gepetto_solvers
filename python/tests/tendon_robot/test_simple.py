import numpy as np
import time

import crest_sparse
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config


def main():
    config = get_base_config()
    
    solver = crest_sparse.TendonRobotSolver(config)
    plotter = TendonRobotPlotter(single_plot_mode=False)

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    for i in range(100):
        tensions_mean = np.zeros(4)
        tensions_mean[0] = 0.1 * i
        
        tip_wrench_mean = np.zeros(6)
        tip_wrench_mean[5] = 0.1 * np.sin(0.1 * i)

        tensions = crest_sparse.Vector4Gaussian(tensions_mean, tensions_cov)
        tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)


if __name__ == "__main__":
    main()
