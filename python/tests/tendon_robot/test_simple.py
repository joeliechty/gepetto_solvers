import numpy as np
import time

import crest_sparse
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config
from .benchmark import TendonRobotSolver

def main():
    config = get_base_config()
    
    solver = crest_sparse.TendonRobotSolver(config)
    
    dummy_solution = solver.solve(crest_sparse.Vector4Gaussian(np.zeros(4), np.eye(4)), crest_sparse.Vector6Gaussian(np.zeros(6), np.eye(6)), None)
    solver_baseline = TendonRobotSolver(config, dummy_solution.marginals.tendon_config.hole_locations)

    plotter = TendonRobotPlotter(single_plot_mode=False)

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    for i in range(100):
        tensions_mean = np.zeros(4)
        tensions_mean[0] = 0.1 * i
        # tensions_mean[0] = 4.0
        
        tip_wrench_mean = np.zeros(6)
        tip_wrench_mean[5] = 0.1 * np.sin(0.1 * i)
        # tip_wrench_mean[3] = 0.2

        tensions = crest_sparse.Vector4Gaussian(tensions_mean, tensions_cov)
        tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)

        solution_baseline = solver_baseline.solve(tensions_mean, tip_wrench_mean[3:])
        p_baseline = solution_baseline[-1]['p']
        p_gt = solution.marginals.rod.states[-1].pose.mean[:3,3]

        print("p_gt: ", p_gt)
        print("p_baseline: ", p_baseline)
        print(f"baseline error: {np.linalg.norm(p_baseline - p_gt)}")



if __name__ == "__main__":
    main()
