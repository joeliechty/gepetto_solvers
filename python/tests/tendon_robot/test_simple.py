import numpy as np
import time

import crest_sparse
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config
from .benchmark import solve_kinematics_bvp

def main():
    config = get_base_config()
    
    solver = crest_sparse.TendonRobotSolver(config)
    plotter = TendonRobotPlotter(single_plot_mode=False)

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    x_guess = None

    for i in range(100):
        tensions_mean = np.zeros(4)
        tensions_mean[0] = 0.1 * i
        
        tip_wrench_mean = np.zeros(6)
        tip_wrench_mean[5] = 0.1 * np.sin(0.1 * i)

        tensions = crest_sparse.Vector4Gaussian(tensions_mean, tensions_cov)
        tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)

        holes = solution.marginals.tendon_config.hole_locations
        shape_baseline, x_guess = solve_kinematics_bvp(tensions_mean, tip_wrench_mean[3:], get_base_config(), holes, x_guess)
        p_gt = solution.marginals.rod.states[-1].pose.mean[:3,3]
        p_baseline = shape_baseline[-1]
        print("p_gt: ", p_gt)
        print("p_baseline: ", p_baseline)
        print(f"baseline error: {np.linalg.norm(p_baseline - p_gt)}")



if __name__ == "__main__":
    main()
