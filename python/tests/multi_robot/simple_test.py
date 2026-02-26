import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse

from .._plotting.multi_robot_plotter import MultiRobotPlotter
from ..tendon_robot.utils import TipForceFunction
from .config import get_base_config


def main():

    config = get_base_config()
    solver = crest_sparse.MultiRobotSolver(config)
    plotter = MultiRobotPlotter(camera_distance=1.7, camera_focal_point=[0.4, 0 , -0.3], camera_azimuth=-60)

    small_wrench_cov = 1e-3 ** 2 * np.eye(6)
    base_pose_cov = 1e-3 ** 2 * np.eye(6)

    tip_force_function = TipForceFunction(max_magnitude=1.0, framerate=10, seed=3)

    for t in np.linspace(0, 100, 1000):
        main_base_pose_mean = np.eye(4)
        main_base_pose_mean[:3,:3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ Rotation.from_rotvec([
            0.0 * np.sin(1.1 * t), 0.0 * np.sin(1.1 * t), 1.0 * np.sin(1.1 * t)]).as_matrix()
        helper_base_pose_mean = np.eye(4)
        helper_base_pose_mean[:3,:3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() #@ Rotation.from_rotvec([
            # 0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t)]).as_matrix()
        
        helper_base_pose_mean[:3,3] = [0.5, 0, 0]

        main_insertion = 1.0 #+ 0.05 * np.sin(1.1 * t)
        helper_insertion = 0.5 #+ 0.05 * np.sin(1.2 * t)

        wrench = np.zeros(6)
        # wrench[3:] = tip_force_function(t)

        solution = solver.solve(
            crest_sparse.Pose3Gaussian(main_base_pose_mean, base_pose_cov),
            main_insertion,
            crest_sparse.Pose3Gaussian(helper_base_pose_mean, base_pose_cov), 
            helper_insertion,
            crest_sparse.Vector6Gaussian(wrench, small_wrench_cov)
        )

        plotter.update(solution)


if __name__ == "__main__":
    main()
