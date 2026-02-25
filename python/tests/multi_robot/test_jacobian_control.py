import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse

from .._plotting.multi_robot_plotter import MultiRobotPlotter
from ..tendon_robot.utils import TipForceFunction
from .config import get_base_config


def get_goal_position(t):
    yc = 0.0
    zc = 0.5
    r = 0.2
    y = r * np.cos(t)
    z = r * np.sin(t)
    x = 0.5

    return np.array([x, y, z])

    
def main():

    config = get_base_config()
    solver = crest_sparse.MultiRobotSolver(config)
    plotter = MultiRobotPlotter(camera_distance=1.7, camera_focal_point=[0.4, 0 , -0.3], camera_azimuth=-60)

    base_pose_cov = 1e-3 ** 2 * np.eye(6)
    wrench = crest_sparse.Vector6Gaussian(np.zeros(6), 1e-3 ** 2 * np.eye(6))

    for t in np.linspace(0, 100, 1000):
        main_base_pose_mean = np.eye(4)
        main_base_pose_mean[:3,:3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ Rotation.from_rotvec([
            0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t)]).as_matrix()
        helper_base_pose_mean = np.eye(4)
        helper_base_pose_mean[:3,:3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ Rotation.from_rotvec([
            0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t)]).as_matrix()
        
        helper_base_pose_mean[:3,3] = [0.5, 0, 0]

        main_insertion = 1.0 + 0.05 * np.sin(1.1 * t)
        helper_insertion = 0.5 + 0.05 * np.sin(1.2 * t)

        solution = solver.solve(
            crest_sparse.Pose3Gaussian(main_base_pose_mean, base_pose_cov),
            main_insertion,
            crest_sparse.Pose3Gaussian(helper_base_pose_mean, base_pose_cov), 
            helper_insertion,
            wrench
        )
        
        # Position Jacobian using all base pose params
        Jp = solution.marginals.J_rod_bases[3:]

        print(Jp)
        plotter.update(solution)


if __name__ == "__main__":
    main()
