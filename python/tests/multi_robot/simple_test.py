import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse

from .._plotting.multi_robot_plotter import MultiRobotPlotter
from ..tendon_robot.utils import TipForceFunction

def get_base_config():
    r = 0.0015 / 2
    I = 0.25 * np.pi * r ** 4
    A = np.pi * r ** 2
    J = 2 * I
    E = 207.0e9
    G = 79.3e9
    
    K_inv = np.diag([
        1 / (E * I), 
        1 / (E * I),
        1 / (J * G),
        1 / (G * A),
        1 / (G * A),
        1 / (E * A)
    ])

    config = crest_sparse.MultiRobotSolverConfig()

    config.base.use_dense = False
    config.nodes_per_rod = 15
    config.K_inv = K_inv
    config.snare_distance_to_tip = 0.2
    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-3
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3

    return config



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
            0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t)]).as_matrix()
        helper_base_pose_mean = np.eye(4)
        helper_base_pose_mean[:3,:3] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix() @ Rotation.from_rotvec([
            0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t), 0.1 * np.sin(1.1 * t)]).as_matrix()
        
        helper_base_pose_mean[:3,3] = [0.5, 0, 0]

        main_insertion = 1.0 + 0.05 * np.sin(1.1 * t)
        helper_insertion = 0.5 + 0.05 * np.sin(1.2 * t)

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
