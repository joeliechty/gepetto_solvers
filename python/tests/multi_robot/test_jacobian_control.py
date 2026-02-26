import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse

from .._plotting.multi_robot_plotter import MultiRobotPlotter
from ..tendon_robot.utils import TipForceFunction
from .config import get_base_config


def get_goal_position(t):
    A = 0.075
    B = 0.1
    C = 0.1
    a, b, c = 0.1, 0.11, 0.12  # frequency in each axis
    delta = np.pi / 2          # phase offset

    x = 0.8 + A * np.sin(2 * np.pi * a * t + delta)
    y = B * np.sin(2 * np.pi * b * t)
    z = -0.5 + C * np.sin(2 * np.pi * c * t)

    return np.array([x, y, z])


def rotation_gradient_body(R, R0):
    # This gradient is associated with frobenius norm cost
    # Equivalent to maximizing Tr(R0.T @ R)
    S = R.T @ R0 - R0.T @ R
    return -0.5 * np.array([
        S[2,1],
        S[0,2],
        S[1,0]
    ])


def main():

    config = get_base_config()
    solver = crest_sparse.MultiRobotSolver(config)
    plotter = MultiRobotPlotter(plot_backbone_frames=True, camera_distance=1.7, camera_focal_point=[0.4, 0 , -0.3], camera_azimuth=-60, plot_rviz_coords=True)

    pose_cov = 1e-3 ** 2 * np.eye(6)
    wrench = crest_sparse.Vector6Gaussian(np.zeros(6), 1e-3 ** 2 * np.eye(6))

    p_main = np.zeros(3)
    R_initial = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    R_main = R_initial.copy()
    insertion_main = 1.0
    p_helper = np.array([0.5, 0, 0])
    R_helper = R_initial.copy()
    insertion_helper = 0.5

    for t in np.linspace(0, 100, 1000):
        # Assemble current poses and solve
        pose_main = np.eye(4)
        pose_main[:3,3] = p_main
        pose_main[:3,:3] = R_main

        pose_helper = np.eye(4)
        pose_helper[:3,3] = p_helper
        pose_helper[:3,:3] = R_helper

        solution = solver.solve(
            crest_sparse.Pose3Gaussian(pose_main, pose_cov),
            insertion_main,
            crest_sparse.Pose3Gaussian(pose_helper, pose_cov), 
            insertion_helper,
            wrench
        )
        
        # Extract stuff from the solution for velocity control
        pose = solution.marginals.end_effector_rod.states[-1].pose.mean
        p = pose[:3,3]
        R = pose[:3,:3]

        p_goal = get_goal_position(t)
        p_error = R.T @ (p_goal - p)

        # Get position jacobian
        J_position_full = solution.marginals.J_rod_bases[3:]

        # We can't actually acutuate all of the DOF, just rotations/insertions
        J = J_position_full[:,[0, 1, 2, 5, 6, 7, 8, 11]]  # main: rx, ry, rz, pz  helper: rx, ry, rz, pz
        
        # Get position input update 
        J_pinv = np.linalg.pinv(J)
        dq_position = J_pinv @ p_error

        # Gradient: increasing rotation increases cost, keeps rotations near nominal
        g = np.zeros(8)
        g[0:3] = rotation_gradient_body(R_main, R_initial)
        g[4:7] = rotation_gradient_body(R_helper, R_initial)

        # Get nullspace motion to keep rotations in check
        N = np.eye(J.shape[1]) - J_pinv @ J
        dq_null = -0.01 * (N @ g)

        # Apply update to actuation variables
        dq = dq_position + dq_null
        R_main = R_main @ Rotation.from_rotvec(dq[:3]).as_matrix()
        insertion_main += dq[3]
        R_helper = R_helper @ Rotation.from_rotvec(dq[4:7]).as_matrix()
        insertion_helper += dq[7]

        plotter.update(solution, p_desired=p_goal)


if __name__ == "__main__":
    main()
