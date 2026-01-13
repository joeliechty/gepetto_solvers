import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from .._plotting.parallel_robot_plotter import ParallelRobotPlotter


def get_base_poses():
    angles = np.array(np.deg2rad([10, 110, 130, 230, 250, 350]))

    radius = 0.3
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    poses = []
    for xi, yi in zip(xs, ys):
        pose = np.eye(4)
        pose[0, 3] = xi
        pose[1, 3] = yi
        poses.append(pose)

    return poses

platform_z_offset = -0.2

def get_tip_poses():
    angles = np.array(np.deg2rad([50, 70, 170, 190, 290, 310]))

    radius = 0.15
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    z = platform_z_offset
    poses = []
    for xi, yi in zip(xs, ys):
        pose = np.eye(4)
        pose[0, 3] = xi
        pose[1, 3] = yi
        pose[2, 3] = z
        poses.append(pose)

    return poses


def get_wrench_prior(t):
    wrench_mean = np.zeros(6)
    f = 5 * np.array([np.cos(0.1 * t), np.sin(0.1 * t), np.sin(0.15 * t)])
    wrench_mean[3:] = f

    wrench_cov = 1e-6 * np.eye(6)
    wrench_cov[3:,3:] = 1e-1 * np.eye(3)

    return wrench_mean, wrench_cov


def get_goal_pose(t):
    xy = 0.1 * np.array([np.cos(0.5 * t), np.sin(0.5 * t)])
    z = 0.7 + 0.1 * np.sin(0.3 * t)
    p = np.hstack([xy, z])

    r = 0.1 * np.array([np.cos(0.5 * t), np.sin(0.5 * t), 0])
    R = Rotation.from_rotvec(r).as_matrix()

    return p, R

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

    config = crest_sparse.ParallelRobotSolverConfig()

    config.nodes_per_rod = 20
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-2
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3
    config.base_end_poses = get_base_poses()
    config.tip_end_poses = get_tip_poses()
    config.sigma_end_pose_pos= 1.0e-4
    config.sigma_end_pose_rot= 1.0e-3

    solver = crest_sparse.ParallelRobotSolver(config)

    plotter = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        platform_z_offset=platform_z_offset,
        camera_azimuth=60, 
        camera_distance=4, 
        camera_focal_point=np.array([0, 0, 0.7])
    )
    
    frame_rate = 5.0
    dt = 1.0 / frame_rate
    t_final = 1200.0
    num_steps = int(t_final / dt)

    rod_lengths = 0.7 * np.ones(6)
    rod_lengths_sigma = 1e-3

    for step in range(num_steps + 1):
        t = step * dt

        wrench_mean, wrench_cov = get_wrench_prior(t)
        solution = solver.solve(rod_lengths, rod_lengths_sigma, wrench_mean, wrench_cov)

        J = solution.marginals.rod_lengths_jacobian

        p = solution.marginals.platform_pose_mean[:3, 3]
        R = solution.marginals.platform_pose_mean[:3,:3]

        p_goal, R_goal = get_goal_pose(t)

        p_error = R.T @ (p_goal - p)
        r_error = Rotation.from_matrix(R.T @ R_goal).as_rotvec()
        
        twist_error = np.hstack((r_error, p_error))

        max_step = 0.05
        d_twist = twist_error
        if np.linalg.norm(d_twist) > max_step:
            d_twist = d_twist / np.linalg.norm(d_twist) * max_step

        d_rod_lengths = np.linalg.pinv(J) @ d_twist
        rod_lengths += d_rod_lengths

        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")


if __name__ == "__main__":
    main()