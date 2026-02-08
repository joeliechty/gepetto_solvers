import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from .._plotting.parallel_robot_plotter import ParallelRobotPlotter
from .baseline_model import ParallelRobotSolver

ang = 10
def get_base_poses():
    angles = np.array(np.deg2rad([ang, 120 - ang, 120 + ang, 240 - ang, 240 + ang, -ang]))

    radius = 0.1
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
    angles = np.array(np.deg2rad([60 - ang, 60 + ang, 180 - ang, 180 + ang, 300 - ang, 300 + ang]))

    radius = 0.1
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


def get_goal_position(t):
    xy_dir = np.array([1, 1])
    xy_dir = xy_dir / np.linalg.norm(xy_dir)
    xy_dist = np.clip(0.1 * t, 0, 0.5)
    return np.hstack([xy_dist * xy_dir, 0.8])


def get_rms_position_error(solution):
    cov = solution.marginals.platform_pose.cov[3:,3:]  # position covariance
    return np.sqrt(np.linalg.trace(cov))


def get_config():
    r = 0.002 / 2
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

    config = crest_sparse.ParallelRobotSolverConfig()

    config.base.use_dense = False
    config.nodes_per_rod = 20
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-3
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3
    config.base_end_poses = get_base_poses()
    config.tip_end_poses = get_tip_poses()
    config.sigma_end_pose_pos= 1.0e-4
    config.sigma_end_pose_rot= 1.0e-3

    return config


def run_sim(alpha, save_frames_dir_name):
    config = get_config()

    solver = crest_sparse.ParallelRobotSolver(config)
    baseline = ParallelRobotSolver(config, plot=False)

    plotter = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_platform_wrench=False,
        single_plot_mode=False,
        save_frames_dir_name=save_frames_dir_name,
        platform_z_offset=platform_z_offset,
        camera_azimuth=-0, 
        camera_distance=3, 
        camera_focal_point=np.array([0, 0, 0.5])
    )
    
    frame_rate = 30.0
    dt = 1.0 / frame_rate
    t_final = 6.0
    num_steps = int(t_final / dt)

    rod_lengths_sigma = 1e-4
    rod_lengths = 0.6 * np.ones(6)

    wrench_mean = np.zeros(6)
    wrench_cov = 1e-6 * np.eye(6)
    wrench_cov[3:,3:] = 1e-1 * np.eye(3)
    wrench = crest_sparse.Vector6Gaussian(wrench_mean, wrench_cov)

    p_solution, p_baseline, p_uncertainty = [], [], []

    for step in range(num_steps + 1):
        solution = solver.solve(rod_lengths, rod_lengths_sigma, wrench)
        comparison = baseline.solve(rod_lengths, tip_force=wrench.mean[3:], tip_moment=wrench.mean[:3])

        p_solution.append(solution.marginals.rods[0].states[-1].pose.mean[:3,3])
        p_baseline.append(comparison[0]['pose'][-1][:3,3])

        h0 = get_rms_position_error(solution)
        grad_h = np.zeros(6)
        fd_step = 1e-5

        for i in range(6):
            dl = np.zeros(6)
            dl[i] = fd_step

            si = solver.solve(rod_lengths + dl, rod_lengths_sigma, wrench)
            grad_h[i] = (get_rms_position_error(si) - h0) / fd_step

        p_uncertainty.append(h0)

        p = solution.marginals.platform_pose.mean[:3, 3]
        R = solution.marginals.platform_pose.mean[:3,:3]

        t = step * dt
        p_goal = get_goal_position(t)
        p_error = R.T @ (p_goal - p)

        J = solution.marginals.rod_lengths_jacobian[3:,:]
        J_pinv = np.linalg.pinv(J)
        J_null = np.eye(J.shape[1]) - J_pinv @ J

        d_rod_lengths = J_pinv @ p_error - alpha * (J_null @ grad_h)

        rod_lengths += d_rod_lengths

        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")

    return np.array(p_solution), np.array(p_baseline), np.array(p_uncertainty)


if __name__ == "__main__":
    _, _, p_rms_nominal = run_sim(0.0, "parallel_robot_nominal")
    p_solution, p_baseline, p_rms_resolved = run_sim(0.07, "parallel_robot_resolved")

    baseline_error = p_solution - p_baseline
    baseline_rms = np.sqrt(np.mean(np.sum(baseline_error**2, axis=1)))

    print("baseline_rms: ", baseline_rms)
    print("nominal_rms: ", p_rms_nominal[-1])
    print("resolved_rms: ", p_rms_resolved[-1])