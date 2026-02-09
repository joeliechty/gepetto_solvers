import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import crest_sparse
from .._plotting.parallel_robot_plotter import ParallelRobotPlotter
from .baseline_model import ParallelRobotSolver


def get_end_poses(angles, radius, z_offset):
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    poses = []
    for xi, yi in zip(xs, ys):
        pose = np.eye(4)
        pose[0, 3] = xi
        pose[1, 3] = yi
        pose[2, 3] = z_offset
        poses.append(pose)

    return poses


def get_base_poses():
    ang = 10
    angles = np.array(np.deg2rad([ang, 120 - ang, 120 + ang, 240 - ang, 240 + ang, -ang]))

    return get_end_poses(angles, radius=0.1, z_offset=0.0)


platform_z_offset = -0.1


def get_tip_poses():
    ang = 10
    angles = np.array(np.deg2rad([60 - ang, 60 + ang, 180 - ang, 180 + ang, 300 - ang, 300 + ang]))

    return get_end_poses(angles, radius=0.1, z_offset=platform_z_offset)


def get_goal_pose(t):
    wt = 2 * np.pi * (0.1) * t
    p_xy = 0.04 * t * np.array([np.cos(wt), np.sin(wt)])
    p = np.hstack([p_xy, 0.8])

    r_xy = 0.04 * t * np.array([-np.sin(wt), np.cos(wt)])
    r = np.hstack([r_xy, 0])

    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_rotvec(r).as_matrix()
    pose[:3,3] = p

    return pose


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


def run_sim(rod_lengths_sigma, save_frames_dir_name=None, plot=False, do_baseline=False):
    config = get_config()

    solver = crest_sparse.ParallelRobotSolver(config)
    baseline = ParallelRobotSolver(config, plot=False)

    plotter = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_platform_wrench=False,
        single_plot_mode=False,
        save_frames_dir_name=save_frames_dir_name,
        platform_z_offset=platform_z_offset,
        camera_azimuth=120, 
        camera_distance=3, 
        camera_focal_point=np.array([0, 0, 0.5])
    )
    
    frame_rate = 10.0
    dt = 1.0 / frame_rate
    t_final = 15
    t = np.arange(0, t_final, dt)

    rod_lengths = 0.5 * np.ones(6)

    max_velocity = 0.15
    max_step = max_velocity * dt

    wrench = crest_sparse.Vector6Gaussian(np.zeros(6), 1e-6 * np.eye(6))

    p_solution, p_baseline, p_uncertainty = [], [], []

    for ti in t:
        solution = solver.solve(rod_lengths, rod_lengths_sigma, wrench)
        p_solution.append(solution.marginals.rods[0].states[-1].pose.mean[:3,3])
        p_uncertainty.append(get_rms_position_error(solution))

        if do_baseline:
            comparison = baseline.solve(rod_lengths, tip_force=wrench.mean[3:], tip_moment=wrench.mean[:3])
            p_baseline.append(comparison[0]['pose'][-1][:3,3])

        p = solution.marginals.platform_pose.mean[:3, 3]
        R = solution.marginals.platform_pose.mean[:3,:3]

        pose_goal = get_goal_pose(ti)
        dp = R.T @ (p_goal - p)

        if np.linalg.norm(dp) > max_step:
            dp = max_step * dp / np.linalg.norm(dp)

        J = solution.marginals.rod_lengths_jacobian[3:,:]
        J_pinv = np.linalg.pinv(J)
        J_null = np.eye(J.shape[1]) - J_pinv @ J

        d_rod_lengths = J_pinv @ dp - dt * (J_null @ grad_h)

        rod_lengths += d_rod_lengths
        
        if plot:
            plotter.update(solution)

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    return t, np.array(p_solution), np.array(p_baseline), np.array(p_uncertainty)


if __name__ == "__main__":
    t, _, _, p_rms_nominal = run_sim(0.0, plot=True)
    # _, _, _, p_rms_resolved = run_sim(1.5)
    # p_solution, p_baseline, p_rms_resolved = run_sim(0.07, "parallel_robot_resolved")

    # baseline_error = p_solution - p_baseline
    # baseline_rms = np.sqrt(np.mean(np.sum(baseline_error**2, axis=1)))

    # print("baseline_rms: ", baseline_rms)
    # print("nominal_rms: ", p_rms_nominal[-1])
    # print("resolved_rms: ", p_rms_resolved[-1])

    plt.figure(figsize=(6, 4))

    plt.plot(t, 1000 * p_rms_nominal, linewidth=2, label="nominal")
    plt.plot(t, 1000 * p_rms_resolved, linewidth=2, label="resolved")

    plt.xlabel("time (sec)")
    plt.ylabel("Predicted RMS position error (mm)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("parallel_robot_rms.png", dpi=300)
    plt.close()