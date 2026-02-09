import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

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

platform_z_offset = -0.1

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
    wt = 2 * np.pi * (0.1) * t
    xy_dir = np.array([np.cos(wt), np.sin(wt)])
    xy_dist = np.clip(0.04 * t, 0, 0.4)
    return np.hstack([xy_dist * xy_dir, 0.8])


def get_rms_position_error(solution):
    cov = solution.marginals.platform_pose.cov[3:,3:]  # position covariance
    cost = np.sqrt(np.linalg.trace(cov))

    # for rod in solution.marginals.rods:
    #     for state in rod.states:
    #         cov = state.pose.cov[3:,3:]
    #         cost += np.sqrt(np.linalg.trace(cov))

    return cost


def get_cost_grad(solver, rod_lengths, rod_lengths_sigma, wrench, alpha):
    def cost(rod_lengths):
        s = solver.solve(rod_lengths, rod_lengths_sigma, wrench)
        p_error = get_rms_position_error(s)
        return alpha * p_error + 0.1 * sum(rod_lengths)

    h0 = cost(rod_lengths)
    grad_h = np.zeros(6)
    fd_step = 1e-5

    for i in range(6):
        dl = np.zeros(6)
        dl[i] = fd_step
        hi = cost(rod_lengths + dl)
        grad_h[i] = (hi - h0) / fd_step

    return h0, grad_h


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


def run_sim(alpha, save_frames_dir_name=None, plot=False, do_baseline=False):
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

    rod_lengths_sigma = 1e-4
    rod_lengths = 0.5 * np.ones(6)

    max_velocity = 0.15
    max_step = max_velocity * dt

    wrench_mean = np.zeros(6)
    wrench_cov = 1e-6 * np.eye(6)
    wrench_cov[3:,3:] = 1e-1 * np.eye(3)
    wrench = crest_sparse.Vector6Gaussian(wrench_mean, wrench_cov)

    p_solution, p_baseline, p_uncertainty = [], [], []

    for ti in t:
        solution = solver.solve(rod_lengths, rod_lengths_sigma, wrench)
        p_solution.append(solution.marginals.rods[0].states[-1].pose.mean[:3,3])

        if do_baseline:
            comparison = baseline.solve(rod_lengths, tip_force=wrench.mean[3:], tip_moment=wrench.mean[:3])
            p_baseline.append(comparison[0]['pose'][-1][:3,3])

        h, grad_h = get_cost_grad(solver, rod_lengths, rod_lengths_sigma, wrench, alpha)
        p_uncertainty.append(h)

        p = solution.marginals.platform_pose.mean[:3, 3]
        R = solution.marginals.platform_pose.mean[:3,:3]

        p_goal = get_goal_position(ti)
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