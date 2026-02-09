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
    wt = 2 * np.pi * (0.105) * t

    p_xy = 0.7 / 20.0 * t * np.array([np.cos(wt), np.sin(wt)])
    p = np.hstack([p_xy, 0.7])

    r_xy = np.radians(80) / 20.0 * t * np.array([-np.sin(wt), np.cos(wt)])
    r = np.hstack([r_xy, 0])
    R = Rotation.from_rotvec(r).as_matrix()

    return p, R


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
    config.nodes_per_rod = 15
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


def get_tip_position_baseline(solution):
    pose_ends = np.array([rod['pose'][-1] for rod in solution])  # num_rods, 4, 4
    p_ends = pose_ends[:,:3,3]
    R_ends = pose_ends[:,:3,:3]
        
    z_offset = R_ends[:,:3,2] * platform_z_offset

    return np.mean(p_ends - z_offset, axis=0)
    

def run_sim(t_final=20.0, frame_rate=30.0, rod_lengths_sigma=0.002, save_frames_dir_name=None, plot=True, do_baseline=True):
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
    
    dt = 1.0 / frame_rate
    t = np.arange(0, t_final, dt)
    rod_lengths = 0.6 * np.ones(6)
    wrench = crest_sparse.Vector6Gaussian(np.zeros(6), 1e-6 * np.eye(6))

    p_solution, p_baseline, p_command, p_rms = [], [], [], []

    for ti in t:
        # Solve using our method and capture uncertainty
        solution = solver.solve(rod_lengths, rod_lengths_sigma, wrench)
        p = solution.marginals.platform_pose.mean[:3, 3]
        R = solution.marginals.platform_pose.mean[:3,:3]
        p_cov = solution.marginals.platform_pose.cov[3:,3:]

        # Compare to baseline model if requested
        if do_baseline:
            comparison = baseline.solve(rod_lengths, tip_force=wrench.mean[3:], tip_moment=wrench.mean[:3])
            p_baseline.append(get_tip_position_baseline(comparison))

        # Compare to current goal pose
        p_goal, R_goal = get_goal_pose(ti)
        p_error = R.T @ (p_goal - p)
        r_error = Rotation.from_matrix(R.T @ R_goal).as_rotvec()
        twist_error = np.hstack((r_error, p_error))

        # Jacobian to step toward the goal
        J = solution.marginals.rod_lengths_jacobian
        d_rod_lengths = np.linalg.pinv(J) @ twist_error
        rod_lengths += d_rod_lengths

        # Collect data, plot, display
        p_solution.append(p)
        p_command.append(p_goal)
        p_rms.append(np.sqrt(np.trace(p_cov)))
        
        if plot:
            plotter.update(solution)

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    return t, np.array(p_solution), np.array(p_baseline), np.array(p_command), np.array(p_rms)


if __name__ == "__main__":
    do_baseline = True
    t, p_solution, p_baseline, p_command, p_rms = run_sim(do_baseline=do_baseline, save_frames_dir_name="parallel_robot_sim", plot=True)

    fig, axs = plt.subplots(
        2, 1,
        figsize=(7, 5),
        sharex=True,
        constrained_layout=True
    )

    axs[0].plot(t, 1000 * p_rms, linewidth=2.5)
    axs[0].set_ylabel("RMS uncertainty (mm)")
    axs[0].grid(True, alpha=0.25)

    if do_baseline:
        baseline_err = 1000 * np.linalg.norm(p_solution - p_baseline, axis=1)
        axs[1].plot(t, baseline_err, linewidth=2.5, label="baseline")

    axs[1].set_ylabel("baseline error (mm)")
    axs[1].set_xlabel("time (sec)")
    axs[1].grid(True, alpha=0.25)

    fig.align_ylabels(axs)
    fig.savefig("parallel_robot_plot.png", dpi=300)
    plt.close(fig)