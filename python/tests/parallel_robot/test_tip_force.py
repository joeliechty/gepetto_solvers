import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import crest_sparse
from .._plotting.parallel_robot_plotter import ParallelRobotPlotter
from .._plotting.utils import setup_plt

from ..tendon_robot.utils import TipForceFunction
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

    p_xy = 0.5 / 20.0 * t * np.array([np.cos(wt), np.sin(wt)])
    p = np.hstack([p_xy, 0.7])

    r_xy = np.radians(60) / 20.0 * t * np.array([-np.sin(wt), np.cos(wt)])
    r = np.hstack([r_xy, 0])
    R = Rotation.from_rotvec(r).as_matrix()

    return p, R


def get_config():
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

    # Simulator to generate actuator forces on rods
    solver_sim = crest_sparse.ParallelRobotSolver(config)
    # baseline = ParallelRobotSolver(config, plot=False)

    # Prior solves the robot with no measurements but with big force prior
    solver_prior = crest_sparse.ParallelRobotSolver(config)
    plotter_prior = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_platform_wrench=True,
        single_plot_mode=False,
        save_frames_dir_name="parallel_robot_prior",
        platform_z_offset=platform_z_offset,
        camera_azimuth=-60, 
        camera_distance=2.7, 
        camera_focal_point=np.array([0, 0, 0.5])
    )
    
    # The actual solver that solves given measurements
    solver_post = crest_sparse.ParallelRobotSolver(config)
    plotter_post = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_platform_wrench=True,
        single_plot_mode=False,
        save_frames_dir_name="parallel_robot_post",
        platform_z_offset=platform_z_offset,
        camera_azimuth=-60, 
        camera_distance=2.7, 
        camera_focal_point=np.array([0, 0, 0.5])
    )

    # Seperate solver just for getting jacobian, dont want to mess up warm starts
    solver_jac = crest_sparse.ParallelRobotSolver(config)

    tip_force_prior_sigma = 1.0
    tip_force_function = TipForceFunction(max_magnitude=2 * tip_force_prior_sigma, seed=7)

    dt = 1.0 / frame_rate
    t = np.arange(0, t_final, dt)
    rod_lengths_cmd = 0.6 * np.ones(6)
    
    # p_solution, p_baseline, p_command, p_rms = [], [], [], []

    small_wrench_cov = 1e-6 * np.eye(6)
    wrench_prior_cov = 1e-6 * np.eye(6)
    wrench_prior_cov[3:,3:] = tip_force_prior_sigma ** 2 * np.eye(3)
    wrench_prior = crest_sparse.Vector6Gaussian(np.zeros(6), wrench_prior_cov)

    for ti in t:
        # Solve using our method and capture uncertainty
        f_gt = tip_force_function(ti)
        wrench_gt = np.zeros(6)
        wrench_gt[3:] = f_gt
        rod_lengths_gt = rod_lengths_cmd # + noise
        solution_sim = solver_sim.solve(
            rod_lengths_gt, 
            rod_lengths_sigma, 
            crest_sparse.Vector6Gaussian(wrench_gt, small_wrench_cov), 
            None
        )

        f_meas = []
        for rod in solution_sim.marginals.rods:
            f_meas.append(rod.states[0].wrench.mean[5]) # z force on base of rod

        solution_prior = solver_prior.solve(rod_lengths_cmd, rod_lengths_sigma, wrench_prior, None)

        force_meas_sigma = 0.1
        solution_post = solver_post.solve(
            rod_lengths_cmd, 
            rod_lengths_sigma, 
            wrench_prior, 
            crest_sparse.ActuationForceMeas(np.array(f_meas), force_meas_sigma)
        )
        
        wrench_post = solution_post.marginals.platform_wrench
        solution_jac = solver_jac.solve(rod_lengths_cmd, rod_lengths_sigma, wrench_post, None)
        # solution_sim.marginals.rods[0].states[0].stress.mean
        
        # p_cov = solution.marginals.platform_pose.cov[3:,3:]

        # Compare to baseline model if requested
        # if do_baseline:
        #     comparison = baseline.solve(rod_lengths, tip_force=wrench.mean[3:], tip_moment=wrench.mean[:3])
        #     p_comparison = get_tip_position_baseline(comparison)
        #     print(f"baseline error: {np.linalg.norm(p_comparison - p)}")
        #     p_baseline.append(p_comparison)

        # Compare to current goal pose
        p = solution_post.marginals.platform_pose.mean[:3, 3]
        R = solution_post.marginals.platform_pose.mean[:3,:3]
        p_goal, R_goal = get_goal_pose(ti)
        p_error = R.T @ (p_goal - p)
        r_error = Rotation.from_matrix(R.T @ R_goal).as_rotvec()
        twist_error = np.hstack((r_error, p_error))

        # Jacobian to step toward the goal
        J = solution_jac.marginals.rod_lengths_jacobian
        d_rod_lengths = np.linalg.pinv(J) @ twist_error
        rod_lengths_cmd += d_rod_lengths

        # Collect data, plot, display
        # p_solution.append(p)
        # p_command.append(p_goal)
        # p_rms.append(np.sqrt(np.trace(p_cov)))
        
        if plot:
            plotter_prior.update(solution_prior)
            plotter_post.update(solution_post)

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    return t, np.array(p_solution), np.array(p_baseline), np.array(p_command), np.array(p_rms)


if __name__ == "__main__":
    do_baseline = True
    plot = True
    dir_name = "parallel_robot_sim"
    t, p_solution, p_baseline, p_command, p_rms = run_sim(do_baseline=do_baseline, save_frames_dir_name=dir_name, plot=plot)

    setup_plt(width=2.0, height=2.0)
    plt.figure()

    plt.plot(t, 1000 * p_rms)
    plt.xlabel("time (sec)")
    plt.ylabel("RMS position uncertainty (mm)")
    plt.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("parallel_robot_uncertainty.pdf", dpi=300)
    plt.close()

    if do_baseline:
        setup_plt()
        plt.figure()
        baseline_err = 1000 * np.linalg.norm(p_solution - p_baseline, axis=1)
        plt.plot(t, baseline_err, label="baseline")
        plt.xlabel("time (sec)")
        plt.ylabel("baseline error (mm)")
        plt.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig("parallel_robot_baseline.pdf", dpi=300)
        plt.close()

        print(f"mean baseline error: {np.mean(baseline_err)} mm")

    