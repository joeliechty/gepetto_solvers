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
    wt = 2 * np.pi * (0.1) * t

    p_xy = 0.2 / 30.0 * t * np.array([np.cos(wt), np.sin(wt)])
    p_z = 0.55 + 0.15 * np.sin(2 * np.pi * (0.333) * t)
    p = np.hstack([p_xy, p_z])

    r_xy = np.radians(30) / 30.0 * t * np.array([-np.sin(wt), np.cos(wt)])
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
    

def run_sim(
        t_final=30.0, 
        frame_rate=30.0, 
        rod_lengths_sigma=0.001,
        small_rod_lengths_sigma=1e-5,
        small_wrench_sigma=1e-3,
        actuator_f_meas_sigma=0.1,
        tip_force_prior_sigma=0.9,
        tip_force_drift_sigma=2.0,
        plot=True, 
        do_baseline=True):

    config = get_config()

    # Simulator to generate actuator forces on rods
    solver_sim = crest_sparse.ParallelRobotSolver(config)
    baseline = ParallelRobotSolver(config, plot=False)

    # Prior solves the robot with no measurements but with big force prior
    solver_prior = crest_sparse.ParallelRobotSolver(config)
    plotter_prior = ParallelRobotPlotter(
        plot_rod_wrenches=False,
        plot_tip_force=False,
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
        plot_tip_force=True,
        single_plot_mode=False,
        save_frames_dir_name="parallel_robot_post",
        platform_z_offset=platform_z_offset,
        camera_azimuth=-60, 
        camera_distance=2.7, 
        camera_focal_point=np.array([0, 0, 0.5])
    )

    # Seperate solver just for getting jacobian, dont want to mess up warm starts
    solver_jac = crest_sparse.ParallelRobotSolver(config)

    tip_force_function = TipForceFunction(max_magnitude=2 * tip_force_prior_sigma, framerate=frame_rate, seed=2)

    dt = 1.0 / frame_rate
    t = np.arange(0, t_final, dt)
    rod_lengths_cmd = 0.6 * np.ones(6)
    
    # p_solution, p_baseline, p_command, p_rms = [], [], [], []
    
    small_wrench_cov = small_wrench_sigma ** 2 * np.eye(6)
    wrench_prior_cov = small_wrench_cov.copy()
    f_prior_cov = tip_force_prior_sigma ** 2 * np.eye(3)
    wrench_prior_cov[3:,3:] = f_prior_cov
    wrench_prior = crest_sparse.Vector6Gaussian(np.zeros(6), wrench_prior_cov)
    
    f_drift_cov = tip_force_drift_sigma ** 2 * dt * np.eye(3)
    f_prev_mean = np.zeros(3)
    f_prev_cov = 10 * np.eye(3)

    data = {
        't': t, 
        'f_gt': [], 'f_mean': [], 'f_std': [], 'f_std_prior': [], 
        'p_goal': [], 'p_gt': [], 'p_mean': [], 'p_std': [], 'p_std_prior': [], 'p_baseline': []
    }

    for ti in t:
        # Solve using our method and capture uncertainty
        f_gt = tip_force_function(ti)
        wrench_gt = np.zeros(6)
        wrench_gt[3:] = f_gt
        rod_lengths_gt = rod_lengths_cmd + rod_lengths_sigma * np.random.randn(6)
        solution_sim = solver_sim.solve(
            rod_lengths_gt, 
            small_rod_lengths_sigma, 
            crest_sparse.Vector6Gaussian(wrench_gt, small_wrench_cov), 
            None
        )
        p_gt = solution_sim.marginals.platform_pose.mean[:3,3]

        # Sample base actuator z forces
        f_meas = []
        for rod in solution_sim.marginals.rods:
            f_meas.append(rod.states[0].wrench.mean[5]) # z force on base of rod
        f_meas = np.array(f_meas) + actuator_f_meas_sigma * np.random.randn(6)

        # Solve prior with no measuremtns
        solution_prior = solver_prior.solve(rod_lengths_cmd, rod_lengths_sigma, wrench_prior, None)

        # Change wrench prior to use drift model
        f_prev_drift_cov = f_prev_cov + f_drift_cov
        f_fused_cov = np.linalg.inv(np.linalg.inv(f_prior_cov) + np.linalg.inv(f_prev_drift_cov))
        f_fused_mean = f_fused_cov @ (np.linalg.inv(f_prev_drift_cov) @ f_prev_mean)
        wrench_fused_mean = np.hstack((np.zeros(3), f_fused_mean))
        wrench_fused_cov = wrench_prior_cov.copy()
        wrench_fused_cov[3:,3:] = f_fused_cov

        solution_post = solver_post.solve(
            rod_lengths_cmd, 
            rod_lengths_sigma, 
            crest_sparse.Vector6Gaussian(wrench_fused_mean, wrench_fused_cov),
            crest_sparse.ActuationForceMeas(np.array(f_meas), actuator_f_meas_sigma)
        )
        
        wrench_post = solution_post.marginals.platform_wrench
        f_cov = wrench_post.cov[3:,3:]
        f_mean = wrench_post.mean[3:]
        f_prev_mean = f_mean
        f_prev_cov = f_cov

        solution_jac = solver_jac.solve(rod_lengths_cmd, rod_lengths_sigma, wrench_post, None)

        # Compare to baseline model if requested
        if do_baseline:
            comparison = baseline.solve(rod_lengths_gt, tip_force=wrench_gt[3:], tip_moment=wrench_gt[:3])
            p_baseline = get_tip_position_baseline(comparison)
            print(f"baseline error: {np.linalg.norm(p_baseline - p_gt)}")
            data['p_baseline'].append(p_baseline)

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
        data['f_gt'].append(f_gt)
        data['f_mean'].append(f_mean)
        data['f_std'].append(np.sqrt(np.diag(f_cov)))
        data['f_std_prior'].append(np.sqrt(np.diag(solution_prior.marginals.platform_wrench.cov[3:,3:])))
        data['p_goal'].append(p_goal)
        data['p_gt'].append(p_gt)
        data['p_mean'].append(solution_post.marginals.platform_pose.mean[:3,3])
        data['p_std'].append(np.sqrt(np.diag(solution_post.marginals.platform_pose.cov[3:,3:])))
        data['p_std_prior'].append(np.sqrt(np.diag(solution_prior.marginals.platform_pose.cov[3:,3:])))

        if plot:
            plotter_prior.update(solution_prior)
            plotter_post.update(solution_post, tip_force_gt=f_gt)

        progress = 100.0 * ti / t[-1]
        print(f"Progress: {progress:5.1f}%", end="\r")

    return {k: np.asarray(v) for k, v in data.items()}


if __name__ == "__main__":
    do_baseline = True
    plot = True
    data = run_sim(do_baseline=do_baseline, plot=plot)


    color_cycle = ['r', 'g', 'b', 'c']

    setup_plt(height=4.0, grid=True)

    fig, axes = plt.subplots(3, 1, sharex=True)

    position_labels = [r'position-$x$ (mm)',
                       r'position-$y$ (mm)',
                       r'position-$z$ (mm)']
    
    for ii, ax in enumerate(axes):
        ax.plot(data['t'], 1000 * data['p_mean'][:, ii], linestyle='-', color=color_cycle[ii], label='mean')
        ax.plot(data['t'], 1000 * data['p_gt'][:, ii], linestyle='--', color=color_cycle[ii], label='truth')
        ax.fill_between(data['t'], 
            1000 * data['p_mean'][:,ii] - 2000 * data['p_std'][:,ii],
            1000 * data['p_mean'][:,ii] + 2000 * data['p_std'][:,ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')

        ax.set_ylabel(position_labels[ii])
        if ii == 1:
            ax.legend(ncol=3, columnspacing=0.5, handletextpad=0.5)

    fig.align_ylabels()
    plt.tight_layout()
    
    plt.savefig("figures/parallel_robot_position.pdf", bbox_inches="tight")



    setup_plt(width=6, height=2, grid=True)

    fig, axes = plt.subplots(1, 3, sharex=True)

    for ii, ax in enumerate(axes):
        ax.plot(data['t'], data['f_gt'][:,ii], 'k--', label='truth')
        ax.plot(data['t'], data['f_mean'][:,ii], color=color_cycle[ii], label='mean')
        ax.fill_between(data['t'], 
            data['f_mean'][:,ii] - 2 * data['f_std'][:,ii],
            data['f_mean'][:,ii] + 2 * data['f_std'][:,ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')
        ax.set_xlabel("time (sec)")
        ax.set_xlim([data['t'][0], data['t'][-1]+1e-1])
        if ii == 0:
            ax.set_ylabel("tip force (N)")
            ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.1), columnspacing=0.2, borderpad=0.0, borderaxespad=0.2, handlelength=1.0, handletextpad=0.2)

    fig.align_ylabels()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.2)

    plt.savefig("figures/parallel_robot_force.pdf", bbox_inches="tight")




    setup_plt(height=2, grid=True)

    fig = plt.figure()

    plt.plot(data['t'], np.sqrt(np.sum(data['p_std']**2, axis=1)), 'k-', label='post')
    plt.plot(data['t'], np.sqrt(np.sum(data['p_std_prior']**2, axis=1)), 'k--', label='prior')
    plt.xlabel('time (sec)')
    plt.ylabel('position uncertainty (mm)')

    fig.align_ylabels()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.2)

    plt.savefig("figures/parallel_robot_uncertainty.pdf", bbox_inches="tight")





    if do_baseline:
        setup_plt()
        plt.figure()
        baseline_err = 1000 * np.linalg.norm(data['p_gt'] - data['p_baseline'], axis=1)
        plt.plot(data['t'], baseline_err, label="baseline")
        plt.xlabel("time (sec)")
        plt.ylabel("baseline error (mm)")
        plt.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig("figures/parallel_robot_baseline.pdf", dpi=300)
        plt.close()

        print(f"mean baseline error: {np.mean(baseline_err)} mm")

    