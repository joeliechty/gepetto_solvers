import numpy as np
import matplotlib.pyplot as plt

from crest_sparse import TendonRobotSolver, Vector6Gaussian, Vector4Gaussian, Vector3Gaussian

from .._plotting.tendon_robot_plotter import TendonRobotPlotter
from .config import get_base_config
from .utils import TipForceFunction, generate_waypoint_trajectory, GaussianProcessNoiseModel


def run_sim(sim_time=3.0, do_plot=True, tip_force_prior_sigma=0.1, tensions_meas_sigma=0.01, tip_position_meas_sigma=0.001):
    config = get_base_config()

    t, positions, tensions_nominal, waypoints = generate_waypoint_trajectory(sim_time, seed=7)
    tip_force_function = TipForceFunction(max_magnitude=2 * tip_force_prior_sigma, seed=7)

    # A solver to simulate the nominal trajectory of the robot, given open loop tensions
    simulator_nominal = TendonRobotSolver(config)
    plotter_nominal = TendonRobotPlotter(
        save_frames_dir_name='tip_force_nominal', 
        plot_tip_force=True, 
        plot_backbone_ellipsoids=False
    )
    
    # Simulator to sample tip pose data
    simulator_tracking = TendonRobotSolver(config)

    # Solver that does the actual inference using tip pose data
    solver_tracking = TendonRobotSolver(config)
    plotter_tracking = TendonRobotPlotter(
        save_frames_dir_name='tip_force_posterior', 
        plot_tip_force=True
    )

    # Use a separate solver for these so it doesn't mess up our performance metrics
    solver_jacobian = TendonRobotSolver(config)
    solver_prior = TendonRobotSolver(config)
    plotter_prior = TendonRobotPlotter(save_frames_dir_name='tip_force_prior', plot_tip_force=False)

    # Setup Jacobian control
    damping = 6e-2
    tensions_min = np.array([0.5, 0.1, 0.1, 0.1])
    tensions_cmd_current = tensions_min
    
    # Continuous time noise models for smoothness
    tensions_nominal_noise_model = GaussianProcessNoiseModel(4, len(t), seed=42)
    tensions_tracking_noise_model = GaussianProcessNoiseModel(4, len(t), seed=42)
    p_nominal_noise_model = GaussianProcessNoiseModel(3, len(t), seed=42)
    p_tracking_noise_model = GaussianProcessNoiseModel(3, len(t), seed=42)

    # Setup data collection
    p_data = {'mean': [], 'std': [], 'gt': [], 'nominal': [], 'desired': []}
    f_data = {'mean': [], 'std': [], 'gt': []}

    small_tensions_cov = 1e-6 * np.eye(4)
    small_wrench_cov = 1e-6 * np.eye(6)

    wrench_prior_cov = small_wrench_cov.copy()
    wrench_prior_cov[3:,3:] = tip_force_prior_sigma ** 2 * np.eye(3)
    position_meas_cov = tip_position_meas_sigma ** 2 * np.eye(3)
    
    for t_i, p_desired, tensions_nominal_i in zip(t, positions, tensions_nominal):
        f_gt = tip_force_function(t_i)

        # Nominal solution with no force correction, still need to add noise and re solve
        tensions_nominal_gt = tensions_nominal_i + tensions_nominal_noise_model.step(tensions_meas_sigma ** 2 * np.eye(4))
        solution_nominal = simulator_nominal.solve(
            Vector4Gaussian(tensions_nominal_gt, small_tensions_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), small_wrench_cov),
            None
        )

        p_nominal_mean = solution_nominal.marginals.rod.states[-1].pose.mean[:3,3]
        R_nominal = solution_nominal.marginals.rod.states[-1].pose.mean[:3,:3]
        p_nominal_cov = R_nominal @ solution_nominal.marginals.rod.states[-1].pose.cov[3:,3:] @ R_nominal.T
        p_nominal_cov += tip_position_meas_sigma ** 2 * np.eye(3)
        p_nominal_gt = p_nominal_mean + p_nominal_noise_model.step(p_nominal_cov)

        # Prior solution, using only prior wrench, no measurement
        solution_prior = solver_prior.solve(
            Vector4Gaussian(tensions_cmd_current, tensions_meas_sigma ** 2 * np.eye(4)),
            Vector6Gaussian(np.zeros(6), wrench_prior_cov),
            None 
        )

        # Simulated solution to sample position from, small covariances
        tensions_tracking_gt = tensions_cmd_current + tensions_tracking_noise_model.step(tensions_meas_sigma ** 2 * np.eye(4))
        solution_sim = simulator_tracking.solve(
            Vector4Gaussian(tensions_tracking_gt, small_tensions_cov),
            Vector6Gaussian(np.hstack((np.zeros(3), f_gt)), small_wrench_cov),
            None
        )

        # Sample the position
        p_sim_mean = solution_sim.marginals.rod.states[-1].pose.mean[:3,3]
        R_sim_mean = solution_sim.marginals.rod.states[-1].pose.mean[:3,:3]
        p_sim_cov = R_sim_mean @ solution_sim.marginals.rod.states[-1].pose.cov[3:,3:] @ R_sim_mean.T
        p_sim_gt = p_sim_mean + p_tracking_noise_model.step(p_sim_cov)
        p_meas = p_sim_gt + tip_position_meas_sigma * np.random.randn(3)

        # Use the sampled position as a prior on tip pose
        solution_post = solver_tracking.solve(
            Vector4Gaussian(tensions_cmd_current, tensions_meas_sigma ** 2 * np.eye(4)),
            Vector6Gaussian(np.zeros(6), wrench_prior_cov), 
            Vector3Gaussian(p_meas, position_meas_cov)
        )
        
        
        # Evaluate the Jacobian for control using the estimated tip wrench
        wrench_post = solution_post.marginals.external_wrenches[-1]
        solution_jacobian = solver_jacobian.solve(
            Vector4Gaussian(tensions_cmd_current, tensions_meas_sigma ** 2 * np.eye(4)),
            wrench_post,
            None
        )
        
        J_position = solution_jacobian.marginals.J_pose_tensions[3:]
        p_error = R_sim_mean.T @ (p_desired - p_meas)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_cmd_current = tensions_cmd_current + d_tensions
        tensions_cmd_current = np.maximum(tensions_cmd_current, tensions_min)

        if do_plot:
            plotter_tracking.update(solution_post, p_desired=p_desired, tip_force_gt=f_gt)
            plotter_nominal.update(solution_nominal, p_desired=p_desired, tip_force_gt=f_gt)
            plotter_prior.update(solution_prior)

        # tip_position_tracking_mean.append(p_tracking_mean)
        # tip_position_tracking_std.append(np.sqrt(np.diag(p_tracking_cov)))
        # tip_position_tracking_gt.append(p_tracking_gt)
        # tip_position_nominal_gt.append(p_nominal_gt)
        # tip_position_desired.append(p_desired)
        # tip_force_gt.append(f_gt)
        # tip_force_mean.append(f_mean)
        # tip_force_std.append(np.sqrt(np.diag(f_cov)))
        # tensions_cmd.append(tensions_cmd_current)
        # tensions_gt.append(tensions_tracking_gt)

    # tip_position_tracking_mean = np.array(tip_position_tracking_mean)
    # tip_position_tracking_std = np.array(tip_position_tracking_std)
    # tip_position_tracking_gt = np.array(tip_position_tracking_gt)
    # tip_position_nominal_gt = np.array(tip_position_nominal_gt)
    # tip_position_desired = np.array(tip_position_desired)
    # tip_force_gt = np.array(tip_force_gt)
    # tip_force_mean = np.array(tip_force_mean)
    # tip_force_std = np.array(tip_force_std)
    # tensions_cmd = np.array(tensions_cmd)
    # tensions_gt = np.array(tensions_gt)

    color_cycle = ['r', 'g', 'b', 'c']

    setup_plt(height=4.0, grid=True)

    fig, axes = plt.subplots(3, 1, sharex=True)

    position_labels = [r'position-$x$ (mm)',
                       r'position-$y$ (mm)',
                       r'position-$z$ (mm)']
    
    for ii, ax in enumerate(axes):
        ax.plot(t, 1000 * tip_position_tracking_mean[:, ii], linestyle='-', color=color_cycle[ii], label='mean')
        ax.plot(t, 1000 * tip_position_tracking_gt[:, ii], linestyle='--', color=color_cycle[ii], label='truth')
        ax.fill_between(t, 
            1000 * tip_position_tracking_mean[:,ii] - 2000 * tip_position_tracking_std[:,ii],
            1000 * tip_position_tracking_mean[:,ii] + 2000 * tip_position_tracking_std[:,ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')

        ax.set_ylabel(position_labels[ii])
        if ii == 1:
            ax.legend(ncol=3, columnspacing=0.5, handletextpad=0.5)

    fig.align_ylabels()
    plt.tight_layout()
    
    plt.savefig("figures/position_noise_model.pdf", bbox_inches="tight")

    setup_plt(width=6, height=2, grid=True)

    fig, axes = plt.subplots(2, 3, sharex=True)

    for ii, ax in enumerate(axes[0,:]):
        ax.plot(t, 1000 * tip_position_desired[:, ii], linestyle=':', color='k', label='desired')
        ax.plot(t, 1000 * tip_position_tracking_gt[:, ii], linestyle='-', color=color_cycle[ii], label='tracking')
        ax.plot(t, 1000 * tip_position_nominal_gt[:, ii], linestyle='--', color=color_cycle[ii], label='OL')
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 0:
            ax.set_ylabel("tip position (mm)")
        if ii == 1:
            ax.legend(ncol=3, loc="upper right", bbox_to_anchor=(1, 1.1), columnspacing=0.2, borderpad=0.0, borderaxespad=0.2, handlelength=1.0, handletextpad=0.2)
            

    for ii, ax in enumerate(axes[1,:]):
        ax.plot(t, tip_force_gt[:,ii], 'k--', label='truth')
        ax.plot(t, tip_force_mean[:,ii], color=color_cycle[ii], label='mean')
        ax.fill_between(t, 
            tip_force_mean[:,ii] - 2 * tip_force_std[:,ii],
            tip_force_mean[:,ii] + 2 * tip_force_std[:,ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')
        ax.set_xlabel("time (sec)")
        ax.set_xlim([t[0], t[-1]+1e-1])
        if ii == 0:
            ax.set_ylabel("tip force (N)")
            ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.1), columnspacing=0.2, borderpad=0.0, borderaxespad=0.2, handlelength=1.0, handletextpad=0.2)
            
    # lines_cmd = axes[6].plot(t, tensions_cmd, linestyle='-')
    # lines_gt  = axes[6].plot(t, tensions_gt, 'k:')
    # handles = [lines_cmd[0], lines_gt[0]]
    # labels  = ['cmd', 'truth']
    # axes[6].legend(handles, labels)
    # axes[6].set_ylabel('tendon tensions (N)')
    
    # axes[7].plot(t, np.linalg.norm(tip_force_gt, axis=1), color='purple')
    # axes[7].set_ylabel('force magnitude (N)')
    # axes[7].set_xlabel('time (sec)')

    fig.align_ylabels()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, hspace=0.2)

    plt.savefig("figures/tip_force_sim_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    run_sim(do_plot=True)
