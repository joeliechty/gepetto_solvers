import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config
from utils import TipForceFunction, setup_plt, generate_waypoint_trajectory, GaussianProcessNoiseModel


def simulation(sim_time, do_plot, save_frames):
    config = get_base_config()
    simulator_tracking = TipForceSolver(config)
    simulator_nominal = TipForceSolver(config)
    solver_jacobian = TipForceSolver(config)
    solver_tracking = TipForceSolver(config)
    config_prior = get_base_config()
    config_prior.tip_position_meas_std = 1.0
    solver_prior = TipForceSolver(config_prior)

    t, positions, tensions_nominal, waypoints = generate_waypoint_trajectory(sim_time, seed=42)
    tip_force_function = TipForceFunction(max_magnitude=2 * config.tip_force_prior_std, seed=42)

    damping = 7e-2
    tensions_min = np.array([0.5, 0.1, 0.1, 0.1])

    tensions_cmd_current = tensions_min
    plotter_prior = TendonRobotPlotter('tip_force_prior', save_frames_mode=save_frames)
    plotter_tracking = TendonRobotPlotter('tip_force_tracking', plot_tip_force=True, save_frames_mode=save_frames)
    plotter_nominal = TendonRobotPlotter('tip_force_nominal', plot_tip_force=True, plot_backbone_ellipsoids=False, save_frames_mode=save_frames)

    tensions_nominal_noise_model = GaussianProcessNoiseModel(4, len(t), seed=42)
    tensions_tracking_noise_model = GaussianProcessNoiseModel(4, len(t), seed=42)
    p_nominal_noise_model = GaussianProcessNoiseModel(3, len(t), seed=42)
    p_tracking_noise_model = GaussianProcessNoiseModel(3, len(t), seed=42)

    tip_position_tracking_mean = []
    tip_position_tracking_std = []
    tip_position_tracking_gt = []
    tip_position_nominal_gt = []
    tip_position_desired = []
    tip_force_gt = []
    tip_force_mean = []
    tip_force_std = []
    tensions_cmd = []
    tensions_gt = []

    for t_i, p_desired, tensions_nominal_i in zip(t, positions, tensions_nominal):
        f_gt = tip_force_function(t_i)

        tensions_nominal_noise = tensions_nominal_noise_model.step(config.tension_meas_std ** 2 * np.eye(4))
        tensions_nominal_gt = tensions_nominal_i + tensions_nominal_noise
        tensions_tracking_noise = tensions_tracking_noise_model.step(config.tension_meas_std ** 2 * np.eye(4))
        tensions_tracking_gt = tensions_cmd_current + tensions_tracking_noise
        
        solution_tracking = simulator_tracking.simulation_step(tensions_tracking_gt, f_gt)
        solution_nominal = simulator_nominal.simulation_step(tensions_nominal_gt, f_gt)

        p_tracking_mean = solution_tracking.backbone_pose_mean[-1][:3,3]
        R = solution_tracking.backbone_pose_mean[-1][:3,:3]
        p_tracking_cov = R @ solution_tracking.backbone_pose_cov[-1][3:,3:] @ R.T
        p_tracking_cov += config.tip_position_meas_std ** 2 * np.eye(3)
        p_tracking_gt = p_tracking_mean + p_tracking_noise_model.step(p_tracking_cov)

        p_nominal_mean = solution_nominal.backbone_pose_mean[-1][:3,3]
        R = solution_nominal.backbone_pose_mean[-1][:3,:3]
        p_nominal_cov = R @ solution_nominal.backbone_pose_cov[-1][3:,3:] @ R.T
        p_nominal_cov += config.tip_position_meas_std ** 2 * np.eye(3)
        p_nominal_gt = p_nominal_mean + p_nominal_noise_model.step(p_nominal_cov)

        solution_tracking = solver_tracking.step(tensions_cmd_current, p_tracking_gt, 1)
        solution_prior = solver_prior.step(tensions_cmd_current, np.zeros(3), 1)

        f_mean = solution_tracking.applied_wrench_mean[-1][3:]
        f_cov = solution_tracking.applied_wrench_cov[-1][3:,3:]

        solution_jacobian = solver_jacobian.simulation_step(tensions_cmd_current, f_mean)
        
        J_position = solution_jacobian.J_pose_tensions[3:]
        p = solution_tracking.backbone_pose_mean[-1][:3,3]
        R = solution_tracking.backbone_pose_mean[-1][:3,:3]
        
        p_error = R.T @ (p_desired - p)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_cmd_current = tensions_cmd_current + d_tensions
        tensions_cmd_current = np.maximum(tensions_cmd_current, tensions_min)

        if do_plot:
            plotter_tracking.update(solution_tracking, p_desired=p_desired, tip_force_gt=f_gt)
            plotter_nominal.update(solution_nominal, p_desired=p_desired, tip_force_gt=f_gt)
            plotter_prior.update(solution_prior)

        tip_position_tracking_mean.append(p_tracking_mean)
        tip_position_tracking_std.append(np.sqrt(np.diag(p_tracking_cov)))
        tip_position_tracking_gt.append(p_tracking_gt)
        tip_position_nominal_gt.append(p_nominal_gt)
        tip_position_desired.append(p_desired)
        tip_force_gt.append(f_gt)
        tip_force_mean.append(f_mean)
        tip_force_std.append(np.sqrt(np.diag(f_cov)))
        tensions_cmd.append(tensions_cmd_current)
        tensions_gt.append(tensions_tracking_gt)
        
    plotter_tracking.plotter.close()
    plotter_nominal.plotter.close()
    plotter_prior.plotter.close()

    tip_position_tracking_mean = np.array(tip_position_tracking_mean)
    tip_position_tracking_std = np.array(tip_position_tracking_std)
    tip_position_tracking_gt = np.array(tip_position_tracking_gt)
    tip_position_nominal_gt = np.array(tip_position_nominal_gt)
    tip_position_desired = np.array(tip_position_desired)
    tip_force_gt = np.array(tip_force_gt)
    tip_force_mean = np.array(tip_force_mean)
    tip_force_std = np.array(tip_force_std)
    tensions_cmd = np.array(tensions_cmd)
    tensions_gt = np.array(tensions_gt)

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

    setup_plt(height=9.0, grid=True)

    fig, axes = plt.subplots(8, 1, sharex=True)
    
    for ii, ax in enumerate(axes[0:3]):
        ax.plot(t, 1000 * tip_position_tracking_gt[:, ii], linestyle='-', color=color_cycle[ii], label='tracking')
        ax.plot(t, 1000 * tip_position_nominal_gt[:, ii], linestyle='--', color=color_cycle[ii], label='open loop')
        ax.plot(t, 1000 * tip_position_desired[:, ii], linestyle=':', color='k', label='desired')
        ax.set_ylabel(position_labels[ii])
        if ii == 1:
            ax.legend(ncol=3, columnspacing=0.5, handletextpad=0.5)
        
    force_labels = [r'force-$x$ (N)',
                    r'force-$y$ (N)',
                    r'force-$z$ (N)']

    for ii, ax in enumerate(axes[3:6]):
        ax.plot(t, tip_force_gt[:,ii], 'k--', label='truth')
        ax.plot(t, tip_force_mean[:,ii], color=color_cycle[ii], label='mean')
        ax.fill_between(t, 
            tip_force_mean[:,ii] - 2 * tip_force_std[:,ii],
            tip_force_mean[:,ii] + 2 * tip_force_std[:,ii], 
            alpha=0.2, color=color_cycle[ii], interpolate=True, label=r'2-$\sigma$')
        ax.set_ylabel(force_labels[ii])
        if ii == 1:
            ax.legend(ncol=3, columnspacing=0.5, handletextpad=0.5)
        
    lines_cmd = axes[6].plot(t, tensions_cmd, linestyle='-')
    # lines_gt  = axes[6].plot(t, tensions_gt, 'k:')
    # handles = [lines_cmd[0], lines_gt[0]]
    # labels  = ['cmd', 'truth']
    # axes[6].legend(handles, labels)
    axes[6].set_ylabel('tendon tensions (N)')
    
    axes[7].plot(t, np.linalg.norm(tip_force_gt, axis=1), color='purple')
    axes[7].set_ylabel('force magnitude (N)')
    axes[7].set_xlabel('time (sec)')

    fig.align_ylabels()
    plt.tight_layout()
    
    plt.savefig("figures/tip_force_sim_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    sim_time = 30
    do_plot = True
    save_frames = True

    simulation(sim_time, do_plot, save_frames)
