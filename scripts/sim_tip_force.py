import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_sim_config, get_base_config
from utils import TipForceFunction, moving_savgol, setup_plt, generate_waypoint_trajectory


def simulation(sim_time, do_plot, save_frames):
    simulator_sim = TipForceSolver(get_sim_config())
    simulator_nominal = TipForceSolver(get_sim_config())

    config = get_base_config()
    solver_inference = TipForceSolver(config)
    solver_jacobian = TipForceSolver(config)

    t, positions, tensions_nominal, waypoints = generate_waypoint_trajectory(sim_time, seed=42)
    tip_force_function = TipForceFunction(max_magnitude=2 * config.tip_force_prior_std, seed=42)

    damping = 6e-2
    tensions_min = np.array([0.1, 0.1, 0.1, 0.1])

    tensions_cmd = tensions_min
    plotter_sim = TendonRobotPlotter('tip_force_sim', save_frames_mode=save_frames)
    plotter_nominal = TendonRobotPlotter('tip_force_nominal', save_frames_mode=save_frames)

    p_meas_filter = moving_savgol()

    tip_position_sim = []
    tip_position_nominal = []
    tip_position_desired = []
    tip_force_gt = []
    tip_force_estimated = []
    tip_force_std = []
    tensions_cmd_all = []

    for t_i, p_desired, tensions_nominal_i in zip(t, positions, tensions_nominal):
        f_gt = tip_force_function(t_i)

        tension_noise = config.tension_meas_std * np.random.randn(*tensions_cmd.shape)
        solution_sim = simulator_sim.simulation_step(tensions_cmd + tension_noise, f_gt)
        solution_nominal = simulator_nominal.simulation_step(tensions_nominal_i + tension_noise, f_gt)

        p_gt = solution_sim.backbone_pose_mean[-1][:3,3]
        p_meas = p_gt + config.tip_position_meas_std * np.random.randn(*p_gt.shape)

        p_meas_filtered = p_meas_filter.update(p_meas)

        solution_inference = solver_inference.step(tensions_cmd, p_meas_filtered, 1)
        tip_force = solution_inference.applied_wrench_mean[-1][3:]
        tip_force_cov = solution_inference.applied_wrench_cov[-1][3:,3:]

        solution_jacobian = solver_jacobian.simulation_step(tensions_cmd, tip_force)
        
        J_position = solution_jacobian.J_pose_tensions[3:]
        p = solution_inference.backbone_pose_mean[-1][:3,3]
        R = solution_inference.backbone_pose_mean[-1][:3,:3]
        
        p_error = R.T @ (p_desired - p)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_cmd = tensions_cmd + d_tensions
        tensions_cmd = np.maximum(tensions_cmd, tensions_min)

        if do_plot:
            plotter_sim.update(solution_inference, p_desired=p_desired, tip_force_gt=f_gt)
            plotter_nominal.update(solution_nominal, p_desired=p_desired, tip_force_gt=f_gt)

        tip_position_sim.append(p_gt)
        tip_position_nominal.append(solution_nominal.backbone_pose_mean[-1][:3,3])
        tip_position_desired.append(p_desired)
        tip_force_gt.append(f_gt)
        tip_force_estimated.append(tip_force)
        tip_force_std.append(np.sqrt(np.diag(tip_force_cov)))
        tensions_cmd_all.append(tensions_cmd)
    
    plotter_sim.plotter.close()
    plotter_nominal.plotter.close()

    tip_position_sim = np.array(tip_position_sim)
    tip_position_nominal = np.array(tip_position_nominal)
    tip_position_desired = np.array(tip_position_desired)
    tip_force_gt = np.array(tip_force_gt)
    tip_force_estimated = np.array(tip_force_estimated)
    tip_force_std = np.array(tip_force_std)
    tensions_cmd_all = np.array(tensions_cmd_all)

    color_cycle = ['r', 'g', 'b', 'c']

    setup_plt(height=9.0, grid=True)

    fig, axes = plt.subplots(8, 1, sharex=True)

    position_labels = [r'position-$x$ (mm)',
                       r'position-$y$ (mm)',
                       r'position-$z$ (mm)']
    
    for ii, ax in enumerate(axes[0:3]):
        ax.plot(t, 1000 * tip_position_sim[:, ii], linestyle='-', color=color_cycle[ii], label='tracking')
        ax.plot(t, 1000 * tip_position_nominal[:, ii], linestyle='--', color=color_cycle[ii], label='open loop')
        ax.plot(t, 1000 * tip_position_desired[:, ii], linestyle=':', color='k', label='desired')
        if ii == 0:
            ax.legend()
        ax.set_ylabel(position_labels[ii])
    
    force_labels = [r'force-$x$ (N)',
                    r'force-$y$ (N)',
                    r'force-$z$ (N)']

    for ii, ax in enumerate(axes[3:6]):
        ax.plot(t, tip_force_gt[:,ii], 'k--')
        ax.plot(t, tip_force_estimated[:,ii], color=color_cycle[ii])
        ax.fill_between(t, tip_force_estimated[:,ii] - 2 * tip_force_std[:,ii], tip_force_estimated[:,ii] + 2 * tip_force_std[:,ii], alpha=0.2, color=color_cycle[ii], interpolate=True)
        ax.set_ylabel(force_labels[ii])
    
    axes[6].plot(t, tensions_cmd_all)
    axes[6].set_ylabel('tensions (N)')
    
    axes[7].plot(t, np.linalg.norm(tip_force_gt, axis=1))
    axes[7].set_ylabel('force magnitude (N)')
    axes[7].set_xlabel('time (sec)')

    fig.align_ylabels()
    plt.tight_layout()
    
    plt.savefig("figures/tip_force_sim.pdf", bbox_inches="tight")


if __name__ == "__main__":
    sim_time = 30
    do_plot = True
    save_frames = True

    simulation(sim_time, do_plot, save_frames)
