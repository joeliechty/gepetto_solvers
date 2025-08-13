import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_sim_config, get_base_config
from utils import TipForceFunction, moving_savgol, setup_plt


def trefoil_knot(t, f_hz=0.05):
    R = 0.03
    omega = 2 * np.pi * f_hz
    tau = omega * t
    x = R * (np.sin(tau) + 2 * np.sin(2 * tau))
    y = R * (np.cos(tau) - 2 * np.cos(2 * tau))
    z = -R * np.sin(3 * tau) / 2
    return np.array([x, y, z + 0.15])

    
def simulation(sim_time, save_frames_mode, frame_rate=30):
    simulator = TipForceSolver(get_sim_config())
    config = get_base_config()
    solver_inference = TipForceSolver(config)
    solver_jacobian = TipForceSolver(config)

    num_steps = sim_time * frame_rate

    damping = 5e-2
    tensions_min = np.array([0.1, 0.1, 0.1, 0.1])
    trajectory_rate_hz = 0.1
    force_magnitude = 0.2

    tensions_cmd = tensions_min
    t_trajectory = np.linspace(0, 1 / trajectory_rate_hz, 300)
    desired_trajectory = [trefoil_knot(t, trajectory_rate_hz) for t in t_trajectory]
    plotter = TendonRobotPlotter('tip_force_sim', desired_trajectory=desired_trajectory, save_frames_mode=save_frames_mode)
    
    p_meas_filter = moving_savgol()

    tip_position_gt = []
    tip_position_desired = []
    tip_force_gt = []
    tip_force_estimated = []
    tip_force_std = []
    tensions_cmd_all = []
    t_all = []

    tip_force_function = TipForceFunction(max_magnitude=force_magnitude)

    for i in range(num_steps):
        t = float(i) / float(frame_rate)
        
        f_gt = tip_force_function(t)  # Define in global frame

        tensions_gt = tensions_cmd + config.tension_meas_std * np.random.randn(*tensions_cmd.shape)
        solution_gt = simulator.simulation_step(tensions_gt, f_gt)

        p_gt = solution_gt.backbone_pose_mean[-1][:3,3]
        p_meas = p_gt + config.tip_position_meas_std * np.random.randn(*p_gt.shape)

        p_meas_filtered = p_meas_filter.update(p_meas)

        solution_inference = solver_inference.step(tensions_cmd, p_meas_filtered, 1)
        tip_force = solution_inference.applied_wrench_mean[-1][3:]
        tip_force_cov = solution_inference.applied_wrench_cov[-1][3:,3:]

        solution_jacobian = solver_jacobian.simulation_step(tensions_cmd, tip_force)
        
        J_position = solution_jacobian.J_pose_tensions[3:]
        p = solution_inference.backbone_pose_mean[-1][:3,3]
        R = solution_inference.backbone_pose_mean[-1][:3,:3]
        
        p_desired = trefoil_knot(t, trajectory_rate_hz)
        p_error = R.T @ (p_desired - p)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_cmd = tensions_cmd + d_tensions
        tensions_cmd = np.maximum(tensions_cmd, tensions_min)

        plotter.update(solution_inference, p_desired=p_desired, tip_force_gt=f_gt)

        tip_position_gt.append(p_gt)
        tip_position_desired.append(p_desired)
        tip_force_gt.append(f_gt)
        tip_force_estimated.append(tip_force)
        tip_force_std.append(np.sqrt(np.diag(tip_force_cov)))
        tensions_cmd_all.append(tensions_cmd)
        t_all.append(t)
    
    plotter.plotter.close()

    tip_position_gt = np.array(tip_position_gt)
    tip_position_desired = np.array(tip_position_desired)
    tip_force_gt = np.array(tip_force_gt)
    tip_force_estimated = np.array(tip_force_estimated)
    tip_force_std = np.array(tip_force_std)
    tensions_cmd_all = np.array(tensions_cmd_all)
    t = np.array(t_all)

    color_cycle = ['r', 'g', 'b', 'c']


    setup_plt(height=9.0, grid=True)

    fig, axes = plt.subplots(6, 1, sharex=True)

    for ii in range(3):
        axes[0].plot(t, tip_position_gt[:, ii], linestyle='--', color=color_cycle[ii])
        axes[0].plot(t, tip_position_desired[:, ii], color=color_cycle[ii])

    axes[0].set_ylabel('tip position (m)')
    
    force_labels = [r'tip force-$x$ (N)',
                    r'tip force-$y$ (N)',
                    r'tip force-$z$ (N)']
    force_ylim = (
        min(tip_force_gt.min(), tip_force_estimated.min()) - 0.05,
        max(tip_force_gt.max(), tip_force_estimated.max()) + 0.05
    )

    for ii, ax in enumerate(axes[1:4]):
        ax.plot(t, tip_force_gt[:,ii], 'k--')
        ax.plot(t, tip_force_estimated[:,ii], color=color_cycle[ii])
        ax.fill_between(t, tip_force_estimated[:,ii] - 2 * tip_force_std[:,ii], tip_force_estimated[:,ii] + 2 * tip_force_std[:,ii], alpha=0.2, color=color_cycle[ii], interpolate=True)
        ax.set_ylabel(force_labels[ii])
        ax.set_ylim(force_ylim)
    
    axes[4].plot(t, tensions_cmd_all)
    axes[4].set_ylabel('tensions (N)')
    
    axes[5].plot(t, np.linalg.norm(tip_force_gt, axis=1))
    axes[5].set_ylabel('force magnitude (N)')
    axes[5].set_xlabel('time (sec)')

    fig.align_ylabels()
    plt.tight_layout()
    
    plt.savefig("figures/tip_force_sim.pdf", bbox_inches="tight")


if __name__ == "__main__":
    sim_time = 5
    save_frames_mode = False

    simulation(sim_time, save_frames_mode)
