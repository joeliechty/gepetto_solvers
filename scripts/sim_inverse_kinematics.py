import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_simulation_config, get_base_config
from utils import tip_force_function, moving_savgol


def trefoil_knot(t, f_hz=0.05):
    R = 0.03
    omega = 2 * np.pi * f_hz
    tau = omega * t
    x = R * (np.sin(tau) + 2 * np.sin(2 * tau))
    y = R * (np.cos(tau) - 2 * np.cos(2 * tau))
    z = -R * np.sin(3 * tau) / 2
    return np.array([x, y, z + 0.15])

    
def simulation(sim_time, save_png_mode, frame_rate=30):
    simulator = TipForceSolver(get_simulation_config())
    config = get_base_config()
    solver_inference = TipForceSolver(config)
    solver_jacobian = TipForceSolver(config)

    num_steps = sim_time * frame_rate
    d_tensions_max = 2.0
    k_p = 0.8
    k_d = 0.6
    damping = 1e-6
    p_prev = np.array([0, 0, 0.25])
    p_desired_prev = np.array([0, 0, 0.25])
    tensions_cmd = np.zeros(4)
    trajectory_rate_hz = 0.1

    plotter = TendonRobotPlotter('Inverse Kinematics', save_png_mode=save_png_mode)
    
    t_trajectory = np.linspace(0, 1 / trajectory_rate_hz, 300)
    desired_trajectory = [trefoil_knot(t, trajectory_rate_hz) for t in t_trajectory]
    
    tip_position_filter = moving_savgol()

    tip_position_gt = []
    tip_position_desired = []
    tip_force_gt = []
    tip_force_estimated = []
    tip_force_std = []
    tensions_cmd_all = []
    t_all = []

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        f_gt = 0.25 * tip_force_function(t)
        tensions_gt = tensions_cmd + config.tension_meas_std * np.random.randn(*tensions_cmd.shape)
        solution_gt = simulator.simulation_step(tensions_gt, f_gt)

        p_gt = solution_gt.backbone_pose_mean[-1][:3,3]
        p_meas = p_gt + config.tip_position_meas_std * np.random.randn(*p_gt.shape)
        p_filtered = tip_position_filter.update(p_meas)

        solution_inference = solver_inference.step(tensions_cmd, p_filtered, 1)
        tip_force = solution_inference.applied_wrench_mean[-1][3:]
        tip_force_cov = solution_inference.applied_wrench_cov[-1][3:,3:]

        solution_jacobian = solver_jacobian.simulation_step(tensions_cmd, tip_force)
        
        J_position = solution_jacobian.J_pose_tensions[3:]
        p = solution_inference.backbone_pose_mean[-1][:3,3]
        R = solution_inference.backbone_pose_mean[-1][:3,:3]
        
        p_desired = trefoil_knot(t, trajectory_rate_hz)

        v_world = p - p_prev
        v = R.T @ v_world

        v_desired_world = p_desired - p_desired_prev
        v_desired = R.T @ v_desired_world

        p_error = R.T @ (p_desired - p)
        v_error = v_desired - v

        pd_error = k_p * p_error + k_d * v_error

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ pd_error
        d_tensions = np.linalg.solve(A, b)

        d_tensions_norm = np.linalg.norm(d_tensions)
        if d_tensions_norm > d_tensions_max:
            d_tensions *= d_tensions_max / d_tensions_norm

        tensions_cmd = tensions_cmd + d_tensions
        
        plotter.update(solution_inference, p_desired=p_desired, desired_trajectory=desired_trajectory, tip_force_gt=f_gt)

        p_prev = p
        p_desired_prev = p_desired

        tip_position_gt.append(p_gt)
        tip_position_desired.append(p_desired)
        tip_force_gt.append(f_gt)
        tip_force_estimated.append(tip_force)
        tip_force_std.append(np.sqrt(np.diag(tip_force_cov)))
        tensions_cmd_all.append(tensions_cmd)
        t_all.append(t)
    
    tip_position_gt = np.array(tip_position_gt)
    tip_position_desired = np.array(tip_position_desired)
    tip_force_gt = np.array(tip_force_gt)
    tip_force_estimated = np.array(tip_force_estimated)
    tip_force_std = np.array(tip_force_std)
    tensions_cmd_all = np.array(tensions_cmd_all)
    t = np.array(t_all)

    color_cycle = ['r', 'g', 'b', 'c']

    plt.figure()

    for ii in range(3):
        plt.plot(t, tip_position_gt[:, ii], linestyle=':', color=color_cycle[ii])
        plt.plot(t, tip_position_desired[:, ii], linestyle='-', color=color_cycle[ii])

    plt.ylabel('tip position (m)')

    plt.figure()

    for ii in range(3):
        plt.subplot(3, 1, ii + 1)
        plt.plot(t, tip_force_gt[:,ii], color='k')
        plt.plot(t, tip_force_estimated[:,ii], color=color_cycle[ii])
        plt.fill_between(t, tip_force_estimated[:,ii] - 2 * tip_force_std[:,ii], tip_force_estimated[:,ii] + 2 * tip_force_std[:,ii], alpha=0.2, color=color_cycle[ii], interpolate=True)

    plt.figure()
    plt.subplot(2,1,1)

    plt.plot(t, tensions_cmd_all)
    plt.ylabel('tendon tensions (N)')

    plt.subplot(2,1,2)
    
    plt.plot(t, np.linalg.norm(tip_force_gt, axis=1))
    plt.ylabel('tip force magnitude (N)')

    plt.show()

    plotter.plotter.close()


if __name__ == "__main__":
    sim_time = 30
    save_png_mode = False

    simulation(sim_time, save_png_mode)
