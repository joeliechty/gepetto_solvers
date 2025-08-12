import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import time
from tendon_robot import DistLoadSolver, TipForceSolver

from plotting import TendonRobotPlotter
from config import get_simulation_config, get_base_config
from utils import moving_savgol


def get_single_contact_force(T, cylinder):
    backbone_points = T[:, :3, 3]

    c = cylinder['center']
    z = cylinder['z'] / np.linalg.norm(cylinder['z'])
    v = backbone_points - c
    axis_projections = np.dot(v, z)
    closest_axis_points = c + np.outer(axis_projections, z)
    radial_vectors = backbone_points - closest_axis_points
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radial_dirs = np.zeros_like(radial_vectors)
    nonzero_mask = radial_distances > 1e-12
    radial_dirs[nonzero_mask] = radial_vectors[nonzero_mask] / radial_distances[nonzero_mask, None]

    k_contact = 10.0
    sigma = 0.004
    mag = np.exp(-np.abs(radial_distances) / sigma)
    forces_world = k_contact * mag[:, None] * radial_dirs

    forces_local = np.empty_like(forces_world)
    for i in range(len(T)):
        R = T[i, :3, :3]
        forces_local[i] = R.T @ forces_world[i]

    return forces_local[1:]


def two_point_trajectory(t, total_time=10.0):
    start_point = np.array([0.05, 0.07, 0.2])
    end_point = np.array([0.12, -0.05, 0.15])
    alpha = np.clip(t / total_time, 0.0, 1.0)  # 0 → 1 over total_time
    return (1 - alpha) * np.array(start_point) + alpha * np.array(end_point)


def generate_trajectory(sim_time, frame_rate=30):
    simulator = TipForceSolver(get_simulation_config())

    num_steps = sim_time * frame_rate

    damping = 5e-2
    tensions_min = np.array([0.1, 0.1, 0.1, 0.1])

    tensions = tensions_min
    tensions_all = []

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        solution = simulator.simulation_step(tensions, np.zeros(3))

        J_position = solution.J_pose_tensions[3:]
        p = solution.backbone_pose_mean[-1][:3,3]
        R = solution.backbone_pose_mean[-1][:3,:3]
        
        p_desired = two_point_trajectory(t, sim_time)
        p_error = R.T @ (p_desired - p)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions = tensions + d_tensions
        tensions = np.maximum(tensions, tensions_min)

        tensions_all.append(tensions)

    return tensions_all


def simulation(tensions_cmd, save_frames_mode, frame_rate=60):
    simulator = DistLoadSolver(get_simulation_config())
    config = get_base_config()
    solver_inference = DistLoadSolver(config)

    cylinders = [
        {'radius': 0.03, 'center': np.array([0.1, 0.04, 0.13]), 'z': np.array([1.0, 1.0, 0.0]), 'length': 0.075}
    ]

    plotter = TendonRobotPlotter('Distributed Load Simulation', save_frames_mode=save_frames_mode, cylinders=cylinders, plot_dist_load=True, d_azimuth=0.5)

    num_poses = config.num_discs + (config.num_discs - 1) * config.poses_between_discs
    forces_gt = np.zeros((num_poses - 1, 3))

    forces_filter = moving_savgol(window_size=30, poly_order=0)
    fbg_signals_filter = moving_savgol()


    for tensions in tensions_cmd:
        tensions_gt = tensions + config.tension_meas_std * np.random.randn(*tensions.shape)
        solution_gt = simulator.step_simulation(tensions_gt, forces_gt)

        forces_noisy = get_single_contact_force(np.array(solution_gt.backbone_pose_mean), cylinders[-1])
        forces_gt = forces_filter.update(forces_noisy)

        fbg_signals_gt = np.array(solution_gt.fbg_array_samples[-1])
        fbg_signals_meas = fbg_signals_gt + config.fbg_strain_meas_std * np.random.randn(*fbg_signals_gt.shape)
        fbg_signals_filtered = fbg_signals_filter.update(fbg_signals_meas)

        solution = solver_inference.step(tensions, fbg_signals_filtered, 1)

        force_mean = np.array(solution.applied_wrench_mean)[:,3:]
        force_cov = np.array(solution.applied_wrench_cov)[:,3:,3:]

        confidence = 0.95
        threshold = chi2.ppf(confidence, df=3)

        max_mahol = -np.inf
        max_mahol_idx = None

        for i, (mu, Sigma) in enumerate(zip(force_mean, force_cov)):
            mahol = mu.T @ np.linalg.inv(Sigma) @ mu
            if mahol > max_mahol:
                max_mahol = mahol
                max_mahol_idx = i

        if max_mahol > threshold:
            detected_location = solution.backbone_pose_mean[max_mahol_idx][:3,3]

            s = np.linspace(0, config.rod_length, len(forces_gt))
            force_two_std = 2 * np.sqrt(np.diagonal(force_cov, axis1=1, axis2=2))

            plt.figure()

            plt.subplot(1,2,1)
            plt.plot(s, forces_gt[:,0], 'g-')
            plt.plot(s, force_mean[:,0], 'b-')
            plt.fill_between(s, force_mean[:,0] - force_two_std[:,0], force_mean[:,0] + force_two_std[:,0], alpha=0.25, interpolate=True)

            plt.subplot(1,2,2)
            plt.plot(s, forces_gt[:,1], 'g-')
            plt.plot(s, force_mean[:,1], 'b-')
            plt.fill_between(s, force_mean[:,1] - force_two_std[:,1], force_mean[:,1] + force_two_std[:,1], alpha=0.25, interpolate=True)

            plt.show()

        else:
            detected_location = None

        plotter.update(solution_gt, p_detected=detected_location)
    
    plotter.plotter.close()



if __name__ == "__main__":
    sim_time = 10
    save_frames_mode = False

    tensions_cmd = generate_trajectory(sim_time)
    tensions_gt, fbg_signals_gt, dist_load_gt = simulation(tensions_cmd, save_frames_mode)