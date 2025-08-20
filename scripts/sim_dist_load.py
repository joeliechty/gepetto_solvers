import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import time
from tendon_robot import DistLoadSolver, TipForceSolver

from plotting import TendonRobotPlotter
from config import get_sim_config, get_base_config
from utils import moving_savgol, setup_plt, generate_trajectory


def get_single_contact_force(T, cylinder, k_contact=5.0, bandwith=0.25):
    backbone_points = T[:, :3, 3]
    tangents = T[:, :3, 2]

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

    sigma = bandwith * cylinder['radius']
    mag = np.exp(-np.abs(radial_distances) / sigma)

    half_len = cylinder['length'] / 2
    inside_mask = np.abs(axis_projections) <= half_len
    mag[~inside_mask] = 0.0

    forces = k_contact * mag[:, None] * radial_dirs
    proj_forces = forces - np.sum(forces * tangents, axis=1, keepdims=True) * tangents

    return proj_forces[1:]


def two_point_trajectory(t, total_time=2.0):
    start_point = np.array([-0.05, 0.1, 0.2])
    end_point = np.array([0.15, -0.1, 0.15])
    alpha = np.clip(t / total_time, 0.0, 1.0)  # 0 → 1 over total_time
    return (1 - alpha) * np.array(start_point) + alpha * np.array(end_point)


def simulation(tensions_cmd, position_cmd, do_plot, save_frames):
    simulator = DistLoadSolver(get_sim_config())
    config = get_base_config()
    solver_inference = DistLoadSolver(config)

    cylinders = [
        {'radius': 0.02, 'center': np.array([0.09, 0.04, 0.14]), 'z': np.array([1.0, 0.5, 0.0]), 'length': 0.125},
        {'radius': 0.02, 'center': np.array([0.07, 0.01, 0.07]), 'z': np.array([1.0, 0.5, 0.0]), 'length': 0.125}
    ]
    
    plotter = TendonRobotPlotter('dist_load_sim', save_frames_mode=save_frames, cylinders=cylinders, plot_dist_load=True, d_azimuth=200 / len(tensions_cmd))

    num_poses = config.num_discs + (config.num_discs - 1) * config.poses_between_discs
    force_gt = np.zeros((num_poses - 1, 3))

    forces_filter = moving_savgol(poly_order=0)
    fbg_signals_filter = moving_savgol(poly_order=1)

    for tensions, position_desired in zip(tensions_cmd, position_cmd):
        tensions_gt = tensions + config.tension_meas_std * np.random.randn(*tensions.shape)
        solution_gt = simulator.step_simulation(tensions_gt, force_gt)

        cylinder_forces = [get_single_contact_force(np.array(solution_gt.backbone_pose_mean), cylinder) for cylinder in cylinders]
        forces_noisy = np.sum(cylinder_forces, axis=0)
        force_gt = forces_filter.update(forces_noisy)

        fbg_signals_gt = np.array(solution_gt.fbg_array_samples[-1])
        fbg_signals_meas = fbg_signals_gt + config.fbg_strain_meas_std * np.random.randn(*fbg_signals_gt.shape)
        fbg_signals_filtered = fbg_signals_filter.update(fbg_signals_meas)

        solution = solver_inference.step(tensions, fbg_signals_filtered, 1)

        if do_plot:
            plotter.update(solution, p_desired=position_desired)
    
    plotter.plotter.close()

    force_mean = []
    force_cov = []
    for ii in range(len(solution.backbone_pose_mean) - 1):
        R = solution.backbone_pose_mean[ii + 1][:3,:3]
        force_mean_spatial = np.array(solution.applied_wrench_mean)[ii,3:]
        force_cov_spatial = np.array(solution.applied_wrench_cov)[ii,3:,3:]
        force_mean.append(R.T @ force_mean_spatial)
        force_cov.append(R.T @ force_cov_spatial @ R)
        force_gt[ii] = R.T @ force_gt[ii]

    force_mean = np.array(force_mean)
    force_cov = np.array(force_cov)

    p_greater = np.zeros(len(force_mean))
    force_thresh = 0.01
    num_samples = 10000

    for i, (mu, Sigma) in enumerate(zip(force_mean, force_cov)):
        L = np.linalg.cholesky(Sigma)
        Z = np.random.standard_normal((num_samples, 3))    # ~ N(0,I)
        F = mu + Z @ L.T                          # samples from N(mu,Sigma)
        norms = np.linalg.norm(F, axis=1)
        p_greater[i] = np.mean(norms > force_thresh)

    s = np.linspace(0, config.rod_length, len(force_gt))
    force_two_std = 2 * np.sqrt(np.diagonal(force_cov, axis1=1, axis2=2))

    setup_plt(height=5, grid=True)

    fig, axes = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 2, 1.5]})

    axes[0].plot(s, force_gt[:,0], 'k--')
    axes[0].plot(s, force_mean[:,0])
    axes[0].fill_between(s, force_mean[:,0] - force_two_std[:,0], force_mean[:,0] + force_two_std[:,0], alpha=0.2, color='blue', interpolate=True)
    axes[0].set_ylabel('force-$x$ (N)')

    axes[1].plot(s, force_gt[:,1], 'k--')
    axes[1].plot(s, force_mean[:,1])
    axes[1].fill_between(s, force_mean[:,1] - force_two_std[:,1], force_mean[:,1] + force_two_std[:,1], alpha=0.2, color='blue', interpolate=True)
    axes[1].set_ylabel('force-$y$ (N)')

    axes[2].plot(s, force_gt[:,2], 'k--')
    axes[2].plot(s, force_mean[:,2])
    axes[2].fill_between(s, force_mean[:,2] - force_two_std[:,2], force_mean[:,2] + force_two_std[:,2], alpha=0.2, color='blue', interpolate=True)
    axes[2].set_ylabel('force-$z$ (N)')

    axes[3].plot(s, p_greater, 'r-')
    axes[3].axhline(0.95, color='k', linestyle='--')
    axes[3].set_ylabel(r"$p(\| \text{force} \| > \tau)$")
    axes[3].set_xlabel('arclength (m)')

    fig.align_ylabels()
    plt.tight_layout()
    
    plt.savefig("figures/dist_load_sim.pdf", bbox_inches="tight")


if __name__ == "__main__":
    sim_time = 8
    do_plot = True
    save_frames = True

    def trajectory(t):
        return two_point_trajectory(t, sim_time)

    t, position_cmd, tensions_cmd = generate_trajectory(trajectory, sim_time)
    simulation(tensions_cmd, position_cmd, do_plot, save_frames)