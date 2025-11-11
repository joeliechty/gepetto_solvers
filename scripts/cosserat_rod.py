import time
import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from plotting import CosseratRodPlotter


def get_tip_force_prior(t):
    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.1 * t)])
    f_xy = 0.5 * xy_dir
    f_z = 3 * np.sin(0.3 * t)

    tip_wrench_mean = np.hstack((np.zeros(3), f_xy, f_z))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.1
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_wrench_cov[3:,3:] = sigma ** 2 * np.eye(3)

    return tip_wrench_mean, tip_wrench_cov, None, None


def get_tip_wrench_prior(t):
    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.1 * t)])
    f_xy = 1.0 * xy_dir
    f_z = 2.0 * np.sin(0.3 * t)

    tip_wrench_mean = np.hstack((f_xy, f_z, np.zeros(3)))

    tip_wrench_cov = 1e-6 * np.eye(6)

    sigma_amplitude = 0.1
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_wrench_cov[:3,:3] = sigma ** 2 * np.eye(3)

    return tip_wrench_mean, tip_wrench_cov, None, None


def get_tip_position_prior(t):
    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.1 * t)])
    p_xy = 0.3 * xy_dir
    p_z = 0.4 + 0.1 * np.sin(0.3 * t)

    tip_pose_mean = np.eye(4)
    tip_pose_mean[:3,3] = np.hstack((p_xy, p_z))

    tip_pose_cov = 1e-6 * np.eye(6)
    tip_pose_cov[:3,:3] = 10 * np.eye(3) # Allow rotation to vary

    tip_wrench_mean = np.zeros(6)

    tip_wrench_cov = 1e-6 * np.eye(6)
    tip_wrench_cov[3:,3:] = 10 * np.eye(3) # Allow tip force to vary

    return tip_wrench_mean, tip_wrench_cov, tip_pose_mean, tip_pose_cov


def get_tip_pose_prior(t):
    x = 0.15
    yz = 0.1 * np.array([np.cos(0.2 * t), np.sin(0.2 * t)])
    yz[1] += 0.25
    p = np.hstack((x, yz))

    r0 = np.array([0, np.pi / 6, 0])
    R0 = Rotation.from_rotvec(r0).as_matrix()
    dr = np.pi / 4 * np.array([np.sin(0.2 * t), np.sin(0.21 * t), np.sin(0.22 * t)])
    dR = Rotation.from_rotvec(dr).as_matrix()
    R = R0 @ dR

    tip_pose_mean = np.eye(4)
    tip_pose_mean[:3,:3] = R
    tip_pose_mean[:3,3] = p

    tip_pose_cov = 1e-6 * np.eye(6)

    return None, None, tip_pose_mean, tip_pose_cov


def main():
    # input_getter = get_tip_force_prior
    # input_getter = get_tip_wrench_prior
    # input_getter = get_tip_position_prior
    prior_getter = get_tip_pose_prior

    config = crest_sparse.CosseratRodConfig()
    config.rod_length = 0.5
    config.num_nodes = 18
    config.k_bending = 0.1
    config.k_torsion = 0.1
    config.k_shear = 1e2
    config.k_extension = 1e2
    config.sigma_twist_pos = 1.0e-3
    config.sigma_twist_rot = 1.0e-3
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3
    config.sigma_base_pose_pos = 1.0e-3
    config.sigma_base_pose_rot = 1.0e-3

    solver = crest_sparse.BasicCosseratSolver(config)
    plotter = CosseratRodPlotter(plot_wrenches=False, plot_backbone_frames=True, plot_tip_plate=True, camera_azimuth=45, camera_distance=1.5, camera_focal_point=np.array([0, 0, 0.3]))

    t0 = time.time()

    while(True):
        t = time.time() - t0

        solution = solver.solve(*prior_getter(t))
        plotter.update(solution)

if __name__ == "__main__":
    main()