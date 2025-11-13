import numpy as np
from scipy.spatial.transform import Rotation

import crest_sparse
from plotting import ParallelRobotPlotter


def get_base_poses():
    angles = np.array(np.deg2rad([10, 110, 130, 230, 250, 350]))

    radius = 0.3
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    poses = []
    for xi, yi in zip(xs, ys):
        pose = np.eye(4)
        pose[0, 3] = xi
        pose[1, 3] = yi
        poses.append(pose)

    return poses


def get_tip_poses():
    angles = np.array(np.deg2rad([50, 70, 170, 190, 290, 310]))

    radius = 0.15
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    
    poses = []
    for xi, yi in zip(xs, ys):
        pose = np.eye(4)
        pose[0, 3] = xi
        pose[1, 3] = yi
        poses.append(pose)

    return poses


def get_rod_lengths(t):
    rod_lengths = np.ones(6) + 0.1 * np.sin(np.array([1.0 * t, 1.1 * t, 1.2 * t, 1.3 * t, 1.4 * t, 1.5 * t]))

    sigma_amplitude = 0.01
    sigma_rod_length = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)

    return rod_lengths, sigma_rod_length


def main():

    k_bending = 0.1
    k_torsion = 0.1
    k_shear = 1e2
    k_extension = 1e2

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    config = crest_sparse.ParallelRobotSolverConfig()

    config.nodes_per_rod = 10
    config.K_inv = K_inv
    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-2
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3
    config.base_end_poses = get_base_poses()
    config.tip_end_poses = get_tip_poses()
    config.sigma_end_pose_pos= 1.0e-4
    config.sigma_end_pose_rot= 1.0e-3

    solver = crest_sparse.ParallelRobotSolver(config)

    plotter = ParallelRobotPlotter(
        camera_azimuth=60, 
        camera_distance=4, 
        camera_focal_point=np.array([0, 0, 0.7])
    )
    
    frame_rate = 5.0
    dt = 1.0 / frame_rate
    t_final = 1200.0
    num_steps = int(t_final / dt)

    rod_lengths = np.ones(6)

    for step in range(num_steps + 1):
        t = step * dt

        solution = solver.solve(rod_lengths, 1e-3)

        J = solution.rod_lengths_jacobian
        
        d_pose = np.zeros(6)
        d_pose[0] = 5e-3

        d_rod_lengths = np.linalg.pinv(J) @ d_pose

        rod_lengths += d_rod_lengths

        print(d_rod_lengths)

        plotter.update(solution)

        progress = 100.0 * step / num_steps
        print(f"Progress: {progress:5.1f}%", end="\r")


if __name__ == "__main__":
    main()