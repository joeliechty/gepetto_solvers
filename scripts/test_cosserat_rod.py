import time
import numpy as np

import crest_sparse
from plotting import CosseratRodPlotter


config = crest_sparse.CosseratRodConfig()
config.rod_length = 0.5
config.num_nodes = 15
config.k_bending = 0.1
config.k_torsion = 0.1
config.k_shear = 1e3
config.k_extension = 1e3
config.sigma_twist_pos = 1.0e-3
config.sigma_twist_rot = 1.0e-3
config.sigma_small_force = 1.0e-3
config.sigma_small_moment = 1.0e-3
config.sigma_base_pose_pos = 1.0e-3
config.sigma_base_pose_rot = 1.0e-3

solver = crest_sparse.BasicCosseratSolver(config)
plotter = CosseratRodPlotter(camera_distance=2.0, camera_focal_point=np.array([0, 0, 0.3]))

t0 = time.time()

while(True):
    t = time.time() - t0

    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.1 * t)])
    f_xy = 0.5 * xy_dir
    f_z = 3 * np.sin(0.3 * t)

    tip_force_mean = np.hstack((f_xy, f_z))

    sigma_amplitude = 0.1
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_force_cov = sigma ** 2 * np.eye(3)

    solution = solver.solve(tip_force_mean, tip_force_cov, None, None)

    plotter.update(solution)
