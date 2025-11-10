import time
import numpy as np

import crest_sparse
from plotting import CosseratRodPlotter


config = crest_sparse.CosseratRodConfig()
config.rod_length = 0.5
config.num_backbone_nodes = 20
config.k_bending = 0.1
config.k_torsion = 0.1
config.k_shear = 1e3
config.k_extension = 1e3
config.sigma_twist_position = 1.0e-3
config.sigma_twist_rotation = 1.0e-3
config.sigma_stress_force = 1.0e-3
config.sigma_stress_moment = 1.0e-3
config.sigma_small_force = 1.0e-3
config.sigma_small_moment = 1.0e-3
config.sigma_base_position = 1.0e-3
config.sigma_base_rotation = 1.03-3

solver = crest_sparse.BasicCosseratSolver(config)
plotter = CosseratRodPlotter(camera_distance=2.0, camera_focal_point=np.array([0, 0, 0.3]))

t0 = time.time()

while(True):
    t = time.time() - t0

    xy_dir = np.array([np.cos(0.1 * t), np.sin(0.1 * t)])
    f_xy = 0.5 * xy_dir
    f_z = 2 * np.sin(t)

    tip_force_mean = np.hstack((f_xy, f_z))

    sigma_amplitude = 0.2
    sigma = 1e-3 + sigma_amplitude - sigma_amplitude * np.cos(0.1 * t)
    tip_force_cov = sigma ** 2 * np.eye(3)

    solution = solver.solve(tip_force_mean, tip_force_cov)

    plotter.update(solution)
