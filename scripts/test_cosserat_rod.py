import numpy as np

import crest_sparse

config = crest_sparse.CosseratRodConfig()
config.rod_length = 0.5
config.num_backbone_nodes = 10
config.k_bending = 1.0
config.k_torsion = 1.0
config.k_shear = 1.0
config.k_extension = 1.0
config.sigma_twist_position = 1.0e-3
config.sigma_twist_rotation = 1.0e-3
config.sigma_stress_force = 1.0e-3
config.sigma_stress_moment = 1.0e-3
config.sigma_small_force = 1.0e-3
config.sigma_small_moment = 1.0e-3
config.sigma_base_position = 1.0e-3
config.sigma_base_rotation = 1.03-3

solver = crest_sparse.BasicCosseratSolver(config)

tip_force = np.array([0.0, 0.0, 0.0])

solution = solver.solve(tip_force)

print(solution.pose_mean)