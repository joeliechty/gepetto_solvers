import numpy as np

import crest_sparse


def get_base_config():
    r = 0.0015 / 2
    I = 0.25 * np.pi * r ** 4
    A = np.pi * r ** 2
    J = 2 * I
    E = 207.0e9
    G = 79.3e9
    
    K_inv = np.diag([
        1 / (E * I), 
        1 / (E * I),
        1 / (J * G),
        1 / (G * A),
        1 / (G * A),
        1 / (E * A)
    ])

    config = crest_sparse.MultiRobotSolverConfig()

    config.base.use_dense = False
    config.nodes_per_rod = 15
    config.K_inv = K_inv

    config.sigma_twist_pos = 1.0e-4
    config.sigma_twist_rot = 1.0e-3
    config.sigma_small_force = 1.0e-3
    config.sigma_small_moment = 1.0e-3

    config.sigma_snare_rot_x = 1e-1
    config.sigma_snare_rot_y = 1e-1
    config.sigma_snare_rot_z = 1e-1
    config.sigma_snare_location = 1e-3
    config.snare_distance_to_tip = 0.25
    
    config.sigma_rod_lengths = 1e-3
    config.sigma_base_rot = 1e-2

    return config