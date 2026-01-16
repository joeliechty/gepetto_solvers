import numpy as np

from crest_sparse import TendonRobotSolverConfig, TendonInput, RoutingAngleFunction, RoutingFunctionParams


def get_sim_K_inv():
    rod_diameter = 0.0012
    youngs_modulus = 40.0e9
    shear_modulus = 15.0e9 
    
    I = (np.pi * rod_diameter**4) / 64.0
    J = 2 * I
    A = (np.pi * rod_diameter**2) / 4.0

    k_bending = youngs_modulus * I
    k_torsion = shear_modulus * J
    k_shear = shear_modulus * A
    k_extension = youngs_modulus * A

    K_inv = np.eye(6)
    K_inv[0,0] = 1 / k_bending
    K_inv[1,1] = 1 / k_bending
    K_inv[2,2] = 1 / k_torsion
    K_inv[3,3] = 1 / k_shear
    K_inv[4,4] = 1 / k_shear
    K_inv[5,5] = 1 / k_extension

    return K_inv


def get_sim_tendon_input():
    tendon_input = TendonInput()

    tendon_input.routing_radius = 0.01

    tendon_input.functions = [
        RoutingAngleFunction.LINEAR,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT
    ]

    tendon_input.params = [
        RoutingFunctionParams(angle_offset=0.0,           total_angle=2 * np.pi),
        RoutingFunctionParams(angle_offset=np.pi,         total_angle=0.0),
        RoutingFunctionParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
        RoutingFunctionParams(angle_offset=0.0,           total_angle=0.0)
    ]

    return tendon_input


def get_base_config():
    config = TendonRobotSolverConfig()

    config.rod_length = 0.25
    config.num_discs = 9
    config.num_between_nodes = 3
    config.K_inv = get_sim_K_inv()
    config.sigma_twist_rot = 1.0e-3
    config.sigma_twist_pos = 1.0e-4
    config.sigma_stress_force = 1.0e-4
    config.sigma_stress_moment = 1.0e-5
    config.sigma_base_pos = 1.0e-4
    config.sigma_base_rot = 1.0e-3
    config.tendon_input = get_sim_tendon_input()
    
    # config.use_midpoint = True
    # config.tip_force_prior_std = 1e-1
    # config.dist_load_prior_std = 2e-2
    # config.dist_load_smoothness_std = 1e-2
    # config.tension_meas_std = 1e-2
    # config.tip_position_meas_std = 1e-3
    # config.fbg_strain_meas_std = 3e-6
    # config.tension_drift_std = 1e-2
    # config.tip_force_drift_std = 1e-1
    # config.dist_load_drift_std = 5e-3

    return config


def get_sim_config():
    config = get_base_config()

    config.tension_meas_std = 1e-4
    config.cosserat_twist_r_std = 1e-4

    return config
