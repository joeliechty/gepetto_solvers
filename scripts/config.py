import numpy as np

from tendon_robot import TendonRobotConfig, RoutingAngleFunction, RoutingFunctionParams


def get_base_config():
    config = TendonRobotConfig()

    config.num_discs = 9
    config.poses_between_discs = 3
    config.rod_length = 0.25
    config.rod_diameter = 0.0012
    config.youngs_modulus = 40.0e9
    config.shear_modulus = 15.0e9 
    config.routing_radius = 0.01

    config.tension_std = 1e-2
    config.cosserat_twist_r_std = 1e-2
    config.small_force_std = 1e-5
    config.small_moment_std = 1e-5
    config.small_r_std = 1e-3
    config.small_p_std = 1e-5
    config.tension_drift_std = 1e-1
    config.tip_pose_jerk_std = 1e0
    config.tip_pose_accel_std = 1e0
    
    config.tip_force_std = 1e-1
    config.tip_force_drift_std = 1e-2
    config.dist_load_magnitude_std = 1e-1
    config.dist_load_jerk_std = 1e-3
    config.dist_load_drift_std = 1e-3

    config.tip_pose_r_meas_std = 3e0
    config.tip_pose_p_meas_std = 1e-3 #0.577e-3  # This makes for a 1mm rms position error
    config.fbg_strain_meas_std = 1e-6


    config.angle_functions = [
        RoutingAngleFunction.LINEAR,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT
    ]

    config.angle_params = [
        RoutingFunctionParams(angle_offset=0.0,           total_angle=2 * np.pi),
        RoutingFunctionParams(angle_offset=np.pi,         total_angle=0.0),
        RoutingFunctionParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
        RoutingFunctionParams(angle_offset=0.0,           total_angle=0.0)
    ]

    return config


def get_simulation_config():
    config = get_base_config()

    # config.youngs_modulus = 40.0e9 * 1.025

    config.tension_std = 1e-4
    config.tip_force_std = 1e-4

    config.cosserat_twist_r_std = 1e-4
    config.tension_drift_std = 1e0
    config.tip_pose_jerk_std = 1e0
    config.tip_pose_accel_std = 1e0

    return config
