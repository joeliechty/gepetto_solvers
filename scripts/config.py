import numpy as np

from tendon_robot import TendonRobotGtsamConfig, RoutingAngleFunction, RoutingParams


def get_base_config():
    config = TendonRobotGtsamConfig()

    config.num_discs = 9
    config.poses_between_discs = 3
    config.rod_length = 0.25
    config.rod_diameter = 0.0012
    config.youngs_modulus = 40.0e9
    config.shear_modulus = 15.0e9 

    config.small_force_std = 1e-3
    config.small_moment_std = 1e-4
    config.cosserat_twist_r_std = 1e-2
    config.small_r_std = 1e-3
    config.small_p_std = 1e-5

    config.routing_radius = 0.01

    config.tip_pose_r_meas_std = 3e0
    config.tip_pose_p_meas_std = 1e-3

    config.angle_functions = [
        RoutingAngleFunction.LINEAR,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT,
        RoutingAngleFunction.CONSTANT
    ]

    config.angle_params = [
        RoutingParams(angle_offset=0.0,           total_angle=2 * np.pi),
        RoutingParams(angle_offset=np.pi,         total_angle=0.0),
        RoutingParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
        RoutingParams(angle_offset=0.0,           total_angle=0.0)
    ]

    return config


def get_simulation_config():
    config = get_base_config()

    config.tension_std = 1e-3
    config.tip_force_std = 1e-3

    # We dont want any lag in sim, so make drift big
    config.pose_drift_p_std = 1e0
    config.pose_drift_r_std = 1e0
    config.tension_drift_std = 1e0
    config.wrench_drift_std = 1e0

    return config


def get_sensing_config():
    config = get_base_config()

    config.tension_std = 1e-2
    config.tip_force_std = 1e-1

    config.pose_drift_p_std = 1e-3
    config.pose_drift_r_std = 1e0
    config.tension_drift_std = 1e-1
    config.wrench_drift_std = 1e-2

    return config
















def get_tip_force_sensing_config():
    config = get_base_config()

    config.tension_std = 1e-1
    config.tip_force_std = 1e-1

    config.pose_drift_p_std = 5e-3
    config.pose_drift_r_std = 1e-1
    config.tension_drift_std = 1e-1
    config.wrench_drift_std = 1e-2

    return config