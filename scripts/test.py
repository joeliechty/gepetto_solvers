import numpy as np
from tendon_robot import TendonRobotGtsam, TendonRobotGtsamConfig, RoutingAngleFunction, RoutingParams

from plotting import plot_robot

cfg = TendonRobotGtsamConfig()

cfg.num_discs = 9
cfg.poses_between_discs = 5
cfg.rod_length = 0.25
cfg.rod_diameter = 1.2e-3
cfg.youngs_modulus = 40.0e9
cfg.shear_modulus = 15.0e9

cfg.tension_std = 5e-2
cfg.small_force_std = 1e-4
cfg.small_moment_std = 1e-4
cfg.small_stress_std = 1e-4
cfg.cosserat_twist_r_std = 1e-1
cfg.small_r_std = 1e-3
cfg.small_p_std = 1e-5
cfg.tip_force_std = 1e-3

cfg.routing_radius = 0.01

cfg.angle_functions = [
    RoutingAngleFunction.LINEAR,
    RoutingAngleFunction.CONSTANT,
    RoutingAngleFunction.CONSTANT,
    RoutingAngleFunction.CONSTANT
]

cfg.angle_params = [
    RoutingParams(angle_offset=0.0,           total_angle=2 * np.pi),
    RoutingParams(angle_offset=np.pi,         total_angle=0.0),
    RoutingParams(angle_offset=3 * np.pi / 2, total_angle=0.0),
    RoutingParams(angle_offset=0.0,           total_angle=0.0)
]

solver = TendonRobotGtsam(cfg)


tip_wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

tensions = np.array([0.0, 0.0, 0.0, 0.0])
solution = solver.solve(tensions, tip_wrench)
# tensions = np.array([1.0, 0.2, 0.0, 0.0])
# solution = solver.solve(tensions, tip_wrench)
# tensions = np.array([2.0, 0.5, 0.0, 0.0])
# solution = solver.solve(tensions, tip_wrench)
# tensions = np.array([5.0, 1.0, 0.0, 0.0])
# solution = solver.solve(tensions, tip_wrench)

plot_robot(solution, title='Test')
