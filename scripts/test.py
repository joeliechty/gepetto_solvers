import numpy as np
import time
from tendon_robot import TendonRobotGtsam, TendonRobotGtsamConfig, RoutingAngleFunction, RoutingParams

from plotting import RobotPlotter

cfg = TendonRobotGtsamConfig()

# cfg.num_discs = 9
# cfg.poses_between_discs = 2
# cfg.rod_length = 250.0  # mm
# cfg.rod_diameter = 1.2  # mm
# cfg.youngs_modulus = 40.0e9 / 1e6 
# cfg.shear_modulus = 15.0e9 / 1e6 

# cfg.tension_std = 1e-2
# cfg.small_force_std = 1e-3
# cfg.small_moment_std = 1e-1
# cfg.cosserat_twist_r_std = 1e-2
# cfg.small_r_std = 1e-3
# cfg.small_p_std = 1e-2
# cfg.tip_force_std = 1e-3

# cfg.routing_radius = 10.0


cfg.num_discs = 9
cfg.poses_between_discs = 3
cfg.rod_length = 0.25
cfg.rod_diameter = 0.0012
cfg.youngs_modulus = 40.0e9
cfg.shear_modulus = 15.0e9 

cfg.tension_std = 1e-1
cfg.small_force_std = 1e-3
cfg.small_moment_std = 1e-4
cfg.cosserat_twist_r_std = 1e-2
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

num_samples = 10
tip_wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

tensions = np.array([0.0, 0.0, 0.0, 0.0])
solution = solver.solve(tensions, tip_wrench, num_samples)
print(f"elapsed time: {solution.total_time_ms:.2f} ms")

plotter = RobotPlotter(solution)

for i in range(100):
    start = time.time()

    tensions = tensions + np.array([0.05, 0.02, 0.04, 0.01])
    # tensions = np.array([3.0, 2.0, 1.0, 1.0])
    solution = solver.solve(tensions, tip_wrench, num_samples)

    python_time = time.time() - start
    start_render = time.time()

    plotter.update(solution)

    render_time = time.time() - start_render
    total_time = time.time() - start

    print(f"cpp solve time: {solution.total_time_ms:.2f} ms")
    print(f"python solve time: {1000 * python_time:.2f} ms")
    print(f"render time: {1000 * render_time:.2f} ms")
    print(f"total time: {1000 * total_time:.2f} ms\n\n")

    # time.sleep(0.5)
