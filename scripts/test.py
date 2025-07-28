import numpy as np
import time
from tendon_robot import TendonRobotGtsam, TendonRobotGtsamConfig, RoutingAngleFunction, RoutingParams

from plotting import TendonRobotPlotter

def simulate_tip_force(tension_meas_std, p_meas_std):
    cfg = TendonRobotGtsamConfig()

    cfg.num_discs = 9
    cfg.poses_between_discs = 3
    cfg.rod_length = 0.25
    cfg.rod_diameter = 0.0012
    cfg.youngs_modulus = 40.0e9
    cfg.shear_modulus = 15.0e9 

    cfg.tension_std = 1e-3  # Small uncertainties for simulation
    cfg.tip_force_std = 1e-3
    cfg.small_force_std = 1e-3
    cfg.small_moment_std = 1e-4
    cfg.cosserat_twist_r_std = 1e-2
    cfg.small_r_std = 1e-3
    cfg.small_p_std = 1e-5

    cfg.pose_drift_p_std = 5e-3
    cfg.pose_drift_r_std = 1e-1
    cfg.tension_drift_std = 1e-1
    cfg.wrench_drift_std = 1e-1

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
    tensions = np.zeros(4)
    tip_force = np.zeros(3)
    solution = solver.solve(tensions, tip_force, num_samples)
    print(f"elapsed time: {solution.total_time_ms:.2f} ms")

    plotter = TendonRobotPlotter(solution)

    base_time = time.time()

    tensions_meas = []
    p_meas = []

    for i in range(1000):
        current_time = time.time() - base_time

        direction_rate_hz = 0.02
        direction = np.array([
            np.sin(2 * np.pi * direction_rate_hz * 1.0 * current_time),
            np.sin(2 * np.pi * direction_rate_hz * 1.1 * current_time),
            np.sin(2 * np.pi * direction_rate_hz * 1.2 * current_time),
        ])

        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])  # default direction if zero
        else:
            direction /= norm

        max_magnitude = 0.1
        force_rate_hz = 0.3
        magnitude = max_magnitude * (0.5 * (1.0 + np.sin(2 * np.pi * force_rate_hz * current_time))) ** 4

        tip_force = magnitude * direction

        # max_tensions = np.array([5.0, 2.0, 2.0, 2.0])
        max_tensions = np.zeros(4)
        tension_rate_hz = 0.02
        tensions_rate_hz = np.array([1.0 * tension_rate_hz, 1.1 * tension_rate_hz, 1.2 * tension_rate_hz, 1.3 * tension_rate_hz])
        tensions = 0.5 * (1.0 + np.sin(2 * np.pi * tensions_rate_hz * current_time))
        tensions *= max_tensions
        # tensions = np.array([5.0, 2.0, 1.0, 1.0])
        solution = solver.solve(tensions, tip_force, num_samples)

        python_time = time.time() - base_time - current_time
        start_render = time.time() - base_time

        plotter.update(solution)

        render_time = time.time() - base_time - start_render
        total_time = time.time() - base_time - current_time

        print(f"cpp solve time: {solution.total_time_ms:.2f} ms")
        print(f"python solve time: {1000 * python_time:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        tensions_meas.append(tensions + tension_meas_std * np.random.randn(4))
        p_meas.append(solution.backbone_pose_samples[0][-1][:3,3] + p_meas_std * np.random.randn(3))


if __name__ == "__main__":
    tension_meas_std = 0.1
    p_meas_std = 1e-3

    simulate_tip_force(tension_meas_std, p_meas_std)
