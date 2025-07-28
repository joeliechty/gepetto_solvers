import numpy as np
import time
from tendon_robot import TipForceSim, TendonRobotGtsamConfig

from plotting import TendonRobotPlotter
from config import get_simulation_config, get_sensing_config


def estimate_tip_force(tensions, tip_positions, tip_forces_gt):
    config = get_sensing_config()

    # Add noise to all measured data
    tensions_meas = tensions + config.tension_std * np.random.randn(*tensions.shape)
    tip_positions_meas = tip_positions + config.tip_position_meas_std * np.random.randn(*tip_positions.shape)

    solver = TendonRobotGtsam(config)

    for tau, p, f_gt in zip(tensions_meas, tip_positions_meas, tip_forces_gt):
        start_solve = time.time()

        solution = solver.update(tau, p, num_samples)

        solve_time = time.time() - start
        start_render = time.time()

        plotter.update(solution, f_gt)

        render_time = time.time() - start_render
        total_time = time.time() - start_solve

        print(f"solve time: {1000 * solve_time:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        # time.sleep(3.0)


def simulate_tip_force():
    solver = TipForceSim(get_simulation_config())

    num_samples = 100

    base_time = time.time()
    tensions_gt = []
    tip_position_gt = []
    tip_force_gt = []

    plotter = TendonRobotPlotter('Ground Truth Simulation')

    for i in range(100):
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
        magnitude = max_magnitude * (0.5 * (1.0 - np.cos(2 * np.pi * force_rate_hz * current_time))) ** 4

        tip_force = magnitude * direction

        max_tensions = np.array([6.0, 2.0, 2.0, 2.0])
        # max_tensions = np.zeros(4)
        tension_rate_hz = 0.02
        tensions_rate_hz = np.array([1.0 * tension_rate_hz, 1.1 * tension_rate_hz, 1.2 * tension_rate_hz, 1.3 * tension_rate_hz])
        tensions = 0.5 * (1.0 - np.cos(2 * np.pi * tensions_rate_hz * current_time))
        tensions *= max_tensions
        # tensions = np.array([5.0, 2.0, 1.0, 1.0])

        solution = solver.step(tensions, tip_force, num_samples)

        python_time = time.time() - base_time - current_time
        start_render = time.time() - base_time

        plotter.update(solution)

        render_time = time.time() - base_time - start_render
        total_time = time.time() - base_time - current_time

        print(f"cpp solve time: {solution.total_time_ms:.2f} ms")
        print(f"python solve time: {1000 * python_time:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        tensions_gt.append(tensions)
        tip_position_gt.append(solution.tip_pose_samples[0][:3,3])
        tip_force_gt.append(tip_force)

        # time.sleep(3.0)

    return np.stack(tensions_gt), np.stack(tip_position_gt), np.stack(tip_force_gt)


if __name__ == "__main__":
    tensions_gt, tip_position_gt, tip_force_gt = simulate_tip_force()
    estimate_tip_force(tensions_gt, tip_position_gt, tip_force_gt)
