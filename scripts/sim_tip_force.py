import time

import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_simulation_config, get_base_config
from utils import TipForceFunction, tensions_function, moving_savgol


def inference(tensions_gt, tip_positions_gt, tip_forces_gt, save_frames_mode):
    config = get_base_config()

    # Add noise to all measured data
    tensions_meas = tensions_gt + config.tension_meas_std * np.random.randn(*tensions_gt.shape)
    tip_positions_meas = tip_positions_gt + config.tip_position_meas_std * np.random.randn(*tip_positions_gt.shape)

    solver = TipForceSolver(config)
    plotter = TendonRobotPlotter('Tip Force Inference', save_frames_mode=save_frames_mode)
    tensions_filter = moving_savgol()
    tip_position_filter = moving_savgol()

    tensions_filtered = []
    tip_positions_filtered = []

    for i, (tensions_meas_i, tip_position_meas_i, tip_force_gt_i) in enumerate(zip(tensions_meas, tip_positions_meas, tip_forces_gt)):
        start_solve = time.time()

        tensions_filtered_i = tensions_filter.update(tensions_meas_i)
        tip_position_filtered_i = tip_position_filter.update(tip_position_meas_i)

        solution = solver.step(tensions_filtered_i, tip_position_filtered_i, 1)

        start_render = time.time()

        plotter.update(solution, tip_force_gt=tip_force_gt_i)

        render_time = time.time() - start_render
        total_time = time.time() - start_solve

        print(f"solve time: {solution.solve_time_ms:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        tensions_filtered.append(tensions_filtered_i)
        tip_positions_filtered.append(tip_position_filtered_i)

    plotter.plotter.close()

    plt.figure()
    plt.plot(tensions_meas, 'ro')
    plt.plot(tensions_filtered, 'b-')
    plt.xlabel("Time step")
    plt.ylabel("Tension")
    
    plt.figure()
    plt.plot(tip_positions_meas, 'ro')
    plt.plot(tip_positions_filtered, 'b-')
    plt.xlabel("Time step")
    plt.ylabel("Position")

    plt.show()


def simulation(sim_time, save_frames_mode, frame_rate=30):
    simulator = TipForceSolver(get_simulation_config())

    num_steps = sim_time * frame_rate

    tensions_gt = []
    tip_position_gt = []
    tip_force_gt = []

    plotter = TendonRobotPlotter('Tip Force Simulation', save_frames_mode=save_frames_mode)

    tip_force_function = TipForceFunction(max_magnitude=0.15)

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        tip_force = tip_force_function(t)
        tensions = tensions_function(t)

        start_solve = time.time()
        solution = simulator.simulation_step(tensions, tip_force)

        start_render = time.time()
        plotter.update(solution)
        render_time = time.time() - start_render

        total_time = time.time() - start_solve

        print(f"solve time: {solution.solve_time_ms:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        tensions_gt.append(tensions)
        # tip_position_gt.append(solution.tip_pose_samples[0][:3,3])
        tip_position_gt.append(solution.backbone_pose_mean[-1][:3,3])
        tip_force_gt.append(tip_force)

    plotter.plotter.close()

    return np.stack(tensions_gt), np.stack(tip_position_gt), np.stack(tip_force_gt)


if __name__ == "__main__":
    sim_time = 5
    save_frames_mode = False

    tensions_gt, tip_position_gt, tip_force_gt = simulation(sim_time, save_frames_mode)
    inference(tensions_gt, tip_position_gt, tip_force_gt, save_frames_mode)
