import numpy as np
import matplotlib.pyplot as plt

import time
from tendon_robot import DistLoadSim, DistLoadSolver

from plotting import TendonRobotPlotter
from config import get_simulation_config, get_sensing_config
from sim_functions import dist_load_function, tensions_function


def inference(tensions_gt, fbg_signals_gt, dist_load_gt):
    config = get_sensing_config()

    # Add noise to all measured data
    tensions_meas = tensions_gt + config.tension_std * np.random.randn(*tensions_gt.shape)
    fbg_signals_meas = fbg_signals_gt + 1e-6 * np.random.randn(*fbg_signals_gt.shape)

    solver = DistLoadSolver(config)
    plotter = TendonRobotPlotter('Distributed Load Inference', plot_dist_load=True)


    for tensions_meas_i, fbg_signals_meas_i, dist_load_gt_i in zip(tensions_meas, fbg_signals_meas, dist_load_gt):
        start_solve = time.time()

        solution = solver.step(tensions_meas_i, fbg_signals_meas_i, 1)

        start_render = time.time()
        plotter.update(solution)

        render_time = time.time() - start_render
        total_time = time.time() - start_solve

        print(f"solve time: {solution.solve_time_ms:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        # if (fbg_signals_meas_i > 0.001).any():
        #     fbg_samples = np.array(solution.fbg_array_samples[-1])
        #     plt.figure()
        #     plt.plot(fbg_signals_meas_i[:,0], 'ro')
        #     plt.plot(fbg_samples[:,0], 'rx')
        #     plt.plot(fbg_signals_meas_i[:,1], 'go')
        #     plt.plot(fbg_samples[:,1], 'gx')
        #     plt.plot(fbg_signals_meas_i[:,2], 'bo')
        #     plt.plot(fbg_samples[:,2], 'bx')

        #     plt.show()


    plotter.plotter.close()


def simulation(sim_time, frame_rate=30):
    config = get_simulation_config()
    simulator = DistLoadSim(config)

    num_steps = sim_time * frame_rate

    plotter = TendonRobotPlotter('Distributed Load Simulation', plot_dist_load=True)

    num_poses = config.num_discs + (config.num_discs - 1) * config.poses_between_discs
    dist_load = dist_load_function(num_poses)

    tensions_gt = []
    dist_load_gt = []
    fbg_signals_gt = []

    for i in range(num_steps):
        
        t = float(i) / float(frame_rate)
        forces = dist_load.update(t)

        tensions = tensions_function(t)
        # tensions = np.zeros(4)

        start_solve = time.time()
        solution = simulator.step(tensions, forces)
        solve_time = time.time() - start_solve

        start_render = time.time()
        plotter.update(solution)
        render_time = time.time() - start_render

        total_time = time.time() - start_solve

        print(f"solve time: {1000 * solve_time:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        tensions_gt.append(tensions)
        fbg_signals_gt.append(np.array(solution.fbg_array_samples[-1]))
        dist_load_gt.append(forces)

    plotter.plotter.close()

    return np.array(tensions_gt), np.array(fbg_signals_gt), np.array(dist_load_gt)


if __name__ == "__main__":
    sim_time = 5

    tensions_gt, fbg_signals_gt, dist_load_gt = simulation(sim_time)
    inference(tensions_gt, fbg_signals_gt, dist_load_gt)