import numpy as np
import matplotlib.pyplot as plt

import time
from tendon_robot import TipForceSim, TipForceSolver, DistLoadSim

from plotting import TendonRobotPlotter
from config import get_simulation_config, get_sensing_config


class dist_load_function:
    def __init__(self, num_poses, framerate, force_rate_hz, mag_min = 0.09, mag_max = 0.1, sigma_min=0.05, sigma_max=0.1):
        self.framerate = framerate
        self.force_rate_hz = force_rate_hz
        self.s = np.linspace(0, 1, num_poses)
        self.steps_per_cycle = framerate / force_rate_hz

        self.mu = 0
        self.sigma = 1
        self.mag = 0
        self.angle = 0

        self.sigma_range = [sigma_min, sigma_max]
        self.mag_range = [mag_min, mag_max]
        self.mu_rage = [0.0, 1.0]
        self.angle_range = [0.0, 2 * np.pi]
        self.step = 0

    def sample_parameters(self):
        self.mu = np.random.uniform(self.mu_rage[0], self.mu_rage[1])
        self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        self.mag = np.random.uniform(self.mag_range[0], self.mag_range[1])
        self.angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

    def update(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()
        
        mag = self.mag * (0.5 * (1.0 - np.cos(2 * np.pi * self.force_rate_hz * t))) ** 4
        gaussian = 1 / (np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(- (self.s - self.mu)**2 / (2 * self.sigma**2))
        gaussian = gaussian / np.sum(gaussian)
        f_1d = mag * gaussian

        f_x = f_1d * np.cos(self.angle)
        f_y = f_1d * np.sin(self.angle)
        f_z = np.zeros_like(f_y)

        self.step = self.step + 1

        return np.column_stack((f_x, f_y, f_z))
    

def simulate(sim_time):
    config = get_simulation_config()

    simulator = DistLoadSim(config)

    frame_rate = 30
    num_steps = sim_time * frame_rate

    plotter = TendonRobotPlotter('Distributed Load Simulation', plot_dist_load=True)

    num_poses = config.num_discs + (config.num_discs - 1) * config.poses_between_discs
    forces = np.zeros([num_poses - 1, 3])
    dist_load = dist_load_function(num_poses, 30, 0.1)

    for i in range(num_steps):
        
        t = float(i) / float(frame_rate)
        forces = dist_load.update(t)

        max_tensions = np.array([6.0, 2.0, 2.0, 2.0])
        # max_tensions = np.zeros(4)
        tension_rate_hz = 0.1
        tensions_rate_hz = np.array([1.0 * tension_rate_hz, 1.1 * tension_rate_hz, 1.2 * tension_rate_hz, 1.3 * tension_rate_hz])
        tensions = 0.5 * (1.0 - np.cos(2 * np.pi * tensions_rate_hz * t))
        tensions *= max_tensions
        # tensions = np.array([5.0, 2.0, 1.0, 1.0])

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

    plotter.plotter.close()


if __name__ == "__main__":
    sim_time = 10

    tensions_gt, tip_position_gt, tip_force_gt = simulate(sim_time)
