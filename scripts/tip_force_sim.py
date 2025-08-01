import numpy as np
import matplotlib.pyplot as plt

import time
from tendon_robot import TipForceSim, TipForceSolver, TendonRobotGtsamConfig

from plotting import TendonRobotPlotter
from config import get_simulation_config, get_sensing_config
from sim_functions import tip_force_function, tensions_function


class MeasurementKalmanFilter:
    def __init__(self, dt, dim=3, position_meas_std=1e-3, accel_prior_std=1e-2):
        self.dt = dt
        self.dim = dim
        
        self.x = np.zeros(2 * self.dim)
        self.P = np.eye(2 * self.dim) * 1e6

        I = np.eye(self.dim)
        self.A = np.block([
            [I, dt * I],
            [np.zeros((self.dim, self.dim)), I]
        ])
        
        q = accel_prior_std**2
        Q_pos = (dt**4)/4 * q * I
        Q_vel = (dt**2) * q * I
        Q_cross = (dt**3)/2 * q * I
        self.Q = np.block([
            [Q_pos, Q_cross],
            [Q_cross, Q_vel]
        ])
        
        self.H = np.block([I, np.zeros((self.dim, self.dim))])
        self.R = (position_meas_std**2) * I

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, position_meas):
        z = np.asarray(position_meas)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2 * self.dim) - K @ self.H) @ self.P

    def step(self, position_meas):
        """Perform a full predict + update cycle"""
        self.predict()
        self.update(position_meas)
        return self.x[:self.dim]  # Return filtered position

    def get_state(self):
        """Return current full state [position, velocity]"""
        return self.x.copy()

    def get_covariance(self):
        """Return full 6x6 covariance"""
        return self.P.copy()


def inference(tensions_gt, tip_positions_gt, tip_forces_gt):
    config = get_sensing_config()

    # Add noise to all measured data
    tensions_meas = tensions_gt + config.tension_std * np.random.randn(*tensions_gt.shape)
    tip_positions_meas = tip_positions_gt + config.tip_pose_p_meas_std * np.random.randn(*tip_positions_gt.shape)

    solver = TipForceSolver(config)
    plotter = TendonRobotPlotter('Tip Force Inference')

    position_filter = MeasurementKalmanFilter(1, dim=3, position_meas_std=config.tip_pose_p_meas_std, accel_prior_std=1e-4)
    tensions_filter = MeasurementKalmanFilter(1, dim=4, position_meas_std=config.tension_std, accel_prior_std=1e-3)

    tip_position_filtered = []
    tensions_filtered = []

    for tensions_meas_i, tip_position_meas_i, tip_force_gt_i in zip(tensions_meas, tip_positions_meas, tip_forces_gt):
        start_solve = time.time()

        position_filtered_i = position_filter.step(tip_position_meas_i)
        tensions_filtered_i = tensions_filter.step(tensions_meas_i)

        tip_position_filtered.append(position_filtered_i)
        tensions_filtered.append(tensions_filtered_i)

        solution = solver.step(tensions_filtered_i, position_filtered_i, 1)

        start_render = time.time()
        plotter.update(solution, tip_force_gt=tip_force_gt_i)

        render_time = time.time() - start_render
        total_time = time.time() - start_solve

        print(f"solve time: {solution.solve_time_ms:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")


    plotter.plotter.close()

    plt.figure()
    plt.plot(tensions_meas, 'ro')
    plt.plot(tensions_filtered, 'b-')
    plt.xlabel("Time step")
    plt.ylabel("Tension")
    
    plt.figure()
    plt.plot(tip_positions_meas, 'ro')
    plt.plot(tip_position_filtered, 'b-')
    plt.xlabel("Time step")
    plt.ylabel("Position")

    plt.show()


def simulation(sim_time, frame_rate=30):
    simulator = TipForceSim(get_simulation_config())

    num_steps = sim_time * frame_rate

    tensions_gt = []
    tip_position_gt = []
    tip_force_gt = []

    plotter = TendonRobotPlotter('Tip Force Simulation')

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        tip_force = tip_force_function(t)
        tensions = tensions_function(t)

        start_solve = time.time()
        solution = simulator.step(tensions, tip_force)

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

    tensions_gt, tip_position_gt, tip_force_gt = simulation(sim_time)
    inference(tensions_gt, tip_position_gt, tip_force_gt)
