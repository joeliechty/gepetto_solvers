import numpy as np
import matplotlib.pyplot as plt

import time
from tendon_robot import TipForceSim, TipForceSolver, TendonRobotGtsamConfig

from plotting import TendonRobotPlotter
from config import get_simulation_config, get_sensing_config


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


def estimate_tip_force(tensions, tip_positions, tip_forces_gt):
    config = get_sensing_config()

    # Add noise to all measured data
    tensions_meas = tensions + config.tension_std * np.random.randn(*tensions.shape)
    tip_positions_meas = tip_positions + config.tip_pose_p_meas_std * np.random.randn(*tip_positions.shape)

    solver = TipForceSolver(config)
    plotter = TendonRobotPlotter('Inference')

    position_filter = MeasurementKalmanFilter(1, dim=3, position_meas_std=config.tip_pose_p_meas_std, accel_prior_std=1e-4)
    tensions_filter = MeasurementKalmanFilter(1, dim=4, position_meas_std=config.tension_std, accel_prior_std=1e-3)

    filtered_position = []
    filtered_tensions = []

    for tensions, tip_position, tip_force in zip(tensions_meas, tip_positions_meas, tip_forces_gt):
        start_solve = time.time()

        filtered_position_i = position_filter.step(tip_position)
        filtered_tensions_i = tensions_filter.step(tensions)

        filtered_position.append(filtered_position_i)
        filtered_tensions.append(filtered_tensions_i)

        solution = solver.step(filtered_tensions_i, filtered_position_i, 1)

        solve_time = time.time() - start_solve
        start_render = time.time()

        plotter.update(solution, tip_force_gt=tip_force)

        render_time = time.time() - start_render
        total_time = time.time() - start_solve

        print(f"solve time: {1000 * solve_time:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")


    plotter.plotter.close()

    plt.figure()
    plt.plot(tensions_meas, 'ro')
    plt.plot(filtered_tensions, 'b-')
    plt.xlabel("Time step")
    plt.ylabel("Tension")
    
    plt.figure()
    plt.plot(tip_positions_meas, 'ro')
    plt.plot(filtered_position, 'b-')
    plt.xlabel("Time step")
    plt.ylabel("Position")

    plt.show()

def simulate_tip_force(sim_time):
    simulator = TipForceSim(get_simulation_config())

    frame_rate = 30
    num_steps = sim_time * frame_rate

    tensions_gt = []
    tip_position_gt = []
    tip_force_gt = []

    plotter = TendonRobotPlotter('Simulation')

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        direction_rate_hz = 0.02
        direction = np.array([
            np.sin(2 * np.pi * direction_rate_hz * 1.0 * t),
            np.sin(2 * np.pi * direction_rate_hz * 1.1 * t),
            np.sin(2 * np.pi * direction_rate_hz * 1.2 * t),
        ])

        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])  # default direction if zero
        else:
            direction /= norm

        max_magnitude = 0.1
        force_rate_hz = 0.3
        magnitude = max_magnitude * (0.5 * (1.0 - np.cos(2 * np.pi * force_rate_hz * t))) ** 4

        tip_force = magnitude * direction
        tip_force = 0.1 * np.array([np.cos(-np.pi / 6), np.sin(-np.pi / 6),0])

        max_tensions = np.array([6.0, 2.0, 2.0, 2.0])
        tension_rate_hz = 0.1
        tensions_rate_hz = np.array([1.0 * tension_rate_hz, 1.1 * tension_rate_hz, 1.2 * tension_rate_hz, 1.3 * tension_rate_hz])
        tensions = 0.5 * (1.0 - np.cos(2 * np.pi * tensions_rate_hz * t))
        tensions *= max_tensions
        tensions = np.array([5.0, 2.0, 1.0, 1.0])
        tensions = np.zeros(4)

        start_solve = time.time()
        solution = simulator.step(tensions, tip_force)
        solve_time = time.time() - start_solve

        start_render = time.time()
        plotter.update(solution)
        render_time = time.time() - start_render

        total_time = time.time() - start_solve

        print(f"solve time: {1000 * solve_time:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

        fbg_samples = np.array(solution.fbg_array_samples[-1])
        plt.figure()
        plt.plot(fbg_samples[:,0], 'ro')
        plt.plot(fbg_samples[:,1], 'go')
        plt.plot(fbg_samples[:,2], 'bo')

        plt.show()


        tensions_gt.append(tensions)
        # tip_position_gt.append(solution.tip_pose_samples[0][:3,3])
        tip_position_gt.append(solution.backbone_pose_mean[-1][:3,3])
        tip_force_gt.append(tip_force)

        # time.sleep(3.0)

    plotter.plotter.close()

    return np.stack(tensions_gt), np.stack(tip_position_gt), np.stack(tip_force_gt)


if __name__ == "__main__":
    sim_time = 10

    tensions_gt, tip_position_gt, tip_force_gt = simulate_tip_force(sim_time)
    estimate_tip_force(tensions_gt, tip_position_gt, tip_force_gt)
