import time

import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_simulation_config, get_base_config
from utils import TipForceFunction, tensions_function, moving_savgol


def trefoil_knot(t, f_hz=0.2):
    R = 0.03
    omega = 2 * np.pi * f_hz  # angular speed (rad/sec)
    tau = omega * t

    x = R * (np.sin(tau) + 2 * np.sin(2 * tau))
    y = R * (np.cos(tau) - 2 * np.cos(2 * tau))
    z = -R * np.sin(3 * tau) / 3
    return np.array([x, y, z + 0.15])


class InverseKinematics:
    def __init__(self, config):
        self.solver = TipForceSolver(config)
        self.max_iter = 100
        self.d_tensions_max = 0.1
        self.dp_tol = 1e-6
        self.tensions = np.zeros(4)

    def solve(self, p_desired, f_tip):
        for ii in range(self.max_iter):
            solution = self.solver.simulation_step(self.tensions, f_tip)

            J_position = solution.J_pose_tensions[3:]
            p = solution.backbone_pose_mean[-1][:3,3]
            R = solution.backbone_pose_mean[-1][:3,:3]
        
            dp = p_desired - p
            dp_body = R.T @ dp
            d_tensions = np.linalg.pinv(J_position) @ dp_body

            d_tensions_norm = np.linalg.norm(d_tensions)
            if d_tensions_norm > self.d_tensions_max:
                d_tensions = self.d_tensions_max * d_tensions / d_tensions_norm
            
            self.tensions = self.tensions + d_tensions

            if np.linalg.norm(dp) < self.dp_tol:
                return solution
        
        print("IK did not converge!!")
    
def simulation(sim_time, save_png_mode, frame_rate=30):
    config = get_simulation_config()
    solver = InverseKinematics(config)

    num_steps = sim_time * frame_rate

    tip_force_function = TipForceFunction()

    plotter = TendonRobotPlotter('Inverse Kinematics', save_png_mode=save_png_mode)
    
    tip_positions = []

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        tip_force = tip_force_function(t)

        start_solve = time.time()

        solution = solver.solve(trefoil_knot(t), tip_force)
        # tensions_meas = tensions + config.tension_meas_std * np.random.randn(*tensions.shape)

        tip_positions.append(solution.backbone_pose_mean[-1][:3,3])
        
        start_render = time.time()
        if len(tip_positions) > 2:
            plotter.update(solution, tip_positions=tip_positions)
        render_time = time.time() - start_render

        total_time = time.time() - start_solve

        print(f"solve time: {solution.solve_time_ms:.2f} ms")
        print(f"render time: {1000 * render_time:.2f} ms")
        print(f"total time: {1000 * total_time:.2f} ms\n\n")

    plotter.plotter.close()


if __name__ == "__main__":
    sim_time = 15
    save_png_mode = True

    simulation(sim_time, save_png_mode)
