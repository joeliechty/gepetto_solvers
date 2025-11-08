import numpy as np
import time
from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config



def main():
    config = get_base_config()
    solver = TipForceSolver(config)

    tip_force = np.zeros(3)
    tensions = np.array([6, 0, 2, 0])
    
    plotter = TendonRobotPlotter('kinematics_sim', plot_tip_force=False, single_plot_mode=True, save_frames_mode=False, plot_backbone_ellipsoids=False)
    solution = solver.simulation_step(tensions, tip_force)

    plotter.update(solution)

    # plotter.plotter.close()

if __name__ == "__main__":
    main()
