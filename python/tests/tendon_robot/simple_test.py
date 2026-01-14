import numpy as np
import time

from crest_sparse import TendonRobotSolver
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config



def main():
    config = get_base_config()
    solver = TendonRobotSolver(config)
    plotter = TendonRobotPlotter('kinematics_sim', single_plot_mode=True)

    tensions_mean = np.array([0.1, 0.0, 0.0, 0.0])
    tensions_cov = 1e-1 *np.eye(4)

    solution = solver.solve(tensions_mean, tensions_cov)
    plotter.update(solution)
    
    
    # tip_force = np.zeros(3)
    
    
    # solution = solver.simulation_step(tensions, tip_force)

    # py::class_<TendonRobotSolver>(m, "TendonRobotSolver")
    #     .def(py::init<const TendonRobotSolverConfig&>())
    #     .def("solve", &TendonRobotSolver::solve,
    #          py::arg("tensions_mean"),
    #          py::arg("tensions_cov"));
    
    # plotter.update(solution)

    # plotter.plotter.close()

if __name__ == "__main__":
    main()
