import numpy as np
import time

import crest_sparse
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config



def main():
    config = get_base_config()
    solver = crest_sparse.TendonRobotSolver(config)
    plotter = TendonRobotPlotter(single_plot_mode=False)

    tensions_cov = (1e-2) ** 2 * np.eye(4)

    for i in range(100):
        tensions_mean = np.array([0.1 * i, 0.0, 0.05 * i, 0.0])
        tensions = crest_sparse.Vector4Gaussian(tensions_mean, tensions_cov)
        solution = solver.solve(tensions, None)
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
