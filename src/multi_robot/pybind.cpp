#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "MultiRobotModel.h"
#include "MultiRobotSolver.h"

namespace py = pybind11;


void bind_multi_robot(py::module& m) {
    py::class_<MultiRobotSolverConfig>(m, "MultiRobotSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &MultiRobotSolverConfig::base)
        .def_readwrite("nodes_per_rod", &MultiRobotSolverConfig::nodes_per_rod)
        .def_readwrite("K_inv", &MultiRobotSolverConfig::K_inv)
        .def_readwrite("snare_distance_to_tip", &MultiRobotSolverConfig::snare_distance_to_tip)
        .def_readwrite("sigma_twist_pos", &MultiRobotSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_twist_rot", &MultiRobotSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_small_force", &MultiRobotSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &MultiRobotSolverConfig::sigma_small_moment);
    
    py::class_<MultiRobotMarginals>(m, "MultiRobotMarginals")
        .def(py::init<>())
        .def_readwrite("main_rod", &MultiRobotMarginals::main_rod)
        .def_readwrite("helper_rod", &MultiRobotMarginals::helper_rod)
        .def_readwrite("end_effector_rod", &MultiRobotMarginals::end_effector_rod);
        
    py::class_<Solution<MultiRobotMarginals>>(m, "MultiRobotSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<MultiRobotMarginals>::meta)
        .def_readwrite("marginals", &Solution<MultiRobotMarginals>::marginals);

    py::class_<MultiRobotSolver>(m, "MultiRobotSolver")
        .def(py::init<const MultiRobotSolverConfig&>(), py::arg("config"))
        .def("solve", &MultiRobotSolver::solve, 
            py::arg("main_base_pose"),
            py::arg("main_insertion"),
            py::arg("helper_base_pose"),
            py::arg("helper_insertion"),
            py::arg("tip_wrench"),
            py::call_guard<py::gil_scoped_release>());
}