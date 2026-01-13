#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "CosseratShellSolver.h"

namespace py = pybind11;
    

void bind_cosserat_shell(py::module& m) {
    py::class_<CosseratShellMarginals>(m, "CosseratShellMarginals")
        .def(py::init<>())
        .def_readwrite("pose_mean", &CosseratShellMarginals::pose_mean)
        .def_readwrite("pose_cov", &CosseratShellMarginals::pose_cov)
        .def_readwrite("stress_mean", &CosseratShellMarginals::stress_mean)
        .def_readwrite("stress_cov", &CosseratShellMarginals::stress_cov);

    py::class_<Solution<CosseratShellMarginals>>(m, "CosseratShellSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<CosseratShellMarginals>::meta)
        .def_readwrite("marginals", &Solution<CosseratShellMarginals>::marginals);

    py::class_<CosseratShellSolverConfig>(m, "CosseratShellSolverConfig")
        .def(py::init<>())
        .def_readwrite("num_nodes_x", &CosseratShellSolverConfig::num_nodes_x)
        .def_readwrite("num_nodes_y", &CosseratShellSolverConfig::num_nodes_y)
        .def_readwrite("element_size", &CosseratShellSolverConfig::element_size)
        .def_readwrite("K_inv", &CosseratShellSolverConfig::K_inv)
        .def_readwrite("sigma_twist_pos", &CosseratShellSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_twist_rot", &CosseratShellSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_stress_force", &CosseratShellSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &CosseratShellSolverConfig::sigma_stress_moment);

    py::class_<CosseratShellSolver>(m, "CosseratShellSolver")
        .def(py::init<const CosseratShellSolverConfig&>())
        .def("solve", &CosseratShellSolver::solve,
            py::arg("displacement_mean"),
            py::arg("displacement_cov"),
            py::call_guard<py::gil_scoped_release>());
}