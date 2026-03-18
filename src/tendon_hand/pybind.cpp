#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonHandModel.h"
#include "TendonHandSolver.h"

namespace py = pybind11;


void bind_tendon_hand(py::module& m) {
    py::class_<TendonHandSolverConfig>(m, "TendonHandSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonHandSolverConfig::base)
        .def_readwrite("sigma_small_force", &TendonHandSolverConfig::sigma_small_force)
        .def_readwrite("sigma_small_moment", &TendonHandSolverConfig::sigma_small_moment);

    py::class_<TendonHandMarginals>(m, "TendonHandMarginals")
        .def(py::init<>())
        .def_readwrite("fingers", &TendonHandMarginals::fingers)
        .def_readwrite("finger_names", &TendonHandMarginals::finger_names);

    py::class_<Solution<TendonHandMarginals>>(m, "TendonHandSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<TendonHandMarginals>::meta)
        .def_readwrite("marginals", &Solution<TendonHandMarginals>::marginals);

    py::class_<TendonHandSolver>(m, "TendonHandSolver")
        .def(py::init<const std::vector<std::pair<std::string, TendonRobotSolverConfig>>&, const TendonHandSolverConfig&>(),
            py::arg("finger_configs"),
            py::arg("config"))
        .def("solve", &TendonHandSolver::solve,
            py::arg("tensions"),
            py::arg("tip_wrenches"),
            py::call_guard<py::gil_scoped_release>())
        .def("num_fingers", &TendonHandSolver::num_fingers);
}
