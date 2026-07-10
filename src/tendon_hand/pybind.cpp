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
        .def_readwrite("wrist_pose", &TendonHandSolverConfig::wrist_pose)
        .def_readwrite("sigma_wrist_pos", &TendonHandSolverConfig::sigma_wrist_pos)
        .def_readwrite("sigma_wrist_rot", &TendonHandSolverConfig::sigma_wrist_rot);

    py::class_<TendonHandMarginals>(m, "TendonHandMarginals")
        .def(py::init<>())
        .def_readwrite("fingers", &TendonHandMarginals::fingers)
        .def_readwrite("finger_names", &TendonHandMarginals::finger_names);

    py::class_<Solution<TendonHandMarginals>>(m, "TendonHandSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<TendonHandMarginals>::meta)
        .def_readwrite("marginals", &Solution<TendonHandMarginals>::marginals);

    py::class_<TendonHandSolver>(m, "TendonHandSolver")
        .def(py::init<
                const std::vector<std::pair<std::string, TendonFingerSolverConfig>>&,
                const TendonHandSolverConfig&>(),
             py::arg("finger_configs"), py::arg("config"))
        .def("solve", &TendonHandSolver::solve,
             py::arg("tensions"), py::arg("tip_wrenches"))
        .def("set_wrist_pose", &TendonHandSolver::set_wrist_pose,
             py::arg("wrist_pose"),
             "Re-aim the shared wrist prior between solves (4x4, world frame) "
             "without rebuilding the solver. solve() then warm-starts from the "
             "previous solution instead of cold-starting from a straight hand.")
        .def("num_fingers", &TendonHandSolver::num_fingers)
        .def("get_factor_error_summary", &TendonHandSolver::get_factor_error_summary);
}
