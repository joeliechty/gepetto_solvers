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
        .def_readwrite("finger_names", &TendonHandMarginals::finger_names)
        .def_readwrite("has_object", &TendonHandMarginals::has_object)
        .def_readwrite("object_pose", &TendonHandMarginals::object_pose);

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
        .def("num_fingers", &TendonHandSolver::num_fingers)
        .def("set_object", [](TendonHandSolver& self,
                              const std::string& vdb_path,
                              const Eigen::Matrix4d& initial_pose_mat,
                              const Eigen::VectorXd& prior_sigmas) {
            gtsam::Pose3 initial_pose(initial_pose_mat);
            self.set_object(vdb_path, initial_pose, prior_sigmas);
        },
        py::arg("vdb_path"),
        py::arg("initial_pose"),
        py::arg("prior_sigmas"),
        "Set an SDF object for contact simulation");
}
