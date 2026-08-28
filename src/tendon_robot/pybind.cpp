#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonRobotModel.h"
#include "TendonRobotSolver.h"

namespace py = pybind11;

void bind_tendon_robot(py::module& m) {
    // RoutingAngleFunction, RoutingFunctionParams, TendonInput and
    // PerDiscTendonInput are registered from tendon_finger/pybind.cpp. The types
    // are declared byte-identically in TendonRobotModel.h and TendonFingerModel.h,
    // so pybind's typeid-keyed registry serves both from that one site.
    py::class_<TendonConfig>(m, "TendonConfig")
        .def(py::init<>())
        .def_readwrite("num_discs", &TendonConfig::num_discs)
        .def_readwrite("num_tendons", &TendonConfig::num_tendons)
        .def_readwrite("routing_radius", &TendonConfig::routing_radius)
        .def_readwrite("disc_pose_idx", &TendonConfig::disc_pose_idx)
        .def_readwrite("no_disc_pose_idx", &TendonConfig::no_disc_pose_idx)
        .def_readwrite("hole_locations", &TendonConfig::hole_locations);

    py::class_<TendonRobotSolverConfig>(m, "TendonRobotSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonRobotSolverConfig::base)
        .def_readwrite("rod_length", &TendonRobotSolverConfig::rod_length)
        .def_readwrite("num_discs", &TendonRobotSolverConfig::num_discs)
        .def_readwrite("num_between_nodes", &TendonRobotSolverConfig::num_between_nodes)
        .def_readwrite("num_tendons", &TendonRobotSolverConfig::num_tendons)
        .def_readwrite("K_inv", &TendonRobotSolverConfig::K_inv)
        .def_readwrite("K_inv_per_segment", &TendonRobotSolverConfig::K_inv_per_segment)
        .def_readwrite("disc_positions_normalized", &TendonRobotSolverConfig::disc_positions_normalized)
        .def_readwrite("sigma_twist_rot", &TendonRobotSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_twist_pos", &TendonRobotSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_stress_force", &TendonRobotSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &TendonRobotSolverConfig::sigma_stress_moment)
        .def_readwrite("sigma_base_pos", &TendonRobotSolverConfig::sigma_base_pos)
        .def_readwrite("sigma_base_rot", &TendonRobotSolverConfig::sigma_base_rot)
        .def_readwrite("tendon_input", &TendonRobotSolverConfig::tendon_input)
        .def_readwrite("per_disc_tendon_input", &TendonRobotSolverConfig::per_disc_tendon_input)
        .def_readwrite("base_pose", &TendonRobotSolverConfig::base_pose);

    py::class_<TendonRobotMarginals>(m, "TendonRobotMarginals")
        .def(py::init<>())
        .def_readwrite("rod", &TendonRobotMarginals::rod)
        .def_readwrite("tendon_config", &TendonRobotMarginals::tendon_config)
        .def_readwrite("external_wrenches", &TendonRobotMarginals::external_wrenches)
        .def_readwrite("tensions", &TendonRobotMarginals::tensions)
        .def_readwrite("J_pose_tensions", &TendonRobotMarginals::J_pose_tensions);

    py::class_<Solution<TendonRobotMarginals>>(m, "TendonRobotSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<TendonRobotMarginals>::meta)
        .def_readwrite("marginals", &Solution<TendonRobotMarginals>::marginals);

    py::class_<TendonRobotSolverDispatch>(m, "TendonRobotSolver")
        .def(py::init<const TendonRobotSolverConfig&>())
        .def("solve", &TendonRobotSolverDispatch::solve,
             py::arg("tensions"),
             py::arg("tip_force"),
             py::arg("tip_meas"));
}
