#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonRobotModel.h"
#include "TendonRobotSolver.h"

namespace py = pybind11;

void bind_tendon_robot(py::module& m) {
    py::enum_<RoutingAngleFunction>(m, "RoutingAngleFunction")
        .value("CONSTANT", RoutingAngleFunction::CONSTANT)
        .value("LINEAR", RoutingAngleFunction::LINEAR)
        .export_values();

    py::class_<RoutingFunctionParams>(m, "RoutingFunctionParams")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("angle_offset"), py::arg("total_angle"))
        .def_readwrite("angle_offset", &RoutingFunctionParams::angle_offset)
        .def_readwrite("total_angle", &RoutingFunctionParams::total_angle);

    py::class_<TendonInput>(m, "TendonInput")
        .def(py::init<>())
        .def_readwrite("functions", &TendonInput::functions)
        .def_readwrite("params", &TendonInput::params)
        .def_readwrite("routing_radius", &TendonInput::routing_radius);

    py::class_<PerDiscTendonInput>(m, "PerDiscTendonInput")
        .def(py::init<>())
        .def_readwrite("num_tendons", &PerDiscTendonInput::num_tendons)
        .def_readwrite("routing_radius", &PerDiscTendonInput::routing_radius)
        .def_readwrite("hole_angles", &PerDiscTendonInput::hole_angles)
        .def_readwrite("hole_radii", &PerDiscTendonInput::hole_radii);

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
