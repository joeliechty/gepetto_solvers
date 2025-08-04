#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "tendon_robot_gtsam.h"

namespace py = pybind11;
using namespace gtsam;


PYBIND11_MODULE(tendon_robot, m) {
    py::class_<TendonDiscConfig>(m, "TendonDiscConfig")
        .def(py::init<>())
        .def_readwrite("num_tendons", &TendonDiscConfig::num_tendons)
        .def_readwrite("num_discs", &TendonDiscConfig::num_discs)
        .def_readwrite("routing_radius", &TendonDiscConfig::routing_radius)
        .def_readwrite("disc_pose_idx", &TendonDiscConfig::disc_pose_idx)
        .def_readwrite("local_holes", &TendonDiscConfig::local_holes);

    py::class_<TendonRobotSolution>(m, "TendonRobotSolution")
        .def(py::init<>())
        .def_readwrite("backbone_pose_mean", &TendonRobotSolution::backbone_pose_mean)
        .def_readwrite("backbone_pose_cov", &TendonRobotSolution::backbone_pose_cov)
        .def_readwrite("tip_pose_samples", &TendonRobotSolution::tip_pose_samples)
        .def_readwrite("fbg_array_samples", &TendonRobotSolution::fbg_array_samples)
        .def_readwrite("applied_wrench_mean", &TendonRobotSolution::applied_wrench_mean)
        .def_readwrite("applied_wrench_cov", &TendonRobotSolution::applied_wrench_cov)
        .def_readwrite("tensions_mean", &TendonRobotSolution::tensions_mean)
        .def_readwrite("tensions_cov", &TendonRobotSolution::tensions_cov)
        .def_readwrite("solve_time_ms", &TendonRobotSolution::solve_time_ms)
        .def_readwrite("extract_time_ms", &TendonRobotSolution::extract_time_ms)
        .def_readwrite("total_time_ms", &TendonRobotSolution::total_time_ms)
        .def_readwrite("tendon_disc_config", &TendonRobotSolution::tendon_disc_config);

    py::enum_<RoutingAngleFunction>(m, "RoutingAngleFunction")
        .value("CONSTANT", RoutingAngleFunction::CONSTANT)
        .value("LINEAR", RoutingAngleFunction::LINEAR)
        .export_values();

    py::class_<RoutingFunctionParams>(m, "RoutingFunctionParams")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("angle_offset"), py::arg("total_angle"))
        .def_readwrite("angle_offset", &RoutingFunctionParams::angle_offset)
        .def_readwrite("total_angle", &RoutingFunctionParams::total_angle);

    py::class_<TendonRobotConfig>(m, "TendonRobotConfig")
        .def(py::init<>())
        .def_readwrite("num_discs", &TendonRobotConfig::num_discs)
        .def_readwrite("poses_between_discs", &TendonRobotConfig::poses_between_discs)
        .def_readwrite("rod_length", &TendonRobotConfig::rod_length)
        .def_readwrite("rod_diameter", &TendonRobotConfig::rod_diameter)
        .def_readwrite("youngs_modulus", &TendonRobotConfig::youngs_modulus)
        .def_readwrite("shear_modulus", &TendonRobotConfig::shear_modulus)
        .def_readwrite("routing_radius", &TendonRobotConfig::routing_radius)

        .def_readwrite("tension_std", &TendonRobotConfig::tension_std)
        .def_readwrite("small_force_std", &TendonRobotConfig::small_force_std)
        .def_readwrite("small_moment_std", &TendonRobotConfig::small_moment_std)
        .def_readwrite("cosserat_twist_r_std", &TendonRobotConfig::cosserat_twist_r_std)
        .def_readwrite("small_r_std", &TendonRobotConfig::small_r_std)
        .def_readwrite("small_p_std", &TendonRobotConfig::small_p_std)
        .def_readwrite("tip_pose_jerk_std", &TendonRobotConfig::tip_pose_jerk_std)
        .def_readwrite("tip_pose_accel_std", &TendonRobotConfig::tip_pose_accel_std)
        .def_readwrite("tension_drift_std", &TendonRobotConfig::tension_drift_std)

        .def_readwrite("tip_force_std", &TendonRobotConfig::tip_force_std)
        .def_readwrite("tip_force_drift_std", &TendonRobotConfig::tip_force_drift_std)
        .def_readwrite("dist_load_magnitude_std", &TendonRobotConfig::dist_load_magnitude_std)
        .def_readwrite("dist_load_jerk_std", &TendonRobotConfig::dist_load_jerk_std)
        .def_readwrite("dist_load_drift_std", &TendonRobotConfig::dist_load_drift_std)
        
        .def_readwrite("tip_pose_r_meas_std", &TendonRobotConfig::tip_pose_r_meas_std)
        .def_readwrite("tip_pose_p_meas_std", &TendonRobotConfig::tip_pose_p_meas_std)
        .def_readwrite("fbg_strain_meas_std", &TendonRobotConfig::fbg_strain_meas_std)

        .def_readwrite("angle_functions", &TendonRobotConfig::angle_functions)
        .def_readwrite("angle_params", &TendonRobotConfig::angle_params);

    py::class_<TipForceSolver>(m, "TipForceSolver")
    .def(py::init<const TendonRobotConfig&>())
    .def("step", &TipForceSolver::step,
         py::arg("tensions_meas"),
         py::arg("tip_position_meas"),
         py::arg("num_samples"),
         py::call_guard<py::gil_scoped_release>())
    .def("simulation_step", &TipForceSolver::simulation_step,
         py::arg("tensions"),
         py::arg("tip_force"),
         py::call_guard<py::gil_scoped_release>());

    py::class_<DistLoadSolver>(m, "DistLoadSolver")
    .def(py::init<const TendonRobotConfig&>())
    .def("step", &DistLoadSolver::step,
         py::arg("tensions_meas"),
         py::arg("fbg_signals_meas"),
         py::arg("num_samples"),
         py::call_guard<py::gil_scoped_release>())
    .def("step_simulation", &DistLoadSolver::step_simulation,
         py::arg("tensions"),
         py::arg("forces"),
         py::call_guard<py::gil_scoped_release>());
}
