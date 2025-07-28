#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "tendon_robot_gtsam.h"

namespace py = pybind11;
using namespace gtsam;

// PYBIND11_MODULE(tendon_robot, m) {
//     m.def("compose_poses", &compose_poses, "compose_poses");

//     py::class_<gtsam::Pose3>(m, "Pose3")
//         .def(py::init<>())
//         // Expose matrix() that returns an Eigen::Matrix4d
//         .def("matrix", &gtsam::Pose3::matrix);
// }





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
        .def_readwrite("backbone_pose_samples", &TendonRobotSolution::backbone_pose_samples)
        .def_readwrite("tip_wrench_mean", &TendonRobotSolution::tip_wrench_mean)
        .def_readwrite("tip_wrench_cov", &TendonRobotSolution::tip_wrench_cov)
        .def_readwrite("tip_wrench_samples", &TendonRobotSolution::tip_wrench_samples)
        .def_readwrite("tensions_mean", &TendonRobotSolution::tensions_mean)
        .def_readwrite("tensions_cov", &TendonRobotSolution::tensions_cov)
        .def_readwrite("tension_samples", &TendonRobotSolution::tension_samples)
        .def_readwrite("build_time_ms", &TendonRobotSolution::build_time_ms)
        .def_readwrite("solve_time_ms", &TendonRobotSolution::solve_time_ms)
        .def_readwrite("extract_time_ms", &TendonRobotSolution::extract_time_ms)
        .def_readwrite("total_time_ms", &TendonRobotSolution::total_time_ms)
        .def_readwrite("tendon_disc_config", &TendonRobotSolution::tendon_disc_config);

    py::enum_<RoutingAngleFunction>(m, "RoutingAngleFunction")
        .value("CONSTANT", RoutingAngleFunction::CONSTANT)
        .value("LINEAR", RoutingAngleFunction::LINEAR)
        .export_values();

    py::class_<RoutingParams>(m, "RoutingParams")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("angle_offset"), py::arg("total_angle"))
        .def_readwrite("angle_offset", &RoutingParams::angle_offset)
        .def_readwrite("total_angle", &RoutingParams::total_angle);

    py::class_<TendonRobotGtsamConfig>(m, "TendonRobotGtsamConfig")
        .def(py::init<>())
        .def_readwrite("num_discs", &TendonRobotGtsamConfig::num_discs)
        .def_readwrite("poses_between_discs", &TendonRobotGtsamConfig::poses_between_discs)
        .def_readwrite("rod_length", &TendonRobotGtsamConfig::rod_length)
        .def_readwrite("rod_diameter", &TendonRobotGtsamConfig::rod_diameter)
        .def_readwrite("youngs_modulus", &TendonRobotGtsamConfig::youngs_modulus)
        .def_readwrite("shear_modulus", &TendonRobotGtsamConfig::shear_modulus)
        .def_readwrite("tension_std", &TendonRobotGtsamConfig::tension_std)
        .def_readwrite("small_force_std", &TendonRobotGtsamConfig::small_force_std)
        .def_readwrite("small_moment_std", &TendonRobotGtsamConfig::small_moment_std)
        .def_readwrite("cosserat_twist_r_std", &TendonRobotGtsamConfig::cosserat_twist_r_std)
        .def_readwrite("small_r_std", &TendonRobotGtsamConfig::small_r_std)
        .def_readwrite("small_p_std", &TendonRobotGtsamConfig::small_p_std)
        .def_readwrite("tip_force_std", &TendonRobotGtsamConfig::tip_force_std)
        .def_readwrite("routing_radius", &TendonRobotGtsamConfig::routing_radius)
        .def_readwrite("angle_functions", &TendonRobotGtsamConfig::angle_functions)
        .def_readwrite("angle_params", &TendonRobotGtsamConfig::angle_params)
        .def_readwrite("pose_drift_p_std", &TendonRobotGtsamConfig::pose_drift_p_std)
        .def_readwrite("pose_drift_r_std", &TendonRobotGtsamConfig::pose_drift_r_std)
        .def_readwrite("tension_drift_std", &TendonRobotGtsamConfig::tension_drift_std)
        .def_readwrite("wrench_drift_std", &TendonRobotGtsamConfig::wrench_drift_std)
        .def_readwrite("p_meas_std", &TendonRobotGtsamConfig::p_meas_std);
    
    m.def("get_default_config", &get_default_config);

    py::class_<TendonRobotGtsam>(m, "TendonRobotGtsam")
    .def(py::init<const TendonRobotGtsamConfig&>())
    .def("solve", &TendonRobotGtsam::solve,
         py::arg("tensions"),
         py::arg("tip_force"),
         py::arg("num_samples"),
         py::call_guard<py::gil_scoped_release>());
}
