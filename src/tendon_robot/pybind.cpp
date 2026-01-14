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

    py::class_<TendonRobotSolverConfig>(m, "TendonRobotSolverConfig")
        .def(py::init<>())
        .def_readwrite("rod_length", &TendonRobotSolverConfig::rod_length)
        .def_readwrite("num_discs", &TendonRobotSolverConfig::num_discs)
        .def_readwrite("num_between_nodes", &TendonRobotSolverConfig::num_between_nodes)
        .def_readwrite("K_inv", &TendonRobotSolverConfig::K_inv)
        .def_readwrite("sigma_twist_rot", &TendonRobotSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_twist_pos", &TendonRobotSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_stress_force", &TendonRobotSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &TendonRobotSolverConfig::sigma_stress_moment)
        .def_readwrite("sigma_base_pos", &TendonRobotSolverConfig::sigma_base_pos)
        .def_readwrite("sigma_base_rot", &TendonRobotSolverConfig::sigma_base_rot)
        .def_readwrite("tendon_input", &TendonRobotSolverConfig::tendon_input);

    py::class_<TendonRobotMarginals>(m, "TendonRobotMarginals")
        .def(py::init<>())
        .def_readwrite("rod", &TendonRobotMarginals::rod)
        .def_readwrite("samples", &TendonRobotMarginals::samples)
        .def_readwrite("tendon_config", &TendonRobotMarginals::tendon_config)
        .def_readwrite("external_wrench_mean", &TendonRobotMarginals::external_wrench_mean)
        .def_readwrite("external_wrench_cov", &TendonRobotMarginals::external_wrench_cov)
        .def_readwrite("tensions_mean", &TendonRobotMarginals::tensions_mean)
        .def_readwrite("tensions_cov", &TendonRobotMarginals::tensions_cov)
        .def_readwrite("J_pose_tensions", &TendonRobotMarginals::J_pose_tensions);

    py::class_<Solution<TendonRobotMarginals>>(m, "TendonRobotSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<TendonRobotMarginals>::meta)
        .def_readwrite("marginals", &Solution<TendonRobotMarginals>::marginals);

    py::class_<TendonRobotSolver>(m, "TendonRobotSolver")
        .def(py::init<const TendonRobotSolverConfig&>())
        .def("solve", &TendonRobotSolver::solve,
             py::arg("tensions_mean"),
             py::arg("tensions_cov"));
}

// OLD

// PYBIND11_MODULE(tendon_robot, m) {
//     py::class_<TendonDiscConfig>(m, "TendonDiscConfig")
//         .def(py::init<>())
//         .def_readwrite("num_tendons", &TendonDiscConfig::num_tendons)
//         .def_readwrite("num_discs", &TendonDiscConfig::num_discs)
//         .def_readwrite("routing_radius", &TendonDiscConfig::routing_radius)
//         .def_readwrite("disc_pose_idx", &TendonDiscConfig::disc_pose_idx)
//         .def_readwrite("local_holes", &TendonDiscConfig::local_holes);

//     py::class_<TendonRobotSolution>(m, "TendonRobotSolution")
//         .def(py::init<>())
//         .def_readwrite("backbone_pose_mean", &TendonRobotSolution::backbone_pose_mean)
//         .def_readwrite("backbone_pose_cov", &TendonRobotSolution::backbone_pose_cov)
//         .def_readwrite("tip_pose_samples", &TendonRobotSolution::tip_pose_samples)
//         .def_readwrite("fbg_array_samples", &TendonRobotSolution::fbg_array_samples)
//         .def_readwrite("applied_wrench_mean", &TendonRobotSolution::applied_wrench_mean)
//         .def_readwrite("applied_wrench_cov", &TendonRobotSolution::applied_wrench_cov)
//         .def_readwrite("tensions_mean", &TendonRobotSolution::tensions_mean)
//         .def_readwrite("tensions_cov", &TendonRobotSolution::tensions_cov)
//         .def_readwrite("J_pose_tensions", &TendonRobotSolution::J_pose_tensions)
//         .def_readwrite("solve_time_ms", &TendonRobotSolution::solve_time_ms)
//         .def_readwrite("extract_time_ms", &TendonRobotSolution::extract_time_ms)
//         .def_readwrite("total_time_ms", &TendonRobotSolution::total_time_ms)
//         .def_readwrite("tendon_disc_config", &TendonRobotSolution::tendon_disc_config);

//     py::enum_<RoutingAngleFunction>(m, "RoutingAngleFunction")
//         .value("CONSTANT", RoutingAngleFunction::CONSTANT)
//         .value("LINEAR", RoutingAngleFunction::LINEAR)
//         .export_values();

//     py::class_<RoutingFunctionParams>(m, "RoutingFunctionParams")
//         .def(py::init<>())
//         .def(py::init<double, double>(), py::arg("angle_offset"), py::arg("total_angle"))
//         .def_readwrite("angle_offset", &RoutingFunctionParams::angle_offset)
//         .def_readwrite("total_angle", &RoutingFunctionParams::total_angle);

//     py::class_<TendonRobotConfig>(m, "TendonRobotConfig")
//         .def(py::init<>())
//         .def_readwrite("num_discs", &TendonRobotConfig::num_discs)
//         .def_readwrite("poses_between_discs", &TendonRobotConfig::poses_between_discs)
//         .def_readwrite("rod_length", &TendonRobotConfig::rod_length)
//         .def_readwrite("rod_diameter", &TendonRobotConfig::rod_diameter)
//         .def_readwrite("youngs_modulus", &TendonRobotConfig::youngs_modulus)
//         .def_readwrite("shear_modulus", &TendonRobotConfig::shear_modulus)
//         .def_readwrite("routing_radius", &TendonRobotConfig::routing_radius)
//         .def_readwrite("use_midpoint", &TendonRobotConfig::use_midpoint)
        
//         .def_readwrite("cosserat_twist_r_std", &TendonRobotConfig::cosserat_twist_r_std)
//         .def_readwrite("small_force_std", &TendonRobotConfig::small_force_std)
//         .def_readwrite("small_moment_std", &TendonRobotConfig::small_moment_std)
//         .def_readwrite("small_r_std", &TendonRobotConfig::small_r_std)
//         .def_readwrite("small_p_std", &TendonRobotConfig::small_p_std)

//         .def_readwrite("tip_force_prior_std", &TendonRobotConfig::tip_force_prior_std)
//         .def_readwrite("dist_load_prior_std", &TendonRobotConfig::dist_load_prior_std)
//         .def_readwrite("dist_load_smoothness_std", &TendonRobotConfig::dist_load_smoothness_std)
        
//         .def_readwrite("tension_meas_std", &TendonRobotConfig::tension_meas_std)
//         .def_readwrite("tip_position_meas_std", &TendonRobotConfig::tip_position_meas_std)
//         .def_readwrite("fbg_strain_meas_std", &TendonRobotConfig::fbg_strain_meas_std)

//         .def_readwrite("tension_drift_std", &TendonRobotConfig::tension_drift_std)
//         .def_readwrite("tip_force_drift_std", &TendonRobotConfig::tip_force_drift_std)
//         .def_readwrite("dist_load_drift_std", &TendonRobotConfig::dist_load_drift_std)

//         .def_readwrite("angle_functions", &TendonRobotConfig::angle_functions)
//         .def_readwrite("angle_params", &TendonRobotConfig::angle_params);

//     py::class_<TipForceSolver>(m, "TipForceSolver")
//     .def(py::init<const TendonRobotConfig&>())
//     .def("step", &TipForceSolver::step,
//          py::arg("tensions_meas"),
//          py::arg("tip_position_meas"),
//          py::arg("num_samples"),
//          py::call_guard<py::gil_scoped_release>())
//     .def("simulation_step", &TipForceSolver::simulation_step,
//          py::arg("tensions"),
//          py::arg("tip_force"),
//          py::call_guard<py::gil_scoped_release>());

//     py::class_<DistLoadSolver>(m, "DistLoadSolver")
//     .def(py::init<const TendonRobotConfig&>())
//     .def("step", &DistLoadSolver::step,
//          py::arg("tensions_meas"),
//          py::arg("fbg_signals_meas"),
//          py::arg("num_samples"),
//          py::call_guard<py::gil_scoped_release>())
//     .def("step_simulation", &DistLoadSolver::step_simulation,
//          py::arg("tensions"),
//          py::arg("forces"),
//          py::call_guard<py::gil_scoped_release>());
// }
