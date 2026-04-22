#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonFingerModel.h"
#include "TendonFingerSolver.h"
#include "TendonFingerTrajectoryPlanner.h"

namespace py = pybind11;

void bind_tendon_finger(py::module& m) {
    py::class_<TendonFingerSolverConfig>(m, "TendonFingerSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonFingerSolverConfig::base)
        .def_readwrite("rod_length", &TendonFingerSolverConfig::rod_length)
        .def_readwrite("num_discs", &TendonFingerSolverConfig::num_discs)
        .def_readwrite("num_between_nodes", &TendonFingerSolverConfig::num_between_nodes)
        .def_readwrite("num_tendons", &TendonFingerSolverConfig::num_tendons)
        .def_readwrite("K_inv", &TendonFingerSolverConfig::K_inv)
        .def_readwrite("K_inv_per_segment", &TendonFingerSolverConfig::K_inv_per_segment)
        .def_readwrite("disc_positions_normalized", &TendonFingerSolverConfig::disc_positions_normalized)
        .def_readwrite("sigma_twist_rot", &TendonFingerSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_twist_pos", &TendonFingerSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_stress_force", &TendonFingerSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &TendonFingerSolverConfig::sigma_stress_moment)
        .def_readwrite("sigma_base_pos", &TendonFingerSolverConfig::sigma_base_pos)
        .def_readwrite("sigma_base_rot", &TendonFingerSolverConfig::sigma_base_rot)
        .def_readwrite("tendon_input", &TendonFingerSolverConfig::tendon_input)
        .def_readwrite("per_disc_tendon_input", &TendonFingerSolverConfig::per_disc_tendon_input)
        .def_readwrite("base_pose", &TendonFingerSolverConfig::base_pose);

    py::class_<TendonFingerMarginals>(m, "TendonFingerMarginals")
        .def(py::init<>())
        .def_readwrite("rod", &TendonFingerMarginals::rod)
        .def_readwrite("tendon_config", &TendonFingerMarginals::tendon_config)
        .def_readwrite("external_wrenches", &TendonFingerMarginals::external_wrenches)
        .def_readwrite("tensions", &TendonFingerMarginals::tensions)
        .def_readwrite("J_pose_tensions", &TendonFingerMarginals::J_pose_tensions)
        .def_readwrite("tendon_lengths", &TendonFingerMarginals::tendon_lengths);

    py::class_<Solution<TendonFingerMarginals>>(m, "TendonFingerSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<TendonFingerMarginals>::meta)
        .def_readwrite("marginals", &Solution<TendonFingerMarginals>::marginals);

    py::class_<TendonFingerSolverDispatch>(m, "TendonFingerSolver")
        .def(py::init<const TendonFingerSolverConfig&>())
        .def("solve", &TendonFingerSolverDispatch::solve,
             py::arg("tensions"),
             py::arg("tip_force"),
             py::arg("tip_meas"));

    // --- Trajectory Planner ---

    py::class_<TrajectoryPlannerConfig>(m, "TrajectoryPlannerConfig")
        .def(py::init<>())
        .def_readwrite("model_config", &TrajectoryPlannerConfig::model_config)
        .def_readwrite("K", &TrajectoryPlannerConfig::K)
        .def_readwrite("dt", &TrajectoryPlannerConfig::dt)
        // Start boundary conditions (set to None to omit; start_tensions replaces bg prior at k=0)
        .def_readwrite("start_pose", &TrajectoryPlannerConfig::start_pose)
        .def_readwrite("start_pose_cov", &TrajectoryPlannerConfig::start_pose_cov)
        .def_readwrite("start_position", &TrajectoryPlannerConfig::start_position)
        .def_readwrite("start_position_cov", &TrajectoryPlannerConfig::start_position_cov)
        .def_readwrite("start_tensions", &TrajectoryPlannerConfig::start_tensions)
        .def_readwrite("start_tensions_cov", &TrajectoryPlannerConfig::start_tensions_cov)
        // Goal boundary conditions (set to None to omit; goal_tensions replaces bg prior at k=K)
        .def_readwrite("goal_pose", &TrajectoryPlannerConfig::goal_pose)
        .def_readwrite("goal_pose_cov", &TrajectoryPlannerConfig::goal_pose_cov)
        .def_readwrite("goal_position", &TrajectoryPlannerConfig::goal_position)
        .def_readwrite("goal_position_cov", &TrajectoryPlannerConfig::goal_position_cov)
        .def_readwrite("goal_tensions", &TrajectoryPlannerConfig::goal_tensions)
        .def_readwrite("goal_tensions_cov", &TrajectoryPlannerConfig::goal_tensions_cov)
        // Background tension prior
        .def_readwrite("background_tensions_mean", &TrajectoryPlannerConfig::background_tensions_mean)
        .def_readwrite("background_tensions_sigmas", &TrajectoryPlannerConfig::background_tensions_sigmas)
        .def_readwrite("gp_tense_Qc", &TrajectoryPlannerConfig::gp_tense_Qc)
        .def_readwrite("gp_len_Qc", &TrajectoryPlannerConfig::gp_len_Qc)
        .def_readwrite("tension_limit_alpha", &TrajectoryPlannerConfig::tension_limit_alpha)
        .def_readwrite("tension_limit_q_min", &TrajectoryPlannerConfig::tension_limit_q_min)
        .def_readwrite("active_tendon_indices", &TrajectoryPlannerConfig::active_tendon_indices)
        .def_readwrite("sigma_ext_wrench_force", &TrajectoryPlannerConfig::sigma_ext_wrench_force)
        .def_readwrite("sigma_ext_wrench_moment", &TrajectoryPlannerConfig::sigma_ext_wrench_moment);

    py::class_<TrajectoryPlannerResult>(m, "TrajectoryPlannerResult")
        .def(py::init<>())
        .def_readwrite("trajectory", &TrajectoryPlannerResult::trajectory)
        .def_readwrite("meta", &TrajectoryPlannerResult::meta);

    py::class_<TendonFingerTrajectoryPlannerDispatch>(m, "TendonFingerTrajectoryPlanner")
        .def(py::init<const TrajectoryPlannerConfig&>())
        .def("plan", &TendonFingerTrajectoryPlannerDispatch::plan);
}
