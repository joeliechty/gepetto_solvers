#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonFingerModel.h"
#include "TendonFingerSolver.h"
#include "TendonFingerTrajectoryPlanner.h"
#include "TendonFingerIterativeSolver.h"
#include "utils/EnvironmentFactors.h"

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

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
        .def_readwrite("tip_radius", &TendonFingerSolverConfig::tip_radius)
        .def_readwrite("sigma_twist_rot", &TendonFingerSolverConfig::sigma_twist_rot)
        .def_readwrite("sigma_twist_pos", &TendonFingerSolverConfig::sigma_twist_pos)
        .def_readwrite("sigma_stress_force", &TendonFingerSolverConfig::sigma_stress_force)
        .def_readwrite("sigma_stress_moment", &TendonFingerSolverConfig::sigma_stress_moment)
        .def_readwrite("sigma_base_pos", &TendonFingerSolverConfig::sigma_base_pos)
        .def_readwrite("sigma_base_rot", &TendonFingerSolverConfig::sigma_base_rot)
        .def_readwrite("tendon_input", &TendonFingerSolverConfig::tendon_input)
        .def_readwrite("per_disc_tendon_input", &TendonFingerSolverConfig::per_disc_tendon_input)
        .def_readwrite("base_pose", &TendonFingerSolverConfig::base_pose)
        .def_readwrite("use_hand_base", &TendonFingerSolverConfig::use_hand_base)
        .def_readwrite("hand_base_offset", &TendonFingerSolverConfig::hand_base_offset)
        .def_readwrite("sphere_contact", &TendonFingerSolverConfig::sphere_contact)
        .def_readwrite("sdf_contact", &TendonFingerSolverConfig::sdf_contact);

    py::class_<SpherePrimitiveContactConfig>(m, "SpherePrimitiveContactConfig")
        .def(py::init<>())
        .def_readwrite("finger_node_index",  &SpherePrimitiveContactConfig::finger_node_index)
        .def_readwrite("finger_node_radius", &SpherePrimitiveContactConfig::finger_node_radius)
        .def_readwrite("sphere_center",      &SpherePrimitiveContactConfig::sphere_center)
        .def_readwrite("sphere_radius",      &SpherePrimitiveContactConfig::sphere_radius)
        .def_readwrite("sphere_pose_cov",    &SpherePrimitiveContactConfig::sphere_pose_cov)
        .def_readwrite("witness",            &SpherePrimitiveContactConfig::witness);

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
             py::arg("tip_meas"))
        .def("get_factor_error_summary",
             &TendonFingerSolverDispatch::get_factor_error_summary)
        .def("get_factor_errors_by_type",
             &TendonFingerSolverDispatch::get_factor_errors_by_type)
        .def("get_initial_factor_error_summary",
             &TendonFingerSolverDispatch::get_initial_factor_error_summary)
        .def("get_hessian_and_gradient",
             &TendonFingerSolverDispatch::get_hessian_and_gradient)
        .def("get_intermediate_solutions",
             &TendonFingerSolverDispatch::get_intermediate_solutions)
        .def("get_initial_solution",
             &TendonFingerSolverDispatch::get_initial_solution);

    // --- Environment (collision/contact) ---

    py::class_<crest_sparse::EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("object_pose_mean",     &crest_sparse::EnvironmentConfig::object_pose_mean)
        .def_readwrite("object_pose_cov",      &crest_sparse::EnvironmentConfig::object_pose_cov)
        .def_readwrite("object_pose_per_step", &crest_sparse::EnvironmentConfig::object_pose_per_step)
        .def_readwrite("collision_avoidance",    &crest_sparse::EnvironmentConfig::collision_avoidance)
        .def_readwrite("collision_epsilon",      &crest_sparse::EnvironmentConfig::collision_epsilon)
        .def_readwrite("collision_sigma",        &crest_sparse::EnvironmentConfig::collision_sigma)
        .def_readwrite("collision_node_indices", &crest_sparse::EnvironmentConfig::collision_node_indices)
        .def_readwrite("collision_node_radii",   &crest_sparse::EnvironmentConfig::collision_node_radii)
        .def_readwrite("collision_node_is_proximal", &crest_sparse::EnvironmentConfig::collision_node_is_proximal)
        .def_readwrite("target_contact_node", &crest_sparse::EnvironmentConfig::target_contact_node)
        .def_readwrite("contact_node_radius", &crest_sparse::EnvironmentConfig::contact_node_radius)
        .def_readwrite("witness_point_seed",  &crest_sparse::EnvironmentConfig::witness_point_seed)
        .def("load_sdf", [](crest_sparse::EnvironmentConfig& self, const std::string& path) {
            openvdb::initialize();
            openvdb::io::File f(path);
            f.open();
            auto names = f.getGrids();
            if (names->empty()) {
                f.close();
                throw std::runtime_error("VDB file contains no grids: " + path);
            }
            self.sdf_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(names->at(0));
            f.close();
            if (!self.sdf_grid) {
                throw std::runtime_error("First grid in VDB file is not a FloatGrid: " + path);
            }
        }, py::arg("path"));

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
        .def_readwrite("sigma_ext_wrench_moment", &TrajectoryPlannerConfig::sigma_ext_wrench_moment)
        // Optional environment for collision/contact (Section 3). None => free-space planner.
        .def_readwrite("environment", &TrajectoryPlannerConfig::environment)
        // Optional analytic sphere-primitive terminal contact (mirrors model_config.sphere_contact).
        .def_readwrite("sphere_contact", &TrajectoryPlannerConfig::sphere_contact);

    py::class_<TrajectoryPlannerResult>(m, "TrajectoryPlannerResult")
        .def(py::init<>())
        .def_readwrite("trajectory", &TrajectoryPlannerResult::trajectory)
        .def_readwrite("meta", &TrajectoryPlannerResult::meta);

    py::class_<TendonFingerTrajectoryPlannerDispatch>(m, "TendonFingerTrajectoryPlanner")
        .def(py::init<const TrajectoryPlannerConfig&>())
        .def("plan", &TendonFingerTrajectoryPlannerDispatch::plan)
        .def("get_factor_error_summary",
             &TendonFingerTrajectoryPlannerDispatch::get_factor_error_summary);

    // --- Iterative (ISAM2) State Estimator ---

    py::class_<TendonFingerEstimatorConfig>(m, "TendonFingerEstimatorConfig")
        .def(py::init<>())
        .def_readwrite("base_config", &TendonFingerEstimatorConfig::base_config)
        .def_readwrite("background_tensions_mean", &TendonFingerEstimatorConfig::background_tensions_mean)
        .def_readwrite("background_tensions_cov",  &TendonFingerEstimatorConfig::background_tensions_cov)
        .def_readwrite("gp_tense_Qc", &TendonFingerEstimatorConfig::gp_tense_Qc)
        .def_readwrite("gp_len_Qc",   &TendonFingerEstimatorConfig::gp_len_Qc)
        .def_readwrite("gp_pose_Qc",  &TendonFingerEstimatorConfig::gp_pose_Qc)
        .def_readwrite("lag_sec",     &TendonFingerEstimatorConfig::lag_sec)
        .def_readwrite("homotopy_steps", &TendonFingerEstimatorConfig::homotopy_steps);

    py::class_<TendonFingerIterativeSolverDispatch>(m, "TendonFingerIterativeSolver")
        .def(py::init<const TendonFingerEstimatorConfig&, double>(),
             py::arg("config"), py::arg("bend_sigma"))
        .def("step", &TendonFingerIterativeSolverDispatch::step,
             py::arg("timestamp_sec"),
             py::arg("tensions_meas")     = std::nullopt,
             py::arg("lengths_meas")      = std::nullopt,
             py::arg("measured_bend")     = std::nullopt,
             py::arg("tip_wrench_meas")   = std::nullopt,
             py::arg("tip_position_meas") = std::nullopt)
        .def("get_current_marginals",
             &TendonFingerIterativeSolverDispatch::get_current_marginals)
        .def("num_tendons", &TendonFingerIterativeSolverDispatch::num_tendons);
}
