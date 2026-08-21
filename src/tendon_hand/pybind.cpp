#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonHandModel.h"
#include "TendonHandSolver.h"
#include "TendonHandTrajectoryPlanner.h"

namespace py = pybind11;


void bind_tendon_hand(py::module& m) {
    py::class_<TendonHandSolverConfig>(m, "TendonHandSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonHandSolverConfig::base)
        .def_readwrite("wrist_pose", &TendonHandSolverConfig::wrist_pose)
        .def_readwrite("sigma_wrist_pos", &TendonHandSolverConfig::sigma_wrist_pos)
        .def_readwrite("sigma_wrist_rot", &TendonHandSolverConfig::sigma_wrist_rot)
        .def_readwrite("goal_positions", &TendonHandSolverConfig::goal_positions)
        .def_readwrite("goal_position_cov", &TendonHandSolverConfig::goal_position_cov)
        .def_readwrite("initial_state", &TendonHandSolverConfig::initial_state,
                       "Optional warm-start posture (TendonHandMarginals from a "
                       "previous solve on the same finger configs). None => the "
                       "straight-rod, zero-tension cold start.");

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
        // GIL released for the duration, as every other solver in this module
        // does. Not an optimization: an AL outer iteration is ~1.4 s of C++, and
        // holding the GIL across it freezes the whole interpreter -- an
        // interactive caller's stop button cannot even be RECEIVED, since the
        // thread that would run its callback is unschedulable. Safe because both
        // arguments are taken by value (see TendonHandSolver.h), so pybind has
        // finished converting them before the guard drops the GIL and the C++
        // retains no Python references.
        .def("solve", &TendonHandSolver::solve,
             py::arg("tensions"), py::arg("tip_wrenches"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_wrist_pose", &TendonHandSolver::set_wrist_pose,
             py::arg("wrist_pose"),
             "Re-aim the shared wrist prior between solves (4x4, world frame) "
             "without rebuilding the solver. solve() then warm-starts from the "
             "previous solution instead of cold-starting from a straight hand.")
        .def("num_fingers", &TendonHandSolver::num_fingers)
        .def("get_factor_error_summary", &TendonHandSolver::get_factor_error_summary)
        .def("get_intermediate_solutions",
             &TendonHandSolver::get_intermediate_solutions,
             "Per-iteration hand snapshots from the last solve(); requires "
             "config.base.record_iterations = True. One entry per AL outer "
             "iteration (subject to iteration_sample_interval). Means only -- "
             "no covariance.")
        .def("get_initial_solution", &TendonHandSolver::get_initial_solution,
             "The initial-guess hand state (start of the last solve()), for the "
             "first frame of a step animation.")
        .def("set_initial_duals", &TendonHandSolver::set_initial_duals,
             py::arg("duals"),
             "Seed this solver's Augmented Lagrangian multipliers from another "
             "solver's get_al_duals(), matched by constraint identity. For "
             "continuing a solve whose CONSTRAINT SET changed: shared "
             "constraints keep their multipliers, new ones start at zero, and "
             "mu is carried clamped by config.base.al_transfer_mu_max.")
        .def("get_al_duals", &TendonHandSolver::get_al_duals,
             py::return_value_policy::copy,
             "The AL multipliers and penalty weight after the last solve, "
             "tagged with the identity of the constraint each belongs to.")
        .def("al_transfer_report", &TendonHandSolver::al_transfer_report,
             py::return_value_policy::copy,
             "How much of the last set_initial_duals() transfer matched.")
        .def("reset_al_duals", &TendonHandSolver::reset_al_duals,
             "Restart the Augmented Lagrangian homotopy from the current "
             "posture: drops the carried multipliers and penalty weight but "
             "keeps the solved values. Only meaningful under "
             "config.base.al_warm_start_duals.")
        .def("constraint_tags_eq",
             [](const TendonHandSolver& s) { return s.constraint_tags().eq; },
             "Identity tag of every EQUALITY constraint in the built graph, in "
             "graph order (\"obj.center|f0\", \"obj.set|f0\", \"obj.witness|f0\", "
             "\"table|f2\", ...). This is how a graph-STRUCTURE change is proved: "
             "get_factor_error_summary() groups by C++ type, so every equality "
             "shows up as one gtsam::ZeroCostConstraint bucket regardless of "
             "which factor it wraps, and solved poses are confounded by whether "
             "the scene converged at all.")
        .def("constraint_tags_ineq",
             [](const TendonHandSolver& s) { return s.constraint_tags().ineq; },
             "The same for every INEQUALITY constraint (\"col.obj|f0|n3\", "
             "\"col.self|f0|f1|...\", \"col.plane|...\").");

    // --- Trajectory Planner (Section 1.4) ---

    py::class_<TendonHandTrajectoryPlannerConfig>(m, "TendonHandTrajectoryPlannerConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonHandTrajectoryPlannerConfig::base)
        .def_readwrite("K", &TendonHandTrajectoryPlannerConfig::K)
        .def_readwrite("dt", &TendonHandTrajectoryPlannerConfig::dt)
        .def_readwrite("wrist_pose", &TendonHandTrajectoryPlannerConfig::wrist_pose)
        .def_readwrite("sigma_wrist_pos", &TendonHandTrajectoryPlannerConfig::sigma_wrist_pos)
        .def_readwrite("sigma_wrist_rot", &TendonHandTrajectoryPlannerConfig::sigma_wrist_rot)
        .def_readwrite("gp_wrist_Qc", &TendonHandTrajectoryPlannerConfig::gp_wrist_Qc)
        .def_readwrite("gp_tense_Qc", &TendonHandTrajectoryPlannerConfig::gp_tense_Qc)
        .def_readwrite("gp_len_Qc", &TendonHandTrajectoryPlannerConfig::gp_len_Qc)
        .def_readwrite("goal_positions", &TendonHandTrajectoryPlannerConfig::goal_positions)
        .def_readwrite("goal_position_cov", &TendonHandTrajectoryPlannerConfig::goal_position_cov)
        .def_readwrite("k_touch", &TendonHandTrajectoryPlannerConfig::k_touch);

    py::class_<TendonHandTrajectoryResult>(m, "TendonHandTrajectoryResult")
        .def(py::init<>())
        .def_readwrite("trajectory", &TendonHandTrajectoryResult::trajectory)
        .def_readwrite("meta", &TendonHandTrajectoryResult::meta);

    py::class_<TendonHandTrajectoryPlanner>(m, "TendonHandTrajectoryPlanner")
        .def(py::init<
                const std::vector<std::pair<std::string, TendonFingerSolverConfig>>&,
                const TendonHandTrajectoryPlannerConfig&>(),
             py::arg("finger_configs"), py::arg("config"))
        .def("plan", &TendonHandTrajectoryPlanner::plan,
             py::arg("tensions"), py::arg("tip_wrenches"),
             py::arg("start_tensions") = std::vector<VectorXGaussian>{})
        .def("num_fingers", &TendonHandTrajectoryPlanner::num_fingers)
        .def("get_factor_error_summary",
             &TendonHandTrajectoryPlanner::get_factor_error_summary)
        .def("get_factor_errors_by_type",
             &TendonHandTrajectoryPlanner::get_factor_errors_by_type)
        .def("get_initial_factor_error_summary",
             &TendonHandTrajectoryPlanner::get_initial_factor_error_summary,
             "Factor-error summary evaluated at the initial guess (before the "
             "solve), to gauge how poor the seed is per factor type.")
        .def("get_hessian_and_gradient",
             &TendonHandTrajectoryPlanner::get_hessian_and_gradient,
             "Dense (Hessian, gradient) of the graph linearized at the final "
             "solution, for conditioning diagnostics (condition number, "
             "near-null eigenvalues flagging gauge freedom).")
        .def("get_intermediate_solutions",
             &TendonHandTrajectoryPlanner::get_intermediate_solutions,
             "Per-iteration trajectory snapshots from the last plan(); requires "
             "config.base.record_iterations = True. One entry per AL outer "
             "iteration (subject to iteration_sample_interval).")
        .def("get_initial_solution",
             &TendonHandTrajectoryPlanner::get_initial_solution,
             "The initial-guess trajectory (start of the last plan()), for the "
             "first frame of a step animation.");
}
