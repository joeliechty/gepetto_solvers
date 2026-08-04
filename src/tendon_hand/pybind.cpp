#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "TendonHandController.h"
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
        .def("solve", &TendonHandSolver::solve,
             py::arg("tensions"), py::arg("tip_wrenches"))
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
             "config.base.al_warm_start_duals.");

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

    // --- Phased real-time controller (Section 1.8) ---

    py::enum_<ControllerPhase>(m, "ControllerPhase",
        "Which Section 1.8 constraint set the controller enforces. These are not "
        "time windows inside one graph (that is the trajectory planner) but three "
        "different constraint sets over the same single-state graph.")
        .value("PreGrasp", ControllerPhase::PreGrasp,
               "Phase 0 (Eq 1.94-1.98): pre-grasp positioning. The only phase "
               "with no equality constraints -- soft pose and tension TARGETS "
               "servo the hand to a collision-free hover posture above the "
               "object so phase 1 starts well-conditioned.")
        .value("SupportContact", ControllerPhase::SupportContact,
               "Phase 1 (Eq 1.96-1.100): contact spheres onto the support surface "
               "inside their opposition half-spaces; everything else avoids the "
               "table and the object.")
        .value("ObjectApproach", ControllerPhase::ObjectApproach,
               "Phase 2 (Eq 1.102-1.106): hold the surface and slide onto the "
               "hyper-ellipsoid proxy (center-direct, no witness point).")
        .value("ObjectServo", ControllerPhase::ObjectServo,
               "Phase 3 (Eq 1.112-1.125): relax the surface equality to an "
               "inequality, swap the proxy for the true SDF, and servo with the "
               "4-residual witness contact.");

    py::enum_<StepAnchor>(m, "StepAnchor",
        "What anchors a control tick to the measured robot state.")
        .value("Tension", StepAnchor::Tension,
               "Tension prior only (Eq 1.95). Pure-simulation default.")
        .value("Length", StepAnchor::Length,
               "Tendon-length prior only (the Eq 1.13 analogue). The "
               "hardware-faithful mode: length is what the motor commands and "
               "what survives a tick, whereas a disturbance contact changes "
               "tension without the robot having moved.")
        .value("Both", StepAnchor::Both,
               "Both priors. Diagnostic; over-constrains a real tick.");

    py::class_<TendonHandControllerConfig>(m, "TendonHandControllerConfig")
        .def(py::init<>())
        .def_readwrite("base", &TendonHandControllerConfig::base)
        .def_readwrite("wrist_pose", &TendonHandControllerConfig::wrist_pose)
        .def_readwrite("sigma_wrist_pos", &TendonHandControllerConfig::sigma_wrist_pos)
        .def_readwrite("sigma_wrist_rot", &TendonHandControllerConfig::sigma_wrist_rot)
        .def_readwrite("phase", &TendonHandControllerConfig::phase)
        .def_readwrite("step_anchor", &TendonHandControllerConfig::step_anchor)
        .def_readwrite("pregrasp_wrist_pose",
                       &TendonHandControllerConfig::pregrasp_wrist_pose)
        .def_readwrite("sigma_pregrasp_pos",
                       &TendonHandControllerConfig::sigma_pregrasp_pos)
        .def_readwrite("sigma_pregrasp_rot",
                       &TendonHandControllerConfig::sigma_pregrasp_rot)
        .def_readwrite("pregrasp_tensions",
                       &TendonHandControllerConfig::pregrasp_tensions)
        .def_readwrite("initial_state",
                       &TendonHandControllerConfig::initial_state,
                       "Theta_curr's ROBOT STATE: the posture the first tick "
                       "starts from, as the marginals of any solve on the same "
                       "finger configs. None => the straight-hand, zero-tension "
                       "cold start, which makes tick 1 travel from a straight "
                       "hand back to wherever the robot actually is.");

    py::class_<TendonHandController>(m, "TendonHandController")
        .def(py::init<
                const std::vector<std::pair<std::string, TendonFingerSolverConfig>>&,
                const TendonHandControllerConfig&>(),
             py::arg("finger_configs"), py::arg("config"))
        .def("step", &TendonHandController::step,
             py::arg("tensions"), py::arg("tip_wrenches"),
             py::arg("lengths") = std::vector<VectorXGaussian>{},
             "One control tick: re-solve the single-state constrained IK problem "
             "anchored to the measured state, warm-started from the previous "
             "tick. lengths is required when step_anchor is Length or Both.")
        .def("set_phase", &TendonHandController::set_phase, py::arg("phase"),
             "Switch the active constraint set, preserving the converged robot "
             "state and seeding only the variables the new phase introduces.")
        .def("phase", &TendonHandController::phase)
        .def("set_wrist_pose", &TendonHandController::set_wrist_pose,
             py::arg("wrist_pose"),
             "Update the base-pose step-prior mean (4x4, world frame) without a "
             "rebuild.")
        .def("set_pregrasp_target", &TendonHandController::set_pregrasp_target,
             py::arg("wrist_pose"), py::arg("sigma_pos"), py::arg("sigma_rot"),
             py::arg("tensions") = std::vector<VectorXGaussian>{},
             "Re-aim the phase-0 Eq 1.94/1.95 targets without a rebuild, so the "
             "clearance height or servo rate can move between ticks.")
        .def("current_wrist_pose", &TendonHandController::current_wrist_pose,
             "The shared base pose T_base at the current retained state (4x4, "
             "world frame). Needed to close the Theta_curr feedback loop: the "
             "marginals carry per-finger state only, so without this the step "
             "prior's mean can never follow the hand.")
        .def("current_tendon_lengths",
             &TendonHandController::current_tendon_lengths,
             "Tendon lengths of every finger at the current retained state. Use "
             "it to seed L_curr for the first tick when anchoring on length, "
             "where there is no measurement yet.")
        .def("current_witness_points",
             &TendonHandController::current_witness_points,
             "The solved witness point p_c,obj of every finger at the current "
             "retained state, or None for a finger that has none. Only phase 3 "
             "instantiates witnesses, so all-None is the normal answer in phases "
             "0-2. This is the actual Symbol('Y', i) variable the Eq 1.114-1.117 "
             "residuals act on -- an analytic surface projection is a look-alike "
             "and cannot show witness drift.")
        .def("phase_violations", &TendonHandController::phase_violations,
             "Worst absolute violation per constraint family at the current "
             "solution, as (name, max_abs) pairs. Covers the equality/goal "
             "families that drive phase advancement; collision penetration is a "
             "whole-hand safety property reported separately.")
        .def("set_state", &TendonHandController::set_state, py::arg("state"),
             "Re-seed the retained robot state from a solved posture mid-run -- "
             "a teleport the step-prior trust region could never absorb in one "
             "tick. Unlike set_wrist_pose (which only re-aims the step prior's "
             "mean) this replaces the values, so it also drops the accumulated "
             "AL duals.")
        .def("reset_al_duals", &TendonHandController::reset_al_duals,
             "Start the next tick's Augmented Lagrangian from a cold outer "
             "loop. set_phase() and set_state() do this for you; call it "
             "directly if you change the constrained problem another way (a "
             "moved object, a new contact mask).")
        .def("num_fingers", &TendonHandController::num_fingers)
        .def("get_factor_error_summary",
             &TendonHandController::get_factor_error_summary);
}
