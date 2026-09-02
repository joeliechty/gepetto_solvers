#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "gepetto_solvers/hand/HandKinematicsRegistry.h"
#include "gepetto_solvers/hand/HandModel.h"
#include "gepetto_solvers/hand/HandSolver.h"
#include "gepetto_solvers/hand/HandSpec.h"
#include "gepetto_solvers/hand/HandTrajectoryPlanner.h"
#include "gepetto_solvers/hand/kinematics/rigid/RigidHandKinematics.h"
#include "gepetto_solvers/hand/kinematics/tendon/TendonHandKinematics.h"  // TendonDigitExtras

namespace py = pybind11;


void bind_hand(py::module& m) {
    // --- The kinematics-agnostic hand description ---
    //
    // HandSpec is what a solver is built from. Its `kinematics` field names a
    // C++ factory (see registered_hand_kinematics()); the payload under
    // `kinematics_config` is interpreted only by that factory, and HandModel
    // never inspects it.

    py::class_<gepetto_solvers::HandKinematicsConfig,
               std::shared_ptr<gepetto_solvers::HandKinematicsConfig>>(
        m, "HandKinematicsConfig",
        "Base class for a hand's kinematics payload. Interpreted only by the "
        "registered factory named in HandSpec.kinematics.");

    py::class_<gepetto_solvers::TendonHandKinematicsConfig,
               gepetto_solvers::HandKinematicsConfig,
               std::shared_ptr<gepetto_solvers::TendonHandKinematicsConfig>>(
        m, "TendonHandKinematicsConfig")
        .def(py::init<>())
        .def_readwrite("fingers", &gepetto_solvers::TendonHandKinematicsConfig::fingers,
                       "One TendonFingerSolverConfig per digit, in digit order.");

    py::class_<gepetto_solvers::RigidDigitSpec>(
        m, "RigidDigitSpec",
        "One digit of a URDF-described hand: its 1-DOF joints base to tip, and "
        "the frame names of sites 1..N. Site 0 is NOT listed -- it is the "
        "digit's fixed mount on the palm, which is the wrist variable itself.")
        .def(py::init<>())
        .def_readwrite("name", &gepetto_solvers::RigidDigitSpec::name)
        .def_readwrite("joints", &gepetto_solvers::RigidDigitSpec::joints)
        .def_readwrite("site_frames", &gepetto_solvers::RigidDigitSpec::site_frames);

    py::class_<gepetto_solvers::RigidHandKinematicsConfig,
               gepetto_solvers::HandKinematicsConfig,
               std::shared_ptr<gepetto_solvers::RigidHandKinematicsConfig>>(
        m, "RigidHandKinematicsConfig")
        .def(py::init<>())
        .def_readwrite("urdf_xml",
                       &gepetto_solvers::RigidHandKinematicsConfig::urdf_xml)
        .def_readwrite("urdf_path",
                       &gepetto_solvers::RigidHandKinematicsConfig::urdf_path)
        .def_readwrite("digits",
                       &gepetto_solvers::RigidHandKinematicsConfig::digits)
        .def_readwrite("sigma_fk",
                       &gepetto_solvers::RigidHandKinematicsConfig::sigma_fk,
                       "Diagonal of the kinematic relaxation covariance "
                       "Sigma_fk, [rot(3), pos(3)]. As it tightens, the "
                       "kinematics likelihood approaches a hard constraint.")
        .def_readwrite("site_sigma_fk",
                       &gepetto_solvers::RigidHandKinematicsConfig::site_sigma_fk,
                       "Per-site override of sigma_fk, [digit][site-1]. Empty "
                       "uses sigma_fk everywhere. Present because the "
                       "formulation defines Sigma_fk,i per FRAME.")
        .def_readwrite("q_init",
                       &gepetto_solvers::RigidHandKinematicsConfig::q_init,
                       "Seed configuration per digit; an empty entry seeds "
                       "that digit at zero.");

    py::class_<gepetto_solvers::HandSpec>(m, "HandSpec")
        .def(py::init<>())
        .def_readwrite("kinematics", &gepetto_solvers::HandSpec::kinematics,
                       "Registry key naming the HandKinematics factory to load.")
        .def_readwrite("digit_names", &gepetto_solvers::HandSpec::digit_names)
        .def_readwrite("opposing_digit", &gepetto_solvers::HandSpec::opposing_digit,
                       "Index of the digit that opposes the others in the "
                       "pre-grasp constraints (the thumb on an anatomical hand). "
                       "-1 means none, and those constraints are not built.")
        .def_readwrite("env", &gepetto_solvers::HandSpec::env,
                       "Per-digit task EnvironmentConfig (contact, collision, "
                       "support plane, half-space, pre-grasp).")
        .def_readwrite("sphere_contact", &gepetto_solvers::HandSpec::sphere_contact)
        .def_readwrite("kinematics_config", &gepetto_solvers::HandSpec::kinematics_config)
        .def("validate", &gepetto_solvers::HandSpec::validate,
             "Raise if the per-digit vectors disagree in length, the kinematics "
             "is unnamed, or the payload is missing.");

    m.def("make_tendon_hand_spec", &gepetto_solvers::make_tendon_hand_spec,
          py::arg("finger_configs"), py::arg("opposing_digit") = -1,
          "Build a HandSpec for the \"tendon\" kinematics from a list of "
          "(name, TendonFingerSolverConfig) pairs, splitting each config's "
          "sdf_contact / sphere_contact off into the spec's task half.");

    m.def("registered_hand_kinematics", &gepetto_solvers::registered_hand_kinematics,
          "Every hand kinematics this build can load, sorted.");

    py::class_<HandSolverConfig>(m, "HandSolverConfig")
        .def(py::init<>())
        .def_readwrite("base", &HandSolverConfig::base)
        .def_readwrite("wrist_pose", &HandSolverConfig::wrist_pose)
        .def_readwrite("sigma_wrist_pos", &HandSolverConfig::sigma_wrist_pos)
        .def_readwrite("sigma_wrist_rot", &HandSolverConfig::sigma_wrist_rot)
        .def_readwrite("goal_positions", &HandSolverConfig::goal_positions)
        .def_readwrite("goal_position_cov", &HandSolverConfig::goal_position_cov)
        .def_readwrite("initial_state", &HandSolverConfig::initial_state,
                       "Optional warm-start posture (HandState from a "
                       "previous solve on the same hand). None => the "
                       "straight-rod, zero-tension cold start.");

    // --- the solved state, and the neutral shape it takes ---

    py::class_<SiteState>(m, "SiteState")
        .def(py::init<>())
        .def_readwrite("pose", &SiteState::pose)
        .def_readwrite("stress", &SiteState::stress,
                       "Continuum-rod stress; zero on a mechanism with none.")
        .def_readwrite("wrench", &SiteState::wrench,
                       "Continuum-rod wrench; zero on a mechanism with none.");

    py::class_<DigitExtras, std::shared_ptr<DigitExtras>>(
        m, "DigitExtras",
        "Base class for per-digit state only one kind of mechanism has. Check "
        "for None, then read the derived type (e.g. TendonDigitExtras).");

    py::class_<gepetto_solvers::TendonDigitExtras, DigitExtras,
               std::shared_ptr<gepetto_solvers::TendonDigitExtras>>(
        m, "TendonDigitExtras")
        .def(py::init<>())
        .def_readwrite("tendon_config",
                       &gepetto_solvers::TendonDigitExtras::tendon_config)
        .def_readwrite("external_wrenches",
                       &gepetto_solvers::TendonDigitExtras::external_wrenches)
        .def_readwrite("J_pose_tensions",
                       &gepetto_solvers::TendonDigitExtras::J_pose_tensions);

    py::class_<DigitState>(m, "DigitState")
        .def(py::init<>())
        .def_readwrite("sites", &DigitState::sites,
                       "One per site, base first and tip last.")
        .def_readwrite("actuation", &DigitState::actuation,
                       "What drives this digit: tendon tensions on the tendon "
                       "hand, joint positions on a rigid-body one.")
        .def_readwrite("displacement", &DigitState::displacement,
                       "The digit's displacement readout where it has one "
                       "distinct from its actuation (tendon lengths). Empty "
                       "when actuation IS position.")
        .def_readwrite("collision_sites", &DigitState::collision_sites,
                       "Indices into `sites` that carry a collision sphere.")
        .def_readwrite("extras", &DigitState::extras,
                       "Mechanism-specific state, or None.");

    py::class_<HandState>(m, "HandState")
        .def(py::init<>())
        .def_readwrite("digits", &HandState::digits,
                       "One entry per digit, in digit order.")
        .def_readwrite("digit_names", &HandState::digit_names)
        .def_readwrite("wrist_pose", &HandState::wrist_pose,
                       "The shared wrist as a 4x4 in the world frame. Carried "
                       "rather than derived: recovering it is a per-mechanism "
                       "question, and this is each kinematics' answer to it.");

    py::class_<Solution<HandState>>(m, "HandSolution")
        .def(py::init<>())
        .def_readwrite("meta", &Solution<HandState>::meta)
        .def_readwrite("marginals", &Solution<HandState>::marginals);

    py::class_<HandSolver>(m, "HandSolver")
        .def(py::init<const gepetto_solvers::HandSpec&, const HandSolverConfig&>(),
             py::arg("spec"), py::arg("config"))
        // GIL released for the duration, as every other solver in this module
        // does. Not an optimization: an AL outer iteration is ~1.4 s of C++, and
        // holding the GIL across it freezes the whole interpreter -- an
        // interactive caller's stop button cannot even be RECEIVED, since the
        // thread that would run its callback is unschedulable. Safe because both
        // arguments are taken by value (see HandSolver.h), so pybind has
        // finished converting them before the guard drops the GIL and the C++
        // retains no Python references.
        .def("solve", &HandSolver::solve,
             py::arg("tensions"), py::arg("tip_wrenches"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_wrist_pose", &HandSolver::set_wrist_pose,
             py::arg("wrist_pose"),
             "Re-aim the shared wrist prior between solves (4x4, world frame) "
             "without rebuilding the solver. solve() then warm-starts from the "
             "previous solution instead of cold-starting from a straight hand.")
        .def("num_fingers", &HandSolver::num_fingers)
        .def("get_factor_error_summary", &HandSolver::get_factor_error_summary)
        .def("get_intermediate_solutions",
             &HandSolver::get_intermediate_solutions,
             "Per-iteration hand snapshots from the last solve(); requires "
             "config.base.record_iterations = True. One entry per AL outer "
             "iteration (subject to iteration_sample_interval). Means only -- "
             "no covariance.")
        .def("get_initial_solution", &HandSolver::get_initial_solution,
             "The initial-guess hand state (start of the last solve()), for the "
             "first frame of a step animation.")
        .def("set_initial_duals", &HandSolver::set_initial_duals,
             py::arg("duals"),
             "Seed this solver's Augmented Lagrangian multipliers from another "
             "solver's get_al_duals(), matched by constraint identity. For "
             "continuing a solve whose CONSTRAINT SET changed: shared "
             "constraints keep their multipliers, new ones start at zero, and "
             "mu is carried clamped by config.base.al_transfer_mu_max.")
        .def("get_al_duals", &HandSolver::get_al_duals,
             py::return_value_policy::copy,
             "The AL multipliers and penalty weight after the last solve, "
             "tagged with the identity of the constraint each belongs to.")
        .def("al_transfer_report", &HandSolver::al_transfer_report,
             py::return_value_policy::copy,
             "How much of the last set_initial_duals() transfer matched.")
        .def("reset_al_duals", &HandSolver::reset_al_duals,
             "Restart the Augmented Lagrangian homotopy from the current "
             "posture: drops the carried multipliers and penalty weight but "
             "keeps the solved values. Only meaningful under "
             "config.base.al_warm_start_duals.")
        .def("constraint_tags_eq",
             [](const HandSolver& s) { return s.constraint_tags().eq; },
             "Identity tag of every EQUALITY constraint in the built graph, in "
             "graph order (\"obj.center|f0\", \"obj.set|f0\", \"obj.witness|f0\", "
             "\"table|f2\", ...). This is how a graph-STRUCTURE change is proved: "
             "get_factor_error_summary() groups by C++ type, so every equality "
             "shows up as one gtsam::ZeroCostConstraint bucket regardless of "
             "which factor it wraps, and solved poses are confounded by whether "
             "the scene converged at all.")
        .def("constraint_tags_ineq",
             [](const HandSolver& s) { return s.constraint_tags().ineq; },
             "The same for every INEQUALITY constraint (\"col.obj|f0|n3\", "
             "\"col.self|f0|f1|...\", \"col.plane|...\").");

    // --- Trajectory Planner (Section 1.4) ---

    py::class_<HandTrajectoryPlannerConfig>(m, "HandTrajectoryPlannerConfig")
        .def(py::init<>())
        .def_readwrite("base", &HandTrajectoryPlannerConfig::base)
        .def_readwrite("K", &HandTrajectoryPlannerConfig::K)
        .def_readwrite("dt", &HandTrajectoryPlannerConfig::dt)
        .def_readwrite("wrist_pose", &HandTrajectoryPlannerConfig::wrist_pose)
        .def_readwrite("sigma_wrist_pos", &HandTrajectoryPlannerConfig::sigma_wrist_pos)
        .def_readwrite("sigma_wrist_rot", &HandTrajectoryPlannerConfig::sigma_wrist_rot)
        .def_readwrite("gp_wrist_Qc", &HandTrajectoryPlannerConfig::gp_wrist_Qc)
        .def_readwrite("gp_actuation_Qc", &HandTrajectoryPlannerConfig::gp_actuation_Qc)
        .def_readwrite("gp_displacement_Qc", &HandTrajectoryPlannerConfig::gp_displacement_Qc)
        .def_readwrite("goal_positions", &HandTrajectoryPlannerConfig::goal_positions)
        .def_readwrite("goal_position_cov", &HandTrajectoryPlannerConfig::goal_position_cov)
        .def_readwrite("k_touch", &HandTrajectoryPlannerConfig::k_touch);

    py::class_<HandTrajectoryResult>(m, "HandTrajectoryResult")
        .def(py::init<>())
        .def_readwrite("trajectory", &HandTrajectoryResult::trajectory)
        .def_readwrite("meta", &HandTrajectoryResult::meta);

    py::class_<HandTrajectoryPlanner>(m, "HandTrajectoryPlanner")
        .def(py::init<const gepetto_solvers::HandSpec&,
                      const HandTrajectoryPlannerConfig&>(),
             py::arg("spec"), py::arg("config"))
        .def("plan", &HandTrajectoryPlanner::plan,
             py::arg("tensions"), py::arg("tip_wrenches"),
             py::arg("start_tensions") = std::vector<VectorXGaussian>{})
        .def("num_fingers", &HandTrajectoryPlanner::num_fingers)
        .def("get_factor_error_summary",
             &HandTrajectoryPlanner::get_factor_error_summary)
        .def("get_factor_errors_by_type",
             &HandTrajectoryPlanner::get_factor_errors_by_type)
        .def("get_initial_factor_error_summary",
             &HandTrajectoryPlanner::get_initial_factor_error_summary,
             "Factor-error summary evaluated at the initial guess (before the "
             "solve), to gauge how poor the seed is per factor type.")
        .def("get_hessian_and_gradient",
             &HandTrajectoryPlanner::get_hessian_and_gradient,
             "Dense (Hessian, gradient) of the graph linearized at the final "
             "solution, for conditioning diagnostics (condition number, "
             "near-null eigenvalues flagging gauge freedom).")
        .def("get_intermediate_solutions",
             &HandTrajectoryPlanner::get_intermediate_solutions,
             "Per-iteration trajectory snapshots from the last plan(); requires "
             "config.base.record_iterations = True. One entry per AL outer "
             "iteration (subject to iteration_sample_interval).")
        .def("get_initial_solution",
             &HandTrajectoryPlanner::get_initial_solution,
             "The initial-guess trajectory (start of the last plan()), for the "
             "first frame of a step animation.");
}
