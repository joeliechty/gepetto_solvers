#pragma once

#include "gepetto_solvers/hand/HandModel.h"
#include "gepetto_solvers/hand/HandSpec.h"
#include "gepetto_solvers/utils/Gaussians.h"
#include "gepetto_solvers/utils/SolverBase.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>


// Trajectory planner for a multi-finger tendon hand (PDF Section 1.4).
//
// The control actions we plan are the wrist pose and per-finger tendon tensions
// over K+1 timesteps (k = 0..K). The planner owns K+1 HandModel instances
// (one per step), each with its own step-indexed wrist variable, and links them
// with Gaussian-process temporal priors:
//   * wrist pose   : BetweenFactor<Pose3>          (Eq 1.41/1.42)
//   * tendon tension: per-finger BetweenFactor<Vec> (Eq 1.11)
//   * tendon length : per-finger BetweenFactor<Vec> (Eq 1.13, optional)
//
// Boundary (Section 1.4.2): the start step (k=0) carries a loose wrist prior
// (the hand-pose prior, Eq 1.40) so the optimizer can reposition the wrist; the
// terminal step (k=K) carries the per-finger contact constraints (contact-as-
// goal). Interior steps carry neither; their wrists are pinned by the GP chain
// and finger kinematics. Because a contact constraint exists at k=K the solve
// runs on SolverBase's Augmented Lagrangian path.
struct HandTrajectoryPlannerConfig {
    SolverBaseConfig base;

    int    K  = 20;    // number of steps (K+1 states, k=0..K)
    double dt = 0.1;   // step duration (scales the GP process-noise covariances)

    // Start wrist pose (world frame). Seeds every step's wrist and is the mean of
    // the k=0 hand-pose prior. In a real receding-horizon run this is the
    // *measured* current wrist pose, so it is strictly constrained by default.
    gtsam::Matrix4 wrist_pose = gtsam::Matrix4::Identity();

    // k=0 hand-pose prior tightness (Eq 1.40). Tight by default so the start wrist
    // is pinned to the measured pose (the known robot state); interior/terminal
    // wrists remain free (GP chain + contact). Loosen these to recover the free-
    // start behavior where the optimizer may also reposition the start wrist.
    double sigma_wrist_pos = 1e-4;
    double sigma_wrist_rot = 1e-3;

    // GP temporal prior process-noise covariances.
    gtsam::Matrix6  gp_wrist_Qc = gtsam::Matrix6::Identity();  // wrist pose (6x6)
    // Per-digit actuation (tendon tensions) and displacement (tendon lengths).
    // NxN each; an empty displacement Qc disables that chain.
    Eigen::MatrixXd gp_actuation_Qc;
    Eigen::MatrixXd gp_displacement_Qc;

    // Optional per-finger terminal tip-position goals (world frame), same order
    // as finger_configs. Empty => no goal priors (legacy behavior). When
    // non-empty, size must equal the number of fingers; each adds a soft
    // PositionPriorFactor on that finger's tip node at k=K -- the point-to-point
    // analogue of contact-as-goal. Because it is a soft prior (not a hard
    // constraint) the solve stays on the plain, non-AL path when no finger has a
    // contact configured.
    std::vector<gtsam::Vector3> goal_positions;
    Eigen::Matrix3d goal_position_cov = 1e-5 * Eigen::Matrix3d::Identity();

    // Slide-and-grasp touch step (Section 1.6.2). Splits the horizon into the
    // three phases about a contact-initialization step. When set (and a finger
    // carries a support plane, see EnvironmentConfig::plane_*), each step's plane
    // fields are scheduled per finger:
    //   k = 0          : no table collision, no table contact (measured start).
    //   1 <= k < k_touch (Phase 1): table collision only (free-space approach).
    //   k >= k_touch     (Phase 2/3): table sliding equality on the tip; the
    //                    terminal object grasp still fires at k=K (corner contact).
    // Unset (default) => no sliding: a configured plane acts purely as collision
    // at every plannable step, and a plane-free run is unchanged.
    std::optional<int> k_touch;
};


struct HandTrajectoryResult {
    std::vector<HandState> trajectory;  // K+1 entries, one per step
    SolutionMetadata meta;
};


class HandTrajectoryPlanner : SolverBase {
public:
    HandTrajectoryPlanner(
        const gepetto_solvers::HandSpec& spec,
        const HandTrajectoryPlannerConfig& config);

    // Plan the trajectory. tensions/tip_wrenches are per finger (same order as
    // finger_configs) and applied as the background-tension / tip-wrench priors.
    //
    // start_tensions (optional, per finger) is the *measured* tendon state at
    // k=0; when non-empty it replaces the background tension prior at k=0 (its
    // VectorXGaussian cov sets how strictly the start state is enforced), so the
    // hand begins in the known kinematic configuration (e.g. flexors slack / open
    // hand) rather than the background target. Empty => k=0 uses the background
    // prior like every other step (legacy behavior).
    HandTrajectoryResult plan(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches,
        const std::vector<VectorXGaussian>& start_tensions = {});

    int num_fingers() const {
        return models_.empty() ? 0 : models_[0]->num_digits();
    }

    // Re-expose SolverBase diagnostics (privately inherited).
    std::vector<std::tuple<std::string, int, double>>
        get_factor_error_summary() const { return SolverBase::get_factor_error_summary(); }

    // Per-factor-type residual lists at the final values_ (graph-traversal order).
    std::vector<std::pair<std::string, std::vector<double>>>
        get_factor_errors_by_type() const {
        return SolverBase::get_factor_errors_by_type();
    }

    // Same grouping as get_factor_error_summary(), but evaluated at the initial
    // guess (initial_values_) so we can see how poor the seed was per factor type.
    std::vector<std::tuple<std::string, int, double>>
        get_initial_factor_error_summary() const {
        return SolverBase::get_initial_factor_error_summary();
    }

    // Dense Hessian H and gradient g of the graph linearized at the final values_.
    // For conditioning diagnostics (condition number, near-null eigenvalues that
    // flag gauge freedom / ill-conditioning of the trajectory solve space).
    std::pair<Eigen::MatrixXd, Eigen::VectorXd>
        get_hessian_and_gradient() const {
        return SolverBase::get_hessian_and_gradient();
    }

    // Per-iteration snapshots for debug visualization of the solve. Populated
    // only when config.base.record_iterations == true (each Augmented Lagrangian
    // outer iteration is one snapshot; see SolverBase::optimize()). Each entry is
    // a full K+1-step trajectory reconstructed from that iteration's Values using
    // means-only marginals (cheap; no covariance factorization). Empty otherwise.
    std::vector<HandTrajectoryResult> get_intermediate_solutions() const;

    // The initial guess (start of the last plan()) as a trajectory, for the first
    // frame of a step animation. Mirrors get_intermediate_solutions()'s extraction.
    HandTrajectoryResult get_initial_solution() const;

private:
    void build_graph() override;
    void extract_solution() override;
    void get_initial_values() override;

    // Reconstruct a full trajectory result from a Values snapshot using means-only
    // marginals. Shared by get_intermediate_solutions()/get_initial_solution().
    HandTrajectoryResult
        extract_trajectory_means_only(const gtsam::Values& values) const;

    HandTrajectoryPlannerConfig config_;

    std::vector<std::unique_ptr<HandModel>> models_;  // K+1 models

    std::vector<VectorXGaussian> tensions_;
    std::vector<Vector6Gaussian> tip_wrenches_;
    std::vector<VectorXGaussian> start_tensions_;  // measured k=0 state (may be empty)

    HandTrajectoryResult result_;
};
