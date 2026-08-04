#pragma once

#include "tendon_hand/TendonHandModel.h"
#include "tendon_finger/TendonFingerSolver.h"   // TendonFingerSolverConfig
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>


struct TendonHandSolverConfig {
    SolverBaseConfig base;

    // Shared floating wrist base pose (world frame). Identity by default; each
    // finger is placed by its own hand_base_offset relative to this.
    gtsam::Matrix4 wrist_pose = gtsam::Matrix4::Identity();

    // Prior tightness on the shared wrist variable. Tight (default) => rigidly
    // anchored gauge; loosen for a more free-floating wrist.
    double sigma_wrist_pos = 1e-4;
    double sigma_wrist_rot = 1e-3;

    // Optional per-finger world-frame tip-position goals (point-to-point). One
    // Vector3 per finger, in config order. Empty (default) => no position goals,
    // so the solve is driven purely by the tendon-tension priors (legacy
    // behavior, unchanged). When non-empty each becomes a soft PositionPriorFactor
    // on that finger's tip node -- the single-shot analogue of
    // TendonHandTrajectoryPlannerConfig::goal_positions. Because these are soft
    // priors (not hard constraints), they do not affect the AL/plain routing:
    // collision/contact still decide whether the Augmented Lagrangian path runs.
    std::vector<gtsam::Vector3> goal_positions;
    Eigen::Matrix3d goal_position_cov = 1e-5 * Eigen::Matrix3d::Identity();

    // Optional warm-start posture: the marginals of any solve on the same finger
    // configs -- an FK pose, or an earlier solve of this same problem. Unset
    // (default) => the straight-rod, zero-tension cold start of
    // TendonHandModel::get_initial_values, unchanged. Set, and the solve begins
    // where the hand already is, which matters for a contact solve: the cold
    // guess is statically inconsistent with a curled rod, and the first
    // iterations are spent hyperextending and crawling back rather than closing
    // the contact. Marginals rather than Values for the same reason
    // TendonHandControllerConfig::initial_state uses them -- it is the state
    // bundle a caller already has from a previous solve.
    std::optional<TendonHandMarginals> initial_state;

    // (Interior/tip external-wrench prior noise is taken per finger from each
    // finger's sigma_stress_moment/force, matching TendonFingerSolver.)
};


// Static/kinematic solver for a multi-finger tendon hand whose fingers share one
// floating wrist base. Thin SolverBase wrapper around TendonHandModel; when any
// finger carries a contact constraint the solve runs on the Augmented Lagrangian
// path (like the single-finger TendonFingerSolver).
class TendonHandSolver : SolverBase {
public:
    TendonHandSolver(
        const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
        const TendonHandSolverConfig& config);

    Solution<TendonHandMarginals> solve(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches);

    // Command a new shared wrist pose (world frame, 4x4) between solves without
    // reconstructing the solver. Because solve() reuses the retained values_ as
    // its initial guess, a set_wrist_pose() + solve() sweep warm-starts each step
    // from the previous solution — far fewer iterations than rebuilding the
    // solver (which cold-starts from a straight hand every frame). Only the first
    // solve after construction pays the cold-start cost.
    void set_wrist_pose(const gtsam::Matrix4& wrist_pose) {
        hand_->set_wrist_pose(gtsam::Pose3(wrist_pose));
    }

    int num_fingers() const { return hand_->num_fingers(); }

    // Re-expose SolverBase diagnostics (privately inherited).
    std::vector<std::tuple<std::string, int, double>>
        get_factor_error_summary() const { return SolverBase::get_factor_error_summary(); }

    // Per-iteration snapshots for debug visualization of the solve. Populated
    // only when config.base.record_iterations == true (each Augmented Lagrangian
    // outer iteration is one snapshot; see SolverBase::optimize()). Each entry is
    // a hand state reconstructed from that iteration's Values using means-only
    // marginals (cheap; no covariance factorization), so reading a covariance
    // off one is invalid. Empty otherwise. Mirrors the trajectory planner's
    // accessor of the same name.
    std::vector<TendonHandMarginals> get_intermediate_solutions() const;

    // The initial guess (start of the last solve()), for the first frame of a
    // step animation. Mirrors get_intermediate_solutions()'s extraction.
    TendonHandMarginals get_initial_solution() const;

    // Restart the Augmented Lagrangian homotopy from the CURRENT posture: drops
    // the carried multipliers and the penalty weight without touching values_.
    //
    // Only meaningful under config.base.al_warm_start_duals, which is what makes
    // repeated solve() calls continue one outer loop instead of each running a
    // fresh one (see SolverBaseConfig::al_warm_start_duals). A caller stepping
    // the loop one outer iteration at a time uses this to re-run the penalty
    // schedule against a pose it has already reached; reconstructing the solver
    // is the stronger reset, since only that restores the initial values too.
    void reset_al_duals() { SolverBase::reset_al_duals(); }

private:
    void build_graph() override;
    void extract_solution() override;
    void get_initial_values() override;

    TendonHandSolverConfig config_;

    std::unique_ptr<TendonHandModel> hand_;

    std::vector<VectorXGaussian> tensions_;
    std::vector<Vector6Gaussian> tip_wrenches_;

    TendonHandMarginals extracted_;
};
