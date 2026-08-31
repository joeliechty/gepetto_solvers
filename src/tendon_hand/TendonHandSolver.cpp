#include "gepetto_solvers/tendon_hand/TendonHandSolver.h"

#include "gepetto_solvers/measurement/PositionPriorFactor.h"
#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <stdexcept>

using namespace gtsam;


TendonHandSolver::TendonHandSolver(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    const TendonHandSolverConfig& config)
:
    SolverBase(config.base),
    config_(config)
{
    SharedDiagonal wrist_noise = get_noise_model_rot_pos(
        config.sigma_wrist_rot, config.sigma_wrist_pos);

    hand_ = std::make_unique<TendonHandModel>(
        finger_configs, Pose3(config.wrist_pose), wrist_noise);

    // A configured per-finger contact is a hard equality constraint and
    // collision avoidance is a hard inequality constraint (Section 1.5), so
    // either routes the solve through SolverBase's Augmented Lagrangian path.
    use_augmented_lagrangian_ = hand_->has_contact() || hand_->has_collision();

    get_initial_values();
}


void TendonHandSolver::get_initial_values() {
    // Seed from a caller-supplied posture when there is one, so the solve starts
    // where the hand already is rather than at the straight-rod, zero-tension
    // cold start (see config_.initial_state). Same warm merge the controller
    // uses: get_initial_values fills in anything the marginals do not carry.
    if (config_.initial_state) {
        const Values warm = hand_->values_from_marginals(*config_.initial_state);
        values_ = hand_->get_initial_values(&warm);
    } else {
        values_ = hand_->get_initial_values();
    }
}


void TendonHandSolver::build_graph() {
    graph_ = hand_->build_graph(tensions_, tip_wrenches_);

    // Optional per-finger tip-position goals (point-to-point). Soft priors on
    // each finger's tip node; only added when goal_positions is non-empty, so
    // legacy tension-driven / contact runs are unaffected. This is the
    // single-shot analogue of TendonHandTrajectoryPlanner's terminal goals.
    if (!config_.goal_positions.empty()) {
        const int num_fingers = hand_->num_fingers();
        if (static_cast<int>(config_.goal_positions.size()) != num_fingers) {
            throw std::runtime_error(
                "TendonHandSolverConfig::goal_positions size must equal the "
                "number of fingers when non-empty.");
        }
        auto goal_noise =
            noiseModel::Gaussian::Covariance(config_.goal_position_cov);
        for (int i = 0; i < num_fingers; ++i) {
            graph_.add(PositionPriorFactor(
                hand_->finger_tip_pose_key(i),
                config_.goal_positions[i],
                goal_noise));
        }
    }
}


void TendonHandSolver::extract_solution() {
    // skip_marginals leaves marginals_ empty, so take the means-only path. A
    // caller that asked to skip the factorization cannot read covariances back.
    if (config_.base.skip_marginals)
        extracted_ = hand_->get_marginals_means_only(values_);
    else
        extracted_ = hand_->get_marginals(values_, marginals_);
}


std::vector<TendonHandMarginals>
TendonHandSolver::get_intermediate_solutions() const {
    std::vector<TendonHandMarginals> out;
    out.reserve(intermediate_values_.size());
    for (const auto& vals : intermediate_values_)
        out.push_back(hand_->get_marginals_means_only(vals));
    return out;
}


TendonHandMarginals TendonHandSolver::get_initial_solution() const {
    return hand_->get_marginals_means_only(initial_values_);
}


Solution<TendonHandMarginals> TendonHandSolver::solve(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    tensions_ = tensions;
    tip_wrenches_ = tip_wrenches;

    Solution<TendonHandMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;
    return solution;
}
