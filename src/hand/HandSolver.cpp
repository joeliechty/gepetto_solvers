#include "gepetto_solvers/hand/HandSolver.h"

#include "gepetto_solvers/measurement/PositionPriorFactor.h"
#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <stdexcept>

using namespace gtsam;


HandSolver::HandSolver(
    const gepetto_solvers::HandSpec& spec,
    const HandSolverConfig& config)
:
    SolverBase(config.base),
    config_(config)
{
    SharedDiagonal wrist_noise = get_noise_model_rot_pos(
        config.sigma_wrist_rot, config.sigma_wrist_pos);

    hand_ = std::make_unique<HandModel>(
        spec, Pose3(config.wrist_pose), wrist_noise);

    // A configured per-digit contact is a hard equality constraint and
    // collision avoidance is a hard inequality constraint (Section 1.5), so
    // either routes the solve through SolverBase's Augmented Lagrangian path.
    use_augmented_lagrangian_ = hand_->has_contact() || hand_->has_collision();

    get_initial_values();
}


void HandSolver::get_initial_values() {
    // Seed from a caller-supplied posture when there is one, so the solve starts
    // where the hand already is rather than at the straight-rod, zero-tension
    // cold start (see config_.initial_state). Same warm merge the controller
    // uses: get_initial_values fills in anything the marginals do not carry.
    if (config_.initial_state) {
        const Values warm = hand_->values_from_state(*config_.initial_state);
        values_ = hand_->get_initial_values(&warm);
    } else {
        values_ = hand_->get_initial_values();
    }
}


void HandSolver::build_graph() {
    graph_ = hand_->build_graph(tensions_, tip_wrenches_);

    // Optional per-finger tip-position goals (point-to-point). Soft priors on
    // each finger's tip node; only added when goal_positions is non-empty, so
    // legacy tension-driven / contact runs are unaffected. This is the
    // single-shot analogue of HandTrajectoryPlanner's terminal goals.
    if (!config_.goal_positions.empty()) {
        const int n_digits = hand_->num_digits();
        if (static_cast<int>(config_.goal_positions.size()) != n_digits) {
            throw std::runtime_error(
                "HandSolverConfig::goal_positions size must equal the "
                "number of digits when non-empty.");
        }
        auto goal_noise =
            noiseModel::Gaussian::Covariance(config_.goal_position_cov);
        for (int i = 0; i < n_digits; ++i) {
            graph_.add(PositionPriorFactor(
                hand_->digit_tip_pose_key(i),
                config_.goal_positions[i],
                goal_noise));
        }
    }
}


void HandSolver::extract_solution() {
    // skip_marginals leaves marginals_ empty, so take the means-only path. A
    // caller that asked to skip the factorization cannot read covariances back.
    if (config_.base.skip_marginals)
        extracted_ = hand_->get_state_means_only(values_);
    else
        extracted_ = hand_->get_state(values_, marginals_);
}


std::vector<HandState>
HandSolver::get_intermediate_solutions() const {
    std::vector<HandState> out;
    out.reserve(intermediate_values_.size());
    for (const auto& vals : intermediate_values_)
        out.push_back(hand_->get_state_means_only(vals));
    return out;
}


HandState HandSolver::get_initial_solution() const {
    return hand_->get_state_means_only(initial_values_);
}


Solution<HandState> HandSolver::solve(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    tensions_ = tensions;
    tip_wrenches_ = tip_wrenches;

    Solution<HandState> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;
    return solution;
}
