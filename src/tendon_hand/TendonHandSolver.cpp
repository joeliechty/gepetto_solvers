#include "TendonHandSolver.h"

#include "measurement/PositionPriorFactor.h"
#include "utils/MiscInline.h"

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
    values_ = hand_->get_initial_values();
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
    extracted_ = hand_->get_marginals(values_, marginals_);
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
