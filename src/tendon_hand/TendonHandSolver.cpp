#include "TendonHandSolver.h"

#include "utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>

using namespace gtsam;


TendonHandSolver::TendonHandSolver(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    const TendonHandSolverConfig& config)
:
    SolverBase(config.base)
{
    SharedDiagonal wrist_noise = get_noise_model_rot_pos(
        config.sigma_wrist_rot, config.sigma_wrist_pos);

    hand_ = std::make_unique<TendonHandModel>(
        finger_configs, Pose3(config.wrist_pose), wrist_noise);

    // A configured per-finger contact is a hard equality constraint, so route
    // the solve through SolverBase's Augmented Lagrangian path.
    use_augmented_lagrangian_ = hand_->has_contact();

    get_initial_values();
}


void TendonHandSolver::get_initial_values() {
    values_ = hand_->get_initial_values();
}


void TendonHandSolver::build_graph() {
    graph_ = hand_->build_graph(tensions_, tip_wrenches_);
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
