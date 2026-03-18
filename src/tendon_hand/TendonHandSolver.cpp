#include "TendonHandSolver.h"
#include "utils/MiscInline.h"

using namespace gtsam;


TendonHandSolver::TendonHandSolver(
    const std::vector<std::pair<std::string, TendonRobotSolverConfig>>& finger_configs,
    const TendonHandSolverConfig& config)
:
    SolverBase(config.base)
{
    SharedDiagonal small_wrench_noise = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force);

    hand_ = std::make_unique<TendonHandModel>(finger_configs, small_wrench_noise);

    get_initial_values();
}


void TendonHandSolver::get_initial_values() {
    values_ = hand_->get_initial_values();
}


void TendonHandSolver::build_graph() {
    graph_ = hand_->build_graph(tensions_, tip_wrenches_, values_);
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
