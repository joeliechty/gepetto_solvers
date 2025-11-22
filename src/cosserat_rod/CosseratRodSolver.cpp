#include "CosseratRodSolver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/MiscInline.h"

using namespace gtsam;


CosseratRodSolver::CosseratRodSolver(const CosseratRodSolverConfig& config) 
:
    rod_length_(config.rod_length)
{
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    base_pose_noise_ = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);
    
    rod_= std::make_unique<CosseratRodModel>(
        config.num_nodes, 
        config.K_inv, 
        twist_noise, 
        small_wrench_noise_);

    get_initial_values();
}


void CosseratRodSolver::extract_solution() {
    extracted_ = rod_->get_marginals(values_, marginals_);
}



void CosseratRodSolver::get_initial_values() {
    values_ = rod_->get_initial_values();
}


Solution<CosseratRodMarginals> CosseratRodSolver::solve(
    const std::optional<Vector6>& tip_wrench_mean, 
    const std::optional<Matrix6>& tip_wrench_cov,
    const std::optional<Matrix4>& tip_pose_mean,
    const std::optional<Matrix6>& tip_pose_cov,
    const std::optional<Vector6>& nominal_strain) 
{
    tip_wrench_mean_ = tip_wrench_mean; 
    tip_wrench_cov_ = tip_wrench_cov;
    tip_pose_mean_ = tip_pose_mean;
    tip_pose_cov_ = tip_pose_cov;
    nominal_strain_ = nominal_strain;

    Solution<CosseratRodMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}


void CosseratRodSolver::build_graph() {
    // Build base rod graph
    graph_ = rod_->build_graph(rod_length_, nominal_strain_);

    // Constrain base pose to identity 
    graph_.add(PriorFactor<Pose3>(
        rod_->get_pose_key(0), 
        Pose3::Identity(), 
        base_pose_noise_));

    // Constrain all wrenches on the interior of the rod to be zero
    std::vector<Key> wrench_keys = rod_->get_wrench_keys();

    // Skip base and tip wrenches
    for (size_t i = 1; i + 1 < wrench_keys.size(); ++i) {
        graph_.add(PriorFactor<Vector6>(
            wrench_keys[i],
            Vector6::Zero(),
            small_wrench_noise_));
    }

    // Set prior on tip wrench/pose based on user input
    if (tip_wrench_mean_) {
        graph_.add(PriorFactor<Vector6>(
            wrench_keys.back(),
            *tip_wrench_mean_,
            noiseModel::Gaussian::Covariance(*tip_wrench_cov_)));
    }

    if (tip_pose_mean_) {
        graph_.add(PriorFactor<Pose3>(
            rod_->get_pose_key(-1),
            Pose3(*tip_pose_mean_),
            noiseModel::Gaussian::Covariance(*tip_pose_cov_)));
    }
}
