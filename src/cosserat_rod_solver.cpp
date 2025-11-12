#include "cosserat_rod_solver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include "gtsam_factors.h"

using namespace gtsam;


inline SharedDiagonal get_noise_model_rot_pos(double sigma_rot, double sigma_pos) {
    SharedDiagonal model = noiseModel::Diagonal::Sigmas((Vector(6) << 
        sigma_rot, sigma_rot, sigma_rot, 
        sigma_pos, sigma_pos, sigma_pos).finished());

    return model;
}


CosseratRodSolver::CosseratRodSolver(const CosseratRodSolverConfig& config) {
    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_cov_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    base_pose_cov_ = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);
    
    rod_= std::make_unique<CosseratRod>(
        config.rod_length, 
        config.num_nodes, 
        config.K_inv, 
        twist_cov, 
        small_wrench_cov_);

    values_ = rod_->get_initial_values();
}


CosseratRodSolution CosseratRodSolver::solve(
    const std::optional<Vector6>& tip_wrench_mean, 
    const std::optional<Matrix6>& tip_wrench_cov,
    const std::optional<Matrix4>& tip_pose_mean,
    const std::optional<Matrix6>& tip_pose_cov) 
{
    auto start = std::chrono::high_resolution_clock::now();

    graph_ = rod_->build_graph();

    add_prior_factors(
        tip_wrench_mean, 
        tip_wrench_cov,
        tip_pose_mean,
        tip_pose_cov);

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    CosseratRodSolution solution;

    auto start_solve = std::chrono::high_resolution_clock::now();

    values_ = optimizer.optimize();

    auto stop_solve = std::chrono::high_resolution_clock::now();

    marginals_ = Marginals(graph_, values_);
    solution.marginals = rod_->get_marginals(values_, marginals_);

    auto stop = std::chrono::high_resolution_clock::now();

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.solve_time_ms = std::chrono::duration<double, std::milli>(stop_solve - start_solve).count();
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}


void CosseratRodSolver::add_prior_factors(
    const std::optional<Vector6>& tip_wrench_mean,
    const std::optional<Matrix6>& tip_wrench_cov,
    const std::optional<Matrix4>& tip_pose_mean,
    const std::optional<Matrix6>& tip_pose_cov)
{
    // Constrain base pose to identity 
    auto base_pose_factor = PriorFactor<Pose3>(
        rod_->get_pose_key(0), 
        Pose3::Identity(), 
        base_pose_cov_);
    
    graph_.add(base_pose_factor);

    // Constrain all wrenches on the interior of the rod to be zero
    std::vector<Key> wrench_keys = rod_->get_wrench_keys();

    // Skip base and tip wrenches
    for (size_t i = 1; i + 1 < wrench_keys.size(); ++i) {
        auto factor = PriorFactor<Vector6>(
            wrench_keys[i],
            Vector6::Zero(),
            small_wrench_cov_);
        
        graph_.add(factor);
    }

    // Set prior on tip wrench/pose based on user input
    if (tip_wrench_mean) {
        auto factor = PriorFactor<Vector6>(
            wrench_keys.back(),
            *tip_wrench_mean,
            noiseModel::Gaussian::Covariance(*tip_wrench_cov));
        
        graph_.add(factor);
    }

    if (tip_pose_mean) {
        auto factor = PriorFactor<Pose3>(
            rod_->get_pose_key(-1),
            Pose3(*tip_pose_mean),
            noiseModel::Gaussian::Covariance(*tip_pose_cov));

        graph_.add(factor);
    }
}
