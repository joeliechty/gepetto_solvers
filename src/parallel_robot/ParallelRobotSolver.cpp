#include "ParallelRobotSolver.h"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

using namespace gtsam;


ParallelRobotSolver::ParallelRobotSolver(const ParallelRobotSolverConfig& config) {
    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_cov_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    robot_ = std::make_unique<ParallelRobot>(
        config.nodes_per_rod, 
        config.K_inv,
        twist_cov,
        small_wrench_cov_,
        config.base_end_poses,
        config.tip_end_poses,
        config.sigma_end_pose_pos,
        config.sigma_end_pose_rot);

    values_ = robot_->get_initial_values();
    
    print_values(values_);
}


ParallelRobotSolution ParallelRobotSolver::solve(
    const std::array<double, NUM_RODS>& rod_lengths, 
    double sigma_rod_lengths,
    const Vector6& wrench_mean,
    const Matrix6& wrench_cov) 
{
    auto start = std::chrono::high_resolution_clock::now();
    auto build_start = start;

    graph_ = robot_->build_graph(
        rod_lengths, 
        sigma_rod_lengths,
        wrench_mean,
        wrench_cov);

    auto build_stop = std::chrono::high_resolution_clock::now();
    auto optimize_start = build_stop;

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    values_ = optimizer.optimize();

    auto optimize_stop = std::chrono::high_resolution_clock::now();
    auto marginalize_start = optimize_stop;

    marginals_ = Marginals(graph_, values_);

    auto marginalize_stop = std::chrono::high_resolution_clock::now();
    auto extract_start = marginalize_stop;

    ParallelRobotSolution solution;
    
    solution.marginals = robot_->get_marginals(values_, marginals_);

    solution.rod_lengths_jacobian = robot_->get_rod_lengths_jacobian(marginals_);

    auto extract_stop = std::chrono::high_resolution_clock::now();
    auto stop = extract_stop;

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.build_time_ms = std::chrono::duration<double, std::milli>(build_stop - build_start).count();
    solution.meta.optimize_time_ms = std::chrono::duration<double, std::milli>(optimize_stop - optimize_start).count();
    solution.meta.marginalize_time_ms = std::chrono::duration<double, std::milli>(marginalize_stop - marginalize_start).count();
    solution.meta.extract_time_ms = std::chrono::duration<double, std::milli>(extract_stop - extract_start).count();
    
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}