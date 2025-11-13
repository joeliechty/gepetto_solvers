#include "parallel_robot_solver.h"

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
    double sigma_rod_lengths) 
{
    auto start = std::chrono::high_resolution_clock::now();

    graph_ = robot_->build_graph(rod_lengths, sigma_rod_lengths);

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    ParallelRobotSolution solution;

    auto start_solve = std::chrono::high_resolution_clock::now();

    values_ = optimizer.optimize();

    auto stop_solve = std::chrono::high_resolution_clock::now();

    marginals_ = Marginals(graph_, values_);
    solution.marginals = robot_->get_marginals(values_, marginals_);

    solution.rod_lengths_jacobian = robot_->get_rod_lengths_jacobian(marginals_);

    auto stop = std::chrono::high_resolution_clock::now();

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.optimize_time_ms = std::chrono::duration<double, std::milli>(stop_solve - start_solve).count();
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}