#include "parallel_robot_solver.h"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>

using namespace gtsam;


inline SharedDiagonal get_noise_model_rot_pos(double sigma_rot, double sigma_pos) {
    SharedDiagonal model = noiseModel::Diagonal::Sigmas((Vector(6) << 
        sigma_rot, sigma_rot, sigma_rot, 
        sigma_pos, sigma_pos, sigma_pos).finished());

    return model;
}


ParallelRobotSolver::ParallelRobotSolver(const ParallelRobotSolverConfig& config) {
    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_cov_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    base_pose_cov_ = get_noise_model_rot_pos(
        config.sigma_end_pose_rot, config.sigma_end_pose_pos);
    
    robot_ = std::make_unique<ParallelRobot>(
        config.num_rods,
        config.nodes_per_rod, 
        config.K_inv,
        twist_cov,
        small_wrench_cov_,
        config.base_end_poses,
        config.tip_end_poses,
        base_pose_cov_);

    values_ = robot_->get_initial_values();
}


ParallelRobotSolution ParallelRobotSolver::solve(const Vector& rod_lengths) {
    auto start = std::chrono::high_resolution_clock::now();

    graph_ = robot_->build_graph(rod_lengths);

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    ParallelRobotSolution solution;

    auto start_solve = std::chrono::high_resolution_clock::now();

    values_ = optimizer.optimize();

    auto stop_solve = std::chrono::high_resolution_clock::now();

    marginals_ = Marginals(graph_, values_);
    solution.marginals = robot_->get_marginals(values_, marginals_);

    auto stop = std::chrono::high_resolution_clock::now();

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.solve_time_ms = std::chrono::duration<double, std::milli>(stop_solve - start_solve).count();
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}