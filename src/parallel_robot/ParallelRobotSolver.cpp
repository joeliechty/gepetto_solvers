#include "ParallelRobotSolver.h"

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "utils/MiscInline.h"

using namespace gtsam;


ParallelRobotSolver::ParallelRobotSolver(const ParallelRobotSolverConfig& config) {
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    robot_ = std::make_unique<ParallelRobot>(
        config.nodes_per_rod, 
        config.K_inv,
        twist_noise,
        small_wrench_noise_,
        config.base_end_poses,
        config.tip_end_poses,
        config.sigma_end_pose_pos,
        config.sigma_end_pose_rot);

    get_initial_values();
}


void ParallelRobotSolver::get_initial_values() {
    values_ = robot_->get_initial_values();
}


void ParallelRobotSolver::build_graph() {
    graph_ = robot_->build_graph(rod_lengths_, sigma_rod_lengths_, wrench_);
}


void ParallelRobotSolver::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
    extracted_.rod_lengths_jacobian = robot_->get_rod_lengths_jacobian(marginals_);
}

    
Solution<ParallelRobotMarginals> ParallelRobotSolver::solve(
    const std::array<double, NUM_RODS>& rod_lengths, 
    double sigma_rod_lengths,
    const Vector6Gaussian& wrench) 
{
    rod_lengths_ = rod_lengths;
    sigma_rod_lengths_ = sigma_rod_lengths;
    wrench_ = wrench;
    
    Solution<ParallelRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}