#include "MultiRobotSolver.h"

#include "multi_robot/MultiRobotModel.h"
#include "utils/MiscInline.h"
#include <gtsam/linear/NoiseModel.h>

using namespace gtsam;


MultiRobotSolver::MultiRobotSolver(const MultiRobotSolverConfig& config) 
:
    SolverBase(config.base)
{
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    SharedDiagonal small_wrench_noise = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    SharedDiagonal snare_constraint_noise = noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
        1e-3, 1e-3, 1e-3, 
        1e-4, 1e-4, 1e-4).finished());

    robot_ = std::make_unique<MultiRobotModel>(
        config.nodes_per_rod, 
        config.K_inv,
        twist_noise,
        small_wrench_noise,
        config.snare_distance_to_tip,
        snare_constraint_noise);  // TODO config these

    get_initial_values();
}


void MultiRobotSolver::get_initial_values() {
    values_ = robot_->get_initial_values();
}


void MultiRobotSolver::build_graph() {
    graph_ = robot_->build_graph(main_base_pose_, main_insertion_, helper_base_pose_, helper_insertion_, tip_wrench_);
}


void MultiRobotSolver::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
    // extracted_.rod_lengths_jacobian = robot_->get_rod_lengths_jacobian(marginals_);
}


Solution<MultiRobotMarginals> MultiRobotSolver::solve(
    const Pose3Gaussian& main_base_pose,
    double main_insertion,
    const Pose3Gaussian& helper_base_pose,
    double helper_insertion,
    const Vector6Gaussian& tip_wrench)
{
    main_base_pose_ = main_base_pose;
    main_insertion_ = main_insertion;
    helper_base_pose_ = helper_base_pose;
    helper_insertion_ = helper_insertion;
    tip_wrench_ = tip_wrench;

    Solution<MultiRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}