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
        config.sigma_snare_rot_x, // In main tip frame
        config.sigma_snare_rot_y,
        config.sigma_snare_rot_z, 
        config.sigma_twist_pos, // small
        config.sigma_twist_pos, // small
        config.sigma_snare_location
    ).finished());

    SharedDiagonal base_pose_noise = noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
        config.sigma_base_rot,
        config.sigma_base_rot,
        config.sigma_base_rot, 
        config.sigma_twist_pos, // small
        config.sigma_twist_pos, // small
        config.sigma_rod_lengths
    ).finished());

    robot_ = std::make_unique<MultiRobotModel>(
        config.nodes_per_rod, 
        config.K_inv,
        twist_noise,
        small_wrench_noise,
        snare_constraint_noise,
        base_pose_noise,
        config.snare_distance_to_tip);  // TODO config these

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
    const Matrix4& main_base_pose,
    double main_insertion,
    const Matrix4& helper_base_pose,
    double helper_insertion,
    const Vector6Gaussian& tip_wrench)
{
    main_base_pose_ = Pose3(main_base_pose);
    main_insertion_ = main_insertion;
    helper_base_pose_ = Pose3(helper_base_pose);
    helper_insertion_ = helper_insertion;
    tip_wrench_ = tip_wrench;

    Solution<MultiRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}