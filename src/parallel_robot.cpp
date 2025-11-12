#include "parallel_robot.h"

#include <gtsam/slam/BetweenFactor.h>

using namespace gtsam;


ParallelRobot::ParallelRobot(
    int num_rods,
    int nodes_per_rod, 
    Matrix6 K_inv,
    SharedDiagonal rod_twist_cov,
    SharedDiagonal small_wrench_cov_,
    std::vector<Matrix4> base_end_poses,
    std::vector<Matrix4> tip_end_poses,
    SharedDiagonal end_pose_cov)
:
    num_rods_(num_rods),
    nodes_per_rod_(nodes_per_rod),
    K_inv_(K_inv),
    rod_twist_cov_(rod_twist_cov),
    small_wrench_cov_(small_wrench_cov_),
    base_end_poses_(base_end_poses),
    tip_end_poses_(tip_end_poses),
    end_pose_cov_(end_pose_cov) 
{
    rods_.resize(num_rods_);
}


Key platform_pose_key() { return Symbol('P', 424242); }


NonlinearFactorGraph ParallelRobot::build_graph(const Vector& rod_lengths) {
    NonlinearFactorGraph graph;

    for (int i = 0; i < num_rods_; i++) {
        // Build base cosserat rod graph
        rods_[i] = std::make_unique<CosseratRod>(
            rod_lengths[i], nodes_per_rod_, K_inv_, rod_twist_cov_, small_wrench_cov_);

        auto rod_graph = rods_[i]->build_graph();
        graph.push_back(rod_graph.begin(), rod_graph.end());

        // Constrain interior wrenches to zero (skip base and tip)
        std::vector<Key> wrench_keys = rods_[i]->get_wrench_keys();
        for (size_t j = 1; j + 1 < wrench_keys.size(); ++j) {
            graph.add(PriorFactor<Vector6>(wrench_keys[j], Vector6::Zero(), small_wrench_cov_));
        }

        // Base pose prior
        graph.add(PriorFactor<Pose3>(rods_[i]->get_pose_key(0), Pose3(base_end_poses_[i]), end_pose_cov_));

        // Tip pose relative to platform
        graph.add(BetweenFactor<Pose3>(
            rods_[i]->get_pose_key(-1),
            platform_pose_key(),
            Pose3(tip_end_poses_[i]),
            end_pose_cov_));
    }

    return graph;
}


Values ParallelRobot::get_initial_values() const {
    Values values;

    for (int i = 0; i < num_rods_; i++) {
        values.insert(rods_[i]->get_initial_values());
    }

    values.insert(platform_pose_key(), Pose3::Identity());

    return values;
}


ParallelRobotMarginals ParallelRobot::get_marginals(
    const Values& values, 
    const Marginals& marginals) const
{
    ParallelRobotMarginals solution;

    for (int i = 0; i < num_rods_; i++) {
        solution.rods[i] = rods_[i]->get_marginals(values, marginals);
    }

    solution.platform_pose_mean = values.at<Pose3>(platform_pose_key()).matrix();
    solution.platform_pose_cov = marginals.marginalCovariance(platform_pose_key());

    return solution;
}