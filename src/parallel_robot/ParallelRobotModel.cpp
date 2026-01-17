#include "ParallelRobotModel.h"

#include <gtsam/base/Vector.h>
#include <gtsam/slam/BetweenFactor.h>

#include "PlatformWrenchBalanceFactor.h"
#include "cosserat_rod/BoundaryStressFactor.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"

using namespace gtsam;


ParallelRobot::ParallelRobot(
    int nodes_per_rod, 
    Matrix6 K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    std::array<Matrix4, NUM_RODS> base_end_poses,
    std::array<Matrix4, NUM_RODS> tip_end_poses,
    double sigma_end_pose_pos,
    double sigma_end_pose_rot)
:
    base_end_poses_(base_end_poses),
    tip_end_poses_(tip_end_poses),
    small_wrench_noise_(stress_noise),
    sigma_end_pose_pos_(sigma_end_pose_pos),
    sigma_end_pose_rot_(sigma_end_pose_rot)
{
    // Make each rod
    for (int i = 0; i < NUM_RODS; i++) {
        rods_[i] = std::make_unique<CosseratRodModel>(
            nodes_per_rod, K_inv, twist_noise, stress_noise);
    }
}


Key platform_pose_key() { return Symbol('P', 424242424242); }


Key platform_stress_key() { return Symbol('S', 424242424242); }


Key platform_wrench_key() { return Symbol('W', 424242424242); }


NonlinearFactorGraph ParallelRobot::build_graph(
    const std::array<double, NUM_RODS>& rod_lengths,
    double sigma_rod_lengths,
    const Vector6Gaussian& wrench_)
{
    NonlinearFactorGraph graph;

    // Tip of the rods relative to platform is relatively certain
    SharedDiagonal tip_pose_noise = get_noise_model_rot_pos(sigma_end_pose_rot_, sigma_end_pose_pos_);

    // Base of rods relative to world is certain, except for z extension, which is uncertain
    gtsam::SharedDiagonal base_pose_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
        sigma_end_pose_rot_, sigma_end_pose_rot_, sigma_end_pose_rot_, 
        sigma_end_pose_pos_, sigma_end_pose_pos_, sigma_rod_lengths).finished());

    // Build each rod
    for (int i = 0; i < NUM_RODS; i++) {
        // Build base cosserat rod graph
        auto rod_graph = rods_[i]->build_graph(rod_lengths[i]);
        graph.push_back(rod_graph.begin(), rod_graph.end());

        // Constrain interior wrenches to zero (skip base and tip)
        std::vector<Key> wrench_keys = rods_[i]->get_wrench_keys();
        for (size_t j = 1; j + 1 < wrench_keys.size(); ++j) {
            graph.add(PriorFactor<Vector6>(wrench_keys[j], Vector6::Zero(), small_wrench_noise_));
        }

        // Base pose prior
        graph.add(PriorFactor<Pose3>(rods_[i]->get_pose_key(0), Pose3(base_end_poses_[i]), base_pose_noise));

        // Tip pose relative to platform
        graph.add(BetweenFactor<Pose3>(
            platform_pose_key(),
            rods_[i]->get_pose_key(-1),
            Pose3(tip_end_poses_[i]),
            tip_pose_noise));
    }

    // Constrain platform stress to be equal to platform wrench
    bool is_base = false;
    graph.add(BoundaryStressFactor(
        platform_stress_key(), 
        platform_wrench_key(),
        platform_pose_key(),
        small_wrench_noise_,
        is_base));
    
    // Put prior on tip wrench based on user input
    graph.add(PriorFactor<Vector6>(
        platform_wrench_key(), 
        wrench_.mean, 
        noiseModel::Gaussian::Covariance(wrench_.cov)));

    // Sum of all transformed tip stresses equals zero (for now)
    graph.add(PlatformWrenchBalanceFactor(
        rods_[0]->get_stress_key(-1),
        rods_[0]->get_pose_key(-1),
        rods_[1]->get_stress_key(-1),
        rods_[1]->get_pose_key(-1),
        rods_[2]->get_stress_key(-1),
        rods_[2]->get_pose_key(-1),
        rods_[3]->get_stress_key(-1),
        rods_[3]->get_pose_key(-1),
        rods_[4]->get_stress_key(-1),
        rods_[4]->get_pose_key(-1),
        rods_[5]->get_stress_key(-1),
        rods_[5]->get_pose_key(-1),
        platform_stress_key(),
        platform_pose_key(),
        small_wrench_noise_));
    
    return graph;
}


Values ParallelRobot::get_initial_values() const {
    Values values;

    // Values for each rod
    for (int i = 0; i < NUM_RODS; i++) {
        values.insert(rods_[i]->get_initial_values());
    }

    // Values for moving platform variables
    values.insert(platform_pose_key(), Pose3::Identity());
    values.insert(platform_stress_key(), Vector6(Vector6::Zero()));
    values.insert(platform_wrench_key(), Vector6(Vector6::Zero()));

    return values;
}


ParallelRobotMarginals ParallelRobot::get_marginals(
    const Values& values, 
    const Marginals& marginals) const
{   
    ParallelRobotMarginals solution;

    for (int i = 0; i < NUM_RODS; i++) {
        solution.rods[i] = rods_[i]->get_marginals(values, marginals);
    }

    solution.platform_pose.mean = values.at<Pose3>(platform_pose_key()).matrix();
    solution.platform_pose.cov = marginals.marginalCovariance(platform_pose_key());

    solution.platform_wrench.mean = values.at<Vector6>(platform_wrench_key()).matrix();
    solution.platform_wrench.cov = marginals.marginalCovariance(platform_wrench_key());
    
    return solution;
}


Matrix6 ParallelRobot::get_rod_lengths_jacobian(const Marginals& marginals) const {
    KeyVector keys;
    for (const auto& rod : rods_) {
        keys.push_back(rod->get_pose_key(0));
    }

    keys.push_back(platform_pose_key());

    JointMarginal joint_marginal = marginals.jointMarginalCovariance(keys);

    const int n = keys.size();
    Eigen::MatrixXd rod_bases_joint = Eigen::MatrixXd::Zero(6 * n, 6 * n);

    int i = 0;
    for (Key& key_i : keys) {
        int j = 0;
        for (Key& key_j : keys) {
            rod_bases_joint.block<6,6>(6 * i, 6 * j) = joint_marginal(key_i, key_j);
            j++;
        }
        i++;
    }

    Eigen::VectorXi indices(12);
    indices << 5, 11, 17, 23, 29, 35, 36, 37, 38, 39, 40, 41;

    Eigen::Matrix<double, 12, 12> rod_lengths_joint;
    for (int r = 0; r < indices.size(); ++r)
        for (int c = 0; c < indices.size(); ++c)
            rod_lengths_joint(r, c) = rod_bases_joint(indices[r], indices[c]);

    Matrix6 sigma_lengths = rod_lengths_joint.block<6,6>(0,0);
    Matrix6 sigma_pose_lengths = rod_lengths_joint.block<6,6>(6,0);

    Eigen::LDLT<Matrix6> ldlt(sigma_lengths);
    return sigma_pose_lengths * ldlt.solve(Matrix6::Identity());
}