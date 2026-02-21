#include "MultiRobotModel.h"

#include <cmath>
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <memory>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"

using namespace gtsam;


class SnareConstraintWrenchFactor: public NoiseModelFactorN<Vector6, Vector6, Vector6> {
    using NoiseModelFactorN<Vector6, Vector6, Vector6>::evaluateError;

public:
    SnareConstraintWrenchFactor(
        Key wrench_1_key, 
        Key wrench_2_key,
        Key wrench_3_key,
        const SharedNoiseModel& model)
    :
        NoiseModelFactorN(model, wrench_1_key, wrench_2_key, wrench_3_key)
    {}
        
        
    Vector evaluateError(
        const Vector6& w1,
        const Vector6& w2,
        const Vector6& w3,
        OptionalMatrixType H1,
        OptionalMatrixType H2,
        OptionalMatrixType H3) const override 
    {
        Vector6 error = w1 + w2 + w3;

        if (H1) { *H1 = Matrix6::Identity(); }

        if (H2) { *H2 = Matrix6::Identity(); }

        if (H3) { *H3 = Matrix6::Identity(); }

        return error;
    }
};



MultiRobotModel::MultiRobotModel(
    int nodes_per_rod, 
    Matrix6 K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    double snare_distance_to_tip,
    SharedDiagonal snare_constraint_noise)
:
    small_wrench_noise_(stress_noise),
    snare_constraint_noise_(snare_constraint_noise),
    snare_distance_to_tip_(snare_distance_to_tip)
{
    main_rod_ = std::make_unique<CosseratRodModel>(nodes_per_rod, K_inv, twist_noise, stress_noise);
    helper_rod_ = std::make_unique<CosseratRodModel>(nodes_per_rod, K_inv, twist_noise, stress_noise);
    end_effector_rod_ = std::make_unique<CosseratRodModel>(nodes_per_rod, K_inv, twist_noise, stress_noise);
}

NonlinearFactorGraph MultiRobotModel::build_graph(
    const Pose3Gaussian& main_base_pose,
    double main_insertion,
    const Pose3Gaussian& helper_base_pose,
    double helper_insertion,
    const Vector6Gaussian& tip_wrench)
{
    NonlinearFactorGraph graph;

    // We separate the physical main rod into main and end effector rods, which makes the snare constraint easier to deal with.
    // All 3 models meet at the constraint point, the main rod and helper rod meet at a right angle
    // The end effector rod and the main rod are actually the same physical rod

    double end_effector_rod_length = snare_distance_to_tip_;
    graph.add(end_effector_rod_->build_graph(end_effector_rod_length));

    double main_rod_length = main_insertion - snare_distance_to_tip_;  // Makes the total main + EE = physical length of that rod
    graph.add(main_rod_->build_graph(main_rod_length));

    double helper_rod_length = helper_insertion;  // Simpler here 
    graph.add(helper_rod_->build_graph(helper_rod_length));

    // Constrain interior wrenches to zero (skip base and tip)
    for (auto* rod : {&main_rod_, &helper_rod_, &end_effector_rod_}) {
        std::vector<Key> wrench_keys = (*rod)->get_wrench_keys();
        for (size_t j = 1; j + 1 < wrench_keys.size(); ++j) {
            graph.add(PriorFactor<Vector6>(wrench_keys[j], Vector6::Zero(), small_wrench_noise_));
        }
    }

    // Main rod base pose prior
    graph.add(PriorFactor<Pose3>(
        main_rod_->get_pose_key(0),
        Pose3(main_base_pose.mean),
        main_base_pose.cov));
    
    // Helper rod base pose prior
    graph.add(PriorFactor<Pose3>(
        helper_rod_->get_pose_key(0),
        Pose3(helper_base_pose.mean),
        helper_base_pose.cov));
    
    // Constrain tip of helper rod to intersect perpendicularly with main rod tip 
    graph.add(BetweenFactor<Pose3>(
        main_rod_->get_pose_key(-1),
        helper_rod_->get_pose_key(-1),
        Pose3(Rot3::Ry(-M_PI / 2.0), Point3::Zero()),  // 90 rotation about x
        snare_constraint_noise_));

    // Constrain base of EE rod to just continue on from main rod tip
    SharedDiagonal small_pose_noise = noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 
        1e-3, 1e-3, 1e-3, 
        1e-4, 1e-4, 1e-4).finished());
        
    graph.add(BetweenFactor<Pose3>(
        main_rod_->get_pose_key(-1),
        end_effector_rod_->get_pose_key(0),
        Pose3::Identity(),  // just keep going in the same direction, since its the same physical rod
        small_pose_noise));  // TODO change noise to be small
    
    // Now the sum of all forces at the constraint needs to be zero
    graph.add(SnareConstraintWrenchFactor(
        main_rod_->get_wrench_key(-1),  // TIP of main rod at intersection
        helper_rod_->get_wrench_key(-1),  // TIP of helper at intersection 
        end_effector_rod_->get_wrench_key(0), // BASE of EE rod at intersection
        small_wrench_noise_));

    // Finally put prior on end effector tip wrench based on user input
    graph.add(PriorFactor<Vector6>(
        end_effector_rod_->get_wrench_key(-1), 
        tip_wrench.mean, 
        noiseModel::Gaussian::Covariance(tip_wrench.cov)));

    return graph;
}


Values MultiRobotModel::get_initial_values() const {
    Values values;

    // Init pointing down since that is how the robots generally are oriented
    Pose3 point_down = Pose3(Rot3::Rx(M_PI), Point3::Zero());

    values.insert(main_rod_->get_initial_values(0.01, point_down));
    values.insert(helper_rod_->get_initial_values(0.01, point_down));
    values.insert(end_effector_rod_->get_initial_values());

    return values;
}


MultiRobotMarginals MultiRobotModel::get_marginals(
    const Values& values, 
    const Marginals& marginals) const
{   
    MultiRobotMarginals solution;

    solution.main_rod = main_rod_->get_marginals(values, marginals);
    solution.helper_rod = helper_rod_->get_marginals(values, marginals);
    solution.end_effector_rod = end_effector_rod_->get_marginals(values, marginals);

    return solution;
}


// Matrix6 ParallelRobot::get_rod_lengths_jacobian(const Marginals& marginals) const {
//     KeyVector keys;
//     for (const auto& rod : rods_) {
//         keys.push_back(rod->get_pose_key(0));
//     }

//     keys.push_back(platform_pose_key());

//     JointMarginal joint_marginal = marginals.jointMarginalCovariance(keys);

//     const int n = keys.size();
//     Eigen::MatrixXd rod_bases_joint = Eigen::MatrixXd::Zero(6 * n, 6 * n);

//     int i = 0;
//     for (Key& key_i : keys) {
//         int j = 0;
//         for (Key& key_j : keys) {
//             rod_bases_joint.block<6,6>(6 * i, 6 * j) = joint_marginal(key_i, key_j);
//             j++;
//         }
//         i++;
//     }

//     Eigen::VectorXi indices(12);
//     indices << 5, 11, 17, 23, 29, 35, 36, 37, 38, 39, 40, 41;

//     Eigen::Matrix<double, 12, 12> rod_lengths_joint;
//     for (int r = 0; r < indices.size(); ++r)
//         for (int c = 0; c < indices.size(); ++c)
//             rod_lengths_joint(r, c) = rod_bases_joint(indices[r], indices[c]);

//     Matrix6 sigma_lengths = rod_lengths_joint.block<6,6>(0,0);
//     Matrix6 sigma_pose_lengths = rod_lengths_joint.block<6,6>(6,0);

//     Eigen::LDLT<Matrix6> ldlt(sigma_lengths);
//     return sigma_pose_lengths * ldlt.solve(Matrix6::Identity());
// }


// Matrix6 ParallelRobot::get_tip_wrench_jacobian(const Marginals& marginals) const {
//     // Get joint marginal between tip wrench and tip pose
//     Key W = platform_wrench_key();
//     Key T = platform_pose_key();

//     KeyVector keys;
//     keys.push_back(W);
//     keys.push_back(T);
//     JointMarginal joint = marginals.jointMarginalCovariance(keys);

//     // Get individual blocks
//     Matrix6 sigma_WW = joint(W, W);
//     Matrix6 sigma_TW = joint(T, W);

//     // Compute J_pose_tensions = sigma_TQ * inv(sigma_QQ)
//     Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma_WW);
//     return sigma_TW * ldlt.solve(Matrix6::Identity());
// }