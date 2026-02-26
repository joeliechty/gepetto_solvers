#include "MultiRobotModel.h"

#include <cmath>
#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
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
    SharedDiagonal snare_constraint_noise,
    SharedDiagonal base_pose_noise,
    double snare_distance_to_tip)
:
    small_wrench_noise_(stress_noise),
    snare_constraint_noise_(snare_constraint_noise),
    twist_noise_(twist_noise),
    base_pose_noise_(base_pose_noise),
    snare_distance_to_tip_(snare_distance_to_tip)
{
    main_rod_ = std::make_unique<CosseratRodModel>(nodes_per_rod, K_inv, twist_noise, stress_noise);
    helper_rod_ = std::make_unique<CosseratRodModel>(nodes_per_rod, K_inv, twist_noise, stress_noise);

    // Note use half as many nodes for this one so it looks more even
    end_effector_rod_ = std::make_unique<CosseratRodModel>(nodes_per_rod / 2, K_inv, twist_noise, stress_noise);
}

NonlinearFactorGraph MultiRobotModel::build_graph(
    const Pose3& main_base_pose,
    double main_insertion,
    const Pose3& helper_base_pose,
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
        main_base_pose,
        base_pose_noise_));  // could use this constructor elsewhere instead of using Gaussian
    
    // Helper rod base pose prior
    graph.add(PriorFactor<Pose3>(
        helper_rod_->get_pose_key(0),
        helper_base_pose,
        base_pose_noise_));
    
    // Constrain tip of helper rod to intersect perpendicularly with main rod tip 
    graph.add(BetweenFactor<Pose3>(
        helper_rod_->get_pose_key(-1),
        main_rod_->get_pose_key(-1),
        Pose3(Rot3::Ry(M_PI / 2.0), Point3::Zero()),  // 90 rotation about x
        snare_constraint_noise_));

    // Constrain base of EE rod to just continue on from main rod tip
    graph.add(BetweenFactor<Pose3>(
        main_rod_->get_pose_key(-1),
        end_effector_rod_->get_pose_key(0),
        Pose3::Identity(),  // just keep going in the same direction, since its the same physical rod
        twist_noise_));
    
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
        tip_wrench.cov));

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

    get_rod_bases_jacobian(marginals, solution.J_rod_bases);

    return solution;
}


void MultiRobotModel::get_rod_bases_jacobian(
    const Marginals& marginals,
    Eigen::Matrix<double, 6, 12>& J_rod_bases) const
{
    Key TM = main_rod_->get_pose_key(0);
    Key TH = helper_rod_->get_pose_key(0);
    Key T = end_effector_rod_->get_pose_key(-1);

    JointMarginal joint = marginals.jointMarginalCovariance({TM, TH, T});

    Eigen::Matrix<double, 12, 12> sigma_QQ;
    sigma_QQ.block<6,6>(0, 0) = joint(TM, TM);
    sigma_QQ.block<6,6>(0, 6) = joint(TM, TH);
    sigma_QQ.block<6,6>(6, 0) = joint(TH, TM);
    sigma_QQ.block<6,6>(6, 6) = joint(TH, TH);

    Eigen::Matrix<double, 6, 12> sigma_TQ;
    sigma_TQ.block<6,6>(0, 0) = joint(T, TM);
    sigma_TQ.block<6,6>(0, 6) = joint(T, TH);

    Eigen::LDLT<Eigen::Matrix<double, 12, 12>> ldlt(sigma_QQ);
    
    J_rod_bases = sigma_TQ * ldlt.solve(Eigen::Matrix<double, 12, 12>::Identity());
}
