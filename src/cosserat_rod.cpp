#include "cosserat_rod.h"

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "gtsam_factors.h"

using namespace gtsam;


CosseratRod::CosseratRod (const CosseratRodConfig& config) 
: 
    config_(config),
    id_(next_id_++)
{
    K_inv_ = Matrix6::Zero();
    K_inv_(0, 0) = 1 / config_.k_bending;
    K_inv_(1, 1) = 1 / config_.k_bending;
    K_inv_(2, 2) = 1 / config_.k_torsion;
    K_inv_(3, 3) = 1 / config_.k_shear;
    K_inv_(4, 4) = 1 / config_.k_shear;
    K_inv_(5, 5) = 1 / config_.k_extension;

    ds_ = config_.rod_length / (config_.num_backbone_nodes - 1);

    small_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
        config_.sigma_small_moment, config_.sigma_small_moment, config_.sigma_small_moment, 
        config_.sigma_small_force, config_.sigma_small_force, config_.sigma_small_force).finished());
        
    cosserat_twist_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
        config_.sigma_twist_rotation, config_.sigma_twist_rotation, config_.sigma_twist_rotation, 
        config_.sigma_twist_position, config_.sigma_twist_position, config_.sigma_twist_position).finished());
}


Values CosseratRod::get_initial_values() const {
    Values values;
    
    for (int i = 0; i < config_.num_backbone_nodes; ++i) {
        values.insert(get_pose_key(i), Pose3(Rot3::Identity(), Point3(0.0, 0.0, i * ds_)));
        values.insert(get_stress_key(i), Vector6(Vector6::Zero()));
        if (i > 0) { values.insert(get_wrench_key(i), Vector6(Vector6::Zero())); }
    }

    return values;
}


NonlinearFactorGraph CosseratRod::build_graph() const {
    NonlinearFactorGraph graph;

    for (int i = 0; i + 1 < config_.num_backbone_nodes; ++i) {
        auto factor = CosseratRodTwistFactor(
            get_pose_key(i), 
            get_pose_key(i + 1), 
            get_stress_key(i), 
            get_stress_key(i + 1), 
            ds_, 
            K_inv_, 
            cosserat_twist_cov_);

        graph.add(factor);
    }
        
    // Cosserat stress factors
    for (int i = 0; i + 1 < config_.num_backbone_nodes; ++i) {
        auto factor = CosseratRodStressFactor(
            get_pose_key(i), 
            get_pose_key(i + 1), 
            get_stress_key(i), 
            get_stress_key(i + 1),
            get_wrench_key(i + 1),
            small_wrench_cov_);
        
        graph.add(factor);
    }

    // Near-zero prior constraint for tip stress
    auto factor = PriorFactor<Vector6>(
        get_stress_key(config_.num_backbone_nodes - 1), Vector6::Zero(), small_wrench_cov_);

    graph.add(factor);

    return graph;
}


CosseratRodSolution CosseratRod::extract_solution(
    const gtsam::Values& values, 
    const gtsam::Marginals& marginals) const 
{
    CosseratRodSolution solution;

    solution.pose_mean.resize(config_.num_backbone_nodes);
    solution.pose_cov.resize(config_.num_backbone_nodes);
    solution.wrench_mean.resize(config_.num_backbone_nodes - 1);
    solution.wrench_cov.resize(config_.num_backbone_nodes - 1);

    for (int i = 0; i < config_.num_backbone_nodes; ++i) {
        solution.pose_mean[i] = values.at<Pose3>(get_pose_key(i)).matrix();
        solution.pose_cov[i] = marginals.marginalCovariance(get_pose_key(i));

        // No applied force at the base pose
        if (i > 0) {
            solution.wrench_mean[i - 1] = values.at<Vector6>(get_wrench_key(i));
            solution.wrench_cov[i - 1] = marginals.marginalCovariance(get_wrench_key(i));
        }
    }

    return solution;
}


Symbol CosseratRod::get_pose_key(int node_idx) const {
    return gtsam::Symbol('T', 1000 * id_ + node_idx);
}


Symbol CosseratRod::get_stress_key(int node_idx) const { 
    return gtsam::Symbol('S', 1000 * id_ + node_idx); 
}


Symbol CosseratRod::get_wrench_key(int node_idx) const { 
    return gtsam::Symbol('F', 1000 * id_ + node_idx); 
}


BasicCosseratSolver::BasicCosseratSolver(const CosseratRodConfig& rod_config)
:
    rod_config_(rod_config), rod_(rod_config)
{
    values_ = rod_.get_initial_values();
}


CosseratRodSolution BasicCosseratSolver::solve(gtsam::Vector3 tip_force) {
    graph_ = rod_.build_graph();

    std::cout << "adding boundary" << std::endl;

    add_boundary_factors();

    std::cout << "adding force" << std::endl;

    add_force_factors(tip_force);

    LevenbergMarquardtParams params;
    params.setVerbosityLM("SUMMARY");
    LevenbergMarquardtOptimizer optimizer(graph_, values_, params);

    std::cout << "running optimization" << std::endl;

    values_ = optimizer.optimize();

    std::cout << "getting marginals" << std::endl;

    marginals_ = Marginals(graph_, values_);

    std::cout << "extracting solution" << std::endl;

    CosseratRodSolution solution = rod_.extract_solution(values_, marginals_);

    return solution;
}


void BasicCosseratSolver::add_boundary_factors() {
    //TODO don't access keys in the solver only interface through rod class
    SharedDiagonal base_cov = noiseModel::Isotropic::Sigma(6, 1e-3);
    auto factor = PriorFactor<Pose3>(rod_.get_pose_key(0), Pose3::Identity(), base_cov);

    graph_.add(factor);
}


void BasicCosseratSolver::add_force_factors(const Vector3& tip_force) {
    
    SharedDiagonal wrench_cov = noiseModel::Isotropic::Sigma(6, 1e-3);

    for (int i = 1; i < rod_config_.num_backbone_nodes; ++i) {
        auto wrench_factor = PriorFactor<Vector6>(
            rod_.get_wrench_key(i), 
            Vector6::Zero(), 
            wrench_cov);

        graph_.add(wrench_factor);
    }


    // SharedDiagonal wrench_cov = noiseModel::Diagonal::Sigmas((Vector(6) << 
    //     rod_config_.sigma_small_moment, rod_config_.sigma_small_moment, rod_config_.sigma_small_moment, 
    //     rod_config_.sigma_small_force, rod_config_.sigma_small_force, rod_config_.sigma_small_force).finished());


    // for (int i = 1; i + 1 < rod_config_.num_backbone_nodes; ++i) {
    //     auto factor = PriorFactor<Vector6>(
    //         wrench_key(i), 
    //         Vector6::Zero(), 
    //         wrench_cov);

    //     graph_.add(factor);
    // }

    // Vector6 tip_wrench_mean = Vector6::Zero();
    // tip_wrench_mean.tail<3>() = tip_force;

    // graph_.add(PriorFactor<Vector6>(wrench_key(rod_config_.num_backbone_nodes - 1), tip_wrench_mean, wrench_cov));
}
