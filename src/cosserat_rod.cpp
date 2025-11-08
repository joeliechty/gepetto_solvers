#include "cosserat_rod.h"

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include "gtsam_factors.h"

using namespace gtsam;


inline Symbol pose_key(int node_idx) { return gtsam::Symbol('T', node_idx); }
inline Symbol stress_key(int node_idx) { return gtsam::Symbol('S', node_idx); }
inline Symbol wrench_key(int node_idx) { return gtsam::Symbol('F', node_idx); }


CosseratRod::CosseratRod (const CosseratRodConfig& config) {
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


Values CosseratRod::get_initial_values() {
    Values values;
    
    for (int i = 0; i < config_.num_backbone_nodes; ++i) {
        values.insert(pose_key(i), Pose3::Identity());
        values.insert(stress_key(i), Vector6(Vector6::Zero()));
        if (i > 0) { values.insert(wrench_key(i), Vector6(Vector6::Zero())); }
    }

    return values;
}


NonlinearFactorGraph CosseratRod::build_graph() const {
    NonlinearFactorGraph graph;

    for (int i = 0; i + 1 < config_.num_backbone_nodes; ++i) {
        auto factor = CosseratRodTwistFactor(
            pose_key(i), 
            pose_key(i + 1), 
            stress_key(i), 
            stress_key(i + 1), 
            ds_, 
            K_inv_, 
            cosserat_twist_cov_);

        graph.add(factor);
    }
        
    // Cosserat stress factors
    for (int i = 0; i + 1 < config_.num_backbone_nodes; ++i) {
        auto factor = CosseratRodStressFactor(
            pose_key(i), 
            pose_key(i + 1), 
            stress_key(i), 
            stress_key(i + 1),
            wrench_key(i + 1),
            small_wrench_cov_);
        
        graph.add(factor);
    }

    // Near-zero prior constraint for tip stress
    auto factor = PriorFactor<Vector6>(
        stress_key(config_.num_backbone_nodes - 1), Vector6::Zero(), small_wrench_cov_);

    graph.add(factor);

    return graph;
}


BasicCosseratSolver::BasicCosseratSolver(const CosseratRodConfig& rod_config)
:
    rod_config_(rod_config), rod_(rod_config)
{
    values_ = rod_.get_initial_values();
}


void BasicCosseratSolver::solve(gtsam::Vector3 tip_force) {
    graph_ = rod_.build_graph();
    add_boundary_conditions(tip_force);

    LevenbergMarquardtOptimizer optimizer(graph_, values_);
    values_ = optimizer.optimize();

    marginals_ = Marginals(graph_, values_);
}


void BasicCosseratSolver::add_boundary_conditions(const Vector3& tip_force) { 
    graph_.add(PriorFactor<Pose3>(pose_key(0), Pose3::Identity()));

    SharedDiagonal tip_wrench_cov = noiseModel::Diagonal::Sigmas((Vector(6) << 
        rod_config_.sigma_small_moment, rod_config_.sigma_small_moment, rod_config_.sigma_small_moment, 
        rod_config_.sigma_small_force, rod_config_.sigma_small_force, rod_config_.sigma_small_force).finished());

    Vector6 tip_wrench_mean = Vector6::Zero();
    tip_wrench_mean.tail<3>() = tip_force;

    graph_.add(PriorFactor<Vector6>(wrench_key(rod_config_.num_backbone_nodes - 1), tip_wrench_mean, tip_wrench_cov));
}

