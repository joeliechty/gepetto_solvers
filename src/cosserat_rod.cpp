#include "cosserat_rod.h"

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
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
        values.insert(get_wrench_key(i), Vector6(Vector6::Zero()));
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
            get_wrench_key(i),
            small_wrench_cov_);
        
        graph.add(factor);
    }

    // Constrain tip strain to be equal to tip force TODO change this to do spatial/body
    auto factor = TipStressWrenchFactor(
        get_stress_key(config_.num_backbone_nodes - 1), 
        get_wrench_key(config_.num_backbone_nodes - 1),
        get_pose_key(config_.num_backbone_nodes - 1),
        small_wrench_cov_);
    
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
    solution.wrench_mean.resize(config_.num_backbone_nodes);
    solution.wrench_cov.resize(config_.num_backbone_nodes);

    for (int i = 0; i < config_.num_backbone_nodes; ++i) {
        solution.pose_mean[i] = values.at<Pose3>(get_pose_key(i)).matrix();
        solution.pose_cov[i] = marginals.marginalCovariance(get_pose_key(i));

        solution.wrench_mean[i] = values.at<Vector6>(get_wrench_key(i));
        solution.wrench_cov[i] = marginals.marginalCovariance(get_wrench_key(i));
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


CosseratRodSolution BasicCosseratSolver::solve(const Vector3& tip_force_mean, const Matrix3& tip_force_cov) {
    graph_ = rod_.build_graph();

    add_boundary_factors();
    add_force_factors(tip_force_mean, tip_force_cov);

    DoglegParams params;
    params.setVerbosity("TERMINATION");
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    values_ = optimizer.optimize();
    marginals_ = Marginals(graph_, values_);

    CosseratRodSolution solution = rod_.extract_solution(values_, marginals_);

    return solution;
}


void BasicCosseratSolver::add_boundary_factors() {
    //TODO don't access keys in the solver only interface through rod class
    SharedDiagonal base_cov = noiseModel::Isotropic::Sigma(6, 1e-3);
    auto factor = PriorFactor<Pose3>(rod_.get_pose_key(0), Pose3::Identity(), base_cov);

    graph_.add(factor);
}


void BasicCosseratSolver::add_force_factors(const Vector3& tip_force_mean, const Matrix3& tip_force_cov) {
    
    SharedDiagonal wrench_cov = noiseModel::Isotropic::Sigma(6, 1e-3);

    for (int i = 0; i + 1 < rod_config_.num_backbone_nodes; ++i) {
        auto wrench_factor = PriorFactor<Vector6>(
            rod_.get_wrench_key(i), 
            Vector6::Zero(),
            wrench_cov);

        graph_.add(wrench_factor);
    }

    Vector6 tip_wrench_mean = Vector6::Zero();
    tip_wrench_mean.tail<3>() = tip_force_mean;

    Matrix6 tip_wrench_cov = 1e-6 * Matrix6::Identity();
    tip_wrench_cov.block<3,3>(3, 3) = tip_force_cov;
    auto tip_wrench_noise = noiseModel::Gaussian::Covariance(tip_wrench_cov);

    graph_.add(PriorFactor<Vector6>(rod_.get_wrench_key(rod_config_.num_backbone_nodes - 1), tip_wrench_mean, tip_wrench_noise));
}
