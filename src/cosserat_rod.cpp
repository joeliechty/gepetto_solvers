#include "cosserat_rod.h"

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include "gtsam_factors.h"

using namespace gtsam;


CosseratRod::CosseratRod (
    double rod_length,
    int num_nodes,
    Matrix6 K_inv,
    SharedDiagonal twist_cov,
    SharedDiagonal stress_cov) 
: 
    id_(next_id_++),
    num_nodes_(num_nodes),
    twist_cov_(twist_cov), 
    stress_cov_(stress_cov)
{
    double ds = rod_length / (num_nodes - 1);
    ds_ = std::vector<double>(num_nodes - 1, ds);

    K_inv_ = std::vector<Matrix6>(num_nodes - 1, K_inv);

    pose_keys_.reserve(num_nodes_);
    stress_keys_.reserve(num_nodes_);
    wrench_keys_.reserve(num_nodes_);

    for (int i = 0; i < num_nodes_; i++) {
        pose_keys_.push_back(  Symbol('T', 1000 * id_ + i));
        stress_keys_.push_back(Symbol('S', 1000 * id_ + i)); 
        wrench_keys_.push_back(Symbol('F', 1000 * id_ + i)); 
    }
}


Key CosseratRod::get_pose_key(int node_idx) const { return pose_keys_[node_idx]; }


Key CosseratRod::get_stress_key(int node_idx) const { return stress_keys_[node_idx]; }


Key CosseratRod::get_wrench_key(int node_idx) const { return wrench_keys_[node_idx]; }


const std::vector<Key>& CosseratRod::get_wrench_keys() const {return wrench_keys_; }


Values CosseratRod::get_initial_values() const {
    Values values;
    
    for (int i = 0; i < num_nodes_; ++i) {
        values.insert(pose_keys_[i], Pose3(Rot3::Identity(), Point3(0.0, 0.0, i * ds_[i])));
        values.insert(stress_keys_[i], Vector6(Vector6::Zero()));
        values.insert(wrench_keys_[i], Vector6(Vector6::Zero()));
    }

    return values;
}


NonlinearFactorGraph CosseratRod::build_graph() const {
    NonlinearFactorGraph graph;

    for (int i = 0; i + 1 < num_nodes_; ++i) {
        auto factor = CosseratRodTwistFactor(
            pose_keys_[i], 
            pose_keys_[i + 1], 
            stress_keys_[i], 
            stress_keys_[i + 1], 
            ds_[i], 
            K_inv_[i], 
            twist_cov_);

        graph.add(factor);
    }
        
    // Cosserat stress factors
    for (int i = 0; i + 1 < num_nodes_; ++i) {
        auto factor = CosseratRodStressFactor(
            pose_keys_[i], 
            pose_keys_[i + 1], 
            stress_keys_[i], 
            stress_keys_[i + 1],
            wrench_keys_[i],
            stress_cov_);
        
        graph.add(factor);
    }

    // Constrain tip strain to be equal to tip force
    auto factor = TipStressWrenchFactor(
        stress_keys_.back(), 
        wrench_keys_.back(),
        pose_keys_.back(),
        stress_cov_);
    
    graph.add(factor);

    return graph;
}


CosseratRodMarginals CosseratRod::get_marginals(
    const gtsam::Values& values, 
    const gtsam::Marginals& marginals) const 
{
    CosseratRodMarginals solution;

    solution.pose_mean.resize(num_nodes_);
    solution.pose_cov.resize(num_nodes_);
    solution.stress_mean.resize(num_nodes_);
    solution.stress_cov.resize(num_nodes_);
    solution.wrench_mean.resize(num_nodes_);
    solution.wrench_cov.resize(num_nodes_);

    for (int i = 0; i < num_nodes_; ++i) {
        solution.pose_mean[i] = values.at<Pose3>(pose_keys_[i]).matrix();
        solution.pose_cov[i] = marginals.marginalCovariance(pose_keys_[i]);

        solution.stress_mean[i] = values.at<Vector6>(stress_keys_[i]);
        solution.stress_cov[i] = marginals.marginalCovariance(stress_keys_[i]);

        solution.wrench_mean[i] = values.at<Vector6>(wrench_keys_[i]);
        solution.wrench_cov[i] = marginals.marginalCovariance(wrench_keys_[i]);
    }

    return solution;
}


inline SharedDiagonal get_noise_model_rot_pos(double sigma_rot, double sigma_pos) {
    SharedDiagonal model = noiseModel::Diagonal::Sigmas((Vector(6) << 
        sigma_rot, sigma_rot, sigma_rot, 
        sigma_pos, sigma_pos, sigma_pos).finished());

    return model;
}


BasicCosseratSolver::BasicCosseratSolver(const CosseratRodConfig& config) {
    Matrix6 K_inv = Matrix6::Zero();
    K_inv(0, 0) = 1 / config.k_bending;
    K_inv(1, 1) = 1 / config.k_bending;
    K_inv(2, 2) = 1 / config.k_torsion;
    K_inv(3, 3) = 1 / config.k_shear;
    K_inv(4, 4) = 1 / config.k_shear;
    K_inv(5, 5) = 1 / config.k_extension;

    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_cov_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    base_pose_cov_ = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);
    
    rod_= std::make_unique<CosseratRod>(
        config.rod_length, 
        config.num_nodes, 
        K_inv, 
        twist_cov, 
        small_wrench_cov_);

    values_ = rod_->get_initial_values();
}


CosseratRodSolution BasicCosseratSolver::solve(
    const std::optional<Vector3>& tip_force_mean, 
    const std::optional<Matrix3>& tip_force_cov,
    const std::optional<Vector3>& tip_pos_mean,
    const std::optional<Matrix3>& tip_pos_cov) 
{
    auto start = std::chrono::high_resolution_clock::now();

    graph_ = rod_->build_graph();

    add_boundary_factors();
    add_wrench_prior_factors(tip_force_mean, tip_force_cov);

    DoglegParams params;
    params.setVerbosity("TERMINATION");
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    CosseratRodSolution solution;

    auto start_solve = std::chrono::high_resolution_clock::now();

    values_ = optimizer.optimize();

    auto stop_solve = std::chrono::high_resolution_clock::now();

    marginals_ = Marginals(graph_, values_);
    solution.marginals = rod_->get_marginals(values_, marginals_);

    auto stop = std::chrono::high_resolution_clock::now();

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.solve_time_ms = std::chrono::duration<double, std::milli>(stop_solve - start_solve).count();

    return solution;
}


void BasicCosseratSolver::add_boundary_factors() {
    auto factor = PriorFactor<Pose3>(rod_->get_pose_key(0), Pose3::Identity(), base_pose_cov_);
    graph_.add(factor);
}


void BasicCosseratSolver::add_wrench_prior_factors(
    const std::optional<Vector3>& tip_force_mean,
    const std::optional<Matrix3>& tip_force_cov)
{
    std::vector<Key> wrench_keys = rod_->get_wrench_keys();

    for (size_t i = 0; i + 1 < wrench_keys.size(); ++i) {
        auto wrench_factor = PriorFactor<Vector6>(
            wrench_keys[i],
            Vector6::Zero(),
            small_wrench_cov_);
        graph_.add(wrench_factor);
    }

    Vector6 tip_wrench_mean = Vector6::Zero();

    if (tip_force_mean) {
        tip_wrench_mean.tail<3>() = *tip_force_mean;
    }

    Matrix6 cov = small_wrench_cov_->sigmas().array().square().matrix().asDiagonal();

    if (tip_force_cov) {
        cov.block<3,3>(3,3) = *tip_force_cov;
    }
    
    auto tip_wrench_cov = noiseModel::Gaussian::Covariance(cov);

    graph_.add(PriorFactor<Vector6>(
        wrench_keys.back(),
        tip_wrench_mean,
        tip_wrench_cov));
}
