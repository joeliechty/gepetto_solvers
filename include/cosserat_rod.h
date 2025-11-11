#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <memory>


struct CosseratRodConfig {
    double rod_length;
    int num_nodes;

    double k_bending;
    double k_torsion;
    double k_shear;
    double k_extension;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_small_force;
    double sigma_small_moment;

    double sigma_base_pose_pos;
    double sigma_base_pose_rot;
};


struct SolutionMetadata {
    double solve_time_ms;
    double total_time_ms;
};


struct CosseratRodMarginals {
    std::vector<gtsam::Matrix4> pose_mean;
    std::vector<gtsam::Matrix6> pose_cov;

    std::vector<gtsam::Vector6> stress_mean;
    std::vector<gtsam::Matrix6> stress_cov;

    std::vector<gtsam::Vector6> wrench_mean;
    std::vector<gtsam::Matrix6> wrench_cov;
};


struct CosseratRodSolution {
    SolutionMetadata meta;
    CosseratRodMarginals marginals;
};


class CosseratRod {
public:
    CosseratRod(
        double rod_length, 
        int num_nodes, 
        gtsam::Matrix6 K_inv, 
        gtsam::SharedDiagonal twist_cov,
        gtsam::SharedDiagonal stress_cov);

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values() const;

    CosseratRodMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

    gtsam::Key get_pose_key(int node_idx) const;

    gtsam::Key get_stress_key(int node_idx) const;
    
    gtsam::Key get_wrench_key(int node_idx) const;
    
    const std::vector<gtsam::Key>& get_wrench_keys() const;

private:
    const int id_;
    inline static int next_id_ = 0;

    const int num_nodes_;
    std::vector<double> ds_;
    std::vector<gtsam::Matrix6> K_inv_;

    gtsam::SharedDiagonal twist_cov_;
    gtsam::SharedDiagonal stress_cov_;

    std::vector<gtsam::Key> pose_keys_;
    std::vector<gtsam::Key> stress_keys_;
    std::vector<gtsam::Key> wrench_keys_;
};


class BasicCosseratSolver {
public:
    BasicCosseratSolver(const CosseratRodConfig& config);

    CosseratRodSolution solve(
        const std::optional<gtsam::Vector3>& tip_force_mean, 
        const std::optional<gtsam::Matrix3>& tip_force_cov,
        const std::optional<gtsam::Vector3>& tip_pos_mean,
        const std::optional<gtsam::Matrix3>& tip_pos_cov);

private:
    void add_boundary_factors();

    void add_wrench_prior_factors(
        const std::optional<gtsam::Vector3>& tip_force_mean, 
        const std::optional<gtsam::Matrix3>& tip_force_cov);

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    gtsam::SharedDiagonal small_wrench_cov_;
    gtsam::SharedDiagonal base_pose_cov_;

    std::unique_ptr<CosseratRod> rod_;
};