#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>


struct CosseratRodConfig {
    double rod_length;
    int num_backbone_nodes;

    double k_bending;
    double k_torsion;
    double k_shear;
    double k_extension;

    double sigma_twist_position;
    double sigma_twist_rotation;
    double sigma_stress_force;
    double sigma_stress_moment;
    double sigma_small_force;
    double sigma_small_moment;
    double sigma_base_position;
    double sigma_base_rotation;
};


struct CosseratRodSolution {
    std::vector<gtsam::Matrix4> pose_mean;
    std::vector<gtsam::Matrix6> pose_cov;

    std::vector<gtsam::Vector6> wrench_mean;
    std::vector<gtsam::Matrix6> wrench_cov;
};


class CosseratRod {
public:
    CosseratRod(const CosseratRodConfig& config);

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values() const;

    CosseratRodSolution extract_solution(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;
    
    gtsam::Symbol get_pose_key(int node_idx) const;
    
    gtsam::Symbol get_stress_key(int node_idx) const;

    gtsam::Symbol get_wrench_key(int node_idx) const;

private:
    const CosseratRodConfig config_;
    double ds_;
    gtsam::Matrix6 K_inv_;
    gtsam::noiseModel::Diagonal::shared_ptr small_wrench_cov_;
    gtsam::noiseModel::Diagonal::shared_ptr cosserat_twist_cov_;

    const int id_;
    inline static int next_id_ = 0;
};


class BasicCosseratSolver {
public:
    BasicCosseratSolver(const CosseratRodConfig& config);

    CosseratRodSolution solve(gtsam::Vector3 tip_force);

private:
    void add_boundary_factors();

    void add_force_factors(const gtsam::Vector3& tip_force);

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    CosseratRodConfig rod_config_;
    CosseratRod rod_;
};