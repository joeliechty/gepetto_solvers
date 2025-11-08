#pragma once

#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


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


class CosseratRod {
public:
    CosseratRod(const CosseratRodConfig& config);
    gtsam::NonlinearFactorGraph build_graph() const;
    gtsam::Values get_initial_values();

private:
    CosseratRodConfig config_;
    double ds_;
    gtsam::Matrix6 K_inv_;
    gtsam::noiseModel::Diagonal::shared_ptr small_wrench_cov_;
    gtsam::noiseModel::Diagonal::shared_ptr cosserat_twist_cov_;
};


class BasicCosseratSolver {
    BasicCosseratSolver(const CosseratRodConfig& config);

    void solve(gtsam::Vector3 tip_force);

private:
    void add_boundary_conditions(const gtsam::Vector3& tip_force);

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    CosseratRodConfig rod_config_;
    CosseratRod rod_;
};