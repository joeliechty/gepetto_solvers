#pragma once

#include "CosseratRodSolver.h"


struct CosseratRodDynamicsConfig {
    CosseratRodSolverConfig rod_config;

    int num_time_steps;
    double dt;
    double linear_damping;
    double rotational_damping;
    double linear_inertia;
    double rotational_inertia;
    double sigma_dynamics_noise;
    
    gtsam::Vector6 initial_tip_wrench;
};


struct CosseratRodDynamicsSolution {
    SolutionMetadata meta;
    std::vector<CosseratRodSolution> marginals;
};

class CosseratRodDynamicsSolver {
public:
    CosseratRodDynamicsSolver(const CosseratRodDynamicsConfig& config);

    CosseratRodDynamicsSolution solve();

private:
    void init_values();
    
    void build_graph();

    const int num_time_steps_;
    const int num_nodes_;
    const double dt_;
    const double rod_length_;
    const double linear_damping_;
    const double rotational_damping_;
    const double linear_inertia_;
    const double rotational_inertia_;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    gtsam::SharedDiagonal small_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;
    gtsam::SharedDiagonal dynamics_noise_;

    CosseratRodSolution static_solution_;
    std::vector<std::unique_ptr<CosseratRodModel>> rods_t_;
};