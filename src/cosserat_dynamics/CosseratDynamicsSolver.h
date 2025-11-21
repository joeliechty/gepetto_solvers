#pragma once

#include "cosserat_rod/CosseratRodModel.h"
#include "cosserat_rod/CosseratRodSolver.h"


struct CosseratDynamicsConfig {
    CosseratRodSolverConfig rod_config;

    int num_time_steps;
    double dt;
    double linear_damping;
    double rotational_damping;
    double linear_inertia;
    double rotational_inertia;
    
    gtsam::Vector6 initial_tip_wrench;
};


struct CosseratDynamicsMarginals {
    std::vector<Solution<CosseratRodMarginals>> rods_t;
};


class CosseratDynamicsSolver {
public:
    CosseratDynamicsSolver(const CosseratDynamicsConfig& config);

    Solution<CosseratDynamicsMarginals> solve();

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

    Solution<CosseratRodMarginals> static_solution_;
    std::vector<std::unique_ptr<CosseratRodModel>> rods_t_;
};