#pragma once

#include "utils/SolverBase.h"
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


class CosseratDynamicsSolver : SolverBase {
public:
    CosseratDynamicsSolver(const CosseratDynamicsConfig& config);

    Solution<CosseratDynamicsMarginals> solve();

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    const int num_time_steps_;
    const int num_nodes_;
    const double dt_;
    const double rod_length_;
    const double linear_damping_;
    const double rotational_damping_;
    const double linear_inertia_;
    const double rotational_inertia_;

    gtsam::SharedDiagonal small_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;

    Solution<CosseratRodMarginals> static_solution_;
    std::vector<std::unique_ptr<CosseratRodModel>> rods_t_;
    CosseratDynamicsMarginals extracted_;
};