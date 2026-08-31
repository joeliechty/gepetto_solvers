#pragma once

#include "gepetto_solvers/utils/SolverBase.h"
#include "gepetto_solvers/cosserat_rod/CosseratRodModel.h"
#include <gtsam/linear/NoiseModel.h>


struct CosseratDynamicsConfig {
    SolverBaseConfig base;
    
    double rod_length;
    int num_nodes;
    gtsam::Matrix6 K_inv;

    double dynamics_noise_sigma;
    double twist_noise_sigma;
    double wrench_noise_sigma;

    int num_time_steps;
    double dt;

    double linear_damping;
    double rotational_damping;

    gtsam::Vector6 init_wrench_mean;
    double init_velocity_sigma;

    double linear_inertia;
    double rotational_inertia;
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

    const gtsam::Vector6 init_wrench_mean_;

    gtsam::SharedDiagonal wrench_noise_;
    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal dynamics_noise_;
    gtsam::SharedDiagonal init_velocity_noise_;
    
    std::vector<std::unique_ptr<CosseratRodModel>> rods_t_;
    CosseratDynamicsMarginals extracted_;
};