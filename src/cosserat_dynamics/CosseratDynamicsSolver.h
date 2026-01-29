#pragma once

#include "utils/SolverBase.h"
#include "cosserat_rod/CosseratRodModel.h"
#include <gtsam/linear/NoiseModel.h>


struct CosseratDynamicsConfig {
    double rod_length;
    int num_nodes;
    gtsam::Matrix6 K_inv;

    double sigma_dynamics_noise;
    double sigma_twist_noise;
    double sigma_wrench_noise;
    double sigma_init_tip_wrench;
    double sigma_init_velocity;

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
    void solve_static_rod(const CosseratDynamicsConfig& config);

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

    const gtsam::Vector6 initial_tip_wrench_;

    gtsam::SharedDiagonal wrench_noise_;
    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal dynamics_noise_;
    gtsam::SharedDiagonal init_tip_wrench_noise_;
    gtsam::SharedDiagonal init_velocity_noise_;
    
    std::vector<std::unique_ptr<CosseratRodModel>> rods_t_;
    CosseratDynamicsMarginals extracted_;
};