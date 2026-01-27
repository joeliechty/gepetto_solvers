#pragma once

#include "utils/Gaussians.h"
#include "utils/SolverBase.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "cosserat_rod/CosseratRodSolver.h"
#include <gtsam/base/Vector.h>


struct CosseratDynamicsConfig {
    CosseratRodSolverConfig rod_config;

    double dt;
    double linear_damping;
    double rotational_damping;
    double linear_inertia;
    double rotational_inertia;
    double dynamics_noise_sigma;
    
    gtsam::Vector6 initial_tip_wrench;
};


struct CosseratDynamicsMarginals {
    CosseratRodMarginals rod;
    std::vector<Vector6Gaussian> velocities;
};


class CosseratDynamicsSolver : SolverBase {
public:
    CosseratDynamicsSolver(const CosseratDynamicsConfig& config);

    Solution<CosseratDynamicsMarginals> solve();

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;
    
    void init_prev_marginals();

    const int num_nodes_;
    const double dt_;
    const double rod_length_;
    const double linear_damping_;
    const double rotational_damping_;
    const double linear_inertia_;
    const double rotational_inertia_;

    gtsam::SharedDiagonal dynamics_noise_;
    gtsam::SharedDiagonal base_pose_noise_;

    Solution<CosseratRodMarginals> static_solution_;

    std::unique_ptr<CosseratRodModel> rod_;
    CosseratDynamicsMarginals rod_marginals_;
};