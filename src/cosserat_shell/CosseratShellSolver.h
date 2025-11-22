#pragma once

#include <gtsam/base/Matrix.h>

#include "CosseratShellModel.h"
#include "utils/SolverBase.h"


struct CosseratShellSolverConfig {
    double num_nodes_x;
    double num_nodes_y;
    double element_size;

    gtsam::Matrix6 K_inv;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_stress_force;
    double sigma_stress_moment;
};


class CosseratShellSolver : SolverBase {
public:
    CosseratShellSolver(const CosseratShellSolverConfig& config);

    Solution<CosseratShellMarginals> solve(
        const gtsam::Matrix4& displacement_mean,
        const gtsam::Matrix6& displacement_cov);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    std::unique_ptr<CosseratShellModel> shell_;

    gtsam::Matrix4 displacement_mean_;
    gtsam::Matrix6 displacement_cov_;

    CosseratShellMarginals extracted_;
};