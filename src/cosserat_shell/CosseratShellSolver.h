#pragma once

#include <gtsam/base/Matrix.h>

#include "CosseratShellModel.h"
#include "utils/SolverBase.h"
#include "utils/Gaussians.h"


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
        const Pose3Gaussian& displacement);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    std::unique_ptr<CosseratShellModel> shell_;

    Pose3Gaussian displacement_;
    CosseratShellMarginals extracted_;
};