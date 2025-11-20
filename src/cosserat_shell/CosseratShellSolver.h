#pragma once

#include <gtsam/base/Matrix.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "CosseratShellModel.h"


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


struct CosseratShellSolution {
    SolutionMetadata meta;
    CosseratShellMarginals marginals;
};


class CosseratShellSolver {
public:
    CosseratShellSolver(const CosseratShellSolverConfig& config);

    CosseratShellSolution solve(
        const gtsam::Matrix4& displacement_mean,
        const gtsam::Matrix6& displacement_cov);

private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    std::unique_ptr<CosseratShellModel> shell_;
};