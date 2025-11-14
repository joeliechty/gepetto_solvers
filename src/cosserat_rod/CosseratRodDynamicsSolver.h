#pragma once

#include "CosseratRodSolver.h"

constexpr int NUM_STEPS = 10;


struct CosseratRodDynamicsSolution {
    SolutionMetadata meta;
    std::vector<CosseratRodMarginals> marginals;
};

class CosseratRodDynamicsSolver {
public:
    CosseratRodDynamicsSolver(const CosseratRodSolverConfig& config);

    CosseratRodDynamicsSolution step();

    void build_graph();

private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    gtsam::SharedDiagonal small_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;

    std::array<std::unique_ptr<CosseratRodModel>, NUM_STEPS> rods_t_;
};