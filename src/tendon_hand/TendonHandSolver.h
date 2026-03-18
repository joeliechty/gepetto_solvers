#pragma once

#include "tendon_hand/TendonHandModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"
#include "tendon_robot/TendonRobotSolver.h"

#include <vector>
#include <string>


struct TendonHandSolverConfig {
    SolverBaseConfig base;

    // Noise for small external wrenches (applied to interior nodes)
    double sigma_small_force = 1e-3;
    double sigma_small_moment = 1e-3;
};


class TendonHandSolver : SolverBase {
public:
    TendonHandSolver(
        const std::vector<std::pair<std::string, TendonRobotSolverConfig>>& finger_configs,
        const TendonHandSolverConfig& config);

    Solution<TendonHandMarginals> solve(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches);

    int num_fingers() const { return hand_->num_fingers(); }

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    std::unique_ptr<TendonHandModel> hand_;

    std::vector<VectorXGaussian> tensions_;
    std::vector<Vector6Gaussian> tip_wrenches_;

    TendonHandMarginals extracted_;
};
