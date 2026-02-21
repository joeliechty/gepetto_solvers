#pragma once

#include <gtsam/base/Matrix.h>

#include "multi_robot/MultiRobotModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"


struct MultiRobotSolverConfig {
    SolverBaseConfig base;
    
    int nodes_per_rod;

    gtsam::Matrix6 K_inv;

    double snare_distance_to_tip;
    
    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_small_force;
    double sigma_small_moment;
};


class MultiRobotSolver : SolverBase {
public:
    MultiRobotSolver(const MultiRobotSolverConfig& config);

    Solution<MultiRobotMarginals> solve(
        const Pose3Gaussian& main_base_pose,
        double main_insertion,
        const Pose3Gaussian& helper_base_pose,
        double helper_insertion,
        const Vector6Gaussian& tip_wrench);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    std::unique_ptr<MultiRobotModel> robot_;
    
    Pose3Gaussian main_base_pose_;
    double main_insertion_;
    Pose3Gaussian helper_base_pose_;
    double helper_insertion_;
    Vector6Gaussian tip_wrench_;

    MultiRobotMarginals extracted_;
};