#pragma once

#include <gtsam/base/Matrix.h>

#include "parallel_robot/ParallelRobotModel.h"
#include "utils/SolverBase.h"


struct ParallelRobotSolverConfig {
    int nodes_per_rod;

    gtsam::Matrix6 K_inv;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_small_force;
    double sigma_small_moment;

    std::array<gtsam::Matrix4, NUM_RODS> base_end_poses;
    std::array<gtsam::Matrix4, NUM_RODS> tip_end_poses;

    double sigma_end_pose_pos;
    double sigma_end_pose_rot;
};


class ParallelRobotSolver {
public:
    ParallelRobotSolver(const ParallelRobotSolverConfig& config);

    Solution<ParallelRobotMarginals> solve(
        const std::array<double, NUM_RODS>& rod_lengths,
        double sigma_rod_lengths,
        const gtsam::Vector6& wrench_mean,
        const gtsam::Matrix6& wrench_cov);

private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    gtsam::SharedDiagonal small_wrench_cov_;
    gtsam::SharedDiagonal base_pose_cov_;

    std::unique_ptr<ParallelRobot> robot_;
};