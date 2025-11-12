#pragma once

#include "parallel_robot.h"


struct ParallelRobotSolverConfig {
    int nodes_per_rod;
    int num_rods;

    gtsam::Matrix6 K_inv;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_small_force;
    double sigma_small_moment;

    std::vector<gtsam::Matrix4> base_end_poses;
    std::vector<gtsam::Matrix4> tip_end_poses;

    double sigma_end_pose_pos;
    double sigma_end_pose_rot;
};


struct SolutionMetadata {
    double solve_time_ms;
    double total_time_ms;
    int iterations;
    int error;
};


struct ParallelRobotSolution {
    SolutionMetadata meta;
    ParallelRobotMarginals marginals;
};


class ParallelRobotSolver {
public:
    ParallelRobotSolver(const ParallelRobotSolverConfig& config);

    ParallelRobotSolution solve(const gtsam::Vector& rod_lengths);

private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    gtsam::SharedDiagonal small_wrench_cov_;
    gtsam::SharedDiagonal base_pose_cov_;

    std::unique_ptr<ParallelRobot> robot_;
};