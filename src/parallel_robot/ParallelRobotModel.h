#pragma once

#include "cosserat_rod/CosseratRodModel.h"
#include <gtsam/linear/NoiseModel.h>


constexpr int NUM_RODS = 6;


struct ParallelRobotMarginals {
    std::array<CosseratRodMarginals, NUM_RODS> rods;

    gtsam::Matrix4 platform_pose_mean;
    gtsam::Matrix6 platform_pose_cov;

    gtsam::Vector6 platform_wrench_mean;
    gtsam::Matrix6 platform_wrench_cov;

    gtsam::Matrix6 rod_lengths_jacobian;
};


class ParallelRobot {
public:
    ParallelRobot(
        int nodes_per_rod, 
        gtsam::Matrix6 K_inv,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        std::array<gtsam::Matrix4, NUM_RODS> base_end_poses,
        std::array<gtsam::Matrix4, NUM_RODS> tip_end_poses, 
        double sigma_end_pose_pos,
        double sigma_end_pose_rot);

    gtsam::NonlinearFactorGraph build_graph(
        const std::array<double, NUM_RODS>& rod_lengths,
        double sigma_rod_lengths,
        const gtsam::Vector6& wrench_mean,
        const gtsam::Matrix6& wrench_cov);

    gtsam::Values get_initial_values() const;

    ParallelRobotMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;
    
    gtsam::Matrix6 get_rod_lengths_jacobian(const gtsam::Marginals& marginals) const;

private:
    const std::array<gtsam::Matrix4, NUM_RODS> base_end_poses_;
    const std::array<gtsam::Matrix4, NUM_RODS> tip_end_poses_;
    const gtsam::SharedDiagonal small_wrench_noise_;
    
    const double sigma_end_pose_pos_;
    const double sigma_end_pose_rot_;
    
    std::array<std::unique_ptr<CosseratRodModel>, NUM_RODS> rods_;
};
