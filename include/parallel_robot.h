#pragma once

#include "cosserat_rod.h"
#include <gtsam/linear/NoiseModel.h>
#include <memory>


struct ParallelRobotMarginals {
    std::vector<CosseratRodMarginals> rods;

    gtsam::Matrix4 plate_pose_mean;
    gtsam::Matrix6 plate_pose_cov;

    gtsam::Vector6 plate_wrench_mean;
    gtsam::Matrix6 plate_wrench_cov;
};


class ParallelRobot {
public:
    ParallelRobot(
        int num_rods,
        int nodes_per_rod, 
        gtsam::Matrix6 K_inv,
        gtsam::SharedDiagonal rod_twist_cov,
        gtsam::SharedDiagonal small_wrench_cov_,
        std::vector<gtsam::Pose3> base_end_poses,
        std::vector<gtsam::Pose3> tip_end_poses,
        gtsam::SharedDiagonal end_pose_cov);

    gtsam::NonlinearFactorGraph build_graph(const std::vector<double>& rod_lengths);

    gtsam::Values get_initial_values() const;

    ParallelRobotMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

private:
    const int num_rods_;
    const int nodes_per_rod_;
    const gtsam::Matrix6 K_inv_;
    const gtsam::SharedDiagonal rod_twist_cov_;
    const gtsam::SharedDiagonal small_wrench_cov_;

    const std::vector<gtsam::Pose3> base_end_poses_;
    const std::vector<gtsam::Pose3> tip_end_poses_;
    const gtsam::SharedDiagonal end_pose_cov_;

    std::vector<std::unique_ptr<CosseratRod>> rods_;
};
