#pragma once

#include <gtsam/base/Matrix.h>

#include "multi_robot/MultiRobotModel.h"
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"


struct MultiRobotSolverConfig {
    SolverBaseConfig base;
    
    int nodes_per_rod;
    
    gtsam::Matrix6 K_inv;
    
    double sigma_twist_pos;
    double sigma_twist_rot;
    double sigma_small_force;
    double sigma_small_moment;

    double sigma_snare_rot_x;
    double sigma_snare_rot_y;
    double sigma_snare_rot_z;
    double sigma_snare_location;
    double snare_distance_to_tip;

    double sigma_rod_lengths;
    double sigma_base_rot;
};


class MultiRobotSolver : SolverBase {
public:
    MultiRobotSolver(const MultiRobotSolverConfig& config);

    Solution<MultiRobotMarginals> solve(
        const gtsam::Matrix4& main_base_pose,
        double main_insertion,
        const gtsam::Matrix4& helper_base_pose,
        double helper_insertion,
        const Vector6Gaussian& tip_wrench);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    std::unique_ptr<MultiRobotModel> robot_;
    
    gtsam::Pose3 main_base_pose_;
    double main_insertion_;
    gtsam::Pose3 helper_base_pose_;
    double helper_insertion_;
    Vector6Gaussian tip_wrench_;

    MultiRobotMarginals extracted_;
};