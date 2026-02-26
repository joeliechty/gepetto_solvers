#pragma once

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include <gtsam/linear/NoiseModel.h>


struct MultiRobotMarginals {
    CosseratRodMarginals main_rod;
    CosseratRodMarginals helper_rod;
    CosseratRodMarginals end_effector_rod;

    Eigen::Matrix<double, 6, 12> J_rod_bases;
};


class MultiRobotModel {
public:
    MultiRobotModel(
        int nodes_per_rod, 
        gtsam::Matrix6 K_inv,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise, 
        gtsam::SharedDiagonal snare_constraint_noise,
        gtsam::SharedDiagonal base_pose_noise,
        double snare_distance_to_tip);

    gtsam::NonlinearFactorGraph build_graph(
        const gtsam::Pose3& main_base_pose,
        double main_insertion,
        const gtsam::Pose3& helper_base_pose,
        double helper_insertion,
        const Vector6Gaussian& tip_wrench);

    gtsam::Values get_initial_values() const;

    MultiRobotMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;
    
    std::unique_ptr<CosseratRodModel> main_rod_;
    std::unique_ptr<CosseratRodModel> helper_rod_;
    std::unique_ptr<CosseratRodModel> end_effector_rod_;

private:
    void get_rod_bases_jacobian(
        const gtsam::Marginals& marginals, 
        Eigen::Matrix<double, 6, 12>& J_rod_bases) const;

    const gtsam::SharedDiagonal small_wrench_noise_;
    const gtsam::SharedDiagonal snare_constraint_noise_;
    const gtsam::SharedDiagonal twist_noise_;
    const gtsam::SharedDiagonal base_pose_noise_;
    
    const double snare_distance_to_tip_;
};
