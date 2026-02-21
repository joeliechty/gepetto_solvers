#pragma once

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"
#include <gtsam/linear/NoiseModel.h>


struct MultiRobotMarginals {
    CosseratRodMarginals main_rod;
    CosseratRodMarginals helper_rod;
    CosseratRodMarginals end_effector_rod;

    // gtsam::Matrix6 tip_jacobian;
};


class MultiRobotModel {
public:
    MultiRobotModel(
        int nodes_per_rod, 
        gtsam::Matrix6 K_inv,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise, 
        double snare_distance_to_tip,
        gtsam::SharedDiagonal snare_constraint_noise);

    gtsam::NonlinearFactorGraph build_graph(
        const Pose3Gaussian& main_base_pose,
        double main_insertion,
        const Pose3Gaussian& helper_base_pose,
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
    const gtsam::SharedDiagonal small_wrench_noise_;
    const gtsam::SharedDiagonal snare_constraint_noise_;
    const double snare_distance_to_tip_;
};
