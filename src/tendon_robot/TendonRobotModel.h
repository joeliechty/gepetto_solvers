#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/Matrix.h>

#include "cosserat_rod/CosseratRodModel.h"


constexpr int NUM_TENDONS = 4;


enum class RoutingAngleFunction {
    CONSTANT = 0,
    LINEAR = 1
};


struct RoutingFunctionParams {
    double angle_offset = 0.0;  // Starting angle (radians)
    double total_angle = 0.0;   // For LINEAR: total angle change across the rod
};


struct TendonRoutingInput {
    std::array<RoutingAngleFunction, NUM_TENDONS> functions;
    std::array<RoutingFunctionParams, NUM_TENDONS> params;
    double routing_radius;
};


struct TendonDiscConfig {
    int num_discs;
    double routing_radius;
    std::vector<int> disc_pose_idx;
    std::vector<int> no_disc_pose_idx;
    std::vector<std::array<gtsam::Vector3, NUM_TENDONS>> hole_locations;
};


struct TendonRobotSamples {
    std::vector<gtsam::Matrix4> tip_pose_samples;
    std::vector<std::vector<gtsam::Vector3>> fbg_array_samples;
};


struct TendonRobotMarginals {
    CosseratRodMarginals rod;
    TendonRobotSamples samples;
    TendonDiscConfig tendon_disc_config;

    std::vector<gtsam::Vector6> external_wrench_mean;
    std::vector<gtsam::Matrix6> external_wrench_cov;

    Eigen::Vector<double, NUM_TENDONS> tensions_mean;
    Eigen::Matrix<double, NUM_TENDONS, NUM_TENDONS> tensions_cov;

    Eigen::Matrix<double, 6, NUM_TENDONS> J_pose_tensions;
};


class TendonRobotModel {
public:
    TendonRobotModel(
        int num_discs,
        int num_between_nodes,
        TendonRoutingInput routing_info,
        const gtsam::Matrix6& K_inv, 
        gtsam::SharedDiagonal tensions_noise,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise);
        
    gtsam::Values get_initial_values();

    gtsam::NonlinearFactorGraph build_graph(const gtsam::Vector4& tensions);

    TendonRobotMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

private:
    void init_tendon_disc_config(TendonRoutingInput routing_info);

    const int num_discs_;
    const int num_nodes_;
    
    gtsam::SharedDiagonal tensions_noise_;
    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    std::unique_ptr<CosseratRodModel> rod_;
    TendonDiscConfig tendon_disc_config_;
};