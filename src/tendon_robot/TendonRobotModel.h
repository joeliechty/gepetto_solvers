#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/Matrix.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"

#include <cmath>
#include <limits>
#include <optional>
#include <vector>


enum class RoutingAngleFunction {
    CONSTANT = 0,
    LINEAR = 1
};


struct RoutingFunctionParams {
    double angle_offset = 0.0;  // Starting angle (radians)
    double total_angle = 0.0;   // For LINEAR: total angle change across the rod
};


// Simple global routing specification (backward-compatible).
// Each tendon gets a single routing function applied across the entire rod.
struct TendonInput {
    std::vector<RoutingAngleFunction> functions;
    std::vector<RoutingFunctionParams> params;
    double routing_radius;
};


// Per-disc routing specification.
// hole_angles[disc_idx][tendon_idx] = angle in radians, or NaN for "no hole" (tendon terminated).
struct PerDiscTendonInput {
    int num_tendons = 0;
    double routing_radius = 0.0;
    std::vector<std::vector<double>> hole_angles;  // [num_discs][num_tendons]
    std::vector<std::vector<double>> hole_radii;   // [num_discs][num_tendons], optional (empty = use routing_radius)

    bool is_populated() const { return num_tendons > 0 && !hole_angles.empty(); }
};


struct TendonConfig {
    int num_discs;
    int num_tendons;
    double routing_radius;
    std::vector<int> disc_pose_idx;
    std::vector<int> no_disc_pose_idx;

    // hole_locations[disc_idx][tendon_idx] = Vector3 position in local frame.
    // nullopt means tendon has no hole at this disc (terminated).
    std::vector<std::vector<std::optional<gtsam::Vector3>>> hole_locations;
};


struct TendonRobotMarginals {
    CosseratRodMarginals rod;
    TendonConfig tendon_config;

    std::vector<Vector6Gaussian> external_wrenches;
    VectorXGaussian tensions;

    Eigen::MatrixXd J_pose_tensions;
};


template<int N>
class TendonRobotModel {
public:
    TendonRobotModel(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        TendonInput tendon_input,
        const gtsam::Matrix6& K_inv,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        gtsam::Pose3 base_pose_mean,
        gtsam::SharedDiagonal base_pose_noise,
        const std::vector<double>& disc_positions_normalized = {});

    // Per-segment compliance: K_inv_per_segment must have (num_discs + (num_discs-1)*num_between_nodes - 1) entries.
    TendonRobotModel(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        TendonInput tendon_input,
        const std::vector<gtsam::Matrix6>& K_inv_per_segment,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        gtsam::Pose3 base_pose_mean,
        gtsam::SharedDiagonal base_pose_noise,
        const std::vector<double>& disc_positions_normalized = {});

    // Per-disc tendon input constructors
    TendonRobotModel(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        PerDiscTendonInput per_disc_input,
        const gtsam::Matrix6& K_inv,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        gtsam::Pose3 base_pose_mean,
        gtsam::SharedDiagonal base_pose_noise,
        const std::vector<double>& disc_positions_normalized = {});

    TendonRobotModel(
        double rod_length,
        int num_discs,
        int num_between_nodes,
        PerDiscTendonInput per_disc_input,
        const std::vector<gtsam::Matrix6>& K_inv_per_segment,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        gtsam::Pose3 base_pose_mean,
        gtsam::SharedDiagonal base_pose_noise,
        const std::vector<double>& disc_positions_normalized = {});

    gtsam::Values get_initial_values() const;

    gtsam::NonlinearFactorGraph build_graph(const VectorNGaussian<N>& tensions) const;

    gtsam::Key get_external_wrench_key(int node_idx) const;

    gtsam::Key get_tensions_key() const;

    gtsam::Key get_disc_wrench_key(int disc_idx) const;

    inline int get_num_nodes() const { return num_nodes_; }

    TendonRobotMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    std::unique_ptr<CosseratRodModel> rod_;

private:
    void init_tendon_disc_config(TendonInput tendon_input, const std::vector<double>& disc_positions_normalized = {});
    void init_tendon_disc_config_per_disc(PerDiscTendonInput per_disc_input, const std::vector<double>& disc_positions_normalized = {});

    void compute_disc_positions_and_segments(const std::vector<double>& disc_positions_normalized);

    void get_J_pose_tensions(const gtsam::Marginals& marginals, TendonRobotMarginals& out) const;

    const double rod_length_;
    const int num_discs_;
    const int num_nodes_;

    std::vector<double> segment_lengths_;

    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    gtsam::Pose3 base_pose_mean_;
    gtsam::SharedDiagonal base_pose_noise_;

    TendonConfig tendon_config_;
};
