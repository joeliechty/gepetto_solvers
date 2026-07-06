#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/base/Matrix.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/Gaussians.h"

#include <cmath>
#include <functional>
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


struct TendonFingerMarginals {
    CosseratRodMarginals rod;
    TendonConfig tendon_config;

    std::vector<Vector6Gaussian> external_wrenches;
    VectorXGaussian tensions;

    Eigen::MatrixXd J_pose_tensions;

    std::vector<double> tendon_lengths;
};


template<int N>
class TendonFingerModel {
public:
    static constexpr int NumTendons = N;

    TendonFingerModel(
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
    TendonFingerModel(
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
    TendonFingerModel(
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

    TendonFingerModel(
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

    // Enable the hand-base reparameterization (paper Section 4). Must be called
    // before get_initial_values()/build_graph(). Replaces the node-0 pose
    // variable with a hand-base variable T_base such that T_0 = T_base o offset;
    // the node-0 base prior moves onto T_base and the factors touching node 0
    // use their Root/offset-aware variants. Off by default (legacy path).
    // shared_key (optional) makes this finger reference a common hand-base
    // variable instead of its own; used by TendonHandModel so several fingers
    // share one floating wrist base, each with its own offset.
    void set_hand_base(const gtsam::Pose3& offset,
                       std::optional<gtsam::Key> shared_key = std::nullopt);

    // When false, build_graph()/build_graph_kinematic() omit this finger's own
    // base-pose prior. Used when an owner (e.g. TendonHandModel) anchors the
    // shared hand base itself with a single prior. Default true (legacy path).
    void set_emit_base_prior(bool emit) { emit_base_prior_ = emit; }

    gtsam::Values get_initial_values() const;

    gtsam::NonlinearFactorGraph build_graph(const VectorNGaussian<N>& tensions) const;

    // Build kinematic graph without tension prior (for use in trajectory planning)
    gtsam::NonlinearFactorGraph build_graph_kinematic() const;

    gtsam::Key get_external_wrench_key(int node_idx) const;

    gtsam::Key get_tensions_key() const;

    gtsam::Key get_lengths_key() const;

    gtsam::Key get_disc_wrench_key(int disc_idx) const;

    Eigen::Vector<double, N> compute_tendon_lengths(const gtsam::Values& values) const;

    inline int get_num_nodes() const { return num_nodes_; }

    inline int get_num_between_nodes() const { return num_between_nodes_; }

    const TendonConfig& get_tendon_config() const { return tendon_config_; }

    TendonFingerMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    // Functor-based overload — accepts callables that return marginal/joint
    // covariances directly. The iterative solver passes lambdas that read
    // from the IncrementalFixedLagSmoother's Bayes tree, avoiding the cost
    // of building a fresh gtsam::Marginals on the entire historical graph.
    using CovFn   = std::function<gtsam::Matrix(gtsam::Key)>;
    using JointFn = std::function<gtsam::Matrix(gtsam::Key, gtsam::Key)>;
    TendonFingerMarginals get_marginals(
        const gtsam::Values& values,
        const CovFn& cov_of,
        const JointFn& joint_of) const;

    std::unique_ptr<CosseratRodModel> rod_;

protected:
    void init_tendon_disc_config(TendonInput tendon_input, const std::vector<double>& disc_positions_normalized = {});
    void init_tendon_disc_config_per_disc(PerDiscTendonInput per_disc_input, const std::vector<double>& disc_positions_normalized = {});

    void compute_disc_positions_and_segments(const std::vector<double>& disc_positions_normalized);

    void get_J_pose_tensions(const gtsam::Marginals& marginals, TendonFingerMarginals& out) const;
    void get_J_pose_tensions(const JointFn& joint_of, TendonFingerMarginals& out) const;

    // Unique ID for key generation to avoid collisions when multiple TendonFingerModels are in the same graph
    const int id_;
    inline static int next_id_ = 0;

    const double rod_length_;
    const int num_discs_;
    const int num_nodes_;
    const int num_between_nodes_;

    std::vector<double> segment_lengths_;

    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    gtsam::Pose3 base_pose_mean_;
    gtsam::SharedDiagonal base_pose_noise_;

    // Hand-base reparameterization (off by default; legacy node-0 prior path).
    bool use_hand_base_ = false;
    gtsam::Pose3 hand_base_offset_;

    // Whether build_graph() emits this finger's own base-pose prior. Set false
    // when an owner (TendonHandModel) anchors the shared hand base itself.
    bool emit_base_prior_ = true;

    double sigma_length_ = 1e-4;

    TendonConfig tendon_config_;
};
