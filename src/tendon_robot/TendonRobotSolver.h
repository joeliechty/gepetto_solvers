#pragma once

#include "utils/Gaussians.h"
#include "utils/SolverBase.h"
#include "TendonRobotModel.h"
#include <gtsam/linear/NoiseModel.h>

#include <memory>
#include <variant>


struct TendonRobotSolverConfig{
    SolverBaseConfig base;

    double rod_length;
    int num_discs;
    int num_between_nodes;
    int num_tendons = 4;  // Default 4 for backward compatibility
    gtsam::Matrix6 K_inv;

    // Optional: per-segment compliance matrices (one per rod segment).
    // If non-empty, overrides K_inv. Must have exactly
    // (num_discs + (num_discs-1)*num_between_nodes - 1) entries.
    // Set near-zero (e.g. 1e-12 * I) for rigid "bone" segments,
    // and the normal K_inv value for flexible "joint" segments.
    std::vector<gtsam::Matrix6> K_inv_per_segment;

    // Optional: custom disc positions along the rod (normalized 0 to 1).
    // If non-empty, must have exactly num_discs entries.
    // First entry should be 0.0, last entry should be 1.0.
    // If empty, discs are uniformly spaced.
    std::vector<double> disc_positions_normalized;

    double sigma_twist_rot;
    double sigma_twist_pos;
    double sigma_stress_force;
    double sigma_stress_moment;
    double sigma_base_pos;
    double sigma_base_rot;

    // Simple global routing (backward-compatible). Used when per_disc_tendon_input is not populated.
    TendonInput tendon_input;

    // Per-disc routing (overrides tendon_input when populated).
    PerDiscTendonInput per_disc_tendon_input;

    // Optional: custom base pose as a 4x4 SE(3) matrix.
    // If all zeros (default), uses legacy hardcoded Rx(-pi/2)*Rz(pi) at origin.
    gtsam::Matrix4 base_pose = gtsam::Matrix4::Zero();
};


template<int N>
class TendonRobotSolver : SolverBase {
public:
    static constexpr int NumTendons = N;

    TendonRobotSolver(const TendonRobotSolverConfig& config);

    Solution<TendonRobotMarginals> solve(
        const VectorNGaussian<N>& tensions,
        const std::optional<Vector6Gaussian>& tip_wrench,
        const std::optional<Vector3Gaussian>& tip_position_meas);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    gtsam::SharedDiagonal small_wrench_noise_;

    std::unique_ptr<TendonRobotModel<N>> robot_;

    VectorNGaussian<N> tensions_;
    std::optional<Vector6Gaussian> tip_wrench_;
    std::optional<Vector3Gaussian> tip_position_meas_;

    TendonRobotMarginals extracted_;
};


// Runtime dispatch wrapper that selects the correct template specialization
// based on config.num_tendons. This is the class exposed to Python.
class TendonRobotSolverDispatch {
public:
    TendonRobotSolverDispatch(const TendonRobotSolverConfig& config);

    Solution<TendonRobotMarginals> solve(
        const VectorXGaussian& tensions,
        const std::optional<Vector6Gaussian>& tip_wrench,
        const std::optional<Vector3Gaussian>& tip_position_meas);

    int num_tendons() const { return num_tendons_; }

private:
    int num_tendons_;

    using SolverVariant = std::variant<
        std::unique_ptr<TendonRobotSolver<1>>,
        std::unique_ptr<TendonRobotSolver<2>>,
        std::unique_ptr<TendonRobotSolver<3>>,
        std::unique_ptr<TendonRobotSolver<4>>,
        std::unique_ptr<TendonRobotSolver<5>>,
        std::unique_ptr<TendonRobotSolver<6>>,
        std::unique_ptr<TendonRobotSolver<7>>,
        std::unique_ptr<TendonRobotSolver<8>>,
        std::unique_ptr<TendonRobotSolver<9>>,
        std::unique_ptr<TendonRobotSolver<10>>
    >;
    SolverVariant solver_;
};
