#pragma once

#include "utils/Gaussians.h"
#include "utils/SolverBase.h"
#include "TendonFingerModel.h"
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>

#include <memory>
#include <optional>
#include <variant>


// Single-state sphere-sphere contact, enforced as a hard equality constraint.
// SphereSphereContactFactor (utils/EnvironmentFactors.h, 1-residual gap form)
// is wrapped in a gtsam::ZeroCostConstraint so the solver's Augmented
// Lagrangian path drives the signed surface gap exactly to zero, pinning one
// rod node sphere of radius r_a tangent to a fixed-world sphere primitive of
// radius r_b. Convergence is governed by the AL parameters on
// SolverBaseConfig (al_initial_mu, al_mu_increase_rate, al_max_iterations).
struct SpherePrimitiveContactConfig {
    int    finger_node_index  = -1;   // -1 = tip alias (clamp_node_idx)
    double finger_node_radius = 0.0;  // r_a

    gtsam::Point3 sphere_center = gtsam::Point3::Zero();  // world frame
    double        sphere_radius = 0.0;                    // r_b

    // Tight prior on the sphere primitive's pose (rigid anchor).
    gtsam::Matrix6 sphere_pose_cov = 1e-8 * gtsam::Matrix6::Identity();
};


struct TendonFingerSolverConfig{
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

    // Optional sphere-sphere contact constraint on a chosen rod node. When
    // unset the solver runs the legacy free-space formulation.
    std::optional<SpherePrimitiveContactConfig> sphere_contact;
};


template<int N>
class TendonFingerSolver : SolverBase {
public:
    static constexpr int NumTendons = N;

    TendonFingerSolver(const TendonFingerSolverConfig& config);

    Solution<TendonFingerMarginals> solve(
        const VectorNGaussian<N>& tensions,
        const std::optional<Vector6Gaussian>& tip_wrench,
        const std::optional<Vector3Gaussian>& tip_position_meas);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    gtsam::SharedDiagonal small_wrench_noise_;

    std::unique_ptr<TendonFingerModel<N>> robot_;

    VectorNGaussian<N> tensions_;
    std::optional<Vector6Gaussian> tip_wrench_;
    std::optional<Vector3Gaussian> tip_position_meas_;

    std::optional<SpherePrimitiveContactConfig> sphere_contact_;
    static gtsam::Key sphere_object_key() { return gtsam::Symbol('O', 0); }

    TendonFingerMarginals extracted_;
};


// Runtime dispatch wrapper that selects the correct template specialization
// based on config.num_tendons. This is the class exposed to Python.
class TendonFingerSolverDispatch {
public:
    TendonFingerSolverDispatch(const TendonFingerSolverConfig& config);

    Solution<TendonFingerMarginals> solve(
        const VectorXGaussian& tensions,
        const std::optional<Vector6Gaussian>& tip_wrench,
        const std::optional<Vector3Gaussian>& tip_position_meas);

    int num_tendons() const { return num_tendons_; }

private:
    int num_tendons_;

    using SolverVariant = std::variant<
        std::unique_ptr<TendonFingerSolver<1>>,
        std::unique_ptr<TendonFingerSolver<2>>,
        std::unique_ptr<TendonFingerSolver<3>>,
        std::unique_ptr<TendonFingerSolver<4>>,
        std::unique_ptr<TendonFingerSolver<5>>,
        std::unique_ptr<TendonFingerSolver<6>>,
        std::unique_ptr<TendonFingerSolver<7>>,
        std::unique_ptr<TendonFingerSolver<8>>,
        std::unique_ptr<TendonFingerSolver<9>>,
        std::unique_ptr<TendonFingerSolver<10>>
    >;
    SolverVariant solver_;
};
