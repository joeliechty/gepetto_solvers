#pragma once

#include "gepetto_solvers/utils/Gaussians.h"
#include "gepetto_solvers/utils/SolverBase.h"
#include "gepetto_solvers/utils/EnvironmentFactors.h"
#include "gepetto_solvers/tendon_finger/TendonFingerModel.h"
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

    // When true, use the 5-residual SphereWitnessContactFactor
    // ([c_R, c_O, c_N, c_T1, c_T2]) with an explicit dummy witness point
    // instead of the 1-residual analytic gap form.
    // This is the analytic counterpart of the SDF witness-point contact and
    // exists mainly to cross-check the witness formulation against the closed
    // form on a pure sphere-sphere problem.
    bool witness = false;
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

    // Radius (m) of the fingertip contact sphere, from the digit's CAD tip
    // width. Carried on the config so it travels with the finger; used to
    // populate the contact node radius (SpherePrimitiveContactConfig::
    // finger_node_radius / EnvironmentConfig::contact_node_radius). 0 => unset.
    double tip_radius = 0.0;

    double sigma_twist_rot;
    double sigma_twist_pos;
    double sigma_stress_force;
    double sigma_stress_moment;
    double sigma_base_pos;
    double sigma_base_rot;

    // Planar-bending approximation. The Cosserat rod bends in any direction and
    // twists freely; the physical finger does not, because its discs are keyed to
    // the backbone. When true, every rod segment gets a PlanarBendFactor
    // penalising the out-of-plane and torsional components of Log(R_i^T R_j)/ds
    // while leaving flexion (rotation about local +y) free.
    //
    // The sigmas are curvatures in rad/m, so they are directly comparable to
    // sigma_twist_rot: at sigma_planar_bend == sigma_twist_rot this is roughly
    // what an anisotropic K_inv already buys (see get_K_inv's
    // lateral_stiffness_scale), and tightening below it is the point of having a
    // separate factor.
    //
    // The defaults are deliberately ASYMMETRIC -- soft bend, tight twist -- and
    // that is the whole trick. Measured over four grasp scenes: TWIST is the
    // cause and out-of-plane bend is the symptom. The spiral-routed lateral
    // tendons inject torsion, torsion rotates the material frame, and the next
    // segment's flexion then lands out of plane. Constrain torsion at the source
    // and the out-of-plane bend collapses with it (13x on big_sphere) while the
    // rod keeps the freedom it needs to reach -- it curls further instead of
    // splaying. Constraining the BEND row hard instead fights the accumulated
    // result rather than the cause: it buys the same planarity but costs ~10 mm
    // of reach, and on the power-drill scene it over-constrains badly enough
    // that the AL stalls at 7 outer iterations. The soft bend row is kept only
    // so a DIRECT lateral load still meets resistance.
    bool   planar_bending      = false;
    double sigma_planar_bend   = 1e-2;
    double sigma_planar_twist  = 1e-4;

    // Simple global routing (backward-compatible). Used when per_disc_tendon_input is not populated.
    TendonInput tendon_input;

    // Per-disc routing (overrides tendon_input when populated).
    PerDiscTendonInput per_disc_tendon_input;

    // Optional: custom base pose as a 4x4 SE(3) matrix.
    // If all zeros (default), uses legacy hardcoded Rx(-pi/2)*Rz(pi) at origin.
    gtsam::Matrix4 base_pose = gtsam::Matrix4::Zero();

    // Hand-base reparameterization (paper Section 4). When true, the finger's
    // node-0 pose is no longer an independent variable: it is the deterministic
    // SE(3) composition T_0 = T_base o T_offset of a new hand-base variable and
    // the fixed offset below (Eq. 43). The rigidity is embedded in the graph
    // topology via "Root" factors instead of a soft base prior, eliminating the
    // soft-rigid null space. When false (default) the legacy node-0 pose prior
    // path is used unchanged. With hand_base_offset = Identity the geometry is
    // identical to the legacy path.
    bool use_hand_base = false;
    gtsam::Matrix4 hand_base_offset = gtsam::Matrix4::Identity();

    // Optional sphere-sphere contact constraint on a chosen rod node. When
    // unset the solver runs the legacy free-space formulation.
    std::optional<SpherePrimitiveContactConfig> sphere_contact;

    // Optional SDF surface contact on a chosen rod node, using the 5-residual
    // witness-point SdfWitnessContactFactor (Section 3, [c_R, c_O, c_N, c_T1, c_T2]) wrapped as a
    // hard AL equality constraint. Reuses EnvironmentConfig as the carrier:
    // sdf_grid, object_pose_mean/cov, target_contact_node, contact_node_radius.
    // Mutually exclusive with sphere_contact (only one contact mode at a time).
    std::optional<gepetto_solvers::EnvironmentConfig> sdf_contact;
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

    // SolverBase is privately inherited, so re-expose its public diagnostics.
    std::vector<std::tuple<std::string, int, double>>
        get_factor_error_summary() const {
        return SolverBase::get_factor_error_summary();
    }
    std::vector<std::pair<std::string, std::vector<double>>>
        get_factor_errors_by_type() const {
        return SolverBase::get_factor_errors_by_type();
    }
    std::vector<std::tuple<std::string, int, double>>
        get_initial_factor_error_summary() const {
        return SolverBase::get_initial_factor_error_summary();
    }
    std::pair<Eigen::MatrixXd, Eigen::VectorXd>
        get_hessian_and_gradient() const {
        return SolverBase::get_hessian_and_gradient();
    }

    // Returns lightweight intermediate solutions (means only, zero covariances)
    // from Values snapshots stored during the last solve's iterate() loop.
    // Only populated when config.base.record_iterations == true and
    // config.base.iteration_sample_interval > 0.
    std::vector<Solution<TendonFingerMarginals>> get_intermediate_solutions() const;

    // Means-only solution at the initial guess (zero covariances). Always
    // available after solve() since initial_values_ is captured unconditionally.
    Solution<TendonFingerMarginals> get_initial_solution() const;

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
    std::optional<gepetto_solvers::EnvironmentConfig> sdf_contact_;
    static gtsam::Key sphere_object_key() { return gtsam::Symbol('O', 0); }
    static gtsam::Key dummy_point_key()   { return gtsam::Symbol('Y', 0); }

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

    std::vector<std::tuple<std::string, int, double>> get_factor_error_summary() const;
    std::vector<std::pair<std::string, std::vector<double>>>
        get_factor_errors_by_type() const;
    std::vector<std::tuple<std::string, int, double>>
        get_initial_factor_error_summary() const;
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> get_hessian_and_gradient() const;

    std::vector<Solution<TendonFingerMarginals>> get_intermediate_solutions() const;
    Solution<TendonFingerMarginals> get_initial_solution() const;

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
