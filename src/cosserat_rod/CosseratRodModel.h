#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

#include <functional>
#include <optional>

#include "utils/Gaussians.h"


struct CosseratRodState {
    Pose3Gaussian pose;
    Vector6Gaussian stress;
    Vector6Gaussian wrench;
};


struct CosseratRodMarginals {
    std::vector<CosseratRodState> states;

    // We could add other things here later
};


class CosseratRodModel {
public:
    CosseratRodModel(
        int num_nodes, 
        const gtsam::Matrix6& K_inv, 
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        bool use_midpoint = true);

    // Per-segment stiffness: K_inv_per_segment must have exactly num_nodes - 1 entries.
    // Use a near-zero matrix (e.g. 1e-12 * I) for effectively rigid "bone" segments
    // and the normal K_inv for flexible "joint" segments.
    CosseratRodModel(
        int num_nodes,
        const std::vector<gtsam::Matrix6>& K_inv_per_segment,
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise,
        bool use_midpoint = true);

    gtsam::NonlinearFactorGraph build_graph(
        double rod_length,
        const std::optional<gtsam::Vector6>& nominal_strain = std::nullopt) const;

    gtsam::NonlinearFactorGraph build_graph(
        const std::vector<double>& ds,
        const std::optional<gtsam::Vector6>& nominal_strain = std::nullopt) const;

    gtsam::Values get_initial_values(
        double rod_length = 0, 
        const gtsam::Pose3& base_pose_init = gtsam::Pose3::Identity()) const;

    gtsam::Values get_initial_values(
        const std::vector<double>& ds,
        const gtsam::Pose3& base_pose_init = gtsam::Pose3::Identity()) const;

    CosseratRodMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    // Functor-based overload — accepts any source of marginal covariances
    // (e.g. ISAM2/IncrementalFixedLagSmoother's Bayes-tree query) so the
    // iterative solver can avoid rebuilding gtsam::Marginals from scratch.
    using CovFn = std::function<gtsam::Matrix(gtsam::Key)>;
    CosseratRodMarginals get_marginals(
        const gtsam::Values& values,
        const CovFn& cov_of) const;

    // Hand-base reparameterization (paper Section 4, Eq. 43-44). When enabled,
    // node 0 is no longer an independent variable: its pose is the deterministic
    // composition T_0 = T_base o offset, where T_base is a new variable keyed by
    // get_root_base_key(). build_graph then emits Root* variants of the factors
    // that touch node 0, and node 0's value is reconstructed (not optimized).
    // When shared_key is provided, that Key is used as the hand-base variable
    // instead of this rod's private Symbol('H', 1000*id_). This lets several rods
    // (e.g. the fingers of one hand) share a single floating wrist base variable,
    // each with its own offset. Default (nullopt) preserves the legacy per-rod key.
    void set_root_reparameterization(const gtsam::Pose3& offset,
                                     std::optional<gtsam::Key> shared_key = std::nullopt);

    bool uses_root() const { return use_root_; }

    // Planar-bending approximation (off by default). The physical finger's discs
    // are keyed to the backbone, so it bends about its local +y axis and reacts
    // out-of-plane bending and torsion structurally instead of deflecting. When
    // enabled, build_graph adds a PlanarBendFactor per segment penalising the
    // out-of-plane and torsional components of Log(R_i^T R_j)/ds; flexion is
    // untouched. Sigmas are curvatures in rad/m, directly comparable to the
    // twist_noise rotation sigma. See PlanarBendFactor.h.
    void set_planar_bending(double sigma_bend, double sigma_twist);

    bool uses_planar_bending() const { return planar_bend_noise_ != nullptr; }

    gtsam::Key get_root_base_key() const { return root_base_key_; }

    const gtsam::Pose3& get_root_offset() const { return root_offset_; }

    gtsam::Key get_pose_key(int node_idx) const;

    gtsam::Key get_stress_key(int node_idx) const;
    
    gtsam::Key get_wrench_key(int node_idx) const;
    
    const std::vector<gtsam::Key>& get_wrench_keys() const;

    const std::vector<gtsam::Key>& get_pose_keys() const;

private:
    int clamp_node_idx(int node_idx) const;
    
    // We need unique rod IDs for unique Keys
    const int id_;
    inline static int next_id_ = 0;

    const int num_nodes_;
    std::vector<gtsam::Matrix6> K_inv_;
    const bool use_midpoint_;

    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    std::vector<gtsam::Key> pose_keys_;
    std::vector<gtsam::Key> stress_keys_;
    std::vector<gtsam::Key> wrench_keys_;
    gtsam::Key dummy_wrench_key_;

    // Planar-bending noise model; null => the factor is not emitted at all.
    gtsam::SharedDiagonal planar_bend_noise_;

    // Hand-base reparameterization state (off by default; legacy node-0 path).
    bool use_root_ = false;
    gtsam::Key root_base_key_;
    gtsam::Pose3 root_offset_;
};
