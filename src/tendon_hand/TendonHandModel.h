#pragma once

#include "tendon_finger/TendonFingerModel.h"
#include "tendon_finger/TendonFingerSolver.h"   // TendonFingerSolverConfig, SpherePrimitiveContactConfig
#include "utils/EnvironmentFactors.h"            // crest_sparse::EnvironmentConfig
#include "utils/Gaussians.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>


// Per-finger marginals collected into one hand solution.
struct TendonHandMarginals {
    std::vector<TendonFingerMarginals> fingers;
    std::vector<std::string> finger_names;
};


// A hand is a set of tendon fingers that all share ONE floating wrist base
// variable (Symbol('W', 0)). Each finger i attaches to the wrist through its own
// fixed offset T_offset_i (config.hand_base_offset), so its node-0 pose is
// T_0^i = T_wrist o T_offset_i. The wrist is anchored by a single prior owned by
// this model; the per-finger base priors are suppressed. This reuses the existing
// TendonFingerModel<N> and its hand-base reparameterization entirely; the only new
// wiring is the shared wrist key + the single wrist prior.
//
// Contact: each finger may carry its own sdf_contact / sphere_contact (from its
// TendonFingerSolverConfig). All contacting fingers touch one shared object
// (Symbol('O', 0)); each gets its own witness point (Symbol('Y', i)). The
// contact factors are wrapped as hard equality constraints (ZeroCostConstraint),
// so the owning solver routes the solve through the Augmented Lagrangian path.
class TendonHandModel {
public:
    TendonHandModel(
        const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
        const gtsam::Pose3& wrist_pose,
        gtsam::SharedDiagonal wrist_noise);

    // Combined graph: each finger's rod+tendon factors, interior/tip wrench
    // priors, the single shared wrist prior, and per-finger contact constraints.
    gtsam::NonlinearFactorGraph build_graph(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches);

    // Initial values for all fingers, with the shared wrist variable inserted
    // exactly once, plus the contact object pose and per-finger witness seeds.
    gtsam::Values get_initial_values() const;

    TendonHandMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    int num_fingers() const { return static_cast<int>(fingers_.size()); }

    // True if any finger has a contact constraint configured (=> AL path).
    bool has_contact() const { return has_contact_; }

    static gtsam::Key wrist_key()          { return gtsam::Symbol('W', 0); }
    static gtsam::Key object_key()         { return gtsam::Symbol('O', 0); }
    static gtsam::Key witness_key(int i)   { return gtsam::Symbol('Y', i); }

private:
    // We use a variant to handle different numbers of tendons per finger.
    using FingerVariant = std::variant<
        std::unique_ptr<TendonFingerModel<1>>,
        std::unique_ptr<TendonFingerModel<2>>,
        std::unique_ptr<TendonFingerModel<3>>,
        std::unique_ptr<TendonFingerModel<4>>,
        std::unique_ptr<TendonFingerModel<5>>,
        std::unique_ptr<TendonFingerModel<6>>,
        std::unique_ptr<TendonFingerModel<7>>,
        std::unique_ptr<TendonFingerModel<8>>,
        std::unique_ptr<TendonFingerModel<9>>,
        std::unique_ptr<TendonFingerModel<10>>
    >;

    std::vector<FingerVariant> fingers_;
    std::vector<std::string> finger_names_;

    // Per-finger contact configuration (either may be empty).
    std::vector<std::optional<crest_sparse::EnvironmentConfig>>   sdf_contacts_;
    std::vector<std::optional<SpherePrimitiveContactConfig>>      sphere_contacts_;

    gtsam::Pose3          wrist_pose_;
    gtsam::SharedDiagonal wrist_noise_;

    // Per-finger interior/tip external-wrench prior noise, derived from each
    // finger's sigma_stress_moment/force (as TendonFingerSolver does). Using the
    // finger's own tight stress noise here is important for conditioning — a loose
    // hand-wide value leaves the wrench variables weakly pinned and makes the
    // contact system indeterminate.
    std::vector<gtsam::SharedDiagonal> small_wrench_noises_;

    bool has_contact_ = false;
};
