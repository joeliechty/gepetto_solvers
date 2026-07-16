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
    // step / emit_wrist_prior default to the single-shot behavior (one wrist
    // variable Symbol('W',0), always anchored). The trajectory planner passes a
    // per-step index so each timestep owns a distinct wrist variable, and
    // suppresses the prior on all but the start step (the GP chain + finger
    // kinematics pin the interior wrists).
    TendonHandModel(
        const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
        const gtsam::Pose3& wrist_pose,
        gtsam::SharedDiagonal wrist_noise,
        int step = 0,
        bool emit_wrist_prior = true);

    // Combined graph: each finger's rod+tendon factors, interior/tip wrench
    // priors, the single shared wrist prior, and per-finger contact constraints.
    gtsam::NonlinearFactorGraph build_graph(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches);

    // Initial values for all fingers, with the shared wrist variable inserted
    // exactly once, plus the contact object pose and per-finger witness seeds.
    gtsam::Values get_initial_values() const;

    // Re-aim the shared wrist prior at a new pose *without* rebuilding the model.
    // Only the wrist-prior mean depends on wrist_pose_ (the finger offsets and
    // the root reparameterization factors are anchored to the shared wrist key,
    // independent of its target), so updating it here is all that build_graph()
    // needs to command a new wrist pose. Intended for warm-started sweeps: keep
    // one solver instance, call set_wrist_pose() + solve() each step, and the
    // solve seeds from the previous solution instead of a straight-hand cold
    // start. Does NOT re-seed get_initial_values() (which still reflects the
    // construction-time wrist), so the first solve after construction is the
    // only cold one.
    void set_wrist_pose(const gtsam::Pose3& wrist_pose) { wrist_pose_ = wrist_pose; }

    TendonHandMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    // Means-only marginals (zero covariance / zero Jacobians). Extracts the same
    // per-finger state as get_marginals() but skips the expensive gtsam::Marginals
    // factorization, so it is cheap enough to call once per solver-iteration
    // snapshot. Used for debug visualization of the optimizer's intermediate
    // states (get_intermediate_solutions()); mirrors TendonFingerModel's
    // functor overload, building the zero functors per finger so each gets its
    // own (6+N)-sized joint block.
    TendonHandMarginals get_marginals_means_only(const gtsam::Values& values) const;

    int num_fingers() const { return static_cast<int>(fingers_.size()); }

    // True if any finger has a contact constraint configured (=> AL path).
    bool has_contact() const { return has_contact_; }

    // True if any finger has collision avoidance configured (=> AL path). A
    // collision-only hand (no contact) must still route through the Augmented
    // Lagrangian optimizer for the inequality constraints to take effect.
    bool has_collision() const { return has_collision_; }

    // Wrist variable key. step defaults to 0 (the single-shot key), so existing
    // callers are unchanged; the trajectory planner uses per-step keys.
    static gtsam::Key wrist_key(int step = 0) { return gtsam::Symbol('W', step); }
    // Object pose variable for THIS model's step. Per-step (Symbol('O', step_))
    // so a trajectory — where collision avoidance anchors the object at EVERY
    // step — gets one object variable per step instead of duplicate insertions
    // of a single shared key. The object is static (same tight prior each step),
    // so per-step variables are equivalent to one shared variable. For the
    // single-shot solve (step_ = 0) this is Symbol('O', 0), unchanged.
    gtsam::Key object_key() const          { return gtsam::Symbol('O', step_); }
    static gtsam::Key witness_key(int i)   { return gtsam::Symbol('Y', i); }

    // This model's own wrist key (step-indexed). Used by the trajectory planner
    // to wire the wrist GP BetweenFactor between consecutive timesteps.
    gtsam::Key wrist_key_instance() const { return wrist_key(step_); }

    // N-agnostic per-finger control keys (visit the finger variant). The
    // trajectory planner uses these to add the tension/length GP BetweenFactors.
    gtsam::Key finger_tension_key(int i) const;
    gtsam::Key finger_length_key(int i) const;

    // Tip (last rod node) pose key for finger i. Used by the trajectory planner
    // to add a terminal PositionPriorFactor when a per-finger position goal is set.
    gtsam::Key finger_tip_pose_key(int i) const;

    // Add the temporal GP priors (Eq 1.11 tensions, Eq 1.13 lengths) linking this
    // model's per-finger tension/length variables to the corresponding variables
    // in the next timestep's model. The per-finger tendon count N is resolved
    // inside the variant visit (where the BetweenFactor<Vector<N>> dimension is
    // known). The length GP is added only when gp_len_Qc is non-empty.
    void add_temporal_gp(
        gtsam::NonlinearFactorGraph& graph,
        const TendonHandModel& next,
        const Eigen::MatrixXd& gp_tense_Qc,
        const Eigen::MatrixXd& gp_len_Qc,
        double dt) const;

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

    // Timestep index (0 for the single-shot solve). Selects this model's wrist
    // key Symbol('W', step_) so per-step wrists don't collide in a trajectory.
    int  step_ = 0;
    // Whether build_graph() emits the wrist PriorFactor. True for the single-shot
    // solve and for the trajectory start step; false for interior/terminal steps.
    bool emit_wrist_prior_ = true;

    // Per-finger interior/tip external-wrench prior noise, derived from each
    // finger's sigma_stress_moment/force (as TendonFingerSolver does). Using the
    // finger's own tight stress noise here is important for conditioning — a loose
    // hand-wide value leaves the wrench variables weakly pinned and makes the
    // contact system indeterminate.
    std::vector<gtsam::SharedDiagonal> small_wrench_noises_;

    bool has_contact_ = false;
    bool has_collision_ = false;
};
