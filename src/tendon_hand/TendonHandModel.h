#pragma once

#include "tendon_finger/TendonFingerModel.h"
#include "tendon_finger/TendonFingerSolver.h"   // TendonFingerSolverConfig, SpherePrimitiveContactConfig
#include "utils/EnvironmentFactors.h"            // gepetto_solvers::EnvironmentConfig
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
// (Symbol('O', 0)); a finger using a witness contact factor also gets its own
// witness point (Symbol('Y', i)) — see uses_center_direct_contact(), which is
// what decides whether a witness exists at all. The contact factors are wrapped
// as hard equality constraints (ZeroCostConstraint), so the owning solver routes
// the solve through the Augmented Lagrangian path.
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
    //
    // warm (optional): a previous solution to take robot poses from instead of
    // the cold-start straight hand. This matters for the witness seeds, which are
    // derived by projecting the CONTACT NODE onto the object surface -- from a
    // converged pose that projection lands where the finger actually is, whereas
    // from the cold start it lands wherever a straight hand happened to point.
    // The Section 1.8 controller passes its retained values here when switching
    // phases. Keys absent from warm fall back to the cold-start value, so passing
    // a partial (or empty) Values is safe.
    gtsam::Values get_initial_values(const gtsam::Values* warm = nullptr) const;

    // Re-key a solved hand state onto THIS model's variables, producing a
    // partial Values suitable as the `warm` argument above.
    //
    // Why marginals and not a gtsam::Values: CosseratRodModel hands out keys from
    // a GLOBAL counter (inline static next_id_), so two separately-constructed
    // TendonHandModels use different Symbols for the same physical variable and a
    // Values from one cannot be merged into the other by key. TendonHandMarginals
    // is the key-independent form -- everything in it is indexed by finger, node
    // and disc -- and every solve already produces one (including the cheap
    // get_marginals_means_only path), so a solver's output re-seeds another
    // solver directly.
    //
    // Covers T/S/F per node, D per disc, Q and L, plus the shared wrist: node 0's
    // pose is not a variable under the hand-base reparameterization (T_0 = T_base
    // o offset), so it is skipped as a pose and inverted into T_base instead. The
    // wrist has to come from the STATE, not from wrist_pose_: it is a variable
    // with a soft prior, so a contact solve moves it tens of millimetres off the
    // commanded pose, and seeding the commanded one would pair a converged finger
    // posture with a base that never went there. Throws if the finger count, node
    // count or tendon count disagrees with this model.
    gtsam::Values values_from_marginals(const TendonHandMarginals& state) const;

    // Stable, semantic identity for every hard constraint this model builds, in
    // build_graph()'s insertion order -- which is exactly the order
    // ConstrainedOptProblem enumerates them in, since it collects constraints by
    // a filtered walk of the graph.
    //
    // The point is transferring Lagrange multipliers across a REBUILD. lambda is
    // indexed by a constraint's POSITION, so adding one constraint renumbers
    // every multiplier after it; a tag like "tbl.contact|f4" names the same
    // physical constraint no matter what else is in the graph. Tags are emitted
    // at the insertion site rather than recovered by introspection because the
    // graph cannot be read back: every equality is wrapped in ZeroCostConstraint
    // and every inequality in CollisionInequalityConstraint, so the factor type
    // says nothing, and e.g. a plane collision and a half-space inequality on the
    // same node carry identical keys.
    struct ConstraintTags {
        std::vector<std::string> eq;
        std::vector<std::string> ineq;
    };
    const ConstraintTags& constraint_tags() const { return tags_; }

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

    // Whether this finger's OBJECT contact is expressed as the witness-free
    // center-direct equality (Eq 1.101) — c_obj(c) = Taubin(T_obj^-1 c) - r = 0,
    // i.e. EllipsoidCollisionGapFactor's residual as an equality — rather than a
    // witness contact factor. Analytic-ellipsoid contact takes the center-direct
    // form BY DEFAULT: it drops the witness variable and goes from 5 residual
    // rows to 1 per finger, and nothing in the 5-row layout is load-bearing for
    // a sphere-on-ellipsoid contact (see the definition for the exceptions).
    //
    // The single source of truth for that choice: build_graph() picks the factor
    // with it and get_initial_values() decides whether to seed a witness point
    // with it, and a disagreement between those two leaves either an orphan
    // variable (indeterminate system) or an unseeded key.
    static bool uses_center_direct_contact(const gepetto_solvers::EnvironmentConfig& env);

    // NOTE there is deliberately no table witness key: the support-plane contact
    // equality constrains the contact node's sphere CENTER directly (one residual,
    // no free point), so the table introduces no variable of its own at any step.

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

    // Pose key of an arbitrary rod node on finger i (negative indices count from
    // the tip, as CosseratRodModel::get_pose_key does). The Section 1.8 controller
    // uses this to re-evaluate its constraint factors on the solved values when
    // reporting per-family violations, without duplicating the geometry math.
    gtsam::Key finger_node_pose_key(int i, int node) const;

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

    // Direct priors on each finger's tendon-length variable (Section 1.8; the
    // trajectory-free analogue of the Eq 1.13 length GP). The §1.8 controller
    // anchors a control tick to the MEASURED state, and on the physical robot
    // that state is the motor position — i.e. tendon length — not tension: the
    // tendons are effectively inextensible so length is what the motor commands
    // and what survives a tick, whereas a disturbance contact changes tension
    // instantly without the robot having moved.
    //
    //   p_step(L^gamma | L^gamma_curr) ~ exp(-1/2 ||L - L_curr||^2_Sigma_L,step)
    //
    // One VectorXGaussian per finger, in config order; the per-finger tendon
    // count N is resolved inside the variant visit (as add_temporal_gp does),
    // which is why this cannot be a free function on the caller's side.
    void add_length_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& lengths) const;

    // ADDITIONAL priors on each finger's tendon-tension variable, on top of the
    // one build_graph() already emits from its `tensions` argument. This is the
    // Section 1.8 phase-0 target Q^gamma_pre of Eq 1.95:
    //
    //   p_pre_tension(Q^gamma) ~ exp(-1/2 ||Q - Q_pre||^2_Sigma_pre,Q)
    //
    // Because the two Gaussians on the same variable multiply, the result is
    // their precision-weighted mean -- a pull toward Q_pre bounded by the step
    // prior, not a replacement for it. Structurally identical to
    // add_length_priors: one VectorXGaussian per finger in config order, with
    // the per-finger tendon count N resolved inside the variant visit (which is
    // why this cannot be a free function on the caller's side).
    void add_tension_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& tensions) const;

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

    // Add one hard constraint AND record its identity, so the two orders cannot
    // drift: every constraint in this model goes through one of these.
    void add_eq(gtsam::NonlinearFactorGraph& graph,
                const gtsam::NoiseModelFactor::shared_ptr& factor,
                std::string tag);
    void add_ineq(gtsam::NonlinearFactorGraph& graph,
                  const gtsam::NoiseModelFactor::shared_ptr& gap,
                  std::string tag);

    ConstraintTags tags_;

    // Each finger's fixed attachment to the wrist (config.hand_base_offset), so
    // T_0^i = T_wrist o T_offset_i. Kept because that relation is the ONLY way
    // back from a state bundle to the wrist: node 0's pose is not a variable
    // under the reparameterization, so values_from_marginals has to invert it.
    std::vector<gtsam::Pose3> hand_base_offsets_;

    // Per-finger contact configuration (either may be empty).
    std::vector<std::optional<gepetto_solvers::EnvironmentConfig>>   sdf_contacts_;
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
