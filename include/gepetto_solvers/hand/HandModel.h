#pragma once

#include "gepetto_solvers/hand/ConstraintTagger.h"
#include "gepetto_solvers/hand/HandKinematics.h"
#include "gepetto_solvers/hand/HandSpec.h"
#include "gepetto_solvers/hand/HandState.h"
#include "gepetto_solvers/utils/EnvironmentFactors.h"   // gepetto_solvers::EnvironmentConfig
#include "gepetto_solvers/utils/Gaussians.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>


// One posed hand: a HandKinematics that supplies the mechanism, plus the task
// constraints that pose it against the world.
//
// The split this class exists to hold:
//
//   * The KINEMATICS is loaded by name from the HandSpec (see
//     HandKinematicsRegistry) and contributes every factor internal to the
//     mechanism. This model never learns what kind of mechanism that is. It
//     addresses the hand only through HandKinematics::site_pose_key, so a hand
//     built from separate digits and one defined as a single whole mechanism
//     are the same thing here.
//
//   * The TASK CONSTRAINTS -- object contact, finger-object and finger-finger
//     collision, the support plane, the opposition half-space and the three
//     pre-grasp constraints -- are built by build_graph() from the per-digit
//     EnvironmentConfigs on the spec. They are the same constraints for any
//     hand.
//
// Together with the single shared wrist prior, that is the whole graph: the
// joint prior over the wrist pose and the digit bases (expressed as one Gaussian
// on the wrist plus each digit's deterministic offset, inside the kinematics)
// solved simultaneously against whichever task constraints are switched on.
//
// Contact: each digit may carry its own env / sphere_contact. All contacting
// digits touch one shared object (Symbol('O', step)); a digit using a witness
// contact factor also gets its own witness point (Symbol('Y', i)) -- see
// uses_center_direct_contact(), which decides whether a witness exists at all.
// Contact factors are wrapped as hard equality constraints (ZeroCostConstraint),
// so the owning solver routes the solve through the Augmented Lagrangian path.
class HandModel {
public:
    // step / emit_wrist_prior default to the single-shot behavior (one wrist
    // variable Symbol('W',0), always anchored). The trajectory planner passes a
    // per-step index so each timestep owns a distinct wrist variable, and
    // suppresses the prior on all but the start step (the GP chain + kinematics
    // pin the interior wrists).
    HandModel(const gepetto_solvers::HandSpec& spec,
              const gtsam::Pose3& wrist_pose,
              gtsam::SharedDiagonal wrist_noise,
              int step = 0,
              bool emit_wrist_prior = true);

    // Combined graph: the kinematics' own factors, the single shared wrist
    // prior, and the task constraints.
    //
    // `actuation` is one Gaussian per digit over that digit's actuation variable
    // (tendon tensions on the tendon hand); `tip_wrenches` the terminal external
    // wrench per digit. Both are passed straight through to the kinematics.
    gtsam::NonlinearFactorGraph build_graph(
        const std::vector<VectorXGaussian>& actuation,
        const std::vector<Vector6Gaussian>& tip_wrenches);

    // Initial values for the whole hand, with the shared wrist variable inserted
    // exactly once, plus the contact object pose and per-digit witness seeds.
    //
    // warm (optional): a previous solution to take poses from instead of the
    // cold start. This matters for the witness seeds, which are derived by
    // projecting the CONTACT SITE onto the object surface -- from a converged
    // pose that projection lands where the digit actually is, whereas from the
    // cold start it lands wherever a straight hand happened to point. Keys
    // absent from warm fall back to the cold-start value, so passing a partial
    // (or empty) Values is safe.
    gtsam::Values get_initial_values(const gtsam::Values* warm = nullptr) const;

    // Re-key a solved hand state onto THIS model's variables, producing a
    // partial Values suitable as the `warm` argument above.
    //
    // Why a HandState and not a gtsam::Values: models hand out keys from a
    // global counter, so two separately-constructed HandModels use different
    // Symbols for the same physical variable and a Values from one cannot be
    // merged into the other by key. HandState is the key-independent form.
    gtsam::Values values_from_state(const HandState& state) const;

    // Every hard constraint's identity, in build_graph()'s insertion order --
    // which is the order ConstrainedOptProblem enumerates them in. See
    // ConstraintTagger.
    const gepetto_solvers::ConstraintTags& constraint_tags() const {
        return tagger_.tags();
    }

    // Re-aim the shared wrist prior at a new pose *without* rebuilding the
    // model. Only the wrist-prior mean depends on wrist_pose_ (the digit offsets
    // and the root reparameterization factors are anchored to the shared wrist
    // key, independent of its target), so updating it here is all build_graph()
    // needs to command a new wrist pose. Intended for warm-started sweeps: keep
    // one solver instance, call set_wrist_pose() + solve() each step, and the
    // solve seeds from the previous solution instead of a cold start. Does NOT
    // re-seed get_initial_values() (which still reflects the construction-time
    // wrist), so the first solve after construction is the only cold one.
    void set_wrist_pose(const gtsam::Pose3& wrist_pose) { wrist_pose_ = wrist_pose; }

    HandState get_state(const gtsam::Values& values,
                        const gtsam::Marginals& marginals) const {
        return kin_->extract(values, &marginals);
    }

    // Means-only state (zero covariance / zero Jacobians): the same per-digit
    // state, skipping the expensive gtsam::Marginals factorization, so it is
    // cheap enough to call once per solver-iteration snapshot. Used for debug
    // visualization of the optimizer's intermediate states.
    HandState get_state_means_only(const gtsam::Values& values) const {
        return kin_->extract(values, nullptr);
    }

    int num_digits() const { return kin_->num_digits(); }

    // True if any digit has a contact constraint configured (=> AL path).
    bool has_contact() const { return has_contact_; }

    // True if any digit has collision avoidance configured (=> AL path). A
    // collision-only hand (no contact) must still route through the Augmented
    // Lagrangian optimizer for the inequality constraints to take effect.
    bool has_collision() const { return has_collision_; }

    // Wrist variable key. step defaults to 0 (the single-shot key); the
    // trajectory planner uses per-step keys.
    static gtsam::Key wrist_key(int step = 0) { return gtsam::Symbol('W', step); }
    // Object pose variable for THIS model's step. Per-step (Symbol('O', step_))
    // so a trajectory -- where collision avoidance anchors the object at EVERY
    // step -- gets one object variable per step instead of duplicate insertions
    // of a single shared key. The object is static (same tight prior each step),
    // so per-step variables are equivalent to one shared variable.
    gtsam::Key object_key() const          { return gtsam::Symbol('O', step_); }
    static gtsam::Key witness_key(int i)   { return gtsam::Symbol('Y', i); }

    // Whether this digit's OBJECT contact is expressed as the witness-free
    // center-direct equality (Eq 1.101) -- c_obj(c) = Taubin(T_obj^-1 c) - r = 0,
    // i.e. EllipsoidCollisionGapFactor's residual as an equality -- rather than a
    // witness contact factor. Analytic-ellipsoid contact takes the center-direct
    // form BY DEFAULT: it drops the witness variable and goes from 5 residual
    // rows to 1 per digit, and nothing in the 5-row layout is load-bearing for a
    // sphere-on-ellipsoid contact (see the definition for the exceptions).
    //
    // The single source of truth for that choice: build_graph() picks the factor
    // with it and get_initial_values() decides whether to seed a witness point
    // with it, and a disagreement between those two leaves either an orphan
    // variable (indeterminate system) or an unseeded key.
    static bool uses_center_direct_contact(const gepetto_solvers::EnvironmentConfig& env);

    // NOTE there is deliberately no table witness key: the support-plane contact
    // equality constrains the contact site's sphere CENTER directly (one
    // residual, no free point), so the table introduces no variable of its own.

    // This model's own wrist key (step-indexed). Used by the trajectory planner
    // to wire the wrist GP BetweenFactor between consecutive timesteps.
    gtsam::Key wrist_key_instance() const { return wrist_key(step_); }

    // The kinematics this model was built with, for the trajectory planner's GP
    // chain and for a caller that needs a variable key by digit.
    const gepetto_solvers::HandKinematics& kinematics() const { return *kin_; }

    gtsam::Key digit_actuation_key(int i) const { return kin_->actuation_key(i); }
    std::optional<gtsam::Key> digit_displacement_key(int i) const {
        return kin_->displacement_key(i);
    }

    // Tip (last node) pose key for digit i. Used by the trajectory planner and
    // the single-shot solver to add a terminal PositionPriorFactor when a
    // per-digit position goal is set.
    gtsam::Key digit_tip_pose_key(int i) const {
        return kin_->site_pose_key({i, -1});
    }

    // Pose key of an arbitrary site on digit i (negative node indices count from
    // the tip). The Section 1.8 controller uses this to re-evaluate its
    // constraint factors on the solved values when reporting per-family
    // violations, without duplicating the geometry math.
    gtsam::Key digit_node_pose_key(int i, int node) const {
        return kin_->site_pose_key({i, node});
    }

    // Add the temporal GP priors (Eq 1.11 actuation, Eq 1.13 displacement)
    // linking this model's per-digit variables to the next timestep's. The
    // displacement GP is added only when gp_displacement_Qc is non-empty.
    void add_temporal_gp(gtsam::NonlinearFactorGraph& graph,
                         const HandModel& next,
                         const Eigen::MatrixXd& gp_actuation_Qc,
                         const Eigen::MatrixXd& gp_displacement_Qc,
                         double dt) const {
        kin_->add_temporal_gp(graph, *next.kin_, gp_actuation_Qc,
                              gp_displacement_Qc, dt);
    }

    // Direct priors on each digit's displacement variable (Section 1.8; the
    // trajectory-free analogue of the Eq 1.13 GP). The controller anchors a
    // control tick to the MEASURED state, and on the physical robot that state
    // is the motor position -- tendon length -- not tension: the tendons are
    // effectively inextensible so length is what the motor commands and what
    // survives a tick, whereas a disturbance contact changes tension instantly
    // without the robot having moved.
    void add_displacement_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& displacement) const {
        kin_->add_displacement_priors(graph, displacement);
    }

    // ADDITIONAL priors on each digit's actuation variable, on top of the one
    // build_graph() already emits from its `actuation` argument. This is the
    // Section 1.8 phase-0 target Q_pre of Eq 1.95. Because the two Gaussians on
    // the same variable multiply, the result is their precision-weighted mean --
    // a pull toward Q_pre bounded by the step prior, not a replacement for it.
    void add_actuation_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& actuation) const {
        kin_->add_actuation_priors(graph, actuation);
    }

private:
    // The mechanism. Loaded by name from the spec; never downcast.
    //
    // A future HandDynamics would sit beside this as a second contributor, and
    // build_graph() would call the two in a fixed order before the task
    // constraints -- appending to the same ConstraintTagger, so the constraint
    // numbering the AL multiplier transfer depends on stays one enumeration.
    std::unique_ptr<gepetto_solvers::HandKinematics> kin_;

    gepetto_solvers::ConstraintTagger tagger_;

    std::vector<std::string> digit_names_;

    // The digit that opposes the others in the pre-grasp constraints, or -1.
    int opposing_digit_ = -1;

    // Per-digit task environment (either may be empty).
    std::vector<std::optional<gepetto_solvers::EnvironmentConfig>> env_;
    std::vector<std::optional<SpherePrimitiveContactConfig>>       sphere_contacts_;

    gtsam::Pose3          wrist_pose_;
    gtsam::SharedDiagonal wrist_noise_;

    // Timestep index (0 for the single-shot solve). Selects this model's wrist
    // key Symbol('W', step_) so per-step wrists don't collide in a trajectory.
    int  step_ = 0;
    // Whether build_graph() emits the wrist PriorFactor. True for the single-shot
    // solve and for the trajectory start step; false for interior/terminal steps.
    bool emit_wrist_prior_ = true;

    bool has_contact_ = false;
    bool has_collision_ = false;
};
