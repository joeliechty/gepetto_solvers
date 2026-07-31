#pragma once

#include "tendon_hand/TendonHandModel.h"
#include "tendon_finger/TendonFingerSolver.h"   // TendonFingerSolverConfig
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>


// The phases of the Section 1.8 manipulation primitive. They are NOT time
// windows inside one graph (that is the §1.6 trajectory planner); they are
// different CONSTRAINT SETS over the same single-state graph, switched between as
// the task progresses.
enum class ControllerPhase {
    // Phase 0 (Eq 1.94-1.98): pre-grasp positioning. The only phase with NO
    // equality constraints: two soft Gaussian TARGETS -- a base pose T_base,pre
    // and a per-finger tension vector Q_pre -- multiply the step priors, giving a
    // first-order servo toward a collision-free hover posture above the object,
    // while every sphere avoids the table (Eq 1.97) and the object (Eq 1.98).
    // Exists so phase 1 starts from a well-conditioned posture instead of
    // whatever pose the caller happened to supply.
    PreGrasp = 0,
    // Phase 1 (Eq 1.96-1.100): free-space approach. Drive the designated contact
    // spheres onto the support surface (c_support = 0) inside their assigned
    // opposition half-spaces (c_half <= 0), while every other sphere avoids the
    // table and the object.
    SupportContact = 1,
    // Phase 2 (Eq 1.102-1.106): sliding approach. Keep the contact spheres on the
    // surface and additionally drive them onto the hyper-ellipsoid object proxy
    // (c_obj = 0, center-direct, no witness point).
    ObjectApproach = 2,
    // Phase 3 (Eq 1.112-1.125): on-object servoing. Relax the support equality
    // back to an inequality so the fingers may lift off, swap the ellipsoid proxy
    // for the object's true SDF geometry, and re-introduce the witness point with
    // the 4-residual [c_R, c_O, c_T1, c_T2] contact constraint.
    ObjectServo = 3
};


// What anchors a control tick to the measured robot state Theta_curr.
enum class StepAnchor {
    // Tension only (Eq 1.95). The pure-simulation default: there is no motor to
    // read a length from, and the tension prior alone determines the posture.
    Tension = 0,
    // Tendon length only (the Eq 1.13 analogue). The hardware-faithful mode: the
    // tendons are effectively inextensible, so length is what the motor actually
    // commands and what survives a tick, whereas a disturbance contact changes
    // tension instantly without the robot having moved. The tension prior relaxes
    // to the §1.1 background hold p_bg so tension stays free to respond.
    Length  = 1,
    // Both priors at once. Useful for diagnosing which anchor is fighting the
    // constraints; over-constrains a real tick, so not a control default.
    Both    = 2
};


struct TendonHandControllerConfig {
    SolverBaseConfig base;

    // Current shared wrist pose (world frame). This doubles as the Eq 1.94 step
    // prior mean: each tick it is re-aimed at the measured base pose.
    gtsam::Matrix4 wrist_pose = gtsam::Matrix4::Identity();

    // Eq 1.94 step prior Sigma_T,step -- the trust region on how far the hand base
    // may move in one control tick. Looser than TendonHandSolver's anchoring prior
    // (1e-4 / 1e-3): the controller WANTS the base to move, just not teleport.
    double sigma_wrist_pos = 1e-3;
    double sigma_wrist_rot = 1e-2;

    // Which constraint set is active. Change it through set_phase(), which also
    // re-seeds any variables the new phase introduces.
    ControllerPhase phase = ControllerPhase::SupportContact;

    // What anchors the tick to Theta_curr (see StepAnchor).
    StepAnchor step_anchor = StepAnchor::Tension;

    // --- Phase 0 targets (Eq 1.94/1.95) ----------------------------------
    // These are emitted ONLY in ControllerPhase::PreGrasp, and they ADD to the
    // step priors rather than replacing them: the two Gaussians on each variable
    // multiply, so the posterior mode is their precision-weighted mean. With the
    // step-prior mean re-aimed at the achieved state each tick, that turns into a
    // first-order servo toward the target.
    //
    // The servo RATE is the sigma RATIO, not either sigma alone. The fraction of
    // remaining error surviving one tick is
    //     rho = sigma_pre^2 / (sigma_pre^2 + sigma_step^2)
    // so sigma_pre = 3 * sigma_step gives rho = 0.9 (~10 %/tick, ~22 ticks to
    // 90 % of the way there) while sigma_pre = sigma_step gives rho = 0.5 -- a
    // jump big enough to break the warm start and stall the amortized AL loop.
    //
    // NOTE this competition is prior-vs-prior, pure linear algebra, so unlike a
    // constraint-driven phase it is NOT limited by the penalty weight mu the AL
    // loop can reach.
    std::optional<gtsam::Matrix4> pregrasp_wrist_pose;   // Eq 1.94 T_base,pre
    double sigma_pregrasp_pos = 3e-3;                    // 3x sigma_wrist_pos
    double sigma_pregrasp_rot = 3e-2;                    // 3x sigma_wrist_rot

    // Eq 1.95 target Q^gamma_pre, one per finger in config order (as step() and
    // add_length_priors take them). Empty => no tension target: a caller that
    // simply COMMANDS Q_pre through the step prior's mean already sits at the
    // target, and a second Gaussian there would add nothing. The prior earns its
    // keep on hardware, where the step prior's mean is a genuine measurement.
    std::vector<VectorXGaussian> pregrasp_tensions;

    // --- Theta_curr's ROBOT STATE (Eq 1.93) ------------------------------
    // The posture the first tick starts from. Unset => the cold-start straight
    // hand with Q = 0 that TendonFingerModel::get_initial_values builds, which
    // is almost never where the robot actually is: a caller that has posed the
    // hand (forward kinematics, or a measurement off the hardware) and commits
    // it as Theta_curr would otherwise see tick 1 spend itself travelling from a
    // straight hand back to the committed posture -- the fingers visibly
    // extending before they curl.
    //
    // wrist_pose above carries Theta_curr's BASE pose and the step priors carry
    // its tensions/lengths; this is the remaining piece, the rod state. Supply
    // the marginals of any solve on the same finger configs (see
    // TendonHandModel::values_from_marginals for why marginals rather than a
    // gtsam::Values).
    std::optional<TendonHandMarginals> initial_state;
};


// Real-time phased inverse-kinematics controller for the tendon hand
// (Section 1.8). Where TendonHandTrajectoryPlanner solves one K+1-step trajectory
// offline, this solves a SINGLE-state constrained IK problem per control tick,
// anchored to the measured robot state by the step priors of Eq 1.93-1.95, and is
// re-solved after each executed step.
//
// The unconstrained posterior of Eq 1.93 needs no new machinery: p_step(T_base) is
// the shared wrist PriorFactor with its mean re-aimed each tick, p_step(Q) is the
// per-finger tension prior with the measured tensions as its mean, and f_kin,
// p_ext, p_bg, p_lim already live in TendonFingerModel. The controller's real job
// is the PHASE-SCHEDULED CONSTRAINT SET plus warm-started re-solving.
//
// Warm starting: values_ is retained across ticks (as in TendonHandSolver), so
// each step() seeds from the previous solution. set_phase() preserves that -- it
// merges only the genuinely new variables from a fresh get_initial_values() and
// keeps the converged robot state, because a cold start on transition would throw
// away exactly the good initial guess §1.8 depends on.
//
// AL budget: the outer Augmented Lagrangian loop is amortized across ticks --
// values, mu AND the Lagrange multipliers -- but only because
// SolverBaseConfig::al_warm_start_duals is set. That is not a default: GTSAM's
// optimize() starts every solve at initialMuEq with zero duals, which caps a
// tick's penalty at al_initial_mu * al_mu_increase_rate^(al_max_iterations - 1)
// -- about 8 at the usual settings, against the ~8e3 an offline solve reaches --
// so the hard constraints are never actually enforced however many ticks run.
// With the duals carried, a small al_max_iterations (3-5) advances the
// mu-homotopy the §1.5 notes call essential while keeping a single tick fast.
//
// The multipliers are indexed by constraint POSITION, so anything that changes
// the constraint set must call reset_al_duals(); set_phase() and set_state() do.
class TendonHandController : SolverBase {
public:
    TendonHandController(
        const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
        const TendonHandControllerConfig& config);

    // Switch the active constraint set. Rebuilds the underlying model from the
    // pristine per-finger environments with the new phase's flags stamped on,
    // then merges in initial values for variables the new phase introduces (the
    // phase-3 witness points) while KEEPING the converged robot state.
    void set_phase(ControllerPhase phase);
    ControllerPhase phase() const { return phase_; }

    // Update Theta_curr's base pose (the Eq 1.94 step-prior mean) without a
    // rebuild. Only the prior's mean depends on it, so this is all build_graph()
    // needs to command a new wrist pose.
    void set_wrist_pose(const gtsam::Matrix4& wrist_pose) {
        config_.wrist_pose = wrist_pose;
        hand_->set_wrist_pose(gtsam::Pose3(wrist_pose));
    }

    // Re-aim the Eq 1.94/1.95 phase-0 targets without a rebuild. Only prior
    // means and covariances depend on them and build_graph() runs every tick, so
    // a caller may move the clearance height or the servo rate between ticks.
    // Symmetric with set_wrist_pose().
    void set_pregrasp_target(const std::optional<gtsam::Matrix4>& wrist_pose,
                             double sigma_pos, double sigma_rot,
                             const std::vector<VectorXGaussian>& tensions = {}) {
        config_.pregrasp_wrist_pose = wrist_pose;
        config_.sigma_pregrasp_pos  = sigma_pos;
        config_.sigma_pregrasp_rot  = sigma_rot;
        config_.pregrasp_tensions   = tensions;
    }

    // Re-seed the retained robot state from a solved posture mid-run -- "the
    // robot is over THERE now", for a teleport the step-prior trust region could
    // never absorb in one tick (a re-posed hand in a GUI, or a resync against the
    // hardware after an unmodelled disturbance).
    //
    // Distinct from set_wrist_pose(), which only re-aims the step prior's MEAN
    // and leaves the retained values alone. This replaces the values, so it also
    // drops the accumulated AL duals: they are a Lagrangian at the old state and
    // carry no information about the new one.
    void set_state(const TendonHandMarginals& state);

    // The shared base pose T_base at the CURRENT retained state.
    //
    // TendonHandMarginals carries per-finger state only, so without this a caller
    // has no way to read back the pose it just solved for -- and therefore no way
    // to close the Theta_curr loop the Eq 1.93 step prior is defined against. Left
    // open, the prior's mean stays at the construction-time pose forever and the
    // base is effectively pinned there. Symmetric with current_tendon_lengths().
    gtsam::Matrix4 current_wrist_pose() const;

    // One control tick.
    //   tensions      Eq 1.95 step prior: mean = measured Q_curr, cov = Sigma_Q,step.
    //   tip_wrenches  external tip wrench priors (zero mean in free space).
    //   lengths       Eq 1.13-analogue step prior: mean = measured L_curr,
    //                 cov = Sigma_L,step. Required when step_anchor is Length or
    //                 Both; ignored (may be empty) under Tension.
    Solution<TendonHandMarginals> step(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches,
        const std::vector<VectorXGaussian>& lengths = {});

    // Tendon lengths of every finger at the CURRENT retained state, computed
    // from the disc poses. Callers anchoring on length (StepAnchor::Length/Both)
    // need an L_curr for the very first tick, before any step() has run and --
    // in simulation -- with no motor to read. This supplies it directly from the
    // controller's own initial values, so the first tick is anchored like every
    // later one instead of needing a special bootstrap solve.
    std::vector<Eigen::VectorXd> current_tendon_lengths() const;

    // The per-finger witness point p_c,obj (Symbol('Y', i)) at the CURRENT
    // retained state, or nullopt for a finger that has none.
    //
    // Only phase 3 instantiates a witness -- phases 0-2 have no such variable at
    // all, and phase 2's contact is center-direct -- so an all-nullopt answer is
    // the normal report outside ObjectServo, not an error. Exposed because the
    // witness is a solved VARIABLE: an analytic surface projection computed from
    // the contact node is a look-alike, not the point the Eq 1.114-1.117
    // residuals are actually written against, so it cannot show witness drift.
    std::vector<std::optional<gtsam::Vector3>> current_witness_points() const;

    // Worst absolute violation per constraint family at the current solution,
    // as (family_name, max|c|) pairs. This is what a caller's phase-advance
    // policy reads: the controller deliberately does NOT advance itself, so the
    // policy can be iterated on in Python without a rebuild.
    std::vector<std::pair<std::string, double>> phase_violations() const;

    int num_fingers() const { return hand_->num_fingers(); }

    // Re-expose SolverBase diagnostics (privately inherited).
    std::vector<std::tuple<std::string, int, double>>
        get_factor_error_summary() const { return SolverBase::get_factor_error_summary(); }

    // Start the next tick's Augmented Lagrangian from a cold outer loop. Called
    // automatically by set_phase() and set_state(); exposed because a caller who
    // changes the constrained problem some other way (a moved object, a new
    // contact mask) has to invalidate the carried multipliers too.
    void reset_al_duals() { SolverBase::reset_al_duals(); }

private:
    void build_graph() override;
    void extract_solution() override;
    void get_initial_values() override;

    // Rebuild hand_ from base_envs_ with the current phase's flags applied.
    void rebuild_model();
    // Per-phase mutation of one finger's environment (the Section 1.8 schedule).
    crest_sparse::EnvironmentConfig phase_env(
        const crest_sparse::EnvironmentConfig& base, int finger_index) const;

    TendonHandControllerConfig config_;
    ControllerPhase phase_;

    // The finger configs exactly as constructed, plus the pristine per-finger
    // environments they carried. phase_env() derives each phase's env from these,
    // so switching phases never compounds edits from a previous phase.
    std::vector<std::pair<std::string, TendonFingerSolverConfig>> finger_configs_;
    std::vector<std::optional<crest_sparse::EnvironmentConfig>>   base_envs_;

    std::unique_ptr<TendonHandModel> hand_;

    std::vector<VectorXGaussian> tensions_;
    std::vector<Vector6Gaussian> tip_wrenches_;
    std::vector<VectorXGaussian> lengths_;

    TendonHandMarginals extracted_;
};
