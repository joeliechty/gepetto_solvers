#include "TendonHandController.h"

#include "utils/EnvironmentFactors.h"
#include "utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace gtsam;
using crest_sparse::EnvironmentConfig;


TendonHandController::TendonHandController(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    const TendonHandControllerConfig& config)
:
    SolverBase(config.base),
    config_(config),
    phase_(config.phase),
    finger_configs_(finger_configs)
{
    // Stash the environments exactly as handed in. Every phase derives its env
    // from these, so switching phases never compounds a previous phase's edits.
    base_envs_.reserve(finger_configs_.size());
    for (const auto& [name, c] : finger_configs_)
        base_envs_.push_back(c.sdf_contact);

    rebuild_model();
    get_initial_values();
}


EnvironmentConfig TendonHandController::phase_env(
    const EnvironmentConfig& base, int /*finger_index*/) const
{
    EnvironmentConfig env = base;

    // The designated contact sphere for this finger, carried on the pristine env
    // as target_contact_node / contact_node_radius. A finger whose contact was
    // toggled off has none: it stays purely in D_free (collision spheres and
    // plane avoidance only) in every phase, which is exactly what we want.
    const auto contact_node   = base.target_contact_node;
    const double contact_r    = base.contact_node_radius;
    const bool   has_contact  = contact_node.has_value();

    // Start from a clean slate of §1.8 flags, then set the ones this phase needs.
    env.support_contact_node.reset();
    env.support_contact_radius        = 0.0;
    env.half_space_enabled            = false;
    env.target_contact_node.reset();
    env.object_contact_center_direct  = false;
    env.contact_drop_normal_row       = false;
    env.witness_target.reset();
    // The §1.6 five-residual sliding witness is never used by the controller;
    // §1.8 replaces it with the center-direct support equality.
    env.table_contact_node.reset();
    env.plane_avoidance               = true;

    switch (phase_) {
    case ControllerPhase::PreGrasp:
        // Eq 1.96-1.98 have NO equality constraints, which is exactly the clean
        // slate above -- so this case is deliberately empty, not unfinished:
        //   - support_contact_node unset + plane_avoidance => the table
        //     inequality covers every non-root sphere, including the contact
        //     tips (Eq 1.97's "for all j in D").
        //   - target_contact_node unset => build_graph()'s is_contact is false
        //     for every sphere, so all of them keep the object inequality
        //     (Eq 1.98's "for all j in D") and no witness point or contact
        //     equality is built at all.
        // What makes this a phase rather than a no-op is the pair of soft
        // targets build_graph() adds (Eq 1.94/1.95).
        break;

    case ControllerPhase::SupportContact:
        // Eq 1.97-1.100. Contact spheres onto the table, inside their opposition
        // half-spaces. No object contact yet, so target_contact_node stays unset
        // -- which also means the contact tips are NOT excluded from the object
        // collision inequality, matching Eq 1.100's D_contact u D_free.
        if (has_contact) {
            env.support_contact_node   = contact_node;
            env.support_contact_radius = contact_r;
            env.half_space_enabled     = base.half_space_enabled;
        }
        break;

    case ControllerPhase::ObjectApproach:
        // Eq 1.103-1.106. Hold the table equality and add the object equality on
        // the sphere center (Eq 1.101, no witness). The half-space has done its
        // job placing the fingers and would now fight the slide, so it is off.
        // target_contact_node set => contact tips drop out of the object
        // collision inequality, matching Eq 1.106's D_free-only.
        if (has_contact) {
            env.support_contact_node          = contact_node;
            env.support_contact_radius        = contact_r;
            env.target_contact_node           = contact_node;
            env.object_contact_center_direct  = true;
        }
        break;

    case ControllerPhase::ObjectServo:
        // Eq 1.113-1.125. The support equality relaxes back to an inequality --
        // achieved simply by leaving support_contact_node unset, which lets the
        // plane-avoidance pass cover every sphere including the tips (Eq 1.117)
        // -- and the ellipsoid proxy is swapped for the object's true SDF with
        // the 4-residual witness contact.
        if (has_contact) {
            env.target_contact_node     = contact_node;
            env.contact_drop_normal_row = true;
            env.witness_target          = base.witness_target;
            // Swap the proxy for the exact geometry. Only when a baked grid is
            // actually available: an analytic-only object (no .vdb) has no exact
            // geometry to switch to, and zeroing the semi-axes would leave the
            // finger with no object surface at all. In that case the ellipsoid
            // stays as the phase-3 surface and only the witness form changes.
            if (base.sdf_grid) env.ellipsoid_semi_axes = gtsam::Vector3::Zero();
        }
        break;
    }

    return env;
}


void TendonHandController::rebuild_model() {
    auto configs = finger_configs_;
    for (size_t i = 0; i < configs.size(); ++i) {
        if (base_envs_[i])
            configs[i].second.sdf_contact = phase_env(*base_envs_[i], static_cast<int>(i));
    }

    SharedDiagonal wrist_noise = get_noise_model_rot_pos(
        config_.sigma_wrist_rot, config_.sigma_wrist_pos);

    hand_ = std::make_unique<TendonHandModel>(
        configs, Pose3(config_.wrist_pose), wrist_noise);

    // Every phase carries hard constraints (phase 1's support equality, phases
    // 2-3's object equality, plus collision inequalities throughout), so the
    // solve always routes through SolverBase's Augmented Lagrangian path.
    use_augmented_lagrangian_ = hand_->has_contact() || hand_->has_collision();

    // Phase 0's ONLY constraints are the Eq 1.97/1.98 inequalities: it has no
    // equality to fall back on, so if the caller attached neither the support
    // plane nor object collision avoidance there is nothing stopping the soft
    // targets from servoing the hand straight through the table. Fail loudly
    // rather than silently solving a free-space problem.
    if (phase_ == ControllerPhase::PreGrasp && !use_augmented_lagrangian_)
        throw std::runtime_error(
            "TendonHandController: the PreGrasp phase is inequality-only "
            "(Eq 1.97/1.98), but no collision constraints are configured. Set "
            "plane_avoidance with a non-zero plane_normal and/or "
            "collision_avoidance with collision_node_indices on the per-finger "
            "environments; otherwise the pre-grasp servo is unconstrained and "
            "will drive the hand through the support surface.");
}


void TendonHandController::set_phase(ControllerPhase phase) {
    if (phase == phase_) return;
    phase_ = phase;

    // Keep the converged robot state and merge in only what the new phase adds
    // (the phase-3 witness points). A cold start on transition would discard
    // exactly the good initial guess the phased formulation depends on.
    const Values previous = values_;
    rebuild_model();

    // Seeded from the previous solution, which matters most at the phase 2 -> 3
    // transition: that swaps the ellipsoid proxy for the exact SDF, moving the
    // constraint manifold, and the new witness points are projected from the
    // CONVERGED contact nodes rather than from a cold-start straight hand.
    //
    // Keyed off the new phase's variable set, so anything it no longer has a
    // factor for (going back a phase drops the object witness) is left behind
    // rather than carried into the linear system as an orphan.
    const Values fresh = hand_->get_initial_values(&previous);
    Values next;
    for (Key k : fresh.keys())
        next.insert(k, previous.exists(k) ? previous.at(k) : fresh.at(k));
    values_ = next;
}


void TendonHandController::get_initial_values() {
    values_ = hand_->get_initial_values();
}


void TendonHandController::build_graph() {
    graph_ = hand_->build_graph(tensions_, tip_wrenches_);

    // Eq 1.94 / Eq 1.95 phase-0 targets. These do NOT replace the step priors
    // already in the graph above -- they multiply them, so the posterior mode is
    // the precision-weighted mean of Theta_curr and Theta_pre, which is what
    // makes the phase a servo rather than a teleport.
    if (phase_ == ControllerPhase::PreGrasp) {
        if (config_.pregrasp_wrist_pose)
            graph_.add(PriorFactor<Pose3>(
                TendonHandModel::wrist_key(0),
                Pose3(*config_.pregrasp_wrist_pose),
                get_noise_model_rot_pos(config_.sigma_pregrasp_rot,
                                        config_.sigma_pregrasp_pos)));
        if (!config_.pregrasp_tensions.empty())
            hand_->add_tension_priors(graph_, config_.pregrasp_tensions);
    }

    // Eq 1.13 / Eq 1.95 length step prior. p_step(T_base) and p_step(Q) are
    // already in the graph above -- the wrist prior and the per-finger tension
    // prior, with their means re-aimed at the measured state each tick -- so this
    // is the only step prior that needs adding.
    const bool want_lengths = config_.step_anchor == StepAnchor::Length ||
                              config_.step_anchor == StepAnchor::Both;
    if (want_lengths) {
        if (lengths_.empty())
            throw std::runtime_error(
                "TendonHandController: step_anchor is Length/Both but step() was "
                "called with no measured tendon lengths.");
        hand_->add_length_priors(graph_, lengths_);
    }
}


void TendonHandController::extract_solution() {
    // skip_marginals leaves marginals_ empty, so take the means-only path. A
    // control tick only ever consumes the means, and skipping the factorization
    // is the single biggest per-tick saving available.
    if (config_.base.skip_marginals)
        extracted_ = hand_->get_marginals_means_only(values_);
    else
        extracted_ = hand_->get_marginals(values_, marginals_);
}


Solution<TendonHandMarginals> TendonHandController::step(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches,
    const std::vector<VectorXGaussian>& lengths)
{
    tensions_     = tensions;
    tip_wrenches_ = tip_wrenches;
    lengths_      = lengths;

    Solution<TendonHandMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;
    return solution;
}


gtsam::Matrix4 TendonHandController::current_wrist_pose() const {
    const Key k = TendonHandModel::wrist_key(0);
    if (!values_.exists(k))
        throw std::runtime_error(
            "TendonHandController::current_wrist_pose: no base pose in the "
            "retained values");
    return values_.at<Pose3>(k).matrix();
}


std::vector<Eigen::VectorXd> TendonHandController::current_tendon_lengths() const {
    // get_marginals_means_only fills tendon_lengths from the disc poses without
    // any Marginals factorization, so this is cheap enough to call per tick.
    const TendonHandMarginals m = hand_->get_marginals_means_only(values_);
    std::vector<Eigen::VectorXd> out;
    out.reserve(m.fingers.size());
    for (const auto& f : m.fingers) {
        Eigen::VectorXd v(f.tendon_lengths.size());
        for (size_t t = 0; t < f.tendon_lengths.size(); ++t)
            v(static_cast<Eigen::Index>(t)) = f.tendon_lengths[t];
        out.push_back(std::move(v));
    }
    return out;
}


std::vector<std::optional<gtsam::Vector3>>
TendonHandController::current_witness_points() const
{
    std::vector<std::optional<gtsam::Vector3>> out;
    out.reserve(static_cast<size_t>(num_fingers()));
    for (int i = 0; i < num_fingers(); ++i) {
        const Key wk = TendonHandModel::witness_key(i);
        if (values_.exists(wk))
            out.emplace_back(values_.at<gtsam::Point3>(wk));
        else
            out.emplace_back(std::nullopt);
    }
    return out;
}


std::vector<std::pair<std::string, double>>
TendonHandController::phase_violations() const
{
    // Re-evaluate the phase's constraint factors on the solved values. We build
    // throwaway factor instances rather than re-deriving the geometry here, so
    // this report can never drift from what the graph actually enforced.
    //
    // Scope: the EQUALITY/goal families that drive phase advancement, all of
    // which act on the single designated contact sphere per finger. Collision
    // penetration is a safety property over every sphere and is reported
    // independently by the Python demos' all-pairs clearance check.
    std::vector<std::pair<std::string, double>> out;

    // Phase 0 has no constraints to violate -- what a caller's advance policy
    // wants there is distance-to-target instead. Reported as SEPARATE families
    // because their units differ from each other and from every family below
    // (metres, radians, newtons); collapsing the pose pair into one
    // Pose3::Logmap norm would mix metres and radians into an unthresholdable
    // scalar.
    if (phase_ == ControllerPhase::PreGrasp) {
        if (config_.pregrasp_wrist_pose) {
            const Key wk = TendonHandModel::wrist_key(0);
            if (values_.exists(wk)) {
                const Pose3 T_pre(*config_.pregrasp_wrist_pose);
                const Pose3 T = values_.at<Pose3>(wk);
                out.emplace_back("pregrasp_pos",
                    (T.translation() - T_pre.translation()).norm());
                out.emplace_back("pregrasp_rot",
                    Rot3::Logmap(T_pre.rotation().between(T.rotation())).norm());
            }
        }
        if (!config_.pregrasp_tensions.empty()) {
            // extracted_ is refreshed by extract_solution() every tick and
            // carries the per-finger tension means, so no variant visit is
            // needed to read them back here.
            double dq = 0.0;
            const size_t n = std::min(extracted_.fingers.size(),
                                      config_.pregrasp_tensions.size());
            for (size_t i = 0; i < n; ++i)
                dq = std::max(dq, (extracted_.fingers[i].tensions.mean -
                                   config_.pregrasp_tensions[i].mean)
                                      .cwiseAbs().maxCoeff());
            out.emplace_back("pregrasp_tension", dq);
        }
    }

    double sup = 0.0, half = 0.0, obj = 0.0;
    bool have_sup = false, have_half = false, have_obj = false;

    auto unit = noiseModel::Isotropic::Sigma(1, 1.0);

    for (size_t i = 0; i < base_envs_.size(); ++i) {
        if (!base_envs_[i]) continue;
        const EnvironmentConfig env = phase_env(*base_envs_[i], static_cast<int>(i));

        if (env.support_contact_node.has_value()) {
            Key k = hand_->finger_node_pose_key(
                static_cast<int>(i), *env.support_contact_node);
            if (values_.exists(k)) {
                crest_sparse::PlaneCollisionGapFactor f(
                    k, env.support_contact_radius,
                    env.plane_origin, env.plane_normal, unit);
                sup = std::max(sup, std::abs(f.unwhitenedError(values_)(0)));
                have_sup = true;

                if (env.half_space_enabled && env.half_space_normal.norm() > 0.0) {
                    crest_sparse::HalfSpaceGapFactor h(
                        k, env.half_space_split_point, env.half_space_normal, unit);
                    // Inequality: only a positive value is a violation.
                    half = std::max(half, std::max(0.0, h.unwhitenedError(values_)(0)));
                    have_half = true;
                }
            }
        }

        if (!env.target_contact_node.has_value()) continue;
        Key tip_key = hand_->finger_node_pose_key(
            static_cast<int>(i), *env.target_contact_node);
        Key obj_key = hand_->object_key();
        if (!values_.exists(tip_key) || !values_.exists(obj_key)) continue;

        if (env.object_contact_center_direct &&
            env.ellipsoid_semi_axes.norm() > 0.0) {
            crest_sparse::EllipsoidCollisionGapFactor f(
                tip_key, obj_key, env.contact_node_radius,
                env.ellipsoid_semi_axes, unit);
            obj = std::max(obj, std::abs(f.unwhitenedError(values_)(0)));
            have_obj = true;
        } else {
            Key wk = TendonHandModel::witness_key(static_cast<int>(i));
            if (!values_.exists(wk)) continue;
            const bool drop_n = env.contact_drop_normal_row;
            const int  n_rows = drop_n ? 4 : 5;
            auto rows = noiseModel::Isotropic::Sigma(n_rows, 1.0);
            Vector e;
            if (env.ellipsoid_semi_axes.norm() > 0.0) {
                crest_sparse::EllipsoidWitnessContactFactor f(
                    tip_key, obj_key, wk, env.contact_node_radius,
                    env.ellipsoid_semi_axes, rows, drop_n);
                e = f.unwhitenedError(values_);
            } else if (env.sdf_grid) {
                crest_sparse::SdfWitnessContactFactor f(
                    tip_key, obj_key, wk, env.contact_node_radius,
                    env.sdf_grid, rows, drop_n);
                e = f.unwhitenedError(values_);
            } else {
                continue;
            }
            obj = std::max(obj, e.cwiseAbs().maxCoeff());
            have_obj = true;
        }
    }

    if (have_sup)  out.emplace_back("support_equality", sup);
    if (have_half) out.emplace_back("half_space", half);
    if (have_obj)  out.emplace_back("object_contact", obj);
    return out;
}
