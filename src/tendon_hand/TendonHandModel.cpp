#include "TendonHandModel.h"

#include "utils/MiscInline.h"

#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <openvdb/tools/Interpolation.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>

using namespace gtsam;


namespace {

// Construct a TendonFingerModel<N> from a TendonFingerSolverConfig, choosing the
// per-disc vs. simple routing path and the K_inv vs. per-segment path exactly as
// TendonFingerSolver does. base_pose_mean is supplied by the caller (= wrist o offset).
template<int N>
std::unique_ptr<TendonFingerModel<N>> make_finger_impl(
    const TendonFingerSolverConfig& c,
    const Pose3& base_pose_mean,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    SharedDiagonal base_pose_noise)
{
    if (c.per_disc_tendon_input.is_populated()) {
        if (c.K_inv_per_segment.empty()) {
            return std::make_unique<TendonFingerModel<N>>(
                c.rod_length, c.num_discs, c.num_between_nodes,
                c.per_disc_tendon_input, c.K_inv, twist_noise, stress_noise,
                base_pose_mean, base_pose_noise, c.disc_positions_normalized);
        }
        return std::make_unique<TendonFingerModel<N>>(
            c.rod_length, c.num_discs, c.num_between_nodes,
            c.per_disc_tendon_input, c.K_inv_per_segment, twist_noise, stress_noise,
            base_pose_mean, base_pose_noise, c.disc_positions_normalized);
    }
    if (c.K_inv_per_segment.empty()) {
        return std::make_unique<TendonFingerModel<N>>(
            c.rod_length, c.num_discs, c.num_between_nodes,
            c.tendon_input, c.K_inv, twist_noise, stress_noise,
            base_pose_mean, base_pose_noise, c.disc_positions_normalized);
    }
    return std::make_unique<TendonFingerModel<N>>(
        c.rod_length, c.num_discs, c.num_between_nodes,
        c.tendon_input, c.K_inv_per_segment, twist_noise, stress_noise,
        base_pose_mean, base_pose_noise, c.disc_positions_normalized);
}

}  // namespace


TendonHandModel::TendonHandModel(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    const Pose3& wrist_pose,
    SharedDiagonal wrist_noise,
    int step,
    bool emit_wrist_prior)
:
    wrist_pose_(wrist_pose),
    wrist_noise_(wrist_noise),
    step_(step),
    emit_wrist_prior_(emit_wrist_prior)
{
    const Key shared_wrist_key = wrist_key(step_);

    fingers_.reserve(finger_configs.size());
    finger_names_.reserve(finger_configs.size());
    sdf_contacts_.reserve(finger_configs.size());
    sphere_contacts_.reserve(finger_configs.size());
    small_wrench_noises_.reserve(finger_configs.size());

    for (const auto& [name, c] : finger_configs) {
        finger_names_.push_back(name);
        sdf_contacts_.push_back(c.sdf_contact);
        sphere_contacts_.push_back(c.sphere_contact);
        // Contact => a terminal witness contact factor (target_contact_node for
        // the SDF path, or any sphere_contact). A collision-only env (collision
        // spheres but no target_contact_node) is NOT contact — it routes through
        // has_collision() instead, so guard on target_contact_node here.
        if ((c.sdf_contact.has_value() && c.sdf_contact->target_contact_node.has_value())
            || c.sphere_contact.has_value())
            has_contact_ = true;
        // Collision => AL inequality constraints on this finger's spheres.
        // The object surface may be a baked SDF grid, an analytic ellipsoid
        // (Section 1.6.3) or an ellipsoid SET (Section 1.2); each provides the
        // finger-object penetration gap.
        if (c.sdf_contact.has_value() &&
            c.sdf_contact->collision_avoidance &&
            crest_sparse::has_object_surface(*c.sdf_contact) &&
            !c.sdf_contact->collision_node_indices.empty())
            has_collision_ = true;
        // Finger-finger, on the same spheres and gated on its own field. Each
        // clause here must match the corresponding gate in build_graph(): a
        // constraint family that is built but does not flip this flag is solved
        // WITHOUT the Augmented Lagrangian, i.e. built and never enforced.
        if (c.sdf_contact.has_value() &&
            c.sdf_contact->self_collision &&
            !c.sdf_contact->collision_node_indices.empty())
            has_collision_ = true;
        // Opposition half-space (Eq 2.16-2.17): an inequality => AL. Standalone,
        // matching build_graph()'s own pass -- it needs neither the support
        // plane nor a contact node of any kind, so a solve whose ONLY constraint
        // is this one still takes the AL path.
        if (c.sdf_contact.has_value() &&
            c.sdf_contact->half_space_enabled &&
            c.sdf_contact->half_space_normal.norm() > 0.0 &&
            (c.sdf_contact->half_space_node.has_value() ||
             c.sdf_contact->table_contact_node.has_value()))
            has_collision_ = true;
        // Support plane (Section 1.6). A configured plane (non-zero normal) with
        // a table_contact_node is a hard equality (sliding contact => AL); with
        // plane_avoidance it is a hard inequality (table collision => AL).
        if (c.sdf_contact.has_value() &&
            c.sdf_contact->plane_normal.norm() > 0.0) {
            if (c.sdf_contact->table_contact_node.has_value()) has_contact_   = true;
            if (c.sdf_contact->plane_avoidance)                has_collision_ = true;
            // Section 1.8 controller phases 1-2: the support-surface equality is
            // a ZeroCostConstraint => AL. Phase 1 has no OBJECT contact at all,
            // so without this the solve would miss the AL path entirely and the
            // support constraint would silently do nothing.
            if (c.sdf_contact->support_contact_node.has_value())
                has_contact_ = true;
        }

        SharedDiagonal twist_noise = get_noise_model_rot_pos(
            c.sigma_twist_rot, c.sigma_twist_pos);
        SharedDiagonal stress_noise = get_noise_model_rot_pos(
            c.sigma_stress_moment, c.sigma_stress_force);
        SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
            c.sigma_base_rot, c.sigma_base_pos);
        // Tight per-finger external-wrench prior noise, matching TendonFingerSolver.
        small_wrench_noises_.push_back(
            get_noise_model_rot_pos(c.sigma_stress_moment, c.sigma_stress_force));

        // This finger's fixed attachment to the wrist. Its node-0 pose mean is
        // T_wrist o T_offset, so the shared base variable seeds to T_wrist for
        // every finger (see get_initial_values / set_root_reparameterization).
        Pose3 offset(c.hand_base_offset);
        hand_base_offsets_.push_back(offset);
        Pose3 base_pose_mean = wrist_pose_ * offset;

        int Nt = c.num_tendons;
        switch (Nt) {
            case 1:  fingers_.push_back(make_finger_impl<1>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 2:  fingers_.push_back(make_finger_impl<2>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 3:  fingers_.push_back(make_finger_impl<3>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 4:  fingers_.push_back(make_finger_impl<4>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 5:  fingers_.push_back(make_finger_impl<5>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 6:  fingers_.push_back(make_finger_impl<6>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 7:  fingers_.push_back(make_finger_impl<7>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 8:  fingers_.push_back(make_finger_impl<8>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 9:  fingers_.push_back(make_finger_impl<9>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 10: fingers_.push_back(make_finger_impl<10>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            default: throw std::invalid_argument(
                "num_tendons must be between 1 and 10, got " + std::to_string(Nt));
        }

        // Share the one wrist variable and let this model own the wrist prior.
        std::visit([&](auto& fp) {
            fp->set_hand_base(offset, shared_wrist_key);
            fp->set_emit_base_prior(false);
        }, fingers_.back());
    }
}


bool TendonHandModel::uses_center_direct_contact(
    const crest_sparse::EnvironmentConfig& env)
{
    // Collision-only env: no object contact of any kind to choose a form for.
    if (!env.target_contact_node.has_value()) return false;

    // An ellipsoid SET (§1.2) has no witness form at all -- the paper defines only
    // the center-direct equality Eq 1.13 for it, and there is no
    // EllipsoidSetWitnessContactFactor to fall back to. So this is not a default
    // that the two witness-only settings may override, the way it is for a single
    // ellipsoid below; asking for either alongside a set is a caller error, and
    // silently ignoring it would solve a different problem than the one requested.
    if (!env.ellipsoid_set.empty()) {
        if (env.contact_drop_normal_row || env.witness_target)
            throw std::invalid_argument(
                "TendonHandModel: contact_drop_normal_row / witness_target select a "
                "witness-point contact form, which an ellipsoid_set does not have "
                "(Section 1.2 defines only the center-direct equality, Eq 1.13). "
                "Clear them, or use ellipsoid_semi_axes for a single ellipsoid.");
        return true;
    }

    // A baked SDF has no closed-form distance to constrain the center against,
    // so it can only be contacted through a witness point.
    if (env.ellipsoid_semi_axes.norm() <= 0.0) return false;
    // Explicit request (§1.8 controller phase 2 sets this).
    if (env.object_contact_center_direct) return true;

    // Default ON for an analytic ellipsoid. The exceptions are the two settings
    // that are meaningless without a witness VARIABLE to attach to, so asking
    // for either is how a caller opts back into the witness form:
    //   contact_drop_normal_row  selects a witness-factor row layout (Eq 1.107-1.110);
    //   witness_target           is a Gaussian prior ON the witness point (Eq 1.111).
    return !env.contact_drop_normal_row && !env.witness_target;
}


void TendonHandModel::add_eq(NonlinearFactorGraph& graph,
                             const gtsam::NoiseModelFactor::shared_ptr& factor,
                             std::string tag) {
    graph.add(gtsam::ZeroCostConstraint(factor));
    tags_.eq.push_back(std::move(tag));
}


void TendonHandModel::add_ineq(NonlinearFactorGraph& graph,
                               const gtsam::NoiseModelFactor::shared_ptr& gap,
                               std::string tag) {
    graph.add(crest_sparse::CollisionInequalityConstraint(gap));
    tags_.ineq.push_back(std::move(tag));
}


NonlinearFactorGraph TendonHandModel::build_graph(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    if (tensions.size() != fingers_.size())
        throw std::invalid_argument("tensions size must match number of fingers");
    if (tip_wrenches.size() != fingers_.size())
        throw std::invalid_argument("tip_wrenches size must match number of fingers");

    NonlinearFactorGraph graph;
    // Rebuilt from scratch alongside the graph: the tags describe THIS graph's
    // constraints, and a stale entry would pair a multiplier with the wrong one.
    tags_ = ConstraintTags{};

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (tensions[i].mean.size() != N)
                throw std::invalid_argument(
                    "Finger " + std::to_string(i) + " expects " + std::to_string(N) +
                    " tendons, got " + std::to_string(tensions[i].mean.size()));

            VectorNGaussian<N> t;
            t.mean = tensions[i].mean;
            t.cov  = tensions[i].cov;

            // Rod + tendon factors (base prior suppressed via set_emit_base_prior).
            graph.add(fp->build_graph(t));

            // Constrain interior external wrenches to zero; the tip wrench uses
            // the caller's value (mirrors TendonFingerSolver::build_graph).
            int num_nodes = fp->get_num_nodes();
            for (int j = 1; j + 1 < num_nodes; ++j) {
                graph.add(PriorFactor<Vector6>(
                    fp->get_external_wrench_key(j), Vector6::Zero(), small_wrench_noises_[i]));
            }
            graph.add(PriorFactor<Vector6>(
                fp->get_external_wrench_key(num_nodes - 1),
                tip_wrenches[i].mean,
                noiseModel::Gaussian::Covariance(tip_wrenches[i].cov)));
        }, fingers_[i]);
    }

    // Single shared floating-wrist prior (rigidly anchors the gauge). In a
    // trajectory only the start step emits it (as the loose hand-pose prior,
    // Eq 1.40); interior/terminal wrists are pinned by the GP chain + kinematics.
    if (emit_wrist_prior_)
        graph.add(PriorFactor<Pose3>(wrist_key(step_), wrist_pose_, wrist_noise_));

    // Per-finger tip contact against one shared object. The object pose is
    // anchored once; each finger drives its own witness point onto the surface.
    // A collision-only sdf env (collision spheres but no target_contact_node)
    // adds no contact factor here — the collision blocks below handle it (and
    // anchor the shared object if no contacting finger did).
    bool object_anchored = false;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i] && !sphere_contacts_[i]) continue;
        std::visit([&](auto& fp) {
            if (sdf_contacts_[i]) {
                const auto& env = *sdf_contacts_[i];
                if (!env.target_contact_node.has_value()) return;  // collision-only
                Key tip_key = fp->rod_->get_pose_key(*env.target_contact_node);
                if (!object_anchored) {
                    graph.add(PriorFactor<Pose3>(
                        object_key(), Pose3(env.object_pose_mean),
                        noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                    object_anchored = true;
                }
                // Center-direct object contact (Eq 1.101): constrain the tip
                // sphere CENTER to the ellipsoid, with no witness point at all.
                // EllipsoidCollisionGapFactor's residual is r - Taubin(x); as an
                // EQUALITY the sign is irrelevant, so its zero set is exactly
                // Eq 1.101. Dropping the witness removes three variables and four
                // residual rows per finger. This is the DEFAULT form for an
                // analytic ellipsoid, not just the controller's phase 2, and the
                // ONLY form for an ellipsoid set — see uses_center_direct_contact().
                if (uses_center_direct_contact(env)) {
                    gtsam::NoiseModelFactor::shared_ptr center_contact;
                    std::string tag;
                    if (!env.ellipsoid_set.empty()) {
                        // Ellipsoid SET (§1.2, Eq 1.13): the same center-direct
                        // equality against the smooth-min distance to the union.
                        // Its own tag, so get_factor_error_summary() tells the two
                        // surface kinds apart.
                        center_contact =
                            std::make_shared<crest_sparse::EllipsoidSetCollisionGapFactor>(
                                tip_key, object_key(), env.contact_node_radius,
                                env.ellipsoid_set, env.ellipsoid_set_beta,
                                noiseModel::Isotropic::Sigma(1, 1.0));
                        tag = "obj.set|f" + std::to_string(i);
                    } else {
                        center_contact =
                            std::make_shared<crest_sparse::EllipsoidCollisionGapFactor>(
                                tip_key, object_key(), env.contact_node_radius,
                                env.ellipsoid_semi_axes,
                                noiseModel::Isotropic::Sigma(1, 1.0));
                        tag = "obj.center|f" + std::to_string(i);
                    }
                    add_eq(graph, center_contact, tag);
                    return;
                }

                // Witness-point contact. drop_normal_row (controller phase 3,
                // Eq 1.107-1.110) omits c_N, so the residual — and the noise
                // model that must match it — is 4 rows instead of 5.
                const bool drop_n = env.contact_drop_normal_row;
                const int  n_rows = drop_n ? 4 : 5;
                gtsam::NoiseModelFactor::shared_ptr contact;
                if (env.ellipsoid_semi_axes.norm() > 0.0) {
                    // Analytic hyper-ellipsoid surface (Section 1.6.3).
                    contact = std::make_shared<crest_sparse::EllipsoidWitnessContactFactor>(
                        tip_key, object_key(), witness_key(static_cast<int>(i)),
                        env.contact_node_radius, env.ellipsoid_semi_axes,
                        noiseModel::Isotropic::Sigma(n_rows, 1.0), drop_n);
                } else {
                    contact = std::make_shared<crest_sparse::SdfWitnessContactFactor>(
                        tip_key, object_key(), witness_key(static_cast<int>(i)),
                        env.contact_node_radius, env.sdf_grid,
                        noiseModel::Isotropic::Sigma(n_rows, 1.0), drop_n);
                }
                add_eq(graph, contact, "obj.witness|f" + std::to_string(i));

                // Soft witness target (Section 1.8, Eq 1.111). Because the AL
                // equality constraints pin the witness to the object surface
                // manifold, this prior acts as a geodesic pull that slides the
                // witness — and the attached finger — along the surface toward a
                // nominal grasp location. Unset => "contact anywhere" (Eq 1.119).
                if (env.witness_target) {
                    graph.add(PriorFactor<Point3>(
                        witness_key(static_cast<int>(i)), *env.witness_target,
                        noiseModel::Gaussian::Covariance(env.witness_target_cov)));
                }
            } else {
                const auto& sc = *sphere_contacts_[i];
                Key finger_key = fp->rod_->get_pose_key(sc.finger_node_index);
                if (!object_anchored) {
                    graph.add(PriorFactor<Pose3>(
                        object_key(), Pose3(Rot3(), sc.sphere_center),
                        noiseModel::Gaussian::Covariance(sc.sphere_pose_cov)));
                    object_anchored = true;
                }
                if (sc.witness) {
                    auto contact = std::make_shared<crest_sparse::SphereWitnessContactFactor>(
                        finger_key, object_key(), witness_key(static_cast<int>(i)),
                        sc.finger_node_radius, sc.sphere_radius,
                        noiseModel::Isotropic::Sigma(5, 1.0));
                    add_eq(graph, contact, "obj.sphwit|f" + std::to_string(i));
                } else {
                    auto contact = std::make_shared<crest_sparse::SphereSphereContactFactor>(
                        finger_key, object_key(),
                        sc.finger_node_radius, sc.sphere_radius,
                        noiseModel::Isotropic::Sigma(1, 1.0));
                    add_eq(graph, contact, "obj.sphere|f" + std::to_string(i));
                }
            }
        }, fingers_[i]);
    }

    // --- Collision avoidance (Section 1.5) -------------------------------
    // Gather each finger's collision spheres once (world-frame pose key, radius,
    // proximal flag). The base disc (node 0) has no free pose variable under the
    // hand-base reparameterization (is_root); it is excluded from BOTH collision
    // passes: the gap factors read the key's translation, which for the shared
    // wrist/root key is the wrist origin — not node-0's position (wrist ∘
    // offset) — and the base disc is rigidly placed by the wrist anyway.
    // is_contact: the tip node that also carries the terminal witness contact
    // factor for this finger (only when this model has target_contact_node, i.e.
    // the terminal step). Its object-collision sphere is skipped so it can't
    // oppose the contact factor — but it is KEPT for finger-finger collision
    // (the contacting tip must still avoid other fingers).
    //
    // The sphere SET is gathered for ANY of its three consumers -- finger-object
    // (needs collision_avoidance and an object surface), the support plane (needs
    // only plane_avoidance) or finger-finger (needs only self_collision). Which
    // constraints are actually built is then decided per family, each on its own
    // field: the finger-object loop below re-checks its own gate, and the
    // finger-finger loop re-checks self_collision.
    // `node` is carried only to name the sphere in a constraint tag: the KEY is
    // model-local (global counter) and useless across a rebuild, while the node
    // index is the same sphere in any model built from the same configs.
    struct CollSphere {
        Key key; double radius; bool proximal; bool is_root; bool is_contact;
        int node;
    };
    std::vector<std::vector<CollSphere>> finger_spheres(fingers_.size());
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i]) continue;
        const auto& env = *sdf_contacts_[i];
        bool wants_object_collision =
            env.collision_avoidance && crest_sparse::has_object_surface(env);
        bool wants_plane_collision  = env.plane_avoidance &&
                                      env.plane_normal.norm() > 0.0;
        bool wants_self_collision   = env.self_collision;
        if ((!wants_object_collision && !wants_plane_collision &&
             !wants_self_collision) ||
            env.collision_node_indices.empty())
            continue;
        std::visit([&](auto& fp) {
            for (size_t j = 0; j < env.collision_node_indices.size(); ++j) {
                int idx = env.collision_node_indices[j];
                bool is_root = fp->rod_->uses_root() &&
                               fp->rod_->get_pose_key(idx) == fp->rod_->get_pose_key(0);
                Key key = is_root ? fp->rod_->get_root_base_key()
                                  : fp->rod_->get_pose_key(idx);
                double r = env.collision_node_radii[j];
                bool prox = (j < env.collision_node_is_proximal.size())
                                ? (env.collision_node_is_proximal[j] != 0) : false;
                bool is_contact = env.target_contact_node.has_value() &&
                    fp->rod_->get_pose_key(idx) ==
                        fp->rod_->get_pose_key(*env.target_contact_node);
                finger_spheres[i].push_back({key, r, prox, is_root, is_contact, idx});
            }
        }, fingers_[i]);
    }

    // Finger-object: keep every collision sphere out of the shared object SDF,
    // except the terminal contact node (its collision would oppose the contact
    // factor; the contact factor already pins it tangent to the surface).
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (finger_spheres[i].empty()) continue;
        const auto& env = *sdf_contacts_[i];
        // Spheres may have been gathered for the support plane alone; only an env
        // that actually asked for OBJECT avoidance builds these. This predicate
        // must stay identical to the has_col guard in get_initial_values(), which
        // decides whether the shared object variable is seeded at all -- anchoring
        // an object that was never seeded is an indeterminate system. Both now
        // call has_object_surface() so they cannot drift apart.
        if (!env.collision_avoidance || !crest_sparse::has_object_surface(env))
            continue;
        if (!object_anchored) {
            graph.add(PriorFactor<Pose3>(
                object_key(), Pose3(env.object_pose_mean),
                noiseModel::Gaussian::Covariance(env.object_pose_cov)));
            object_anchored = true;
        }
        auto col_noise = noiseModel::Isotropic::Sigma(1, env.collision_sigma);
        // Surface precedence, as documented on EnvironmentConfig::ellipsoid_set:
        // set > single ellipsoid > baked SDF.
        bool is_set       = !env.ellipsoid_set.empty();
        bool is_ellipsoid = env.ellipsoid_semi_axes.norm() > 0.0;
        for (const auto& s : finger_spheres[i]) {
            if (s.is_contact || s.is_root) continue;
            gtsam::NoiseModelFactor::shared_ptr gap;
            if (is_set) {
                // Eq 1.12: the same residual as the contact equality above, read
                // as an inequality c_pen = r - d_E <= 0.
                gap = std::make_shared<crest_sparse::EllipsoidSetCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.ellipsoid_set,
                    env.ellipsoid_set_beta, col_noise);
            } else if (is_ellipsoid) {
                gap = std::make_shared<crest_sparse::EllipsoidCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.ellipsoid_semi_axes, col_noise);
            } else {
                gap = std::make_shared<crest_sparse::SdfCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.sdf_grid, col_noise);
            }
            add_ineq(graph, gap, "col.obj|f" + std::to_string(i) +
                                 "|n" + std::to_string(s.node));
        }
    }

    // Finger-finger: keep collision spheres of distinct fingers apart. Built for
    // a pair iff BOTH its fingers asked for self_collision -- the constraint
    // belongs to the two of them jointly, so one finger opting out drops every
    // pair it is part of. Skip a pair iff BOTH spheres are proximal
    // (rigidly-attached base bones), and skip any base-disc (root) sphere (no
    // distinct per-finger pose variable). This leaves distal-distal and
    // distal-proximal cross-finger pairs.
    // When collision_cull_margin >= 0, additionally skip pairs whose gap at the
    // initial values already exceeds the margin (see the caveats on
    // EnvironmentConfig::collision_cull_margin — heuristic, verified by the
    // tests' independent all-pairs penetration report).
    std::optional<Values> init_vals;
    auto initial_position = [&](Key key) {
        if (!init_vals) init_vals = get_initial_values();
        return init_vals->at<Pose3>(key).translation();
    };
    for (size_t a = 0; a < fingers_.size(); ++a) {
        if (finger_spheres[a].empty()) continue;
        if (!sdf_contacts_[a]->self_collision) continue;
        auto col_noise = noiseModel::Isotropic::Sigma(
            1, sdf_contacts_[a]->collision_sigma);
        const double cull_margin = sdf_contacts_[a]->collision_cull_margin;
        for (size_t b = a + 1; b < fingers_.size(); ++b) {
            if (finger_spheres[b].empty()) continue;
            if (!sdf_contacts_[b]->self_collision) continue;
            for (const auto& sa : finger_spheres[a]) {
                if (sa.is_root) continue;
                for (const auto& sb : finger_spheres[b]) {
                    if (sb.is_root) continue;
                    if (sa.proximal && sb.proximal) continue;
                    if (cull_margin >= 0.0) {
                        const double gap0 =
                            (initial_position(sa.key) - initial_position(sb.key))
                                .norm() - sa.radius - sb.radius;
                        if (gap0 > cull_margin) continue;
                    }
                    auto gap = std::make_shared<crest_sparse::SphereSphereCollisionGapFactor>(
                        sa.key, sb.key, sa.radius, sb.radius, col_noise);
                    // a < b by construction, so the pair name is canonical.
                    add_ineq(graph, gap,
                             "col.ff|f" + std::to_string(a) + "n" +
                             std::to_string(sa.node) + "|f" + std::to_string(b) +
                             "n" + std::to_string(sb.node));
                }
            }
        }
    }

    // --- Support plane / "table" (Section 1.6) ---------------------------
    // A world-fixed analytic half-space (env.plane_origin, env.plane_normal). It
    // is independent of the object SDF machinery above (no object variable), so
    // it walks env.collision_node_indices directly rather than reusing
    // finger_spheres -- which carries the root/contact exclusions the OBJECT
    // constraints want, not the (different) ones the plane wants.
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i]) continue;
        const auto& env = *sdf_contacts_[i];
        if (env.plane_normal.norm() <= 0.0) continue;   // no table configured
        std::visit([&](auto& fp) {
            // Table sliding equality: place the contact node's sphere on the
            // plane and let it slide, as ONE residual on the sphere CENTER --
            // c_table(c) = Dist_plane(c) = 0 -- with no witness point at all.
            // PlaneCollisionGapFactor's residual is r - (c - p).n; as an EQUALITY
            // the sign is irrelevant, so its zero set is exactly that, in the
            // signed form (pins the sphere on the +n free side and stays smooth
            // at the contact point, where the paper's |.| has a kink).
            //
            // This replaces the original Eq 1.60-1.64 five-residual witness form.
            // The witness there bought nothing: four of its five rows only pinned
            // the gauge of the free point it introduced, and for a PLANE a single
            // scalar on the center leaves no rotational freedom to brick the
            // solver -- the same argument §1.8 already makes for
            // support_contact_node, whose factor this now matches exactly.
            if (env.table_contact_node.has_value()) {
                Key tip_key = fp->rod_->get_pose_key(*env.table_contact_node);
                auto contact = std::make_shared<crest_sparse::PlaneCollisionGapFactor>(
                    tip_key, env.table_contact_radius,
                    env.plane_origin, env.plane_normal,
                    noiseModel::Isotropic::Sigma(1, 1.0));
                add_eq(graph, contact, "tbl.contact|f" + std::to_string(i));
            }

            // --- Section 1.8 controller phases 1-2 ---------------------------
            // Support-surface contact EQUALITY on the sphere CENTER (Eq 1.97),
            // the witness-free counterpart of the §1.6 sliding equality above.
            // PlaneCollisionGapFactor's residual is r - (c - p).n; as an EQUALITY
            // the sign is irrelevant, so its zero set is exactly Dist_plane = 0 —
            // in the signed form, which pins the sphere on the +n (free) side and
            // stays smooth at the contact point (the paper's |.| has a kink
            // exactly where the solver operates). Kept for the dormant phased
            // controller use case; independent of the table_contact_node
            // equality above.
            if (env.support_contact_node.has_value()) {
                Key sup_key = fp->rod_->get_pose_key(*env.support_contact_node);
                auto support = std::make_shared<crest_sparse::PlaneCollisionGapFactor>(
                    sup_key, env.support_contact_radius,
                    env.plane_origin, env.plane_normal,
                    noiseModel::Isotropic::Sigma(1, 1.0));
                add_eq(graph, support, "sup.contact|f" + std::to_string(i));
            }

            // Table collision (Eq 1.59): keep every non-root collision sphere out
            // of the half-space, except the table contact node (its collision
            // would oppose the sliding equality that already pins it to the plane).
            // The same exclusion applies to support_contact_node: §1.8 warns that
            // stacking the inequality on a sphere that already carries the
            // equality gives linearly dependent Jacobians once contact is met,
            // artificially rank-deficient Hessian, destabilized inner LM.
            if (env.plane_avoidance && !env.collision_node_indices.empty()) {
                auto col_noise = noiseModel::Isotropic::Sigma(1, env.collision_sigma);
                for (size_t j = 0; j < env.collision_node_indices.size(); ++j) {
                    int idx = env.collision_node_indices[j];
                    bool is_root = fp->rod_->uses_root() &&
                        fp->rod_->get_pose_key(idx) == fp->rod_->get_pose_key(0);
                    if (is_root) continue;
                    if (env.table_contact_node.has_value() &&
                        fp->rod_->get_pose_key(idx) ==
                            fp->rod_->get_pose_key(*env.table_contact_node))
                        continue;
                    if (env.support_contact_node.has_value() &&
                        fp->rod_->get_pose_key(idx) ==
                            fp->rod_->get_pose_key(*env.support_contact_node))
                        continue;
                    double r = env.collision_node_radii[j];
                    auto gap = std::make_shared<crest_sparse::PlaneCollisionGapFactor>(
                        fp->rod_->get_pose_key(idx), r,
                        env.plane_origin, env.plane_normal, col_noise);
                    add_ineq(graph, gap, "col.plane|f" + std::to_string(i) +
                                         "|n" + std::to_string(idx));
                }
            }
        }, fingers_[i]);
    }

    // --- Opposition half-space (Eq 2.16-2.17 / Eq 1.92) ------------------
    // Keep each participating finger's sphere center on its designated half of
    // the splitting line, so the thumb lands opposite the fingers. A pass of its
    // OWN, gated on nothing but its own fields: the residual is a statement
    // about one node's position relative to a line (constant Jacobian), and
    // needs neither a support plane nor a contact node of any kind. It used to
    // live inside the table-contact branch above, which silently made checking
    // "opposition" alone build nothing at all.
    //
    // half_space_node is this constraint's own opt-in field; table_contact_node
    // is the fallback for a caller still writing the env the old way.
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i]) continue;
        const auto& env = *sdf_contacts_[i];
        if (!env.half_space_enabled || env.half_space_normal.norm() <= 0.0)
            continue;
        std::optional<int> node = env.half_space_node.has_value()
                                      ? env.half_space_node
                                      : env.table_contact_node;
        if (!node.has_value()) continue;
        std::visit([&](auto& fp) {
            auto half = std::make_shared<crest_sparse::HalfSpaceGapFactor>(
                fp->rod_->get_pose_key(*node),
                env.half_space_split_point, env.half_space_normal,
                noiseModel::Isotropic::Sigma(1, env.collision_sigma),
                env.half_space_margin);
            // Tag unchanged from when this lived in the table block: the AL
            // dual transfer matches constraints by tag, so renaming it here
            // would silently break every warm start across a rebuild.
            add_ineq(graph, half, "half|f" + std::to_string(i));
        }, fingers_[i]);
    }

    // Pre-grasp hand-centering (Eq 2.18-2.19): spans the thumb + every other
    // participating finger, so — unlike every block above — this collects
    // across ALL fingers first rather than building inside the per-finger
    // visit. Thumb identified by name (the existing hand-wide convention:
    // config.py's get_default_hand_configs always appends "thumb" last).
    {
        std::optional<Key> thumb_key;
        std::vector<Key> finger_keys;
        double h_clear = 0.0;
        gtsam::Vector3 n_hat = gtsam::Vector3::Zero();
        for (size_t i = 0; i < fingers_.size(); ++i) {
            if (!sdf_contacts_[i]) continue;
            const auto& env = *sdf_contacts_[i];
            if (!env.pregrasp_center_node.has_value()) continue;
            std::visit([&](auto& fp) {
                Key k = fp->rod_->get_pose_key(*env.pregrasp_center_node);
                if (finger_names_[i] == "thumb") thumb_key = k;
                else finger_keys.push_back(k);
            }, fingers_[i]);
            h_clear = env.pregrasp_clearance_height;
            n_hat = env.pregrasp_clearance_normal;
            // Anchor the shared object pose if nothing else has yet -- this
            // constraint touches object_key() too, but only its translation,
            // so it cannot by itself fix the object's orientation gauge.
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(env.object_pose_mean),
                    noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                object_anchored = true;
            }
        }
        if (thumb_key.has_value() && !finger_keys.empty() && n_hat.norm() > 0.0) {
            auto center = std::make_shared<crest_sparse::PreGraspHandCenteringFactor>(
                *thumb_key, finger_keys, object_key(), h_clear, n_hat,
                noiseModel::Isotropic::Sigma(3, 1.0));
            add_eq(graph, center, "pregrasp.center");
        }
    }

    // Pre-grasp short-axis alignment (companion to Eq 2.16-2.17): also a
    // hand-level pass, collected the same way as the hand-centering block
    // above, but on the SEPARATE pregrasp_align_node/pregrasp_align_axis
    // fields so it stays independently toggleable. No object-pose anchoring
    // needed here -- PreGraspAxisAlignmentFactor never touches object_key().
    {
        std::optional<Key> thumb_key;
        std::vector<Key> finger_keys;
        gtsam::Vector3 axis = gtsam::Vector3::Zero();
        for (size_t i = 0; i < fingers_.size(); ++i) {
            if (!sdf_contacts_[i]) continue;
            const auto& env = *sdf_contacts_[i];
            if (!env.pregrasp_align_node.has_value()) continue;
            std::visit([&](auto& fp) {
                Key k = fp->rod_->get_pose_key(*env.pregrasp_align_node);
                if (finger_names_[i] == "thumb") thumb_key = k;
                else finger_keys.push_back(k);
            }, fingers_[i]);
            axis = env.pregrasp_align_axis;
        }
        if (thumb_key.has_value() && !finger_keys.empty() && axis.norm() > 0.0) {
            auto align = std::make_shared<crest_sparse::PreGraspAxisAlignmentFactor>(
                *thumb_key, finger_keys, axis,
                noiseModel::Isotropic::Sigma(1, 1.0));
            add_eq(graph, align, "pregrasp.align");
        }
    }

    // Pre-grasp PINCH-CENTROID centering: the hardcoded-point sibling of the
    // hand-centering block above. Hand-level like the other two, but no finger
    // opts in -- the point is a constant in the WRIST frame, so the factor
    // keys off wrist_key() directly and the fields are simply duplicated
    // across every finger's env (first one found wins, matching how
    // h_clear/n_hat are read above).
    {
        std::optional<gtsam::Vector3> centroid;
        double h_clear = 0.0;
        gtsam::Vector3 n_hat = gtsam::Vector3::Zero();
        for (size_t i = 0; i < fingers_.size(); ++i) {
            if (!sdf_contacts_[i]) continue;
            const auto& env = *sdf_contacts_[i];
            if (!env.pregrasp_centroid_point.has_value()) continue;
            centroid = *env.pregrasp_centroid_point;
            h_clear = env.pregrasp_centroid_clearance;
            n_hat = env.pregrasp_centroid_normal;
            // Same reasoning as the hand-centering block: this constraint
            // reads object_key()'s translation, so the object needs an anchor
            // if no other block has supplied one.
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(env.object_pose_mean),
                    noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                object_anchored = true;
            }
            break;
        }
        if (centroid.has_value() && n_hat.norm() > 0.0) {
            auto pinch = std::make_shared<crest_sparse::PreGraspCentroidFactor>(
                wrist_key(step_), object_key(), gtsam::Point3(*centroid),
                h_clear, n_hat, noiseModel::Isotropic::Sigma(3, 1.0));
            add_eq(graph, pinch, "pregrasp.centroid");
        }
    }

    return graph;
}


Values TendonHandModel::get_initial_values(const Values* warm) const {
    Values values;

    // Merge each finger's values; the shared wrist variable appears in every
    // finger's values (identical), so keep only the first and drop the rest.
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            Values fv = fp->get_initial_values();
            if (i > 0) fv.erase(wrist_key(step_));
            values.insert(fv);
        }, fingers_[i]);
    }

    // Adopt any warm-start poses BEFORE the witness seeding below, so the
    // projections that derive each witness from its contact node start from
    // where the finger actually converged rather than from a straight hand.
    if (warm) {
        for (Key k : values.keys())
            if (warm->exists(k)) values.update(k, warm->at(k));
    }

    // Contact object pose (once) + per-finger witness point seeds.
    bool object_seeded = false;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i] && !sphere_contacts_[i]) continue;
        std::visit([&](const auto& fp) {
            if (sdf_contacts_[i]) {
                const auto& env = *sdf_contacts_[i];
                // Only seed the shared object when this env actually contributes
                // a factor in build_graph: a terminal contact and/or active
                // collision. An inert sdf env seeds nothing (avoids an orphan
                // object variable with no factors).
                bool has_col = env.collision_avoidance &&
                               crest_sparse::has_object_surface(env) &&
                               !env.collision_node_indices.empty();
                if (!env.target_contact_node.has_value() && !has_col) return;

                Pose3 obj_mean(env.object_pose_mean);
                if (!object_seeded) { values.insert(object_key(), obj_mean); object_seeded = true; }

                // Collision-only env (no target_contact_node): the object is
                // seeded above; there is no witness point to seed.
                if (!env.target_contact_node.has_value()) return;

                // Center-direct contact (Eq 1.101) constrains the sphere center
                // directly, so build_graph creates no witness variable here —
                // seeding one would leave an orphan value with no factors. Must
                // stay in lockstep with build_graph, hence the shared predicate.
                if (uses_center_direct_contact(env)) return;

                Point3 seed_local;
                if (env.witness_target) {
                    // Controller phase 3 (Eq 1.111): start the witness at its
                    // nominal grasp target (given in the WORLD frame) so the
                    // geodesic pull has a short way to travel.
                    seed_local = obj_mean.transformTo(*env.witness_target);
                } else if (env.witness_point_seed) {
                    // Caller-provided seed (object-local frame); skip the march.
                    seed_local = *env.witness_point_seed;
                } else if (env.ellipsoid_semi_axes.norm() > 0.0) {
                    // Analytic ellipsoid (Section 1.6.3): project the tip radially
                    // onto the surface x^T M x = 1. seed = tip_local / sqrt(tip^T M tip).
                    int i_node = *env.target_contact_node;
                    Point3 tip_world = values.at<Pose3>(fp->rod_->get_pose_key(i_node)).translation();
                    Point3 tip_local = obj_mean.transformTo(tip_world);
                    const Vector3& a = env.ellipsoid_semi_axes;
                    Vector3 m_diag(1.0 / (a.x() * a.x()), 1.0 / (a.y() * a.y()),
                                   1.0 / (a.z() * a.z()));
                    double q = tip_local.cwiseProduct(m_diag.cwiseProduct(tip_local)).sum();
                    if (q > 1e-12) seed_local = Point3(tip_local / std::sqrt(q));
                    else           seed_local = Point3(a.x(), 0.0, 0.0);
                } else {
                    // Ray-march in the object-local frame from the local origin
                    // toward the tip until the SDF crosses zero (mirrors
                    // TendonFingerSolver).
                    int i_node = *env.target_contact_node;
                    Point3 tip_world = values.at<Pose3>(fp->rod_->get_pose_key(i_node)).translation();
                    Point3 tip_local = obj_mean.transformTo(tip_world);
                    double tip_local_norm = tip_local.norm();
                    Point3 dir_local = (tip_local_norm > 1e-8)
                                           ? Point3(tip_local / tip_local_norm)
                                           : Point3(0.0, 0.0, 1.0);

                    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>
                        sampler(*env.sdf_grid);
                    const double step = 5e-4;
                    const int max_it = 4000;
                    double t = 0.0;
                    double prev_sdf = sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0));
                    double t_surface = -1.0;
                    for (int it = 1; it <= max_it; ++it) {
                        double tt = it * step;
                        Point3 q = tt * dir_local;
                        double sdf = sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z()));
                        if (std::isfinite(prev_sdf) && std::isfinite(sdf) && prev_sdf * sdf < 0.0) {
                            double alpha = prev_sdf / (prev_sdf - sdf);
                            t_surface = t + alpha * step;
                            break;
                        }
                        prev_sdf = sdf;
                        t = tt;
                    }
                    if (t_surface < 0.0)
                        t_surface = std::abs(sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0)));
                    seed_local = Point3(t_surface * dir_local);
                }
                Point3 seed_world = obj_mean.transformFrom(seed_local);
                values.insert(witness_key(static_cast<int>(i)), seed_world);
            } else {
                const auto& sc = *sphere_contacts_[i];
                if (!object_seeded) {
                    values.insert(object_key(), Pose3(Rot3(), sc.sphere_center));
                    object_seeded = true;
                }
                if (sc.witness) {
                    Point3 finger_pos = values
                        .at<Pose3>(fp->rod_->get_pose_key(sc.finger_node_index)).translation();
                    Vector3 d = finger_pos - sc.sphere_center;
                    double dn = d.norm();
                    Vector3 dir = (dn > 1e-8) ? Vector3(d / dn) : Vector3(0.0, 0.0, 1.0);
                    values.insert(witness_key(static_cast<int>(i)),
                                  Point3(sc.sphere_center + sc.sphere_radius * dir));
                }
            }
        }, fingers_[i]);
    }

    // No support-plane seeding: the table contact equality constrains the contact
    // node's sphere CENTER directly (see build_graph), so there is no witness
    // variable to seed. Must stay in lockstep with build_graph -- seeding one
    // here would leave an orphan value with no factors, i.e. an indeterminate
    // system.

    return values;
}


Values TendonHandModel::values_from_marginals(const TendonHandMarginals& state) const {
    if (state.fingers.size() != fingers_.size())
        throw std::invalid_argument(
            "values_from_marginals: state has " +
            std::to_string(state.fingers.size()) + " fingers, this model has " +
            std::to_string(fingers_.size()));

    Values values;

    for (size_t i = 0; i < fingers_.size(); ++i) {
        const TendonFingerMarginals& fm = state.fingers[i];
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            const int num_nodes = fp->get_num_nodes();
            if (static_cast<int>(fm.rod.states.size()) != num_nodes)
                throw std::invalid_argument(
                    "values_from_marginals: finger " + std::to_string(i) +
                    " has " + std::to_string(fm.rod.states.size()) +
                    " rod states, this model has " + std::to_string(num_nodes) +
                    " nodes");
            if (fm.tensions.mean.size() != N)
                throw std::invalid_argument(
                    "values_from_marginals: finger " + std::to_string(i) +
                    " has " + std::to_string(fm.tensions.mean.size()) +
                    " tendons, this model has " + std::to_string(N));

            // Rod chain. Node 0's pose is NOT a variable under the hand-base
            // reparameterization (see the header note); its stress and wrench
            // still are, so only the pose insert is skipped.
            const bool skip_node0_pose = fp->rod_->uses_root();
            for (int j = 0; j < num_nodes; ++j) {
                const auto& s = fm.rod.states[j];
                if (!(skip_node0_pose && j == 0))
                    values.insert(fp->rod_->get_pose_key(j), Pose3(s.pose.mean));
                values.insert(fp->rod_->get_stress_key(j), Vector6(s.stress.mean));
                values.insert(fp->rod_->get_wrench_key(j), Vector6(s.wrench.mean));
            }

            // External disc wrenches. external_wrenches is indexed by NODE and
            // resolves through get_external_wrench_key, which aliases the rod's
            // own wrench key at non-disc nodes -- already written above. Only the
            // genuine Symbol('D', ...) variables are left, so walk the discs.
            const auto& disc_pose_idx = fp->get_tendon_config().disc_pose_idx;
            for (size_t d = 1; d < disc_pose_idx.size(); ++d) {
                const int node = disc_pose_idx[d];
                if (node < 0 || node >= static_cast<int>(fm.external_wrenches.size()))
                    continue;
                values.insert(fp->get_disc_wrench_key(static_cast<int>(d)),
                              Vector6(fm.external_wrenches[node].mean));
            }

            values.insert(fp->get_tensions_key(),
                          Eigen::Vector<double, N>(fm.tensions.mean));

            if (static_cast<int>(fm.tendon_lengths.size()) == N) {
                Eigen::Vector<double, N> L;
                for (int t = 0; t < N; ++t) L(t) = fm.tendon_lengths[t];
                values.insert(fp->get_lengths_key(), L);
            }
        }, fingers_[i]);
    }

    // The shared wrist. No finger carries it directly: under the hand-base
    // reparameterization node 0's pose is not a variable but the composition
    // T_0 = T_wrist o T_offset, so the loop above deliberately skipped it and
    // the wrist would otherwise be missing from the bundle entirely. A warm
    // start built from that would hold every rod pose from the state and the
    // wrist at whatever the receiving model was constructed with -- an
    // inconsistent guess that the Root factors and the wrist prior immediately
    // tear back apart, i.e. the hand snapping to the commanded base pose on the
    // first iteration. Invert the relation instead and carry it.
    const bool uses_root = !fingers_.empty() && std::visit(
        [](const auto& fp) { return fp->rod_->uses_root(); }, fingers_[0]);
    if (uses_root && !hand_base_offsets_.empty()) {
        const Pose3 T0(state.fingers[0].rod.states[0].pose.mean);
        values.insert(wrist_key(step_), T0 * hand_base_offsets_[0].inverse());
    }

    return values;
}


Key TendonHandModel::finger_tension_key(int i) const {
    return std::visit(
        [](const auto& fp) { return fp->get_tensions_key(); }, fingers_.at(i));
}


Key TendonHandModel::finger_length_key(int i) const {
    return std::visit(
        [](const auto& fp) { return fp->get_lengths_key(); }, fingers_.at(i));
}


Key TendonHandModel::finger_tip_pose_key(int i) const {
    return std::visit(
        [](const auto& fp) { return fp->rod_->get_pose_key(-1); }, fingers_.at(i));
}


Key TendonHandModel::finger_node_pose_key(int i, int node) const {
    return std::visit(
        [node](const auto& fp) { return fp->rod_->get_pose_key(node); }, fingers_.at(i));
}


void TendonHandModel::add_temporal_gp(
    NonlinearFactorGraph& graph,
    const TendonHandModel& next,
    const Eigen::MatrixXd& gp_tense_Qc,
    const Eigen::MatrixXd& gp_len_Qc,
    double dt) const
{
    const bool has_len_gp = gp_len_Qc.size() > 0;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            // Tension GP (Eq 1.11): identity transition, zero-mean between factor.
            Eigen::Matrix<double, N, N> Qc = gp_tense_Qc.topLeftCorner<N, N>();
            graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(),
                next.finger_tension_key(static_cast<int>(i)),
                Eigen::Vector<double, N>::Zero(),
                noiseModel::Gaussian::Covariance(Qc * dt)));

            // Length GP (Eq 1.13), optional.
            if (has_len_gp) {
                Eigen::Matrix<double, N, N> Qc_len = gp_len_Qc.topLeftCorner<N, N>();
                graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                    fp->get_lengths_key(),
                    next.finger_length_key(static_cast<int>(i)),
                    Eigen::Vector<double, N>::Zero(),
                    noiseModel::Gaussian::Covariance(Qc_len * dt)));
            }
        }, fingers_[i]);
    }
}


void TendonHandModel::add_length_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& lengths) const
{
    if (lengths.size() != fingers_.size())
        throw std::invalid_argument(
            "add_length_priors: lengths size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (lengths[i].mean.size() != N)
                throw std::invalid_argument(
                    "add_length_priors: finger " + std::to_string(i) + " expects " +
                    std::to_string(N) + " tendon lengths, got " +
                    std::to_string(lengths[i].mean.size()));

            Eigen::Vector<double, N> mean = lengths[i].mean;
            Eigen::Matrix<double, N, N> cov = lengths[i].cov.topLeftCorner<N, N>();
            graph.add(PriorFactor<Eigen::Vector<double, N>>(
                fp->get_lengths_key(), mean, noiseModel::Gaussian::Covariance(cov)));
        }, fingers_[i]);
    }
}


void TendonHandModel::add_tension_priors(
    NonlinearFactorGraph& graph,
    const std::vector<VectorXGaussian>& tensions) const
{
    if (tensions.size() != fingers_.size())
        throw std::invalid_argument(
            "add_tension_priors: tensions size must match number of fingers");

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (tensions[i].mean.size() != N)
                throw std::invalid_argument(
                    "add_tension_priors: finger " + std::to_string(i) + " expects " +
                    std::to_string(N) + " tendon tensions, got " +
                    std::to_string(tensions[i].mean.size()));

            Eigen::Vector<double, N> mean = tensions[i].mean;
            Eigen::Matrix<double, N, N> cov = tensions[i].cov.topLeftCorner<N, N>();
            graph.add(PriorFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(), mean, noiseModel::Gaussian::Covariance(cov)));
        }, fingers_[i]);
    }
}


TendonHandMarginals TendonHandModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    TendonHandMarginals out;
    out.fingers.reserve(fingers_.size());
    out.finger_names = finger_names_;
    for (const auto& finger : fingers_) {
        std::visit([&](const auto& fp) {
            out.fingers.push_back(fp->get_marginals(values, marginals));
        }, finger);
    }
    return out;
}


TendonHandMarginals TendonHandModel::get_marginals_means_only(
    const Values& values) const
{
    TendonHandMarginals out;
    out.fingers.reserve(fingers_.size());
    out.finger_names = finger_names_;
    // Zero-returning functors: extract means only, skipping the Marginals solve.
    // cov_of returns a 6x6 (pose block; the tension cov it also feeds is unused
    // for visualization). joint_of must be sized (6+N)x(6+N) per finger because
    // TendonFingerModel::get_J_pose_tensions reads block<6,N>(0,6)/block<N,N>(6,6),
    // so it is built inside the visit where the finger's N (NumTendons) is known.
    auto zero_cov = [](gtsam::Key) { return gtsam::Matrix::Zero(6, 6); };
    for (const auto& finger : fingers_) {
        std::visit([&](const auto& fp) {
            constexpr int N = std::remove_reference_t<decltype(*fp)>::NumTendons;
            auto zero_joint = [N](gtsam::Key, gtsam::Key) {
                return gtsam::Matrix::Zero(6 + N, 6 + N);
            };
            out.fingers.push_back(fp->get_marginals(values, zero_cov, zero_joint));
        }, finger);
    }
    return out;
}
