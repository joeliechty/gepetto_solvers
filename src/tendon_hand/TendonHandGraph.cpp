// TendonHandModel::build_graph -- every factor in the system, in the order it
// is added.
//
// THAT ORDER IS LOAD-BEARING. The Augmented Lagrangian indexes multipliers by
// a constraint's position in the enumeration, which is graph insertion order,
// so reordering these blocks re-seats every carried multiplier onto the wrong
// constraint. tests/core/test_constraint_tags.py guards it.
//
// The emission order is: per-finger factors, external-wrench priors, the
// shared wrist prior, object contact, finger-object collision, finger-finger
// collision, the support plane, then pre-grasp positioning.

#include "gepetto_solvers/tendon_hand/TendonHandModel.h"

#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <openvdb/tools/Interpolation.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>

using namespace gtsam;


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
                    if (env.object_contact_in_plane) {
                        // Eq 13: the same center-direct equality, with the
                        // distance measured inside the finger's pulling plane
                        // (Eq 11) instead of in 3D. A single analytic ellipsoid
                        // goes in as a one-member set -- K=1 with an identity
                        // local_pose reduces exactly to it, so there is one code
                        // path rather than two.
                        //
                        // The plane's third point, the metacarpal base, is taken
                        // from hand_base_offsets_[i] rather than from the env:
                        // it IS this finger's mounting offset, so reading it here
                        // makes it impossible for the plane to be built about a
                        // base the finger is not actually on.
                        std::vector<gepetto_solvers::EllipsoidPrimitive> members;
                        if (!env.ellipsoid_set.empty()) {
                            // Narrowed by contact_ellipsoid_subset when one is
                            // set; the free spheres below keep the whole union.
                            members = gepetto_solvers::contact_ellipsoid_members(env);
                        } else {
                            gepetto_solvers::EllipsoidPrimitive one;
                            one.semi_axes = env.ellipsoid_semi_axes;
                            members.push_back(one);
                        }
                        center_contact =
                            std::make_shared<gepetto_solvers::EllipsoidSetPlanarGapFactor>(
                                tip_key, object_key(), wrist_key(step_),
                                env.contact_node_radius, members,
                                env.ellipsoid_set_beta,
                                hand_base_offsets_[i].translation(),
                                gtsam::Point3(*env.contact_plane_centroid),
                                noiseModel::Isotropic::Sigma(1, 1.0),
                                env.contact_plane_rho_lo, env.contact_plane_rho_hi,
                                env.contact_plane_gap_lo, env.contact_plane_gap_hi);
                        // Its own tag, and that matters twice over:
                        // get_factor_error_summary() tells the two forms apart,
                        // and the AL dual transfer matches by tag -- so switching
                        // forms correctly DROPS the old multipliers instead of
                        // carrying them onto a constraint that means something else.
                        tag = "obj.planar|f" + std::to_string(i);
                    } else if (!env.ellipsoid_set.empty()) {
                        // Ellipsoid SET (§1.2, Eq 1.13): the same center-direct
                        // equality against the smooth-min distance to the union.
                        // Its own tag, so get_factor_error_summary() tells the two
                        // surface kinds apart.
                        //
                        // The union here is the CONTACT one -- narrowed by
                        // contact_ellipsoid_subset to the shells that are grasp
                        // targets, where the collision inequality below keeps
                        // every member. The tag does not encode the subset, so a
                        // warm start would carry duals across a change of it;
                        // callers that change the subset should reset them, the
                        // same as for a change of object.
                        center_contact =
                            std::make_shared<gepetto_solvers::EllipsoidSetCollisionGapFactor>(
                                tip_key, object_key(), env.contact_node_radius,
                                gepetto_solvers::contact_ellipsoid_members(env),
                                env.ellipsoid_set_beta,
                                noiseModel::Isotropic::Sigma(1, 1.0));
                        tag = "obj.set|f" + std::to_string(i);
                    } else {
                        center_contact =
                            std::make_shared<gepetto_solvers::EllipsoidCollisionGapFactor>(
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
                    contact = std::make_shared<gepetto_solvers::EllipsoidWitnessContactFactor>(
                        tip_key, object_key(), witness_key(static_cast<int>(i)),
                        env.contact_node_radius, env.ellipsoid_semi_axes,
                        noiseModel::Isotropic::Sigma(n_rows, 1.0), drop_n);
                } else {
                    contact = std::make_shared<gepetto_solvers::SdfWitnessContactFactor>(
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
                    auto contact = std::make_shared<gepetto_solvers::SphereWitnessContactFactor>(
                        finger_key, object_key(), witness_key(static_cast<int>(i)),
                        sc.finger_node_radius, sc.sphere_radius,
                        noiseModel::Isotropic::Sigma(5, 1.0));
                    add_eq(graph, contact, "obj.sphwit|f" + std::to_string(i));
                } else {
                    auto contact = std::make_shared<gepetto_solvers::SphereSphereContactFactor>(
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
            env.collision_avoidance && gepetto_solvers::has_object_surface(env);
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
        if (!env.collision_avoidance || !gepetto_solvers::has_object_surface(env))
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
                gap = std::make_shared<gepetto_solvers::EllipsoidSetCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.ellipsoid_set,
                    env.ellipsoid_set_beta, col_noise);
            } else if (is_ellipsoid) {
                gap = std::make_shared<gepetto_solvers::EllipsoidCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.ellipsoid_semi_axes, col_noise);
            } else {
                gap = std::make_shared<gepetto_solvers::SdfCollisionGapFactor>(
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
                    auto gap = std::make_shared<gepetto_solvers::SphereSphereCollisionGapFactor>(
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
                auto contact = std::make_shared<gepetto_solvers::PlaneCollisionGapFactor>(
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
                auto support = std::make_shared<gepetto_solvers::PlaneCollisionGapFactor>(
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
                    auto gap = std::make_shared<gepetto_solvers::PlaneCollisionGapFactor>(
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
            auto half = std::make_shared<gepetto_solvers::HalfSpaceGapFactor>(
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
            auto center = std::make_shared<gepetto_solvers::PreGraspHandCenteringFactor>(
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
            auto align = std::make_shared<gepetto_solvers::PreGraspAxisAlignmentFactor>(
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
            auto pinch = std::make_shared<gepetto_solvers::PreGraspCentroidFactor>(
                wrist_key(step_), object_key(), gtsam::Point3(*centroid),
                h_clear, n_hat, noiseModel::Isotropic::Sigma(3, 1.0));
            add_eq(graph, pinch, "pregrasp.centroid");
        }
    }

    return graph;
}
