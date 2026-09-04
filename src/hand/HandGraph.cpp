// HandModel::build_graph -- every factor in the system, in the order it is added.
//
// THAT ORDER IS LOAD-BEARING. The Augmented Lagrangian indexes multipliers by a
// constraint's position in the enumeration, which is graph insertion order, so
// reordering these blocks re-seats every carried multiplier onto the wrong
// constraint. tests/core/test_constraint_tags.py guards it.
//
// The emission order is: the kinematics' own factors, the shared wrist prior,
// object contact, digit-object collision, digit-digit collision, the support
// plane, then pre-grasp positioning.
//
// Nothing below knows what kind of mechanism it is posing. Every constraint
// reaches the hand through kin_->site_pose_key({digit, node}), and the node
// indices come from the EnvironmentConfig the caller wrote -- so a hand built
// from separate digits and one defined as a single whole mechanism are handled
// by the same code.

#include "gepetto_solvers/hand/HandModel.h"

#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/slam/PriorFactor.h>

#include <openvdb/tools/Interpolation.h>

#include <cmath>
#include <optional>
#include <stdexcept>

using namespace gtsam;
using gepetto_solvers::HandSite;


NonlinearFactorGraph HandModel::build_graph(
    const std::vector<VectorXGaussian>& actuation,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    const int n_digits = num_digits();

    NonlinearFactorGraph graph;
    // Rebuilt from scratch alongside the graph: the tags describe THIS graph's
    // constraints, and a stale entry would pair a multiplier with the wrong one.
    tagger_.clear();

    // The mechanism itself. Whatever factors this hand's kinematics is made of
    // -- rod and tendon factors, joint and linkage factors -- enter here, and
    // any hard constraint among them is numbered by the same tagger the task
    // constraints below use, so the two halves share one enumeration.
    kin_->add_kinematics_factors(graph, tagger_, actuation, tip_wrenches);

    // Single shared floating-wrist prior (rigidly anchors the gauge). In a
    // trajectory only the start step emits it (as the loose hand-pose prior,
    // Eq 1.40); interior/terminal wrists are pinned by the GP chain + kinematics.
    if (emit_wrist_prior_)
        graph.add(PriorFactor<Pose3>(wrist_key(step_), wrist_pose_, wrist_noise_));

    // Per-digit tip contact against one shared object. The object pose is
    // anchored once; each digit drives its own witness point onto the surface.
    // A collision-only env (collision spheres but no target_contact_node) adds no
    // contact factor here -- the collision blocks below handle it (and anchor the
    // shared object if no contacting digit did).
    bool object_anchored = false;
    for (int i = 0; i < n_digits; ++i) {
        if (!env_[i] && !sphere_contacts_[i]) continue;

        if (env_[i]) {
            const auto& env = *env_[i];
            if (!env.target_contact_node.has_value()) continue;  // collision-only
            Key tip_key = kin_->site_pose_key({i, *env.target_contact_node});
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(env.object_pose_mean),
                    noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                object_anchored = true;
            }
            // Center-direct object contact (Eq 1.101): constrain the tip sphere
            // CENTER to the ellipsoid, with no witness point at all.
            // EllipsoidCollisionGapFactor's residual is r - d(x); as an
            // EQUALITY the sign is irrelevant, so its zero set is exactly
            // Eq 1.101. Dropping the witness removes three variables and four
            // residual rows per digit. This is the DEFAULT form for an analytic
            // ellipsoid, not just the controller's phase 2, and the ONLY form for
            // an ellipsoid set -- see uses_center_direct_contact().
            if (uses_center_direct_contact(env)) {
                gtsam::NoiseModelFactor::shared_ptr center_contact;
                std::string tag;
                if (env.object_contact_in_plane) {
                    // Eq 13: the same center-direct equality, with the distance
                    // measured inside the digit's pulling plane (Eq 11) instead
                    // of in 3D. A single analytic ellipsoid goes in as a
                    // one-member set -- K=1 with an identity local_pose reduces
                    // exactly to it, so there is one code path rather than two.
                    //
                    // The plane's third point, the digit's mounting base, is
                    // taken from the kinematics rather than from the env: it IS
                    // this digit's attachment offset, so reading it there makes
                    // it impossible for the plane to be built about a base the
                    // digit is not actually on.
                    std::vector<gepetto_solvers::EllipsoidPrimitive> members;
                    if (!env.ellipsoid_set.empty()) {
                        // Narrowed by contact_ellipsoid_subset when one is set;
                        // the free spheres below keep the whole union.
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
                            kin_->digit_base_offset(i).translation(),
                            gtsam::Point3(*env.contact_plane_centroid),
                            noiseModel::Isotropic::Sigma(1, 1.0),
                            env.contact_plane_rho_lo, env.contact_plane_rho_hi,
                            env.contact_plane_gap_lo, env.contact_plane_gap_hi,
                            env.ellipsoid_taubin);
                    // Its own tag, and that matters twice over:
                    // get_factor_error_summary() tells the two forms apart, and
                    // the AL dual transfer matches by tag -- so switching forms
                    // correctly DROPS the old multipliers instead of carrying
                    // them onto a constraint that means something else.
                    tag = "obj.planar|f" + std::to_string(i);
                } else if (!env.ellipsoid_set.empty()) {
                    // Ellipsoid SET (§1.2, Eq 1.13): the same center-direct
                    // equality against the smooth-min distance to the union. Its
                    // own tag, so get_factor_error_summary() tells the two
                    // surface kinds apart.
                    //
                    // The union here is the CONTACT one -- narrowed by
                    // contact_ellipsoid_subset to the shells that are grasp
                    // targets, where the collision inequality below keeps every
                    // member. The tag does not encode the subset, so a warm start
                    // would carry duals across a change of it; callers that
                    // change the subset should reset them, the same as for a
                    // change of object.
                    center_contact =
                        std::make_shared<gepetto_solvers::EllipsoidSetCollisionGapFactor>(
                            tip_key, object_key(), env.contact_node_radius,
                            gepetto_solvers::contact_ellipsoid_members(env),
                            env.ellipsoid_set_beta,
                            noiseModel::Isotropic::Sigma(1, 1.0),
                            env.ellipsoid_taubin);
                    tag = "obj.set|f" + std::to_string(i);
                } else {
                    center_contact =
                        std::make_shared<gepetto_solvers::EllipsoidCollisionGapFactor>(
                            tip_key, object_key(), env.contact_node_radius,
                            env.ellipsoid_semi_axes,
                            noiseModel::Isotropic::Sigma(1, 1.0),
                            env.ellipsoid_taubin);
                    tag = "obj.center|f" + std::to_string(i);
                }
                tagger_.add_eq(graph, center_contact, tag);
                continue;
            }

            // Witness-point contact. drop_normal_row (controller phase 3,
            // Eq 1.107-1.110) omits c_N, so the residual -- and the noise model
            // that must match it -- is 4 rows instead of 5.
            const bool drop_n = env.contact_drop_normal_row;
            const int  n_rows = drop_n ? 4 : 5;
            gtsam::NoiseModelFactor::shared_ptr contact;
            // object_contact_exact is tested FIRST, ahead of the surface
            // precedence: it exists precisely to contact the grid while a proxy
            // ellipsoid stays attached for the collision blocks, so the
            // precedence -- which would pick that proxy -- must not run here.
            // uses_center_direct_contact() has already rejected the flag without
            // a grid, so sdf_grid is known good.
            if (env.object_contact_exact) {
                contact = std::make_shared<gepetto_solvers::SdfWitnessContactFactor>(
                    tip_key, object_key(), witness_key(i),
                    env.contact_node_radius, env.sdf_grid,
                    noiseModel::Isotropic::Sigma(n_rows, 1.0), drop_n);
            } else if (env.ellipsoid_semi_axes.norm() > 0.0) {
                // Analytic hyper-ellipsoid surface (Section 1.6.3).
                contact = std::make_shared<gepetto_solvers::EllipsoidWitnessContactFactor>(
                    tip_key, object_key(), witness_key(i),
                    env.contact_node_radius, env.ellipsoid_semi_axes,
                    noiseModel::Isotropic::Sigma(n_rows, 1.0), drop_n,
                    env.ellipsoid_taubin);
            } else {
                contact = std::make_shared<gepetto_solvers::SdfWitnessContactFactor>(
                    tip_key, object_key(), witness_key(i),
                    env.contact_node_radius, env.sdf_grid,
                    noiseModel::Isotropic::Sigma(n_rows, 1.0), drop_n);
            }
            tagger_.add_eq(graph, contact, "obj.witness|f" + std::to_string(i));

            // Soft witness target (Section 1.8, Eq 1.111). Because the AL
            // equality constraints pin the witness to the object surface
            // manifold, this prior acts as a geodesic pull that slides the
            // witness -- and the attached digit -- along the surface toward a
            // nominal grasp location. Unset => "contact anywhere" (Eq 1.119).
            if (env.witness_target) {
                graph.add(PriorFactor<Point3>(
                    witness_key(i), *env.witness_target,
                    noiseModel::Gaussian::Covariance(env.witness_target_cov)));
            }
        } else {
            const auto& sc = *sphere_contacts_[i];
            Key digit_key = kin_->site_pose_key({i, sc.finger_node_index});
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(Rot3(), sc.sphere_center),
                    noiseModel::Gaussian::Covariance(sc.sphere_pose_cov)));
                object_anchored = true;
            }
            if (sc.witness) {
                auto contact = std::make_shared<gepetto_solvers::SphereWitnessContactFactor>(
                    digit_key, object_key(), witness_key(i),
                    sc.finger_node_radius, sc.sphere_radius,
                    noiseModel::Isotropic::Sigma(5, 1.0));
                tagger_.add_eq(graph, contact, "obj.sphwit|f" + std::to_string(i));
            } else {
                auto contact = std::make_shared<gepetto_solvers::SphereSphereContactFactor>(
                    digit_key, object_key(),
                    sc.finger_node_radius, sc.sphere_radius,
                    noiseModel::Isotropic::Sigma(1, 1.0));
                tagger_.add_eq(graph, contact, "obj.sphere|f" + std::to_string(i));
            }
        }
    }

    // --- Collision avoidance (Section 1.5) -------------------------------
    // Gather each digit's collision spheres once (world-frame pose key, radius,
    // proximal flag). A site that resolves to the shared wrist/root variable has
    // no free pose of its own (is_root); it is excluded from BOTH collision
    // passes, because the gap factors read the key's translation, which for the
    // root key is the wrist origin -- not the place the site names -- and such a
    // site is rigidly placed by the wrist anyway.
    // is_contact: the site that also carries the terminal witness contact factor
    // for this digit (only when this model has target_contact_node, i.e. the
    // terminal step). Its object-collision sphere is skipped so it can't oppose
    // the contact factor -- but it is KEPT for digit-digit collision (the
    // contacting tip must still avoid other digits).
    //
    // The sphere SET is gathered for ANY of its three consumers -- digit-object
    // (needs collision_avoidance and an object surface), the support plane (needs
    // only plane_avoidance) or digit-digit (needs only self_collision). Which
    // constraints are actually built is then decided per family, each on its own
    // field: the digit-object loop below re-checks its own gate, and the
    // digit-digit loop re-checks self_collision.
    // `node` is carried only to name the sphere in a constraint tag: the KEY is
    // model-local (global counter) and useless across a rebuild, while the node
    // index is the same sphere in any model built from the same configs.
    struct CollSphere {
        Key key; double radius; bool proximal; bool is_root; bool is_contact;
        int node;
    };
    std::vector<std::vector<CollSphere>> digit_spheres(n_digits);
    for (int i = 0; i < n_digits; ++i) {
        if (!env_[i]) continue;
        const auto& env = *env_[i];
        bool wants_object_collision =
            env.collision_avoidance && gepetto_solvers::has_object_surface(env);
        bool wants_plane_collision  = env.plane_avoidance &&
                                      env.plane_normal.norm() > 0.0;
        bool wants_self_collision   = env.self_collision;
        if ((!wants_object_collision && !wants_plane_collision &&
             !wants_self_collision) ||
            env.collision_node_indices.empty())
            continue;
        for (size_t j = 0; j < env.collision_node_indices.size(); ++j) {
            int idx = env.collision_node_indices[j];
            bool is_root = kin_->site_is_root({i, idx});
            Key key = kin_->site_pose_key({i, idx});
            double r = env.collision_node_radii[j];
            bool prox = (j < env.collision_node_is_proximal.size())
                            ? (env.collision_node_is_proximal[j] != 0) : false;
            bool is_contact = env.target_contact_node.has_value() &&
                key == kin_->site_pose_key({i, *env.target_contact_node});
            digit_spheres[i].push_back({key, r, prox, is_root, is_contact, idx});
        }
    }

    // Digit-object: keep every collision sphere out of the shared object SDF,
    // except the terminal contact site (its collision would oppose the contact
    // factor; the contact factor already pins it tangent to the surface).
    for (int i = 0; i < n_digits; ++i) {
        if (digit_spheres[i].empty()) continue;
        const auto& env = *env_[i];
        // Spheres may have been gathered for the support plane alone; only an env
        // that actually asked for OBJECT avoidance builds these. This predicate
        // must stay identical to the has_col guard in get_initial_values(), which
        // decides whether the shared object variable is seeded at all -- anchoring
        // an object that was never seeded is an indeterminate system. Both call
        // has_object_surface() so they cannot drift apart.
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
        for (const auto& s : digit_spheres[i]) {
            if (s.is_contact || s.is_root) continue;
            gtsam::NoiseModelFactor::shared_ptr gap;
            if (is_set) {
                // Eq 1.12: the same residual as the contact equality above, read
                // as an inequality c_pen = r - d_E <= 0.
                gap = std::make_shared<gepetto_solvers::EllipsoidSetCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.ellipsoid_set,
                    env.ellipsoid_set_beta, col_noise, env.ellipsoid_taubin);
            } else if (is_ellipsoid) {
                gap = std::make_shared<gepetto_solvers::EllipsoidCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.ellipsoid_semi_axes, col_noise,
                    env.ellipsoid_taubin);
            } else {
                gap = std::make_shared<gepetto_solvers::SdfCollisionGapFactor>(
                    s.key, object_key(), s.radius, env.sdf_grid, col_noise);
            }
            tagger_.add_ineq(graph, gap, "col.obj|f" + std::to_string(i) +
                                         "|n" + std::to_string(s.node));
        }
    }

    // Digit-digit: keep collision spheres of distinct digits apart. Built for a
    // pair iff BOTH its digits asked for self_collision -- the constraint belongs
    // to the two of them jointly, so one digit opting out drops every pair it is
    // part of. Skip a pair iff BOTH spheres are proximal (rigidly-attached base
    // bones), and skip any root sphere (no distinct per-digit pose variable).
    // This leaves distal-distal and distal-proximal cross-digit pairs.
    // When collision_cull_margin >= 0, additionally skip pairs whose gap at the
    // initial values already exceeds the margin (see the caveats on
    // EnvironmentConfig::collision_cull_margin -- heuristic, verified by the
    // tests' independent all-pairs penetration report).
    std::optional<Values> init_vals;
    auto initial_position = [&](Key key) {
        if (!init_vals) init_vals = get_initial_values();
        return init_vals->at<Pose3>(key).translation();
    };
    for (int a = 0; a < n_digits; ++a) {
        if (digit_spheres[a].empty()) continue;
        if (!env_[a]->self_collision) continue;
        auto col_noise = noiseModel::Isotropic::Sigma(
            1, env_[a]->collision_sigma);
        const double cull_margin = env_[a]->collision_cull_margin;
        for (int b = a + 1; b < n_digits; ++b) {
            if (digit_spheres[b].empty()) continue;
            if (!env_[b]->self_collision) continue;
            for (const auto& sa : digit_spheres[a]) {
                if (sa.is_root) continue;
                for (const auto& sb : digit_spheres[b]) {
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
                    tagger_.add_ineq(graph, gap,
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
    // digit_spheres -- which carries the root/contact exclusions the OBJECT
    // constraints want, not the (different) ones the plane wants.
    for (int i = 0; i < n_digits; ++i) {
        if (!env_[i]) continue;
        const auto& env = *env_[i];
        if (env.plane_normal.norm() <= 0.0) continue;   // no table configured

        // Table sliding equality: place the contact site's sphere on the plane
        // and let it slide, as ONE residual on the sphere CENTER --
        // c_table(c) = Dist_plane(c) = 0 -- with no witness point at all.
        // PlaneCollisionGapFactor's residual is r - (c - p).n; as an EQUALITY the
        // sign is irrelevant, so its zero set is exactly that, in the signed form
        // (pins the sphere on the +n free side and stays smooth at the contact
        // point, where the paper's |.| has a kink).
        //
        // This replaces the original Eq 1.60-1.64 five-residual witness form. The
        // witness there bought nothing: four of its five rows only pinned the
        // gauge of the free point it introduced, and for a PLANE a single scalar
        // on the center leaves no rotational freedom to brick the solver -- the
        // same argument §1.8 already makes for support_contact_node, whose factor
        // this now matches exactly.
        if (env.table_contact_node.has_value()) {
            Key tip_key = kin_->site_pose_key({i, *env.table_contact_node});
            auto contact = std::make_shared<gepetto_solvers::PlaneCollisionGapFactor>(
                tip_key, env.table_contact_radius,
                env.plane_origin, env.plane_normal,
                noiseModel::Isotropic::Sigma(1, 1.0));
            tagger_.add_eq(graph, contact, "tbl.contact|f" + std::to_string(i));
        }

        // --- Section 1.8 controller phases 1-2 ---------------------------
        // Support-surface contact EQUALITY on the sphere CENTER (Eq 1.97), the
        // witness-free counterpart of the §1.6 sliding equality above.
        // PlaneCollisionGapFactor's residual is r - (c - p).n; as an EQUALITY the
        // sign is irrelevant, so its zero set is exactly Dist_plane = 0 -- in the
        // signed form, which pins the sphere on the +n (free) side and stays
        // smooth at the contact point (the paper's |.| has a kink exactly where
        // the solver operates). Kept for the dormant phased controller use case;
        // independent of the table_contact_node equality above.
        if (env.support_contact_node.has_value()) {
            Key sup_key = kin_->site_pose_key({i, *env.support_contact_node});
            auto support = std::make_shared<gepetto_solvers::PlaneCollisionGapFactor>(
                sup_key, env.support_contact_radius,
                env.plane_origin, env.plane_normal,
                noiseModel::Isotropic::Sigma(1, 1.0));
            tagger_.add_eq(graph, support, "sup.contact|f" + std::to_string(i));
        }

        // Table collision (Eq 1.59): keep every non-root collision sphere out of
        // the half-space, except the table contact site (its collision would
        // oppose the sliding equality that already pins it to the plane). The
        // same exclusion applies to support_contact_node: §1.8 warns that
        // stacking the inequality on a sphere that already carries the equality
        // gives linearly dependent Jacobians once contact is met, artificially
        // rank-deficient Hessian, destabilized inner LM.
        if (env.plane_avoidance && !env.collision_node_indices.empty()) {
            auto col_noise = noiseModel::Isotropic::Sigma(1, env.collision_sigma);
            for (size_t j = 0; j < env.collision_node_indices.size(); ++j) {
                int idx = env.collision_node_indices[j];
                if (kin_->site_is_root({i, idx})) continue;
                const Key node_key = kin_->site_pose_key({i, idx});
                if (env.table_contact_node.has_value() &&
                    node_key == kin_->site_pose_key({i, *env.table_contact_node}))
                    continue;
                if (env.support_contact_node.has_value() &&
                    node_key == kin_->site_pose_key({i, *env.support_contact_node}))
                    continue;
                double r = env.collision_node_radii[j];
                auto gap = std::make_shared<gepetto_solvers::PlaneCollisionGapFactor>(
                    node_key, r,
                    env.plane_origin, env.plane_normal, col_noise);
                tagger_.add_ineq(graph, gap, "col.plane|f" + std::to_string(i) +
                                             "|n" + std::to_string(idx));
            }
        }
    }

    // --- Opposition half-space (Eq 2.16-2.17 / Eq 1.92) ------------------
    // Keep each participating digit's sphere center on its designated half of the
    // splitting line, so the opposing digit lands opposite the rest. A pass of
    // its OWN, gated on nothing but its own fields: the residual is a statement
    // about one site's position relative to a line (constant Jacobian), and needs
    // neither a support plane nor a contact node of any kind. It used to live
    // inside the table-contact branch above, which silently made checking
    // "opposition" alone build nothing at all.
    //
    // half_space_node is this constraint's own opt-in field; table_contact_node
    // is the fallback for a caller still writing the env the old way.
    for (int i = 0; i < n_digits; ++i) {
        if (!env_[i]) continue;
        const auto& env = *env_[i];
        if (!env.half_space_enabled || env.half_space_normal.norm() <= 0.0)
            continue;
        std::optional<int> node = env.half_space_node.has_value()
                                      ? env.half_space_node
                                      : env.table_contact_node;
        if (!node.has_value()) continue;
        auto half = std::make_shared<gepetto_solvers::HalfSpaceGapFactor>(
            kin_->site_pose_key({i, *node}),
            env.half_space_split_point, env.half_space_normal,
            noiseModel::Isotropic::Sigma(1, env.collision_sigma),
            env.half_space_margin);
        // Tag unchanged from when this lived in the table block: the AL dual
        // transfer matches constraints by tag, so renaming it here would silently
        // break every warm start across a rebuild.
        tagger_.add_ineq(graph, half, "half|f" + std::to_string(i));
    }

    // Pre-grasp hand-centering (Eq 2.18-2.19): spans the opposing digit + every
    // other participating digit, so -- unlike every block above -- this collects
    // across ALL digits first rather than building inside the per-digit body.
    //
    // The opposing digit is an INDEX the hand declared (HandSpec::opposing_digit),
    // not a name matched against the literal "thumb": a hand with two opposable
    // digits, or one whose digits are numbered, has no such string to match. A
    // hand that declares none (-1) never builds this constraint, because there is
    // no opposition to center about.
    {
        std::optional<Key> opposing_key;
        std::vector<Key> other_keys;
        double h_clear = 0.0;
        gtsam::Vector3 n_hat = gtsam::Vector3::Zero();
        for (int i = 0; i < n_digits; ++i) {
            if (!env_[i]) continue;
            const auto& env = *env_[i];
            if (!env.pregrasp_center_node.has_value()) continue;
            Key k = kin_->site_pose_key({i, *env.pregrasp_center_node});
            if (i == opposing_digit_) opposing_key = k;
            else other_keys.push_back(k);
            h_clear = env.pregrasp_clearance_height;
            n_hat = env.pregrasp_clearance_normal;
            // Anchor the shared object pose if nothing else has yet -- this
            // constraint touches object_key() too, but only its translation, so
            // it cannot by itself fix the object's orientation gauge.
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(env.object_pose_mean),
                    noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                object_anchored = true;
            }
        }
        if (opposing_key.has_value() && !other_keys.empty() && n_hat.norm() > 0.0) {
            auto center = std::make_shared<gepetto_solvers::PreGraspHandCenteringFactor>(
                *opposing_key, other_keys, object_key(), h_clear, n_hat,
                noiseModel::Isotropic::Sigma(3, 1.0));
            tagger_.add_eq(graph, center, "pregrasp.center");
        }
    }

    // Pre-grasp short-axis alignment (companion to Eq 2.16-2.17): also a
    // hand-level pass, collected the same way as the hand-centering block above,
    // but on the SEPARATE pregrasp_align_node/pregrasp_align_axis fields so it
    // stays independently toggleable. No object-pose anchoring needed here --
    // PreGraspAxisAlignmentFactor never touches object_key().
    {
        std::optional<Key> opposing_key;
        std::vector<Key> other_keys;
        gtsam::Vector3 axis = gtsam::Vector3::Zero();
        for (int i = 0; i < n_digits; ++i) {
            if (!env_[i]) continue;
            const auto& env = *env_[i];
            if (!env.pregrasp_align_node.has_value()) continue;
            Key k = kin_->site_pose_key({i, *env.pregrasp_align_node});
            if (i == opposing_digit_) opposing_key = k;
            else other_keys.push_back(k);
            axis = env.pregrasp_align_axis;
        }
        if (opposing_key.has_value() && !other_keys.empty() && axis.norm() > 0.0) {
            auto align = std::make_shared<gepetto_solvers::PreGraspAxisAlignmentFactor>(
                *opposing_key, other_keys, axis,
                noiseModel::Isotropic::Sigma(1, 1.0));
            tagger_.add_eq(graph, align, "pregrasp.align");
        }
    }

    // Pre-grasp PINCH-CENTROID centering: the hardcoded-point sibling of the
    // hand-centering block above. Hand-level like the other two, but no digit
    // opts in -- the point is a constant in the WRIST frame, so the factor keys
    // off wrist_key() directly and the fields are simply duplicated across every
    // digit's env (first one found wins, matching how h_clear/n_hat are read
    // above).
    {
        std::optional<gtsam::Vector3> centroid;
        double h_clear = 0.0;
        gtsam::Vector3 n_hat = gtsam::Vector3::Zero();
        for (int i = 0; i < n_digits; ++i) {
            if (!env_[i]) continue;
            const auto& env = *env_[i];
            if (!env.pregrasp_centroid_point.has_value()) continue;
            centroid = *env.pregrasp_centroid_point;
            h_clear = env.pregrasp_centroid_clearance;
            n_hat = env.pregrasp_centroid_normal;
            // Same reasoning as the hand-centering block: this constraint reads
            // object_key()'s translation, so the object needs an anchor if no
            // other block has supplied one.
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
            tagger_.add_eq(graph, pinch, "pregrasp.centroid");
        }
    }

    // --- Geometric grasp alignment (h_grasp) -----------------------------
    // The contacts must SURROUND the object, not merely touch it. One Vector6
    // equality over every contacting digit's witness point at once:
    //
    //   sum_i [ -n_i ; -(p_i - t_obj) x n_i ] = 0
    //
    // Hand-level, so it is collected here in a pass of its own like the three
    // pre-grasp blocks above rather than built inside the per-digit body. It is
    // the piece the per-contact witness factors cannot express, because each of
    // those sees exactly one contact.
    //
    // Its POSITION in the emission order is load-bearing and must not move: the
    // AL indexes multipliers by graph position, so shifting this block would
    // re-seat every carried multiplier of every solve that predates it. Only the
    // ellipsoid block below it may be appended after. See the header note at the
    // top of this file.
    {
        std::vector<Key> point_keys;
        openvdb::FloatGrid::Ptr grid;
        double sigma_f = 1.0, sigma_t = 1.0, curv_step = 0.0, grad_step = 0.0;
        for (int i = 0; i < n_digits; ++i) {
            if (!env_[i]) continue;
            const auto& env = *env_[i];
            if (!env.grasp_alignment_enabled) continue;
            if (!env.target_contact_node.has_value()) continue;  // collision-only
            // A witness point is the variable this factor keys off, so a digit on
            // a center-direct form has nothing to contribute -- and cannot be
            // quietly dropped, because the caller asked for a wrench balance over
            // a contact set that would then be missing a member. Same reasoning
            // as uses_center_direct_contact's own throws.
            if (uses_center_direct_contact(env))
                throw std::invalid_argument(
                    "HandModel: grasp_alignment_enabled needs a WITNESS-point "
                    "contact to key off, but digit " + std::to_string(i) +
                    " is on the center-direct form. Set object_contact_exact (or "
                    "contact a baked SDF) on every participating digit.");
            if (!env.sdf_grid)
                throw std::invalid_argument(
                    "HandModel: grasp_alignment_enabled reads the object's own "
                    "surface normal at each witness point, which needs sdf_grid; "
                    "digit " + std::to_string(i) + " has none. Bake one "
                    "(scripts/objects/setup_objects.py).");
            point_keys.push_back(witness_key(i));
            grid      = env.sdf_grid;
            sigma_f   = env.grasp_alignment_sigma_force;
            sigma_t   = env.grasp_alignment_sigma_torque;
            curv_step = env.grasp_alignment_curvature_step;
            grad_step = env.grasp_alignment_gradient_step;
            // This factor reads the object pose, so it needs an anchor if no
            // block above supplied one -- same reasoning as the hand-centering
            // block. In practice a contacting digit has always anchored it by
            // now (this constraint requires one), but the guard is free and the
            // alternative is an indeterminate system.
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(env.object_pose_mean),
                    noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                object_anchored = true;
            }
        }
        // Fewer than two contacts is skipped rather than thrown: a single unit
        // force cannot sum to zero against anything, so the constraint is not
        // merely hard but unsatisfiable -- and WHICH digits contact is the
        // caller's standing selection (contact_fingers), not a mis-request about
        // this constraint. Throwing would make unchecking a finger blow up a
        // solve that is otherwise perfectly well posed.
        if (point_keys.size() >= 2) {
            // Two sigmas, not one: the top three rows are a sum of unit normals
            // (dimensionless) and the bottom three a sum of moment arms (metres),
            // so an isotropic model would weight the halves by whatever the
            // object's size happens to be. See GraspAlignmentFactor's header.
            Vector6 sigmas;
            sigmas << sigma_f, sigma_f, sigma_f, sigma_t, sigma_t, sigma_t;
            auto grasp = std::make_shared<gepetto_solvers::GraspAlignmentFactor>(
                point_keys, object_key(), grid,
                noiseModel::Diagonal::Sigmas(sigmas), curv_step, grad_step);
            tagger_.add_eq(graph, grasp, "grasp.align");
        }
    }

    // --- Approximate geometric grasp alignment (h_grasp,E) ---------------
    // The same Vector6 wrench equilibrium as the block above, but keyed off the
    // contact sphere CENTERS and reading the normal from the analytic ellipsoid
    // set rather than the baked SDF:
    //
    //   sum_i [ -n_i ; -(c_i - t_obj) x n_i ] = 0
    //
    // A separate constraint, not a mode of the one above, and the two may both
    // be on. The block above REFUSES a digit on the center-direct contact form,
    // because a witness point is the variable it keys off; this one is built for
    // exactly that case, so the wrench balance is available during the
    // approximation phase -- before any exact contact exists. The sphere radius
    // cancels analytically out of the torque, which is what lets it key off the
    // center with no witness variable at all.
    //
    // LAST in the emission order, and it must stay last: the AL indexes
    // multipliers by graph position, so inserting this anywhere but the end would
    // re-seat every carried multiplier of every solve that predates it -- which
    // is also why it was appended HERE rather than beside its sibling. See the
    // header note at the top of this file.
    {
        std::vector<Key> center_keys;
        std::vector<gepetto_solvers::EllipsoidPrimitive> members;
        double sigma_f = 1.0, sigma_t = 1.0, beta = 1000.0, curv_step = 0.0;
        bool taubin = false;
        for (int i = 0; i < n_digits; ++i) {
            if (!env_[i]) continue;
            const auto& env = *env_[i];
            if (!env.ellipsoid_grasp_alignment_enabled) continue;
            if (!env.target_contact_node.has_value()) continue;  // collision-only
            // This constraint reads the object's ANALYTIC surface normal, so it
            // needs an ellipsoid surface and cannot fall back to a baked grid the
            // way the contact block can -- the whole point of it is to run on the
            // smooth proxy. Silently substituting the SDF would build a different
            // constraint than the caller asked for, so raise like the sibling.
            if (env.ellipsoid_set.empty() && !(env.ellipsoid_semi_axes.norm() > 0.0))
                throw std::invalid_argument(
                    "HandModel: ellipsoid_grasp_alignment_enabled reads the "
                    "object's ellipsoid surface normal at each sphere center, "
                    "which needs ellipsoid_set or ellipsoid_semi_axes; digit " +
                    std::to_string(i) + " has neither. Fit one "
                    "(scripts/objects/setup_objects.py), or use the SDF form "
                    "(grasp_alignment_enabled) instead.");
            center_keys.push_back(kin_->site_pose_key({i, *env.target_contact_node}));
            // The FULL set, never contact_ellipsoid_subset: the normal field is a
            // property of the object's geometry, not of which shells this finger
            // was cleared to touch. Same choice EllipsoidSetCollisionGapFactor
            // makes, and for the same reason.
            members = !env.ellipsoid_set.empty()
                ? env.ellipsoid_set
                : std::vector<gepetto_solvers::EllipsoidPrimitive>{
                      gepetto_solvers::EllipsoidPrimitive{env.ellipsoid_semi_axes,
                                                          Pose3()}};
            beta      = env.ellipsoid_set_beta;
            taubin    = env.ellipsoid_taubin;
            sigma_f   = env.ellipsoid_grasp_alignment_sigma_force;
            sigma_t   = env.ellipsoid_grasp_alignment_sigma_torque;
            curv_step = env.ellipsoid_grasp_alignment_curvature_step;
            // Reads the object pose, so it needs an anchor if no block above
            // supplied one -- same reasoning as the hand-centering block.
            if (!object_anchored) {
                graph.add(PriorFactor<Pose3>(
                    object_key(), Pose3(env.object_pose_mean),
                    noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                object_anchored = true;
            }
        }
        // Fewer than two contacts is skipped rather than thrown, for the reason
        // spelled out on the sibling: a single unit force cannot sum to zero, and
        // WHICH digits contact is the caller's standing selection.
        if (center_keys.size() >= 2) {
            Vector6 sigmas;
            sigmas << sigma_f, sigma_f, sigma_f, sigma_t, sigma_t, sigma_t;
            auto grasp = std::make_shared<gepetto_solvers::EllipsoidGraspAlignmentFactor>(
                center_keys, object_key(), members, beta,
                noiseModel::Diagonal::Sigmas(sigmas), taubin, curv_step);
            tagger_.add_eq(graph, grasp, "grasp.align.ell");
        }
    }

    return graph;
}
