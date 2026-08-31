// TendonHandModel: construction, keys, and the tagged constraint inserters.
//
// add_eq/add_ineq are the ONLY way a hard constraint may enter the graph.
// They record a semantic tag beside every one, in insertion order, which is
// the order ConstrainedOptProblem enumerates constraints in -- so tag k names
// the constraint multiplier k belongs to. Adding a constraint any other way
// silently misaligns every tag after it.

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
            gepetto_solvers::has_object_surface(*c.sdf_contact) &&
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
        // `cfg` rather than the structured binding `c` directly: capturing a
        // structured binding in a lambda is C++20, and this builds as C++17
        // (clang takes it as an extension, with a warning).
        const auto& cfg = c;
        std::visit([&](auto& fp) {
            fp->set_hand_base(offset, shared_wrist_key);
            fp->set_emit_base_prior(false);
            // Per-finger, like the rod sigmas above: the planar-bending switch
            // rides on the FINGER config, so a hand can mix keyed and free rods.
            if (cfg.planar_bending)
                fp->set_planar_bending(cfg.sigma_planar_bend, cfg.sigma_planar_twist);
        }, fingers_.back());
    }
}


bool TendonHandModel::uses_center_direct_contact(
    const gepetto_solvers::EnvironmentConfig& env)
{
    // Collision-only env: no object contact of any kind to choose a form for.
    if (!env.target_contact_node.has_value()) return false;

    // In-plane contact (Eq 13) is a center-direct form -- it constrains the tip
    // sphere's CENTER, with no witness variable -- so it is decided here with the
    // others. Its two incompatibilities are rejected rather than ignored, on the
    // same reasoning as the ellipsoid_set case below: silently dropping the
    // caller's request would solve a different problem than the one asked for.
    if (env.object_contact_in_plane) {
        if (env.contact_drop_normal_row || env.witness_target)
            throw std::invalid_argument(
                "TendonHandModel: contact_drop_normal_row / witness_target select a "
                "witness-point contact form, which the in-plane contact (Eq 13) is "
                "not -- it constrains the sphere center against the plane's "
                "cross-section. Clear them, or clear object_contact_in_plane.");
        if (env.ellipsoid_set.empty() && env.ellipsoid_semi_axes.norm() <= 0.0)
            throw std::invalid_argument(
                "TendonHandModel: object_contact_in_plane needs an ellipsoid surface "
                "(ellipsoid_set or ellipsoid_semi_axes) -- a baked SDF has no "
                "closed-form cross-section for the pulling plane to cut.");
        if (!env.contact_plane_centroid.has_value())
            throw std::invalid_argument(
                "TendonHandModel: object_contact_in_plane needs "
                "contact_plane_centroid, the wrist-frame point Eq 11 spans the "
                "pulling plane with; without it the plane is undefined.");
        return true;
    }

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
    graph.add(gepetto_solvers::CollisionInequalityConstraint(gap));
    tags_.ineq.push_back(std::move(tag));
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
