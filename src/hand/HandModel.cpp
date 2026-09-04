// HandModel: construction and the contact-form predicate.
//
// Construction does two things and no more: load the kinematics named by the
// spec, and read the per-digit task environment. Everything that used to build
// tendon fingers here now lives behind HandKinematics.

#include "gepetto_solvers/hand/HandModel.h"

#include "gepetto_solvers/hand/HandKinematicsRegistry.h"
#include "gepetto_solvers/utils/MiscInline.h"

#include <stdexcept>

using namespace gtsam;


HandModel::HandModel(
    const gepetto_solvers::HandSpec& spec,
    const Pose3& wrist_pose,
    SharedDiagonal wrist_noise,
    int step,
    bool emit_wrist_prior)
:
    digit_names_(spec.digit_names),
    opposing_digit_(spec.opposing_digit),
    env_(spec.env),
    sphere_contacts_(spec.sphere_contact),
    wrist_pose_(wrist_pose),
    wrist_noise_(wrist_noise),
    step_(step),
    emit_wrist_prior_(emit_wrist_prior)
{
    kin_ = gepetto_solvers::load_hand_kinematics(spec, wrist_pose_, wrist_key(step_));

    // Which optimizer path this hand needs, decided purely from the task
    // environment -- it was already environment-driven, and now it is visibly
    // so: nothing below asks the kinematics anything.
    for (size_t i = 0; i < env_.size(); ++i) {
        const auto& e = env_[i];

        // Contact => a terminal witness contact factor (target_contact_node for
        // the SDF path, or any sphere_contact). A collision-only env (collision
        // spheres but no target_contact_node) is NOT contact -- it routes through
        // has_collision() instead, so guard on target_contact_node here.
        if ((e.has_value() && e->target_contact_node.has_value())
            || sphere_contacts_[i].has_value())
            has_contact_ = true;

        if (!e.has_value()) continue;

        // Collision => AL inequality constraints on this digit's spheres.
        // The object surface may be a baked SDF grid, an analytic ellipsoid
        // (Section 1.6.3) or an ellipsoid SET (Section 1.2); each provides the
        // digit-object penetration gap.
        if (e->collision_avoidance &&
            gepetto_solvers::has_object_surface(*e) &&
            !e->collision_node_indices.empty())
            has_collision_ = true;
        // Digit-digit, on the same spheres and gated on its own field. Each
        // clause here must match the corresponding gate in build_graph(): a
        // constraint family that is built but does not flip this flag is solved
        // WITHOUT the Augmented Lagrangian, i.e. built and never enforced.
        if (e->self_collision && !e->collision_node_indices.empty())
            has_collision_ = true;
        // Opposition half-space (Eq 2.16-2.17): an inequality => AL. Standalone,
        // matching build_graph()'s own pass -- it needs neither the support
        // plane nor a contact node of any kind, so a solve whose ONLY constraint
        // is this one still takes the AL path.
        if (e->half_space_enabled &&
            e->half_space_normal.norm() > 0.0 &&
            (e->half_space_node.has_value() ||
             e->table_contact_node.has_value()))
            has_collision_ = true;
        // Approximate geometric grasp alignment (h_grasp,E): a ZeroCostConstraint
        // => AL. Gated on the same pair build_graph() collects on, and it needs
        // its own clause rather than riding on the SDF sibling's: this one runs
        // in the approximation phase, where the object contact may be the
        // center-direct form -- which grasp_alignment_enabled refuses outright.
        if (e->ellipsoid_grasp_alignment_enabled && e->target_contact_node.has_value())
            has_contact_ = true;
        // Support plane (Section 1.6). A configured plane (non-zero normal) with
        // a table_contact_node is a hard equality (sliding contact => AL); with
        // plane_avoidance it is a hard inequality (table collision => AL).
        if (e->plane_normal.norm() > 0.0) {
            if (e->table_contact_node.has_value()) has_contact_   = true;
            if (e->plane_avoidance)                has_collision_ = true;
            // Section 1.8 controller phases 1-2: the support-surface equality is
            // a ZeroCostConstraint => AL. Phase 1 has no OBJECT contact at all,
            // so without this the solve would miss the AL path entirely and the
            // support constraint would silently do nothing.
            if (e->support_contact_node.has_value())
                has_contact_ = true;
        }
    }
}


bool HandModel::uses_center_direct_contact(
    const gepetto_solvers::EnvironmentConfig& env)
{
    // Collision-only env: no object contact of any kind to choose a form for.
    if (!env.target_contact_node.has_value()) return false;

    // Contact the EXACT geometry (phases 3-4), leaving the proxy to the collision
    // blocks. Decided first, ahead of every surface test below, because that is
    // precisely what it means: the caller has said which surface the contact
    // reads, so the precedence that would otherwise pick one for it does not run.
    //
    // A grid can only be contacted through a witness point -- it has no
    // closed-form distance to constrain a center against -- so this is always the
    // witness form, and its two incompatibilities are rejected rather than
    // ignored, on the same reasoning as the ellipsoid_set case below.
    if (env.object_contact_exact) {
        if (!env.sdf_grid)
            throw std::invalid_argument(
                "HandModel: object_contact_exact contacts the baked SDF, but no "
                "sdf_grid is attached -- bake one for this object "
                "(scripts/objects/setup_objects.py), or clear the flag to "
                "contact the ellipsoid proxy instead.");
        if (env.object_contact_in_plane)
            throw std::invalid_argument(
                "HandModel: object_contact_exact and object_contact_in_plane are "
                "two different contact FORMS on two different surfaces (the baked "
                "SDF vs. the ellipsoid proxy's cross-section). Set at most one.");
        return false;
    }

    // In-plane contact (Eq 13) is a center-direct form -- it constrains the tip
    // sphere's CENTER, with no witness variable -- so it is decided here with the
    // others. Its two incompatibilities are rejected rather than ignored, on the
    // same reasoning as the ellipsoid_set case below: silently dropping the
    // caller's request would solve a different problem than the one asked for.
    if (env.object_contact_in_plane) {
        if (env.contact_drop_normal_row || env.witness_target)
            throw std::invalid_argument(
                "HandModel: contact_drop_normal_row / witness_target select a "
                "witness-point contact form, which the in-plane contact (Eq 13) is "
                "not -- it constrains the sphere center against the plane's "
                "cross-section. Clear them, or clear object_contact_in_plane.");
        if (env.ellipsoid_set.empty() && env.ellipsoid_semi_axes.norm() <= 0.0)
            throw std::invalid_argument(
                "HandModel: object_contact_in_plane needs an ellipsoid surface "
                "(ellipsoid_set or ellipsoid_semi_axes) -- a baked SDF has no "
                "closed-form cross-section for the pulling plane to cut.");
        if (!env.contact_plane_centroid.has_value())
            throw std::invalid_argument(
                "HandModel: object_contact_in_plane needs "
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
                "HandModel: contact_drop_normal_row / witness_target select a "
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


Values HandModel::values_from_state(const HandState& state) const {
    Values values;
    kin_->insert_from_state(values, state);
    return values;
}
