#pragma once

// The routing table: which fields are set decides which factors get built.

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/constrained/NonlinearInequalityConstraint.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gepetto_solvers {

// One member of a hyper-ellipsoid SET (Section 1.2, Eq 1.10).
//
// semi_axes are (a, b, c) in THIS ellipsoid's own frame, so its shape matrix is
// M_k = diag(a^-2, b^-2, c^-2) -- same convention as EllipsoidCollisionGapFactor's
// single ellipsoid. local_pose is T_k, this ellipsoid's CONSTANT pose in the
// OBJECT frame; Eq 1.10's world pose is therefore T_Ek = T_obj o T_k, with T_obj
// the one optimized object Pose3 the rest of this header already shares. The set
// is thus one rigid body made of K primitives -- it moves with the object
// variable, and adds no variables of its own.
//
// Defined here, ahead of EnvironmentConfig, because that config owns a vector of
// these; the factor that consumes them is further down (EllipsoidSetCollisionGapFactor).
struct EllipsoidPrimitive {
    gtsam::Vector3 semi_axes;
    gtsam::Pose3   local_pose = gtsam::Pose3();
};

// Configuration for an OpenVDB-backed environment used by trajectory planners.
// Implements the math in Section 3 of the underactuated object manipulation
// formulation (cubic-polynomial barrier collision + dummy-point surface
// contact).
struct EnvironmentConfig {
    openvdb::FloatGrid::Ptr sdf_grid;
    gtsam::Matrix4 object_pose_mean = gtsam::Matrix4::Identity();
    gtsam::Matrix6 object_pose_cov  = 1e-8 * gtsam::Matrix6::Identity();
    bool object_pose_per_step = false;

    // Collision avoidance (Section 1.5). The finger nodes listed in
    // collision_node_indices (with radii collision_node_radii) are kept out of
    // the object by sphere-to-SDF inequality constraints c_pen <= 0, built from
    // SdfCollisionGapFactor wrapped in a CollisionInequalityConstraint and
    // driven to feasibility by the Augmented Lagrangian optimizer.
    // collision_sigma scales the constraint rows (1.0 = same whitening as the
    // contact constraint rows).
    //
    // collision_avoidance switches the finger-OBJECT inequalities on and off
    // (default true), at every trajectory step and in the hand. It does NOT
    // govern the other two consumers of the same collision-sphere set: the
    // support plane has plane_avoidance, and finger-finger has self_collision
    // below. Each constraint family is gated on its own field alone, so a
    // caller can ask for any combination of the three.
    //
    // self_collision (default true) switches the FINGER-FINGER pairs on and
    // off -- cross-finger sphere pairs in the hand, SphereSphereCollisionGap-
    // Factor. Only HandModel builds these (a single finger cannot
    // collide with another), so it is inert for the single-finger solvers.
    // Needs collision_node_indices to be non-empty like the other two, since
    // the spheres are what it constrains.
    //
    // collision_node_is_proximal is a parallel vector to collision_node_indices
    // (1 = proximal, 0 = distal). It only matters for finger-finger collision in
    // the hand: a sphere pair is skipped iff BOTH spheres are proximal (the
    // metacarpal/base bones are rigidly attached to the shared hand base, so a
    // constraint between them is constant and useless). An empty vector means
    // every node is treated as non-proximal (no pair excluded).
    bool collision_avoidance = true;
    bool self_collision      = true;
    double collision_sigma   = 1e-3;
    std::vector<int>    collision_node_indices;
    std::vector<double> collision_node_radii;
    std::vector<int>    collision_node_is_proximal;

    // Finger-finger pair culling (>= 0 enables). A cross-finger sphere pair is
    // skipped when its gap at the INITIAL values exceeds this margin (m).
    // Profiling showed ~half the 5-finger trajectory graph is inequality
    // constraints that stay inactive (error ~1e-16) through the whole solve,
    // yet each one adds three factors (penalty + BiasedFactor + AntiFactor) to
    // every AL merit graph. Heuristic, not sound: fingers curling toward a
    // shared object roughly preserve their lateral separation, but a culled
    // pair is unprotected if the solve does bring it together — pick a margin
    // comfortably above the expected relative motion, and rely on the tests'
    // independent all-pairs penetration report to catch a cull that was too
    // aggressive. Finger-OBJECT constraints are never culled: every step's
    // initial guess starts far from the object by construction, so an
    // initial-gap cull would strip exactly the protection the trajectory
    // needs. Disabled (< 0) by default.
    double collision_cull_margin = -1.0;

    // Contact-as-goal terminal constraint (Eq 33-35). When target_contact_node
    // is set, the planner adds a hard equality constraint on that node — the
    // 5-residual SdfWitnessContactFactor [c_R, c_O, c_N, c_T1, c_T2] wrapped in
    // a gtsam::ZeroCostConstraint — and solves with the Augmented Lagrangian
    // optimizer, which drives all five residuals exactly to zero. Convergence
    // is governed by the AL parameters on SolverBaseConfig, not a covariance.
    std::optional<int> target_contact_node;
    double contact_node_radius = 0.0;

    // Optional explicit initial value for this contact's witness point, given
    // in the object-local frame. When unset (the default), the witness is
    // seeded by ray-marching from the object-local origin toward the finger tip
    // until the SDF crosses zero (see HandModel::get_initial_values). Set
    // it to override that heuristic -- e.g. to seed an opposing thumb's witness
    // on the far side of the object so the solver starts in the enclosing grasp
    // configuration instead of collapsing every witness onto one side. Only an
    // initial guess; the Augmented-Lagrangian solve still drives it onto the
    // true surface.
    std::optional<gtsam::Point3> witness_point_seed;

    // --- Analytic hyper-ellipsoid object surface (Section 1.6.3) ---------
    // Semi-axes (a, b, c) of the bounding ellipsoid in the OBJECT-LOCAL frame;
    // the shape matrix is M = diag(a^-2, b^-2, c^-2). World orientation comes
    // from object_pose_mean (Eq 1.90's R_obj), so the ellipsoid needs no VDB
    // grid. When set (norm > 0), the object contact/collision factors evaluate
    // the analytic surface (EllipsoidWitnessContactFactor / EllipsoidCollisionGapFactor)
    // instead of sampling sdf_grid, giving a C-infinity surface with
    // non-vanishing gradients everywhere -- avoiding the flat-face / sharp-edge
    // local minima of a baked SDF. norm()==0 (the default) => not an ellipsoid,
    // so existing SDF/plane envs build exactly the pre-existing graph.
    gtsam::Vector3 ellipsoid_semi_axes = gtsam::Vector3::Zero();

    // --- Hyper-ellipsoid SET object surface (Section 1.2, Eq 1.10-1.13) ---
    // K primitives rigidly placed in the OBJECT frame, whose UNION is the object:
    // the generalization of ellipsoid_semi_axes to a shape one ellipsoid cannot
    // represent (a screwdriver's fat handle joined to its thin shaft is two
    // scales in one body, and the single MVEE over both is mostly air).
    // EllipsoidSetCollisionGapFactor evaluates it, fusing the per-member signed
    // distances (see ellipsoid_taubin) with a LogSumExp smooth min so the
    // surface stays C-infinity
    // across the seams where the members meet -- which is where a sliding finger
    // spends its time.
    //
    // PRECEDENCE, in the order has_object_surface() and the HandModel
    // builders test it: a non-empty ellipsoid_set wins over ellipsoid_semi_axes,
    // which wins over sdf_grid. Empty (the default) => every existing env builds
    // exactly the pre-existing graph.
    //
    // A set has NO witness-point contact form. The paper defines only the
    // center-direct equality (Eq 1.13) for it, so HandModel always takes
    // that form here and rejects the two witness-only settings outright rather
    // than silently falling back -- see uses_center_direct_contact().
    std::vector<EllipsoidPrimitive> ellipsoid_set;

    // LogSumExp sharpness for the set above. Distances are in METRES, so this is
    // O(100-1000), not O(1). The smooth min understates by up to ln(K)/beta, so
    // the constraint surface sits that far OUTSIDE the true union: at K=4 that is
    // 1.4 mm here, 0.7 mm at beta=2000. Raising it shrinks the bias at the cost of
    // a sharper (more min-like) gradient at the seams -- which is the smoothness
    // the formulation exists to buy. See EllipsoidSetCollisionGapFactor's
    // SMOOTH-MIN BIAS note for the full trade-off.
    double ellipsoid_set_beta = 1000.0;

    // Measure ellipsoid distance with Taubin's first-order algebraic
    // approximation instead of the exact orthogonal distance. Applies to every
    // ellipsoid factor at once -- the single ellipsoid, the set, and the 3D half
    // of the in-plane form -- because a graph that measured the same object two
    // ways in the collision and contact rows would be optimizing a shape that
    // exists nowhere.
    //
    // False (the default) = exact: the true distance to the closest surface
    // point, whose gradient is the unit surface normal. Its Jacobian rows are
    // conditioned identically at every eccentricity, which is what lets a flat or
    // coin-like object be resolved at the same AL step size as a sphere -- the
    // Taubin gradient's norm drifts from 1 as the shape gets more eccentric, and
    // that drift IS the ill-conditioning.
    //
    // True = the approximation every result before this flag existed was produced
    // with. Kept for reproducing those, and for the one property it has that the
    // exact form does not: it is C-infinity everywhere, where the exact distance's
    // gradient is only C^0 across the interior medial axis (the closest surface
    // point jumps there). Both share the same zero set exactly, so the surface the
    // constraints pin to is identical either way; only the field off it moves.
    //
    // See the TWO METRICS note on EllipsoidDistance.
    bool ellipsoid_taubin = false;

    // Which members of ellipsoid_set the CONTACT equality (Eq 1.13 / Eq 13) may
    // target. Empty (the default) => all of them, so every existing env builds
    // exactly the pre-existing graph.
    //
    // A decomposition is not all handles. The shells covering a drill's housing
    // or a pitcher's body are there to BOUND the geometry, and a fingertip sent
    // to the nearest point of the union lands on one of them as readily as on
    // the grip -- so which members are grasp targets is a property of the object
    // that only a human looking at it can supply. That choice is authored per
    // object and travels in the fit file as `grasp_subset`.
    //
    // COLLISION IS NEVER SUBSET. Eq 12 keeps the whole union for every free
    // sphere, and this field is read at the contact sites only: dropping the
    // excluded members from collision would let the rod pass straight through
    // the part the subset exists to say "do not grab this", which is the
    // opposite of what it asks for. Same asymmetry as object_contact_in_plane's
    // COLLISION IS NEVER PROJECTED note below, and for the same reason -- a
    // narrowed CONTACT target is a planning choice, a narrowed COLLISION set is
    // a lie about where the object is.
    std::vector<int> contact_ellipsoid_subset;

    // --- Tendon-aligned in-plane object contact (Eq 11 / Eq 13) ----------
    // Swap the object CONTACT equality from the full 3D distance to the distance
    // measured inside the finger's pulling plane -- EllipsoidSetPlanarGapFactor
    // instead of EllipsoidSetCollisionGapFactor, everything else identical:
    // the same center-direct form, the same residual convention (zero set
    // d = contact_node_radius), and the same exemption of the contact sphere from
    // the Eq 12 collision inequality. Only the distance metric differs.
    //
    // COLLISION IS NEVER PROJECTED. Eq 12 keeps the 3D distance for every free
    // sphere, here and everywhere else: the in-plane distance is always the
    // LARGER of the two, so an inequality on it would report clearance while the
    // finger is really inside the object.
    //
    // Needs an ellipsoid surface (a set, or ellipsoid_semi_axes, which is used as
    // a one-member set) -- a baked SDF has no plane cross-section to cut, and
    // uses_center_direct_contact() rejects the combination rather than silently
    // contacting something else.
    bool object_contact_in_plane = false;

    // Eq 11's p_centroid: the point, CONSTANT IN THE WRIST FRAME, where the
    // participating digits are measured to meet (HAND_PINCH_POSES). Required when
    // object_contact_in_plane is set -- it is the third point of the plane, and
    // which point it is depends on which digits are pinching, something the C++
    // side has no way to know. The other two points are the fingertip (a variable)
    // and the finger's metacarpal base, which is NOT a field here: that is
    // hand_base_offsets_[i], the model's own mounting data, so it cannot drift
    // from where the finger is actually mounted.
    std::optional<gtsam::Vector3> contact_plane_centroid;

    // The factor's two smoothstep fallback bands, defaulted to its own values.
    // Exposed here so they can be tuned from Python without a recompile:
    //   rho in [lo, hi] blends to 3D as the plane stops reaching a member;
    //   gap in [lo, hi] (metres) blends to 3D as the tip approaches the
    //   base->centroid axis, where Eq 11's normal stops being defined.
    double contact_plane_rho_lo = 0.90;
    double contact_plane_rho_hi = 1.00;
    double contact_plane_gap_lo = 0.002;
    double contact_plane_gap_hi = 0.010;

    // --- Support plane / "table" (Section 1.6) --------------------------
    // A world-fixed analytic half-space support surface, defined by an origin
    // point and an OUTWARD unit normal: SDF_table(p) = (p - plane_origin) .
    // plane_normal. The plane is treated as "absent" whenever plane_normal has
    // zero norm (the default), so an env with no table configured builds exactly
    // the pre-existing graph.
    //
    // plane_avoidance (Eq 1.59): when true, every non-contact, non-root collision
    // sphere gets a PlaneCollisionGapFactor wrapped in a CollisionInequalityConstraint
    // (c_pen = r - SDF_table <= 0) -- the free-space approach that keeps the
    // fingers off the table.
    //
    // table_contact_node: the finger node whose sphere is placed on the table and
    // slid across it during the sliding phase. When set, that node gets a SINGLE
    // residual on the sphere CENTER -- c_table(c) = Dist_plane(c) = 0, i.e.
    // PlaneCollisionGapFactor wrapped in a gtsam::ZeroCostConstraint -- with no
    // witness point of its own. (This replaces §1.6's five-residual
    // PlaneWitnessContactFactor form, Eq 1.60-1.64; it is the same factor
    // support_contact_node below uses, and the same argument applies: a plane
    // needs no witness.) table_contact_radius is that node's contact sphere
    // radius. The trajectory planner schedules these fields per step around the
    // k_touch touch step, so the same env is reused for approach
    // (plane_avoidance on, table_contact_node cleared) and slide
    // (plane_avoidance off, table_contact_node set).
    gtsam::Vector3 plane_origin = gtsam::Vector3::Zero();
    gtsam::Vector3 plane_normal = gtsam::Vector3::Zero();  // norm 0 => no table
    bool plane_avoidance = false;
    std::optional<int> table_contact_node;
    double table_contact_radius = 0.0;

    // --- Section 1.8 phased real-time controller -------------------------
    // The §1.8 formulation drops the trajectory index k and solves a single-state
    // constrained IK problem per control tick, switching the CONSTRAINT SET between
    // three phases. Every field below is inert by default, so an env configured for
    // the §1.3-1.6 solvers/planners builds exactly the pre-existing graph.

    // Support-surface contact EQUALITY on the contact sphere CENTER (Eq 1.97),
    // used in controller phases 1 and 2. Same factor and same residual as
    // table_contact_node above, on a separately designated node: there is no
    // witness point at all, just c_support(c) = Dist_plane(c) = 0 on the sphere
    // center. The witness is unnecessary because a single scalar residual on the
    // center leaves no rotational gauge freedom to brick the solver -- the null
    // space §1.6 worried about only appears once a free p_c variable is
    // introduced.
    //
    // NOTE on sign: the paper writes Dist_plane(c) = |(c - p).n| - r. We use the
    // SIGNED form (which is what PlaneCollisionGapFactor already computes, wrapped
    // in a gtsam::ZeroCostConstraint): it pins the sphere on the +n_table (free)
    // side of the table and is smooth at the contact point, whereas the absolute
    // value has a kink exactly where the solver operates.
    std::optional<int> support_contact_node;
    double support_contact_radius = 0.0;

    // Opposition half-space (Eq 1.92), controller phase 1. Splits the support
    // surface in half so the thumb lands opposite the fingers:
    //   c_half(c) = -(c - p_split) . m_hat + half_space_margin <= 0
    // m_hat is a unit vector lying IN the support plane (n_table . m_hat = 0)
    // pointing into the valid half-space for THIS finger, so each finger carries
    // its own direction.
    //
    // half_space_node is the node whose sphere CENTER the constraint acts on --
    // its own opt-in field, the same shape as pregrasp_center_node/
    // pregrasp_align_node below. The constraint is a statement about where one
    // node sits relative to a splitting line: it needs no support plane and no
    // table contact, and HandModel builds it in a pass of its own gated
    // on nothing but the fields here. Unset falls back to table_contact_node,
    // which is where this constraint used to live, so a caller that has not
    // been updated keeps working. Inert unless half_space_enabled, a node
    // resolves, and half_space_normal has non-zero norm.
    //
    // half_space_margin (m, >= 0) is the minimum standoff this finger must keep
    // from the splitting line -- see HalfSpaceGapFactor's d_min. 0 (the default)
    // is the plain half-space, satisfied by a center sitting exactly on the
    // split; a positive value opens a corridor of width 2*margin between the
    // thumb's side and the opposing fingers'.
    bool               half_space_enabled     = false;
    std::optional<int> half_space_node;
    gtsam::Vector3     half_space_split_point = gtsam::Vector3::Zero();
    gtsam::Vector3     half_space_normal      = gtsam::Vector3::Zero();  // m_hat
    double             half_space_margin      = 0.0;

    // Constrain the contact sphere CENTER directly to the hyper-ellipsoid instead
    // of introducing a witness point (Eq 1.101) --
    //   c_obj(c) = Taubin(T_obj^-1 c) - r = 0
    // -- which is EllipsoidCollisionGapFactor's residual (up to an irrelevant
    // sign, since this is an equality) wrapped in a ZeroCostConstraint. Dropping
    // the witness removes three variables and four residual rows per finger (5
    // -> 1), which is the point of the real-time sliding phase.
    //
    // NOTE this flag now only FORCES the form: HandModel already uses it by
    // default for any ellipsoid contact (see uses_center_direct_contact()), so
    // setting it there is redundant but harmless. Read only by HandModel;
    // the single-finger TendonFingerSolver and TendonFingerTrajectoryPlanner
    // ignore it and always use the witness form.
    // Requires ellipsoid_semi_axes to be set; ignored when target_contact_node
    // is unset.
    bool object_contact_center_direct = false;

    // Contact the baked SDF even though an ellipsoid proxy is also attached
    // (phases 3-4 of the staged pipeline). The one field on this struct that
    // separates the surface CONTACT uses from the surface COLLISION uses:
    //
    //   h_rad/h_sdf/h_tan  ->  sdf_grid          (the exact geometry)
    //   h_pen              ->  ellipsoid_set / ellipsoid_semi_axes  (E_obj)
    //
    // Everything else in this header shares one surface, resolved by the
    // precedence documented on ellipsoid_set (set > single ellipsoid > SDF).
    // That precedence is what the collision blocks keep using, untouched: they
    // read the proxy whenever one is attached, which is exactly the phase-3/4
    // formulation. Only the CONTACT block consults this flag, and only to look
    // past the proxy it would otherwise have picked.
    //
    // Why the split is wanted rather than just clearing the proxy: the approach
    // slid along E_obj and the free spheres are still steered by it, so dropping
    // it would swap a smooth everywhere-differentiable bound for a baked grid in
    // the middle of a solve -- and a baked grid is exactly what §1.7 introduced
    // the proxy to avoid for the parts of the hand that are NOT servoing on the
    // surface.
    //
    // Requires sdf_grid. Selects the WITNESS contact form by construction (the
    // grid has no closed-form distance to constrain a center against), so it is
    // rejected alongside object_contact_in_plane -- see
    // HandModel::uses_center_direct_contact, which raises rather than silently
    // contacting something other than what was asked for.
    bool object_contact_exact = false;

    // --- Geometric grasp alignment (h_grasp) -----------------------------
    // Net virtual-wrench equilibrium over this solve's witness points:
    //
    //   h_grasp({p_i}, T_obj) = sum_i [ -n_i ; -(p_i - t_obj) x n_i ] = 0
    //
    // A HAND-LEVEL constraint spanning every contacting digit at once, like the
    // three pregrasp_* families below, so build_graph() collects it in a pass of
    // its own after the per-digit loop rather than building one factor per digit
    // here. A digit opts in by setting this flag alongside its target_contact_node;
    // the remaining fields are shared constants duplicated across every
    // participating digit's env (first one found wins), the same convention
    // pregrasp_clearance_* uses.
    //
    // Needs WITNESS POINTS to key off, so it composes only with the witness
    // contact form (object_contact_exact, or a baked SDF) -- a center-direct
    // contact has no p_i at all. It also needs sdf_grid, since n_i is the
    // object's own surface normal at p_i. Both are rejected loudly rather than
    // skipped.
    //
    // TWO sigmas, where every sibling constraint here takes one: the residual is
    // dimensionless in its top three rows (a sum of unit normals) and has units
    // of length in its bottom three (a moment arm), so a shared isotropic model
    // would weight them by whatever the object's size happens to be. See the
    // factor's own header for the sizing note.
    //
    // The two finite-difference steps default (0.0) to half the grid voxel, which
    // is the measured sweet spot -- a sub-voxel stencil resolves the trilinear
    // interpolant rather than the geometry. Raise them on a noisy grid; the cost
    // is a smoothed normal, which for a constraint Jacobian is the safe direction
    // to err in.
    bool   grasp_alignment_enabled        = false;
    double grasp_alignment_sigma_force    = 1.0;
    double grasp_alignment_sigma_torque   = 1.0;
    double grasp_alignment_curvature_step = 0.0;
    double grasp_alignment_gradient_step  = 0.0;

    // --- Approximate geometric grasp alignment (h_grasp,E) ---------------
    // The APPROXIMATION-PHASE counterpart of the block above, and an independent
    // constraint rather than a mode of it: same Vector6 virtual-wrench
    // equilibrium, but keyed off the contact sphere CENTERS and reading the
    // normal from the analytic ellipsoid set instead of the baked SDF.
    //
    //   h_grasp,E({c_i}) = sum_i [ -n_i ; -(c_i - t_obj) x n_i ] = 0
    //   n_i = R_obj * normalize(grad d_E(T_obj^{-1} c_i))
    //
    // grasp_alignment_enabled REFUSES a digit on the center-direct contact form
    // (it has no witness point to key off). This one IS that form, so it is what
    // makes the wrench balance available before any exact contact exists -- i.e.
    // while the hand is still being steered by the smooth ellipsoid proxy. The
    // sphere radius cancels analytically out of the torque, so no radius and no
    // witness variable appear anywhere in it. Both flags may be set at once.
    //
    // Reads the FULL ellipsoid_set (not contact_ellipsoid_subset): the normal
    // field is a property of the object's geometry, not of which shells a finger
    // has been cleared to touch, and blending only the grasp shells would report
    // a normal that points into a housing the object actually has. Same choice
    // EllipsoidSetCollisionGapFactor makes. Falls back to ellipsoid_semi_axes as
    // a single identity-posed member; there is NO fall back to sdf_grid, and
    // build_graph raises rather than silently substituting one.
    //
    // The two sigmas split the residual's halves for the reason the block above
    // documents. The curvature step defaults (0.0) to 1e-5 m rather than to half
    // a voxel: this field is a closed form, so the stencil has no interpolation
    // floor to clear. See the factor's own header.
    bool   ellipsoid_grasp_alignment_enabled        = false;
    double ellipsoid_grasp_alignment_sigma_force    = 1.0;
    double ellipsoid_grasp_alignment_sigma_torque   = 1.0;
    double ellipsoid_grasp_alignment_curvature_step = 0.0;

    // Controller phase 3 (Eq 1.107-1.110): drop the normal-alignment row c_N from
    // the witness contact factor, leaving [c_R, c_O, c_T1, c_T2]. Justified in
    // §1.8: with collision geometry modeled exclusively as spheres, the tangential
    // slip rows already force the relative vector -- and hence the outward sphere
    // normal -- collinear with the object surface normal, so c_N is redundant.
    bool contact_drop_normal_row = false;

    // Controller phase 3 soft witness target (Eq 1.111): a Gaussian prior pulling
    // this finger's object witness point toward a nominal grasp location, given in
    // the WORLD frame. Because the AL equality constraints restrict the witness to
    // the object surface manifold, this acts as a geodesic pull that slides the
    // witness (and the attached finger) along the surface. Unset => no prior, i.e.
    // the "contact anywhere on the surface" formulation (Eq 1.119-1.125).
    std::optional<gtsam::Point3> witness_target;
    Eigen::Matrix3d witness_target_cov = 1e-4 * Eigen::Matrix3d::Identity();

    // --- Pre-grasp hand-centering (Section 2.2.1, Eq 2.18-2.19) ----------
    // Opt-in per finger: a finger participates in PreGraspHandCenteringFactor
    // by setting pregrasp_center_node (the node whose sphere center enters
    // the constraint), mirroring table_contact_node/support_contact_node.
    // UNLIKE every other field on this struct, this constraint spans
    // MULTIPLE fingers at once (the thumb + every other participating
    // finger), so build_graph() collects it in a separate hand-level pass
    // after the per-finger loop rather than building one factor per finger
    // here. pregrasp_clearance_height / pregrasp_clearance_normal are shared
    // constants, duplicated across every participating finger's env (same
    // convention as plane_origin/plane_normal). clearance_normal norm 0 (the
    // default) => not configured.
    std::optional<int> pregrasp_center_node;
    double pregrasp_clearance_height = 0.0;
    gtsam::Vector3 pregrasp_clearance_normal = gtsam::Vector3::Zero();

    // --- Pre-grasp short-axis alignment (companion to Eq 2.16-2.17) ------
    // Opt-in per finger, same shape as pregrasp_center_node above: a finger
    // participates in PreGraspAxisAlignmentFactor by setting
    // pregrasp_align_node. A SEPARATE field from pregrasp_center_node on
    // purpose -- this constraint must be independently toggleable from
    // pre-grasp centering, so it cannot piggyback on that field being set.
    // pregrasp_align_axis is the shared target direction (duplicated across
    // every participating finger's env): the SAME m_hat the opposition
    // half-space uses (perpendicular to the object's longest in-plane axis),
    // NOT an independently-derived 3D short axis -- deliberately reuses
    // half_space_normal's geometry rather than half_space_normal's FIELD, so
    // this constraint doesn't depend on half_space_enabled being on either.
    // Also a hand-level (multi-finger) constraint, so build_graph() collects
    // it in its own pass, same as pregrasp_center_node. Zero norm (the
    // default) => not configured.
    std::optional<int> pregrasp_align_node;
    gtsam::Vector3 pregrasp_align_axis = gtsam::Vector3::Zero();

    // --- Pre-grasp PINCH-CENTROID centering -------------------------------
    // The hardcoded-point sibling of pregrasp_center_node above. Centering
    // drives the MEASURED midpoint of the thumb's and the opposing fingers'
    // contact spheres onto the object; this drives a point that is CONSTANT
    // in the wrist frame onto the same target. That point is where a given
    // finger combination's fingertips are known to meet (measured offline
    // per combination -- see python/tests/tendon_hand/fk_pinch_centroids.py
    // and the HAND_PINCH_POSES table in tendon_hand/config.py), so satisfying
    // this constraint positions the hand such that CLOSING those digits
    // closes them on the object, without the fingers having to be curled yet.
    //
    // Consequences of the point being constant rather than pose-derived:
    //   * the residual depends only on the wrist and object poses, so
    //     PreGraspCentroidFactor is a fixed-arity NoiseModelFactorN<Pose3,
    //     Pose3> rather than the runtime-arity KeyVector the other two
    //     pre-grasp factors need, and
    //   * no finger opts in, so there is no *_node field here. The point
    //     itself being set IS the opt-in.
    // Hand-level all the same, so build_graph() collects it in its own pass;
    // the fields are duplicated across every finger's env like
    // pregrasp_clearance_* (first one found wins).
    //
    // SEPARATE clearance/normal fields rather than reusing
    // pregrasp_clearance_* on purpose: those are only written when the
    // pre-grasp centering block runs, so sharing them would leave this
    // constraint silently inert whenever that toggle happens to be off.
    // Zero-norm normal (the default) => not configured, same convention.
    std::optional<gtsam::Vector3> pregrasp_centroid_point;   // WRIST frame
    double pregrasp_centroid_clearance = 0.0;
    gtsam::Vector3 pregrasp_centroid_normal = gtsam::Vector3::Zero();
};

// Does this env carry an object surface to contact or avoid, in any of its three
// representations? THE one definition of that question.
//
// It used to be spelled inline at five sites in HandModel, two of which
// carry comments warning they must stay identical -- get_initial_values decides
// whether the shared object variable is seeded at all, and build_graph decides
// whether it is anchored, so a predicate that drifts between them anchors an
// object that was never seeded and the system comes out indeterminate. Adding a
// third representation to five copies is exactly how that drift happens, hence
// the shared function.
inline bool has_object_surface(const EnvironmentConfig& env) {
    return !env.ellipsoid_set.empty() ||
           env.ellipsoid_semi_axes.norm() > 0.0 ||
           static_cast<bool>(env.sdf_grid);
}

// The members a CONTACT factor may target: contact_ellipsoid_subset applied to
// ellipsoid_set, or the whole set when no subset was asked for.
//
// Shared for the same reason has_object_surface() is: the two contact sites in
// HandModel::build_graph (the 3D equality and its in-plane Eq 13 variant)
// must narrow identically, or switching contact FORM would silently also change
// which shells are being touched.
//
// An index that addresses no member THROWS rather than being skipped. Silently
// dropping it would leave the caller believing a shell is a contact target when
// it is not, and the resulting grasp would read as a solver failure rather than
// as the mis-request it is -- the same reasoning attach_ellipsoid_set documents
// on the Python side.
inline std::vector<EllipsoidPrimitive>
contact_ellipsoid_members(const EnvironmentConfig& env) {
    if (env.contact_ellipsoid_subset.empty()) return env.ellipsoid_set;

    std::vector<EllipsoidPrimitive> members;
    members.reserve(env.contact_ellipsoid_subset.size());
    for (int index : env.contact_ellipsoid_subset) {
        if (index < 0 || index >= static_cast<int>(env.ellipsoid_set.size()))
            throw std::invalid_argument(
                "EnvironmentConfig::contact_ellipsoid_subset holds index " +
                std::to_string(index) + ", but ellipsoid_set has " +
                std::to_string(env.ellipsoid_set.size()) + " member(s)");
        members.push_back(env.ellipsoid_set[static_cast<std::size_t>(index)]);
    }
    return members;
}

}  // namespace gepetto_solvers
