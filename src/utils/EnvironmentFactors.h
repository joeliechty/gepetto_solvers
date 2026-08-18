#pragma once

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

namespace crest_sparse {

// Frisvad/Hughes-Moller Householder basis: maps +Z onto the unit normal n and
// returns the two orthonormal tangent vectors spanning n's tangent plane. This
// is the explicitly-unrolled Householder reflection -- a few lines of arithmetic
// with no matrix allocation, suitable for a factor evaluated thousands of times
// per second. The single singularity at the south pole (n ~ -Z) is handled
// explicitly. Used by the witness-point contact factors to deterministically
// build the local Contact Frame (Section 3, Eq 30-31).
inline void frisvad_tangent_basis(const gtsam::Vector3& n,
                                  gtsam::Vector3& t1, gtsam::Vector3& t2) {
    if (n.z() < -0.9999999) {
        t1 = gtsam::Vector3( 0.0, -1.0,  0.0);
        t2 = gtsam::Vector3(-1.0,  0.0,  0.0);
    } else {
        const double a = 1.0 / (1.0 + n.z());
        const double b = -n.x() * n.y() * a;
        t1 = gtsam::Vector3(1.0 - n.x() * n.x() * a, b, -n.x());
        t2 = gtsam::Vector3(b, 1.0 - n.y() * n.y() * a, -n.y());
    }
}

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
    // Factor. Only TendonHandModel builds these (a single finger cannot
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
    // until the SDF crosses zero (see TendonHandModel::get_initial_values). Set
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
    // table contact, and TendonHandModel builds it in a pass of its own gated
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
    // NOTE this flag now only FORCES the form: TendonHandModel already uses it by
    // default for any ellipsoid contact (see uses_center_direct_contact()), so
    // setting it there is redundant but harmless. Read only by TendonHandModel;
    // the single-finger TendonFingerSolver and TendonFingerTrajectoryPlanner
    // ignore it and always use the witness form.
    // Requires ellipsoid_semi_axes to be set; ignored when target_contact_node
    // is unset.
    bool object_contact_center_direct = false;

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


// ---------------------------------------------------------------------------
// Collision-avoidance factors (Section 1.5).
//
// Collision is modeled as a hard inequality constraint c_pen(x) <= 0 handled
// natively by GTSAM's AugmentedLagrangianOptimizer -- NOT as a soft cubic /
// quadratic penalty (the old SdfCollisionFactor). Each geometry pair is split
// into two pieces:
//   * a "gap factor" (a plain NoiseModelFactor) whose *unwhitened* error is the
//     raw signed penetration depth c_pen(x) with analytical Jacobians, and
//   * a CollisionInequalityConstraint wrapper that presents that gap factor to
//     the AL solver as the one-sided constraint c_pen(x) <= 0.
//
// Under the AL two-loop formulation:
//   Free space  (c_pen <  0): inactive -- the constraint ramps to zero error
//                             and zero Jacobian, adding nothing to the linear
//                             system and preserving graph sparsity.
//   Collision   (c_pen >= 0): active -- the inner loop applies a smooth
//                             quadratic penalty pushing the sphere back to the
//                             surface; mu / lambda are updated in the outer loop.
// This mirrors gtsam::ZeroCostConstraint (which does the equality analogue for
// the terminal contact factors) but for the one-sided collision case.
// ---------------------------------------------------------------------------

// Wraps a scalar "gap factor" -- any NoiseModelFactor whose unwhitened error is
// a penetration depth c_pen(x) -- into the inequality constraint c_pen(x) <= 0
// consumed by GTSAM's AugmentedLagrangianOptimizer. The constraint's sigma and
// keys are inherited from the wrapped factor, exactly as gtsam::ZeroCostConstraint
// does for equality constraints. The base class supplies the ramp (inactive
// branch), the active() test, and the L2 penalty; we additionally expose the
// g(x)=0 equality form used to build the Lagrange-multiplier term.
class CollisionInequalityConstraint : public gtsam::NonlinearInequalityConstraint {
private:
    gtsam::NoiseModelFactor::shared_ptr gap_factor_;

public:
    explicit CollisionInequalityConstraint(const gtsam::NoiseModelFactor::shared_ptr& gap_factor)
        : gtsam::NonlinearInequalityConstraint(
              constrainedNoise(gap_factor->noiseModel()->sigmas()), gap_factor->keys()),
          gap_factor_(gap_factor) {}

    // g(x) = raw penetration depth; the base class ramps this to enforce <= 0.
    gtsam::Vector unwhitenedExpr(const gtsam::Values& x,
                                 gtsam::OptionalMatrixVecType H = nullptr) const override {
        return gap_factor_->unwhitenedError(x, H);
    }

    // The corresponding g(x) = 0 equality constraint, used by the AL optimizer
    // to build the Lagrange-multiplier (linear) term for this inequality.
    gtsam::NonlinearEqualityConstraint::shared_ptr createEqualityConstraint() const override {
        return std::make_shared<gtsam::ZeroCostConstraint>(gap_factor_);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new CollisionInequalityConstraint(*this)));
    }
};


// Finger-object (sphere-to-SDF) penetration gap (Section 1.5):
//   c_pen(p_i, T_obj) = r_i - SDF(T_obj^{-1} p_i)
// where p_i is the world-frame center of the collision sphere on the finger
// node and r_i its radius. c_pen > 0 means the sphere penetrates the object
// surface. The analytical Jacobian follows the writeup:
//   d c_pen / d p_i = -grad SDF(T_obj^{-1} p_i),
// chained through node_pose.translation() and object_pose.transformTo(). Wrap
// an instance in a CollisionInequalityConstraint to enforce c_pen <= 0.
class SdfCollisionGapFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    double radius_;
    openvdb::FloatGrid::Ptr sdf_grid_;

public:
    SdfCollisionGapFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                          double radius, const openvdb::FloatGrid::Ptr& sdf_grid,
                          const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, node_pose_key, object_key),
          radius_(radius), sdf_grid_(sdf_grid) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_pworld_finger;
        gtsam::Point3 p_world = node_pose.translation(H1 ? &D_pworld_finger : nullptr);

        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_pworld;
        gtsam::Point3 p_local = object_pose.transformTo(p_world,
            H2 ? &D_plocal_obj    : nullptr,
            H1 ? &D_plocal_pworld : nullptr);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R q(p_local.x(), p_local.y(), p_local.z());
        double sdf = sampler.wsSample(q);
        double c_pen = radius_ - sdf;   // > 0  <=>  penetration

        if (H1 || H2) {
            // Central-difference SDF gradient in the object-local frame.
            double h = 1e-4;
            double dx = sampler.wsSample(openvdb::Vec3R(q.x() + h, q.y(), q.z())) -
                        sampler.wsSample(openvdb::Vec3R(q.x() - h, q.y(), q.z()));
            double dy = sampler.wsSample(openvdb::Vec3R(q.x(), q.y() + h, q.z())) -
                        sampler.wsSample(openvdb::Vec3R(q.x(), q.y() - h, q.z()));
            double dz = sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z() + h)) -
                        sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z() - h));
            gtsam::Vector3 grad(dx, dy, dz);
            grad /= (2.0 * h);

            // d c_pen / d p_local = -grad^T (dc_pen/dsdf = -1, dsdf/dp = grad^T)
            gtsam::Matrix13 dcpen_dplocal = -grad.transpose();
            if (H1) *H1 = dcpen_dplocal * D_plocal_pworld * D_pworld_finger;
            if (H2) *H2 = dcpen_dplocal * D_plocal_obj;
        }

        return gtsam::Vector1(c_pen);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SdfCollisionGapFactor(*this)));
    }
};


// Finger-ellipsoid (sphere-to-analytic-ellipsoid) penetration gap
// (Section 1.6.3, Eq 1.91). The analytic analog of SdfCollisionGapFactor for a
// hyper-ellipsoid object surface with shape matrix M = diag(a^-2, b^-2, c^-2).
// Because the raw algebraic value x^T M x - 1 warps space non-uniformly (it is
// not a Euclidean distance), we use the Taubin first-order distance
// approximation of the implicit surface:
//   x       = T_obj^{-1} p_i          (object-local sphere center)
//   dist    = (x^T M x - 1) / (2 ||M x||)
//   c_pen   = r_i - dist              (> 0 <=> penetration)
// The gradient is fully analytic (no SDF sampling). Wrap an instance in a
// CollisionInequalityConstraint to enforce c_pen <= 0.
class EllipsoidCollisionGapFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    double radius_;
    gtsam::Vector3 m_diag_;   // (1/a^2, 1/b^2, 1/c^2)

public:
    EllipsoidCollisionGapFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                                double radius, const gtsam::Vector3& semi_axes,
                                const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, node_pose_key, object_key),
          radius_(radius),
          m_diag_(1.0 / (semi_axes.x() * semi_axes.x()),
                  1.0 / (semi_axes.y() * semi_axes.y()),
                  1.0 / (semi_axes.z() * semi_axes.z())) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_pworld_finger;
        gtsam::Point3 p_world = node_pose.translation(H1 ? &D_pworld_finger : nullptr);

        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_pworld;
        gtsam::Point3 x = object_pose.transformTo(p_world,
            H2 ? &D_plocal_obj    : nullptr,
            H1 ? &D_plocal_pworld : nullptr);

        gtsam::Vector3 Mx = m_diag_.cwiseProduct(x);   // M x
        double f = x.dot(Mx) - 1.0;                    // x^T M x - 1
        double g = Mx.norm();                          // ||M x||
        if (g < 1e-9) g = 1e-9;
        double dist  = f / (2.0 * g);                  // Taubin first-order distance
        double c_pen = radius_ - dist;                 // > 0 <=> penetration

        if (H1 || H2) {
            // dist = f / (2 g), f = x^T M x - 1, g = ||M x||.
            //   df/dx = 2 M x  (row: 2 Mx^T)
            //   dg/dx = (M x)^T M / g = (m_diag ∘ M x)^T / g
            //   d dist/dx = f'/(2g) - f g'/(2 g^2)
            gtsam::Vector3 mMx = m_diag_.cwiseProduct(Mx);   // M (M x)
            gtsam::Matrix13 ddist_dx =
                  (Mx.transpose() / g)
                - (f / (2.0 * g * g * g)) * mMx.transpose();
            gtsam::Matrix13 dcpen_dplocal = -ddist_dx;
            if (H1) *H1 = dcpen_dplocal * D_plocal_pworld * D_pworld_finger;
            if (H2) *H2 = dcpen_dplocal * D_plocal_obj;
        }

        return gtsam::Vector1(c_pen);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new EllipsoidCollisionGapFactor(*this)));
    }
};


// One member of a hyper-ellipsoid SET (Section 1.2, Eq 1.10).
//
// semi_axes are (a, b, c) in THIS ellipsoid's own frame, so its shape matrix is
// M_k = diag(a^-2, b^-2, c^-2) -- same convention as EllipsoidCollisionGapFactor's
// single ellipsoid. local_pose is T_k, this ellipsoid's CONSTANT pose in the
// OBJECT frame; Eq 1.10's world pose is therefore T_Ek = T_obj o T_k, with T_obj
// the one optimized object Pose3 the rest of this header already shares. The set
// is thus one rigid body made of K primitives -- it moves with the object
// variable, and adds no variables of its own.
struct EllipsoidPrimitive {
    gtsam::Vector3 semi_axes;
    gtsam::Pose3   local_pose = gtsam::Pose3();
};


// Finger-ellipsoid-SET penetration gap (Section 1.2, Eq 1.10-1.13). The K-primitive
// generalization of EllipsoidCollisionGapFactor: an object too complicated for one
// hyper-ellipsoid is modeled as the union of a set E = {E_1, ..., E_K}, each member
// an EllipsoidPrimitive rigidly placed in the object frame.
//
//   x_k     = T_k^{-1} T_obj^{-1} p_i            (Eq 1.10, sphere center in E_k's frame)
//   d_k     = (x_k^T M_k x_k - 1) / (2 ||M_k x_k||)     (Taubin, per member)
//   d_E     = -(1/beta) ln sum_k exp(-beta d_k)  (Eq 1.11, LogSumExp smooth min)
//   c_pen   = r_i - d_E                          (> 0 <=> penetration)
//
// The LogSumExp fusion is the point of the formulation: a hard min over the members
// is gradient-discontinuous exactly at the seams where two ellipsoids meet, which is
// where a sliding finger spends its time. Blending the fields instead keeps the
// collision manifold C-infinity, so the AL solver slides across surface primitives
// without stalling on an internal boundary.
//
// TWO USES, ONE RESIDUAL. Eq 1.12 and Eq 1.13 differ only in sign, which an equality
// does not see:
//   * Eq 1.12, collision inequality c_pen_set = r_i - d_E <= 0: wrap an instance in
//     CollisionInequalityConstraint (above).
//   * Eq 1.13, contact equality c_obj_set = d_E - r_i = 0: wrap the SAME instance in
//     gtsam::ZeroCostConstraint -- its zero set is exactly Eq 1.13. This is the
//     center-direct form (it constrains the sphere CENTER c_i, with no witness point),
//     matching the paper, which defines no witness variant for the set. The
//     single-ellipsoid precedent is TendonHandModel::build_graph, which already wraps
//     EllipsoidCollisionGapFactor as the Eq 1.101 center-direct contact equality.
//
// Per-member distance is Taubin's first-order approximation rather than the raw
// algebraic x^T M x - 1 for the reason spelled out on EllipsoidCollisionGapFactor and
// EllipsoidWitnessContactFactor: the raw value scales as ~1/min(semi_axis)^2, so under
// a shared noise model it swamps every other row and the AL inner solve stagnates.
// Taubin restores an O(1) Euclidean-like distance with an exact analytic Jacobian --
// and it is what makes the members COMMENSURATE here, which the smooth min needs: a
// LogSumExp over differently-warped algebraic values would blend quantities that are
// not the same thing.
//
// SMOOTH-MIN BIAS -- how to pick beta. LogSumExp-min understates:
//   min_k d_k - ln(K)/beta  <=  d_E  <=  min_k d_k
// so the constraint surface {d_E = r} sits up to ln(K)/beta OUTSIDE the true union.
// Distances here are in METRES, so beta is O(100-1000), not O(1): at K=2, beta=500
// the bias is 1.4 mm; at beta=2000 it is 0.35 mm. Raising beta shrinks the bias at the
// cost of a sharper (more min-like) gradient near the seams, which is the smoothness
// this factor exists to buy -- so beta is a constructor argument, and the caller owns
// the trade-off. The bias is conservative for collision (Eq 1.12 keeps the sphere
// slightly further out than required) and a small standoff for contact (Eq 1.13).
//
// K = 1 with an identity local_pose reduces exactly to EllipsoidCollisionGapFactor,
// for any beta.
class EllipsoidSetCollisionGapFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    double radius_;
    double beta_;
    std::vector<gtsam::Vector3> m_diag_;   // per-k (1/a^2, 1/b^2, 1/c^2)
    std::vector<gtsam::Matrix3> Rk_T_;     // per-k R_k^T, precomputed
    std::vector<gtsam::Vector3> tk_;       // per-k t_k

public:
    EllipsoidSetCollisionGapFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                                   double radius,
                                   const std::vector<EllipsoidPrimitive>& ellipsoids,
                                   double beta,
                                   const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, node_pose_key, object_key),
          radius_(radius), beta_(beta)
    {
        // These are silent-garbage cases, not recoverable ones: an empty set has no
        // distance to report, and a non-positive beta or semi-axis divides by zero or
        // flips the smooth min into a smooth MAX. Fail where the mistake was made.
        if (ellipsoids.empty())
            throw std::invalid_argument(
                "EllipsoidSetCollisionGapFactor: the ellipsoid set is empty");
        if (!(beta > 0.0))
            throw std::invalid_argument(
                "EllipsoidSetCollisionGapFactor: beta must be > 0 (got " +
                std::to_string(beta) + ")");

        m_diag_.reserve(ellipsoids.size());
        Rk_T_.reserve(ellipsoids.size());
        tk_.reserve(ellipsoids.size());
        for (size_t k = 0; k < ellipsoids.size(); ++k) {
            const gtsam::Vector3& a = ellipsoids[k].semi_axes;
            if (!(a.x() > 0.0 && a.y() > 0.0 && a.z() > 0.0))
                throw std::invalid_argument(
                    "EllipsoidSetCollisionGapFactor: ellipsoid " + std::to_string(k) +
                    " has a non-positive semi-axis");
            m_diag_.emplace_back(1.0 / (a.x() * a.x()),
                                 1.0 / (a.y() * a.y()),
                                 1.0 / (a.z() * a.z()));
            // x_k = T_k^{-1} p_obj = R_k^T (p_obj - t_k); T_k is constant, so cache
            // the two pieces and skip a Pose3 inverse per member per evaluation.
            Rk_T_.push_back(ellipsoids[k].local_pose.rotation().matrix().transpose());
            tk_.push_back(ellipsoids[k].local_pose.translation());
        }
    }

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_pworld_finger;
        gtsam::Point3 p_world = node_pose.translation(H1 ? &D_pworld_finger : nullptr);

        gtsam::Matrix36 D_pobj_obj;
        gtsam::Matrix33 D_pobj_pworld;
        gtsam::Point3 p_obj = object_pose.transformTo(p_world,
            H2 ? &D_pobj_obj    : nullptr,
            H1 ? &D_pobj_pworld : nullptr);

        const size_t K = m_diag_.size();
        const bool need_jac = (H1 || H2);

        // --- Per-member Taubin distance d_k and (optionally) d d_k / d p_obj ------
        std::vector<double> d(K);
        std::vector<gtsam::Matrix13> dd_dpobj(need_jac ? K : 0);
        double d_min = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < K; ++k) {
            gtsam::Vector3 x  = Rk_T_[k] * (p_obj - tk_[k]);   // x_k = T_k^{-1} p_obj
            gtsam::Vector3 Mx = m_diag_[k].cwiseProduct(x);    // M_k x_k
            double f = x.dot(Mx) - 1.0;                        // x^T M x - 1
            double g = Mx.norm();                              // ||M x||
            if (g < 1e-9) g = 1e-9;
            d[k] = f / (2.0 * g);                              // Taubin distance
            if (d[k] < d_min) d_min = d[k];

            if (need_jac) {
                // d d_k / d x = Mx^T/g - (f/(2 g^3)) (M (M x))^T, then into the
                // OBJECT frame through the constant rotation: d x / d p_obj = R_k^T.
                gtsam::Vector3 mMx = m_diag_[k].cwiseProduct(Mx);   // M (M x)
                gtsam::Matrix13 dd_dx =
                      (Mx.transpose() / g)
                    - (f / (2.0 * g * g * g)) * mMx.transpose();
                dd_dpobj[k] = dd_dx * Rk_T_[k];
            }
        }

        // --- LogSumExp smooth min (Eq 1.11) --------------------------------------
        // Shifted by d_min so every exponent is <= 0: exp() cannot overflow, the sum
        // is >= 1, and an underflowing far member simply contributes weight 0.
        double s = 0.0;
        std::vector<double> w(need_jac ? K : 0);
        for (size_t k = 0; k < K; ++k) {
            double e = std::exp(-beta_ * (d[k] - d_min));
            if (need_jac) w[k] = e;
            s += e;
        }
        double d_set = d_min - std::log(s) / beta_;
        double c_pen = radius_ - d_set;   // > 0  <=>  penetration

        if (need_jac) {
            // d(LSE)/dx = sum_k w_k * d d_k/dx with w_k the softmin weights (they sum
            // to 1) -- exact, no locally-frozen-weight approximation needed.
            gtsam::Matrix13 dset_dpobj = gtsam::Matrix13::Zero();
            for (size_t k = 0; k < K; ++k)
                dset_dpobj += (w[k] / s) * dd_dpobj[k];

            gtsam::Matrix13 dcpen_dpobj = -dset_dpobj;
            if (H1) *H1 = dcpen_dpobj * D_pobj_pworld * D_pworld_finger;
            if (H2) *H2 = dcpen_dpobj * D_pobj_obj;
        }

        return gtsam::Vector1(c_pen);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new EllipsoidSetCollisionGapFactor(*this)));
    }
};


// Finger-plane (sphere-to-half-space) penetration gap (Section 1.6, Eq 1.59).
// The support surface ("table") is a world-fixed analytic half-space defined by
// an origin point p_table and an OUTWARD unit normal n_table:
//   SDF_table(p) = (p - p_table) . n_table   (>0 in the free half-space)
//   c_pen(p_i)   = r_i - SDF_table(p_i)       (>0 <=> the sphere penetrates)
// where p_i is the world-frame center of the collision sphere (the node pose's
// translation). Closed-form Jacobian:
//   d c_pen / d p_world = -n_table^T ,
// chained through node_pose.translation(). No object pose variable -- the plane
// is a constant. Wrap an instance in a CollisionInequalityConstraint to enforce
// c_pen <= 0.
class PlaneCollisionGapFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
private:
    double radius_;
    gtsam::Vector3 p_table_;
    gtsam::Vector3 n_table_;

public:
    PlaneCollisionGapFactor(gtsam::Key node_pose_key, double radius,
                            const gtsam::Vector3& p_table, const gtsam::Vector3& n_table,
                            const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, node_pose_key),
          radius_(radius), p_table_(p_table), n_table_(n_table.normalized()) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                gtsam::OptionalMatrixType H1) const override
    {
        gtsam::Matrix36 D_pworld_pose;
        gtsam::Point3 p_world = node_pose.translation(H1 ? &D_pworld_pose : nullptr);

        double sdf = (p_world - p_table_).dot(n_table_);
        double c_pen = radius_ - sdf;   // > 0  <=>  penetration

        // d c_pen / d p_world = -n_table^T (dc_pen/dsdf = -1, dsdf/dp = n^T)
        if (H1) *H1 = -n_table_.transpose() * D_pworld_pose;   // 1x6

        return gtsam::Vector1(c_pen);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PlaneCollisionGapFactor(*this)));
    }
};


// Opposition half-space constraint (Section 1.8, Eq 1.92). Keeps a finger's
// contact sphere on its designated half of the support surface, so the thumb
// opposes the other grasping fingers:
//
//   c_half(c) = -(c - p_split) . m_hat + d_min   <= 0
//
// where c is the world-frame center of the contact sphere (the node pose's
// translation), p_split is a point on the splitting line (e.g. the object's
// centroid projected onto the support surface), and m_hat is a unit vector lying
// IN the support plane (n_table . m_hat = 0) pointing into the valid half-space
// for this finger. Because m_hat is orthogonal to the plane normal, the sphere
// radius cancels out entirely -- the constraint depends only on the center, and
// the Jacobian is the CONSTANT row
//
//   d c_half / d c = -m_hat^T ,
//
// chained through node_pose.translation(). That constant Jacobian makes this the
// cheapest constraint in the graph to evaluate. Wrap an instance in a
// CollisionInequalityConstraint to enforce c_half <= 0.
//
// d_min (>= 0, default 0) is a MINIMUM STANDOFF: the distance the sphere center
// must clear the splitting line by, along this finger's own m_hat. At the
// default 0 the constraint is the bare half-space above, which a center sitting
// exactly ON the split already satisfies -- so opposition alone does not stop
// the thumb and the opposing fingers from closing onto each other. A positive
// d_min holds each side that far off the split, i.e. holds a corridor of width
// 2*d_min open between the two groups, which is what makes this usable as a
// pre-grasp opening. It shifts the residual by a constant, so the Jacobian --
// and the cost of evaluating it -- is unchanged.
class HalfSpaceGapFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
private:
    gtsam::Vector3 p_split_;
    gtsam::Vector3 m_hat_;
    double         d_min_;

public:
    HalfSpaceGapFactor(gtsam::Key node_pose_key,
                       const gtsam::Vector3& p_split, const gtsam::Vector3& m_hat,
                       const gtsam::SharedNoiseModel& noise_model,
                       double d_min = 0.0)
        : NoiseModelFactorN(noise_model, node_pose_key),
          p_split_(p_split), m_hat_(m_hat.normalized()), d_min_(d_min) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                gtsam::OptionalMatrixType H1) const override
    {
        gtsam::Matrix36 D_pworld_pose;
        gtsam::Point3 p_world = node_pose.translation(H1 ? &D_pworld_pose : nullptr);

        // > 0  <=>  the center is on the WRONG side of the splitting line, or is
        // on the right side but closer to it than the required standoff.
        double c_half = -(p_world - p_split_).dot(m_hat_) + d_min_;

        if (H1) *H1 = -m_hat_.transpose() * D_pworld_pose;   // 1x6, constant in p

        return gtsam::Vector1(c_half);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new HalfSpaceGapFactor(*this)));
    }
};


// Pre-grasp hand-centering constraint (Section 2.2.1, Eq 1.92 numbering ->
// paper Eq 2.18-2.19). Centers the hand over the object prior to initiating
// surface contact by aligning the midpoint of the thumb's contact-sphere
// center and the mean of the opposing fingers' contact-sphere centers with
// the object centroid, raised by a fixed clearance offset along the support
// surface normal:
//
//   c_hand = (1/2) * ( c_thumb + (1/|F|) sum_{i in F} c_i )
//   c_center(c_thumb, F, p_obj) = c_hand - (p_obj + h_clear * n_hat) = 0
//
// where c_thumb and each c_i are world-frame sphere centers (the translations
// of their node Pose3 variables -- same convention as HalfSpaceGapFactor and
// PlaneCollisionGapFactor), p_obj is the object pose's translation, and
// h_clear / n_hat are constructor-supplied constants (n_hat need not be the
// table normal specifically -- whatever clearance axis the caller wants).
//
// Variable arity: the number of opposing fingers |F| is runtime-determined,
// so -- like TendonLengthFactor -- this derives from gtsam::NoiseModelFactor
// directly (not NoiseModelFactorN) and hand-builds its KeyVector in the
// initializer list.
//
// Keys: [thumb_pose_key, finger_pose_key_0, ..., finger_pose_key_{|F|-1}, object_key]
// Residual: Vector3.
class PreGraspHandCenteringFactor : public gtsam::NoiseModelFactor {
private:
    size_t num_fingers_;
    double h_clear_;
    gtsam::Vector3 n_hat_;

public:
    using NoiseModelFactor::unwhitenedError;

    PreGraspHandCenteringFactor(gtsam::Key thumb_pose_key,
                                const std::vector<gtsam::Key>& finger_pose_keys,
                                gtsam::Key object_key,
                                double h_clear,
                                const gtsam::Vector3& n_hat,
                                const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor(noise_model, /* keys: */ [&]() {
              gtsam::KeyVector keys;
              keys.push_back(thumb_pose_key);
              keys.insert(keys.end(), finger_pose_keys.begin(), finger_pose_keys.end());
              keys.push_back(object_key);
              return keys;
          }()),
          num_fingers_(finger_pose_keys.size()),
          h_clear_(h_clear),
          n_hat_(n_hat.normalized()) {}

    gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                  gtsam::OptionalMatrixVecType H = nullptr) const override
    {
        // Key layout: [0] = thumb, [1 .. num_fingers_] = fingers, [1+num_fingers_] = object.
        gtsam::Matrix36 D_cthumb_pose;
        const gtsam::Pose3& thumb_pose = x.at<gtsam::Pose3>(keys()[0]);
        gtsam::Point3 c_thumb = thumb_pose.translation(H ? &D_cthumb_pose : nullptr);

        std::vector<gtsam::Matrix36> D_ci_pose(num_fingers_);
        gtsam::Vector3 c_fingers_sum = gtsam::Vector3::Zero();
        for (size_t j = 0; j < num_fingers_; ++j) {
            const gtsam::Pose3& finger_pose = x.at<gtsam::Pose3>(keys()[1 + j]);
            gtsam::Point3 c_i = finger_pose.translation(H ? &D_ci_pose[j] : nullptr);
            c_fingers_sum += c_i;
        }
        gtsam::Vector3 c_fingers_mean = (num_fingers_ > 0)
            ? gtsam::Vector3(c_fingers_sum / double(num_fingers_))
            : gtsam::Vector3::Zero();

        gtsam::Matrix36 D_pobj_pose;
        const gtsam::Pose3& object_pose = x.at<gtsam::Pose3>(keys()[1 + num_fingers_]);
        gtsam::Point3 p_obj = object_pose.translation(H ? &D_pobj_pose : nullptr);

        gtsam::Vector3 c_hand = 0.5 * (c_thumb + c_fingers_mean);
        gtsam::Vector3 target = p_obj + h_clear_ * n_hat_;
        gtsam::Vector3 e = c_hand - target;

        if (H) {
            H->resize(2 + num_fingers_);
            (*H)[0] = 0.5 * D_cthumb_pose;   // d(c_hand)/d(thumb), 3x6
            double w = (num_fingers_ > 0) ? 0.5 / double(num_fingers_) : 0.0;
            for (size_t j = 0; j < num_fingers_; ++j)
                (*H)[1 + j] = w * D_ci_pose[j];   // d(c_hand)/d(finger_j), 3x6
            (*H)[1 + num_fingers_] = -D_pobj_pose; // d(-target)/d(object), 3x6
        }

        return e;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PreGraspHandCenteringFactor(*this)));
    }
};


// Pre-grasp short-axis alignment (companion to the opposition half-space,
// Eq 2.16-2.17): the vector between the thumb's and the opposing fingers'
// contact centroids should align with the PERPENDICULAR to the half-space
// split plane -- the same in-plane axis m_hat the opposition split itself
// uses (perpendicular to the object's longest in-plane axis, hence
// implicitly its shortest in-plane axis) -- NOT the object's raw 3D shortest
// axis, which can be entirely out of the table plane (e.g. a coin's
// thickness). Direction-agnostic: squaring the cosine means it does not
// matter which of the two antiparallel directions m_hat happens to point.
//
//   v = c_thumb - mean_{i in F} c_i
//   v_hat = v / |v|
//   c_align(v, m_hat) = 1 - (v_hat . m_hat)^2 = 0
//
// m_hat is a FROZEN constant, supplied by the caller (Python computes it once
// via config.opposition_axis_from_object, the same helper the opposition
// half-space itself uses) -- matching HalfSpaceGapFactor's own convention for
// this axis: it is not re-derived from a live object Pose3 each iteration, so
// this factor carries no object key at all.
//
// Variable arity like PreGraspHandCenteringFactor -- spans the thumb and an
// arbitrary number of opposing fingers, so it derives from
// gtsam::NoiseModelFactor directly (not NoiseModelFactorN) with a hand-built
// KeyVector, following the same TendonLengthFactor-style pattern.
//
// Keys: [thumb_pose_key, finger_pose_key_0, ..., finger_pose_key_{|F|-1}]
// Residual: scalar.
class PreGraspAxisAlignmentFactor : public gtsam::NoiseModelFactor {
private:
    size_t num_fingers_;
    gtsam::Vector3 m_hat_;

public:
    using NoiseModelFactor::unwhitenedError;

    PreGraspAxisAlignmentFactor(gtsam::Key thumb_pose_key,
                                const std::vector<gtsam::Key>& finger_pose_keys,
                                const gtsam::Vector3& target_axis,
                                const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor(noise_model, /* keys: */ [&]() {
              gtsam::KeyVector keys;
              keys.push_back(thumb_pose_key);
              keys.insert(keys.end(), finger_pose_keys.begin(), finger_pose_keys.end());
              return keys;
          }()),
          num_fingers_(finger_pose_keys.size()),
          m_hat_(target_axis.normalized()) {}

    gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                  gtsam::OptionalMatrixVecType H = nullptr) const override
    {
        // Key layout: [0] = thumb, [1 .. num_fingers_] = opposing fingers.
        gtsam::Matrix36 D_cthumb_pose;
        const gtsam::Pose3& thumb_pose = x.at<gtsam::Pose3>(keys()[0]);
        gtsam::Point3 c_thumb = thumb_pose.translation(H ? &D_cthumb_pose : nullptr);

        std::vector<gtsam::Matrix36> D_ci_pose(num_fingers_);
        gtsam::Vector3 c_sum = gtsam::Vector3::Zero();
        for (size_t j = 0; j < num_fingers_; ++j) {
            const gtsam::Pose3& finger_pose = x.at<gtsam::Pose3>(keys()[1 + j]);
            gtsam::Point3 c_i = finger_pose.translation(H ? &D_ci_pose[j] : nullptr);
            c_sum += c_i;
        }
        gtsam::Vector3 c_mean = (num_fingers_ > 0)
            ? gtsam::Vector3(c_sum / double(num_fingers_))
            : gtsam::Vector3::Zero();

        gtsam::Vector3 v = c_thumb - c_mean;
        double vn = v.norm();
        if (vn < 1e-9) vn = 1e-9;
        gtsam::Vector3 v_hat = v / vn;

        double d = v_hat.dot(m_hat_);
        double e = 1.0 - d * d;   // 0 <=> colinear with m_hat, either direction

        if (H) {
            H->resize(1 + num_fingers_);
            // de/dv_hat = -2 d * m_hat^T ; dv_hat/dv = (I - v_hat v_hat^T)/|v|
            const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
            gtsam::Matrix3 P = (I3 - v_hat * v_hat.transpose()) / vn;
            gtsam::Matrix13 de_dv = (-2.0 * d * m_hat_.transpose()) * P;

            (*H)[0] = de_dv * D_cthumb_pose;   // d(e)/d(thumb), 1x6
            double w = (num_fingers_ > 0) ? -1.0 / double(num_fingers_) : 0.0;
            for (size_t j = 0; j < num_fingers_; ++j)
                (*H)[1 + j] = (w * de_dv) * D_ci_pose[j];   // d(e)/d(finger_j), 1x6
        }

        return gtsam::Vector1(e);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PreGraspAxisAlignmentFactor(*this)));
    }
};


// Pre-grasp PINCH-CENTROID centering: put a point that is FIXED in the wrist
// frame onto the object (raised by a clearance along n_hat).
//
//   c_world(T_wrist) = T_wrist * c_local
//   target(T_obj)    = p_obj + h_clear * n_hat
//   c_centroid       = c_world - target = 0        (Vector3)
//
// The hardcoded-point counterpart of PreGraspHandCenteringFactor: that factor
// averages the thumb's and the opposing fingers' ACHIEVED sphere centers, so
// it only says something once the fingers are already near the grasp. c_local
// here is instead the offline-measured point where a given finger combination
// meets (HAND_PINCH_POSES, keyed by which fingers are checked), so this
// constrains where the HAND must be for that pinch to land on the object --
// a statement about the wrist alone, true whatever the fingers are doing.
//
// Because c_local is constant, only two variables enter and the arity is fixed
// -- so unlike its two siblings this is a plain NoiseModelFactorN and gets
// evaluateError() rather than a hand-built KeyVector and unwhitenedError().
//
// It also means the factor references the wrist variable DIRECTLY
// (TendonHandModel::wrist_key), sidestepping the root-reparameterization trap
// that node-0 of a finger has no pose key of its own when uses_root() -- there
// is no finger node here to remap.
//
// Keys: [wrist_pose_key, object_key].  Residual: Vector3.
class PreGraspCentroidFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    gtsam::Point3 c_local_;
    double h_clear_;
    gtsam::Vector3 n_hat_;

public:
    PreGraspCentroidFactor(gtsam::Key wrist_key,
                           gtsam::Key object_key,
                           const gtsam::Point3& centroid_local,
                           double h_clear,
                           const gtsam::Vector3& n_hat,
                           const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, wrist_key, object_key),
          c_local_(centroid_local),
          h_clear_(h_clear),
          n_hat_(n_hat.normalized()) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& wrist_pose,
                                const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H_wrist,
                                gtsam::OptionalMatrixType H_object) const override
    {
        // Constant body-frame point pushed to the world, with its 3x6
        // Jacobian -- the same primitive TendonLengthFactor uses for a
        // routing hole.
        gtsam::Matrix36 D_cworld_wrist;
        gtsam::Point3 c_world =
            wrist_pose.transformFrom(c_local_, H_wrist ? &D_cworld_wrist : nullptr);

        gtsam::Matrix36 D_pobj_pose;
        gtsam::Point3 p_obj =
            object_pose.translation(H_object ? &D_pobj_pose : nullptr);

        if (H_wrist) *H_wrist = D_cworld_wrist;
        if (H_object) *H_object = -D_pobj_pose;

        return gtsam::Vector3(c_world - (p_obj + h_clear_ * n_hat_));
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PreGraspCentroidFactor(*this)));
    }
};


// Finger-finger (sphere-to-sphere) penetration gap (Section 1.5):
//   c_pen(p_i, p_j) = (r_i + r_j) - ||p_i - p_j||
// where p_i, p_j are the world-frame centers of the two collision spheres
// (the translations of the two Pose3 variables) and r_i, r_j their radii.
// c_pen > 0 means the two spheres overlap. Analytical Jacobians (writeup):
//   d c_pen / d p_i = -(p_i - p_j)/||p_i - p_j||,
//   d c_pen / d p_j = +(p_i - p_j)/||p_i - p_j||.
// Only the translations enter the residual; the rotation Jacobian blocks are
// zero. Wrap an instance in a CollisionInequalityConstraint to enforce
// c_pen <= 0.
class SphereSphereCollisionGapFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3>
{
private:
    double r_a_, r_b_;

public:
    SphereSphereCollisionGapFactor(gtsam::Key pose_a_key, gtsam::Key pose_b_key,
                                   double r_a, double r_b,
                                   const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, pose_a_key, pose_b_key),
          r_a_(r_a), r_b_(r_b) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_a,
                                const gtsam::Pose3& pose_b,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_ca_pose, D_cb_pose;
        gtsam::Point3 c_a = pose_a.translation(H1 ? &D_ca_pose : nullptr);
        gtsam::Point3 c_b = pose_b.translation(H2 ? &D_cb_pose : nullptr);

        gtsam::Vector3 d = c_a - c_b;
        double dn = d.norm();
        if (dn < 1e-7) dn = 1e-7;
        gtsam::Vector3 n = d / dn;      // unit vector from c_b toward c_a

        double c_pen = (r_a_ + r_b_) - dn;   // > 0  <=>  overlap

        // d c_pen / d c_a = -n^T ,  d c_pen / d c_b = +n^T
        if (H1) *H1 = -n.transpose() * D_ca_pose;   // 1x6
        if (H2) *H2 =  n.transpose() * D_cb_pose;   // 1x6

        return gtsam::Vector1(c_pen);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SphereSphereCollisionGapFactor(*this)));
    }
};


// Surface-to-surface witness-point contact factor (Eq 30-31).
// Connects:
//   - node_pose_key  (Pose3)   : finger node whose sphere should touch the surface
//   - object_key     (Pose3)   : object pose
//   - point_key      (Point3)  : dummy contact point p_c in world frame
//
// 5D residual = [ ||p_c - p_i||_2 - R,            (c_R)
//                 SDF(T_obj^{-1} p_c),            (c_O)
//                 1 + N_i . N_obj,                (c_N)
//                 (p_c - p_i) . t1(N_obj),        (c_T1)
//                 (p_c - p_i) . t2(N_obj) ].      (c_T2)
// Rows 0-1 drive p_c onto both the body sphere (radius R around the tip) and
// the object surface — tangential contact. Row 2 enforces antiparallel
// alignment of the body-sphere outward normal (N_i = (p_c - c_i)/||.||) and
// the object surface normal (N_obj = R_obj * normalize(grad SDF)). Rows 3-4
// are the C-frame gauge-fixing residuals: t1, t2 span the object surface's
// tangent plane (Frisvad basis of N_obj), so penalizing the projection of
// (p_c - p_i) onto them pins p_c strictly along the contact normal axis. This
// removes the residual gauge freedom of p_c and yields a full-rank gradient,
// so no Tikhonov regularizer / stabilizing prior on p_c is required.
//
// drop_normal_row (Section 1.8, Eq 1.107-1.110): when true the c_N row is omitted
// and the residual is the 4-vector [c_R, c_O, c_T1, c_T2]. Justified in §1.8:
// because the robot's collision geometry is modeled exclusively as spheres, the
// tangential-slip rows already force (p_c - p_i) -- and hence the outward sphere
// normal -- collinear with the object surface normal, making c_N redundant. The
// caller must size its noise model to match (Isotropic::Sigma(4, ...)).
class SdfWitnessContactFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> {
private:
    double R_;
    openvdb::FloatGrid::Ptr sdf_grid_;
    bool drop_normal_row_;

public:
    SdfWitnessContactFactor(gtsam::Key node_pose_key, gtsam::Key object_key, gtsam::Key point_key,
                            double radius, const openvdb::FloatGrid::Ptr& sdf_grid,
                            const gtsam::SharedNoiseModel& noise_model,
                            bool drop_normal_row = false)
        : NoiseModelFactor3(noise_model, node_pose_key, object_key, point_key),
          R_(radius), sdf_grid_(sdf_grid), drop_normal_row_(drop_normal_row) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        // --- e1 = ||p_c - c_i|| - R --------------------------------------
        gtsam::Matrix36 D_center_pose;
        gtsam::Point3 center = node_pose.translation(H1 ? &D_center_pose : nullptr);

        gtsam::Vector3 diff = dummy_point - center;
        double d = diff.norm();
        if (d < 1e-7) d = 1e-7;
        double e1 = d - R_;
        gtsam::Vector3 n_i = diff / d;  // body-sphere outward normal (world frame)

        // --- e2 = SDF(T_obj^{-1} p_c) ------------------------------------
        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_point;
        gtsam::Point3 p_local = object_pose.transformTo(dummy_point,
            (H2 || H3) ? &D_plocal_obj   : nullptr,
            (H2 || H3) ? &D_plocal_point : nullptr);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R vdb_pt(p_local.x(), p_local.y(), p_local.z());
        double e2 = sampler.wsSample(vdb_pt);

        // FD SDF gradient in object-local frame. Reused for e2 Jacobian and
        // to build N_obj for e3.
        double h = 1e-4;
        double dx = sampler.wsSample(openvdb::Vec3R(vdb_pt.x() + h, vdb_pt.y(), vdb_pt.z())) -
                    sampler.wsSample(openvdb::Vec3R(vdb_pt.x() - h, vdb_pt.y(), vdb_pt.z()));
        double dy = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() + h, vdb_pt.z())) -
                    sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() - h, vdb_pt.z()));
        double dz = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() + h)) -
                    sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() - h));
        gtsam::Vector3 n_obj_local(dx, dy, dz);
        double g_norm = n_obj_local.norm();
        if (g_norm > 1e-8) n_obj_local /= g_norm;
        else               n_obj_local = gtsam::Vector3(0.0, 0.0, 1.0);

        // --- e3 = 1 + N_i . N_obj_world ----------------------------------
        gtsam::Matrix3 R_obj = object_pose.rotation().matrix();
        gtsam::Vector3 n_obj_world = R_obj * n_obj_local;
        double e3 = 1.0 + n_i.dot(n_obj_world);

        // --- e4, e5 = C-frame gauge fixing (Eq 30-31) --------------------
        // Build a deterministic tangent basis (t1, t2) of the object surface
        // normal and penalize the projection of v = (p_c - c_i) onto it, so
        // p_c is pinned along the contact normal axis. t1, t2 are treated as
        // constant within the local Gauss-Newton step (C-frame held fixed --
        // standard SOTA contact convention), so their Jacobian contribution
        // reduces to the tangent vectors themselves.
        gtsam::Vector3 t1, t2;
        frisvad_tangent_basis(n_obj_world, t1, t2);
        gtsam::Vector3 v = diff;  // p_c - c_i
        double e4 = v.dot(t1);
        double e5 = v.dot(t2);

        // Row layout: [c_R, c_O, (c_N), c_T1, c_T2]. Dropping c_N shifts the two
        // tangent rows up by one and shrinks the residual to 4.
        const int dim = drop_normal_row_ ? 4 : 5;
        const int rT1 = drop_normal_row_ ? 2 : 3;

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(dim, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(dim, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(dim, 3);

            // Row 0: e1 -- only c_i (via node_pose) and p_c.
            if (H1) H1->row(0) = -n_i.transpose() * D_center_pose;
            if (H3) H3->row(0) =  n_i.transpose();

            // Row 1: e2 -- via p_local(object_pose, p_c).
            gtsam::Matrix13 de2_dplocal = n_obj_local.transpose();
            if (H2) H2->row(1) = de2_dplocal * D_plocal_obj;
            if (H3) H3->row(1) = de2_dplocal * D_plocal_point;

            // Row 2: e3. Chain rule with the locally-constant-gradient
            // approximation -- treat n_obj_local as p_c-independent (the
            // standard locally-constant-gradient contact convention).
            // n_i = (p_c - c_i)/d, projector P = (I - n_i n_i^T)/d.
            // dn_i/dc_i = -P,   dn_i/dp_c = P.
            if (!drop_normal_row_) {
                const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
                gtsam::Matrix3 P = (I3 - n_i * n_i.transpose()) / d;
                Eigen::RowVector3d nobjT = n_obj_world.transpose();

                if (H1) H1->row(2) = -(nobjT * P) * D_center_pose;
                if (H3) H3->row(2) =  nobjT * P;

                // d(n_obj_world)/d(xi_obj) under GTSAM's Pose3 tangent order
                // [omega(3), upsilon(3)]:  d(R v)/d(omega) = -R * skew(v).
                // Translation has no effect on the normal under the
                // locally-constant-gradient approximation.
                if (H2) {
                    gtsam::Matrix3 dRv_dxiR = -R_obj * gtsam::skewSymmetric(n_obj_local);
                    H2->block<1, 3>(2, 0) = n_i.transpose() * dRv_dxiR;
                    H2->block<1, 3>(2, 3) = Eigen::RowVector3d::Zero();
                }
            }

            // Tangent rows: e4, e5. With t1, t2 constant, de/dp_c = t^T and
            // de/dc_i = -t^T (v = p_c - c_i). Object pose has no effect on v
            // under the fixed-C-frame approximation, so those H2 rows stay zero.
            if (H3) H3->row(rT1)     = t1.transpose();
            if (H3) H3->row(rT1 + 1) = t2.transpose();
            if (H1) H1->row(rT1)     = -t1.transpose() * D_center_pose;
            if (H1) H1->row(rT1 + 1) = -t2.transpose() * D_center_pose;
        }

        if (drop_normal_row_)
            return (gtsam::Vector(4) << e1, e2, e4, e5).finished();
        return (gtsam::Vector(5) << e1, e2, e3, e4, e5).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SdfWitnessContactFactor(*this)));
    }
};


// Surface-to-surface witness-point contact against an analytic hyper-ellipsoid
// (Section 1.6.3, Eq 1.89-1.90). The analytic analog of SdfWitnessContactFactor
// for an object whose surface is the ellipsoid x^T M x = 1 in the object-local
// frame, M = diag(a^-2, b^-2, c^-2). Connects:
//   - node_pose_key (Pose3)  : finger node whose sphere should touch the surface
//   - object_key    (Pose3)  : object pose (supplies R_obj for Eq 1.90)
//   - point_key     (Point3) : dummy contact point p_c in world frame
//
// 5D residual = [ ||p_c - c_i|| - R,               (c_R)
//                 (x^T M x - 1) / (2 ||M x||),      (c_O,  x = T_obj^{-1} p_c)
//                 1 + N_i . N_obj,                 (c_N,  N_obj = R_obj * normalize(M x))
//                 (p_c - c_i) . t1(N_obj),         (c_T1)
//                 (p_c - c_i) . t2(N_obj) ].       (c_T2)
// The c_O row uses the Taubin first-order distance to the surface x^T M x = 1
// rather than the raw algebraic value of Eq 1.89. Both share the identical zero
// set, but the raw form warps space non-uniformly (the paper itself switches to
// this Taubin form for the collision inequality, Eq 1.91): its residual and
// gradient scale as ~1/min(semi_axis)^2, ~40x the Euclidean distance on a 5 cm
// sphere and ~10^6x along a coin's thin axis, so under a shared unit noise model
// the raw c_O row swamps the others and the AL inner solve stagnates. The Taubin
// normalization makes c_O a well-scaled O(1) Euclidean-like distance with an
// exact analytic Jacobian, recovering SDF-level conditioning. The surface-normal
// rows reuse the standard locally-constant-gradient contact convention (N held
// fixed within the Gauss-Newton step), matching SdfWitnessContactFactor.
//
// drop_normal_row (Section 1.8, Eq 1.107-1.110): as on SdfWitnessContactFactor,
// omits the c_N row and returns the 4-vector [c_R, c_O, c_T1, c_T2].
class EllipsoidWitnessContactFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
{
private:
    double R_;
    gtsam::Vector3 m_diag_;   // (1/a^2, 1/b^2, 1/c^2)
    bool drop_normal_row_;

public:
    EllipsoidWitnessContactFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                                  gtsam::Key point_key, double radius,
                                  const gtsam::Vector3& semi_axes,
                                  const gtsam::SharedNoiseModel& noise_model,
                                  bool drop_normal_row = false)
        : NoiseModelFactor3(noise_model, node_pose_key, object_key, point_key),
          R_(radius),
          m_diag_(1.0 / (semi_axes.x() * semi_axes.x()),
                  1.0 / (semi_axes.y() * semi_axes.y()),
                  1.0 / (semi_axes.z() * semi_axes.z())),
          drop_normal_row_(drop_normal_row) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        // --- e1 = ||p_c - c_i|| - R --------------------------------------
        gtsam::Matrix36 D_center_pose;
        gtsam::Point3 center = node_pose.translation(H1 ? &D_center_pose : nullptr);

        gtsam::Vector3 diff = dummy_point - center;
        double d = diff.norm();
        if (d < 1e-7) d = 1e-7;
        double e1 = d - R_;
        gtsam::Vector3 n_i = diff / d;  // body-sphere outward normal (world frame)

        // --- e2 = (x^T M x - 1) / (2 ||M x||)  (Taubin), x = T_obj^{-1} p_c ---
        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_point;
        gtsam::Point3 x_local = object_pose.transformTo(dummy_point,
            (H2 || H3) ? &D_plocal_obj   : nullptr,
            (H2 || H3) ? &D_plocal_point : nullptr);

        gtsam::Vector3 Mx = m_diag_.cwiseProduct(x_local);   // M x
        double f_alg = x_local.dot(Mx) - 1.0;                // raw x^T M x - 1
        double g_alg = Mx.norm();                            // ||M x||
        if (g_alg < 1e-9) g_alg = 1e-9;
        double e2 = f_alg / (2.0 * g_alg);                   // Taubin distance

        // Object-local surface normal: grad(x^T M x) = 2 M x, normalized.
        gtsam::Vector3 n_obj_local = Mx;
        double g_norm = n_obj_local.norm();
        if (g_norm > 1e-8) n_obj_local /= g_norm;
        else               n_obj_local = gtsam::Vector3(0.0, 0.0, 1.0);

        // --- e3 = 1 + N_i . N_obj_world  (Eq 1.90) ----------------------
        gtsam::Matrix3 R_obj = object_pose.rotation().matrix();
        gtsam::Vector3 n_obj_world = R_obj * n_obj_local;
        double e3 = 1.0 + n_i.dot(n_obj_world);

        // --- e4, e5 = C-frame gauge fixing ------------------------------
        gtsam::Vector3 t1, t2;
        frisvad_tangent_basis(n_obj_world, t1, t2);
        gtsam::Vector3 v = diff;  // p_c - c_i
        double e4 = v.dot(t1);
        double e5 = v.dot(t2);

        // Row layout: [c_R, c_O, (c_N), c_T1, c_T2]. Dropping c_N shifts the two
        // tangent rows up by one and shrinks the residual to 4.
        const int dim = drop_normal_row_ ? 4 : 5;
        const int rT1 = drop_normal_row_ ? 2 : 3;

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(dim, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(dim, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(dim, 3);

            // Row 0: e1 -- only c_i (via node_pose) and p_c.
            if (H1) H1->row(0) = -n_i.transpose() * D_center_pose;
            if (H3) H3->row(0) =  n_i.transpose();

            // Row 1: e2 -- Taubin distance, exact analytic Jacobian.
            // e2 = f/(2g), f = x^T M x - 1, g = ||M x||.
            //   df/dx = 2 M x  (row 2 Mx^T),  dg/dx = (m_diag ∘ M x)^T / g
            //   de2/dx = f'/(2g) - f g'/(2 g^2)
            gtsam::Vector3 mMx = m_diag_.cwiseProduct(Mx);   // M (M x)
            gtsam::Matrix13 de2_dxlocal =
                  (Mx.transpose() / g_alg)
                - (f_alg / (2.0 * g_alg * g_alg * g_alg)) * mMx.transpose();
            if (H2) H2->row(1) = de2_dxlocal * D_plocal_obj;
            if (H3) H3->row(1) = de2_dxlocal * D_plocal_point;

            // Row 2: e3 -- locally-constant-gradient normal convention
            // (n_obj_local treated as p_c-independent within the GN step).
            // n_i = (p_c - c_i)/d, projector P = (I - n_i n_i^T)/d.
            if (!drop_normal_row_) {
                const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
                gtsam::Matrix3 P = (I3 - n_i * n_i.transpose()) / d;
                Eigen::RowVector3d nobjT = n_obj_world.transpose();

                if (H1) H1->row(2) = -(nobjT * P) * D_center_pose;
                if (H3) H3->row(2) =  nobjT * P;

                // d(n_obj_world)/d(xi_obj): d(R v)/d(omega) = -R * skew(v);
                // translation has no effect on the normal.
                if (H2) {
                    gtsam::Matrix3 dRv_dxiR = -R_obj * gtsam::skewSymmetric(n_obj_local);
                    H2->block<1, 3>(2, 0) = n_i.transpose() * dRv_dxiR;
                    H2->block<1, 3>(2, 3) = Eigen::RowVector3d::Zero();
                }
            }

            // Tangent rows: e4, e5. With t1, t2 constant, de/dp_c = t^T and
            // de/dc_i = -t^T. Object pose has no effect on v under the
            // fixed-C-frame approximation, so those H2 rows stay zero.
            if (H3) H3->row(rT1)     = t1.transpose();
            if (H3) H3->row(rT1 + 1) = t2.transpose();
            if (H1) H1->row(rT1)     = -t1.transpose() * D_center_pose;
            if (H1) H1->row(rT1 + 1) = -t2.transpose() * D_center_pose;
        }

        if (drop_normal_row_)
            return (gtsam::Vector(4) << e1, e2, e4, e5).finished();
        return (gtsam::Vector(5) << e1, e2, e3, e4, e5).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new EllipsoidWitnessContactFactor(*this)));
    }
};


// NOTE the Section 1.6 five-residual PlaneWitnessContactFactor (Eq 1.60-1.64)
// that used to live here has been REMOVED. The table sliding equality now
// constrains the contact sphere's CENTER directly, as the single residual
// PlaneCollisionGapFactor wrapped in a gtsam::ZeroCostConstraint (see
// TendonHandModel::build_graph and the support_contact_node notes above).
// Four of the witness form's five rows existed only to pin the gauge of the
// free contact point it introduced; for a PLANE that point buys nothing, since
// a scalar residual on the center leaves no rotational freedom to brick the
// solver and still lets the tip slide laterally.


// Sphere-sphere contact factor (analytical, 1-residual gap form). Use when
// both bodies are spheres (e.g. finger vs. spherical primitive). Connects
// two Pose3 variables whose translations are the sphere centers:
//
//   e = ||c_a - c_b|| - (r_a + r_b)
//
// e == 0 means tangent contact; e > 0 separated; e < 0 inter-penetrating.
// Only the translations enter the residual; rotation Jacobian blocks are
// zero. Single-residual form avoids the rank-deficient slack subspace that
// the 3-residual (p_c-bearing) form introduces when both surfaces are
// analytic spheres -- p_c is uniquely determined by c_a, c_b, r_a, r_b and
// is not a real degree of freedom here.
class SphereSphereContactFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
private:
    double r_a_, r_b_;

public:
    SphereSphereContactFactor(gtsam::Key pose_a_key, gtsam::Key pose_b_key,
                              double r_a, double r_b,
                              const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor2(noise_model, pose_a_key, pose_b_key),
          r_a_(r_a), r_b_(r_b) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_a,
                                const gtsam::Pose3& pose_b,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_ca_pose, D_cb_pose;
        gtsam::Point3 c_a = pose_a.translation(H1 ? &D_ca_pose : nullptr);
        gtsam::Point3 c_b = pose_b.translation(H2 ? &D_cb_pose : nullptr);

        gtsam::Vector3 d = c_a - c_b;
        double dn = d.norm();
        if (dn < 1e-7) dn = 1e-7;
        gtsam::Vector3 n = d / dn;       // unit vector from c_b toward c_a

        double e = dn - (r_a_ + r_b_);

        if (H1) *H1 =  n.transpose() * D_ca_pose;   // 1x6
        if (H2) *H2 = -n.transpose() * D_cb_pose;   // 1x6

        return (gtsam::Vector(1) << e).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SphereSphereContactFactor(*this)));
    }
};

// 5-residual sphere-to-sphere witness-point contact factor (Eq 30-31).
// Serves as an analytical counterpart of the 1-DoF SphereSphereContactFactor
// and the SDF-backed SdfWitnessContactFactor.
//
// Connects:
//   - pose_a_key  (Pose3)  : Body A (e.g., finger node)
//   - pose_b_key  (Pose3)  : Body B (e.g., primitive sphere -- the contacted object)
//   - point_key   (Point3) : Dummy contact point p_c in world frame
//
// 5D residual = [ ||p_c - c_a|| - r_a,        (c_R)
//                 ||p_c - c_b|| - r_b,        (c_O)
//                 1 + N_a . N_b,              (c_N)
//                 (p_c - c_a) . t1(N_b),      (c_T1)
//                 (p_c - c_a) . t2(N_b) ].    (c_T2)
// Rows 3-4 are the C-frame gauge-fixing residuals: t1, t2 span the tangent
// plane of body B's outward normal N_b (the contacted object), so penalizing
// the projection of (p_c - c_a) onto them pins p_c along the contact normal
// axis. This removes the genuine 1-DoF gauge freedom that rows 0-2 alone leave
// behind (rotating p_c about the center-to-center axis is invariant to them),
// so no stabilizing prior on p_c is required.
class SphereWitnessContactFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
{
private:
    double r_a_, r_b_;

public:
    SphereWitnessContactFactor(gtsam::Key pose_a_key, gtsam::Key pose_b_key, gtsam::Key point_key,
                               double r_a, double r_b,
                               const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor3(noise_model, pose_a_key, pose_b_key, point_key),
          r_a_(r_a), r_b_(r_b) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_a,
                                const gtsam::Pose3& pose_b,
                                const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        // --- Get centers and their Jacobians wrt Pose ---
        gtsam::Matrix36 D_ca_pose, D_cb_pose;
        gtsam::Point3 c_a = pose_a.translation(H1 ? &D_ca_pose : nullptr);
        gtsam::Point3 c_b = pose_b.translation(H2 ? &D_cb_pose : nullptr);

        // --- Sphere A Geometry ---
        gtsam::Vector3 d_a = dummy_point - c_a;
        double norm_a = d_a.norm();
        if (norm_a < 1e-7) norm_a = 1e-7;
        gtsam::Vector3 n_a = d_a / norm_a; // Outward unit normal from A

        // --- Sphere B Geometry ---
        gtsam::Vector3 d_b = dummy_point - c_b;
        double norm_b = d_b.norm();
        if (norm_b < 1e-7) norm_b = 1e-7;
        gtsam::Vector3 n_b = d_b / norm_b; // Outward unit normal from B

        // --- Residuals ---
        double e1 = norm_a - r_a_;
        double e2 = norm_b - r_b_;
        double e3 = 1.0 + n_a.dot(n_b);

        // --- e4, e5 = C-frame gauge fixing (Eq 30-31) --------------------
        // Tangent basis of body B's normal (the contacted object), used to pin
        // p_c along the contact normal axis. t1, t2 held constant within the
        // local Gauss-Newton step (C-frame fixed), so their Jacobian reduces to
        // the tangent vectors themselves.
        gtsam::Vector3 t1, t2;
        frisvad_tangent_basis(n_b, t1, t2);
        gtsam::Vector3 v = d_a;  // p_c - c_a
        double e4 = v.dot(t1);
        double e5 = v.dot(t2);

        // --- Jacobians ---
        if (H1 || H2 || H3) {
            // GTSAM passes these in as default-constructed 0x0 matrices
            // (see NoiseModelFactor::linearize: `std::vector<Matrix> A(size())`).
            // We must ASSIGN a correctly-sized matrix to resize the storage --
            // H->setZero() does NOT resize a dynamic Eigen matrix, so the later
            // H->row()/H->block() writes would scribble past a 0-byte allocation
            // and corrupt the heap.
            if (H1) *H1 = gtsam::Matrix::Zero(5, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(5, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(5, 3);

            // Row 0: e1 (Tangent to Sphere A)
            // de1/dc_a = -n_a^T, de1/dp_c = n_a^T
            if (H1) H1->row(0) = -n_a.transpose() * D_ca_pose;
            if (H3) H3->row(0) =  n_a.transpose();

            // Row 1: e2 (Tangent to Sphere B)
            // de2/dc_b = -n_b^T, de2/dp_c = n_b^T
            if (H2) H2->row(1) = -n_b.transpose() * D_cb_pose;
            if (H3) H3->row(1) =  n_b.transpose();

            // // Row 2: e3 (Normal alignment)
            // // Projectors for unit vectors: P = (I - n*n^T)/norm
            // const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
            // gtsam::Matrix3 P_a = (I3 - n_a * n_a.transpose()) / norm_a;
            // gtsam::Matrix3 P_b = (I3 - n_b * n_b.transpose()) / norm_b;

            // // dn_a/dp_c = P_a,  dn_a/dc_a = -P_a
            // // dn_b/dp_c = P_b,  dn_b/dc_b = -P_b
            // Eigen::RowVector3d naT = n_a.transpose();
            // Eigen::RowVector3d nbT = n_b.transpose();

            // // Chain rule: de3 = n_b^T * dn_a + n_a^T * dn_b
            // if (H1) H1->row(2) = (-nbT * P_a) * D_ca_pose;
            // if (H2) H2->row(2) = (-naT * P_b) * D_cb_pose;
            // if (H3) H3->row(2) =  nbT * P_a + naT * P_b;
            // Row 2: e3 (Normal alignment)
            const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
            gtsam::Matrix3 P_a = (I3 - n_a * n_a.transpose()) / norm_a;
            gtsam::Matrix3 P_b = (I3 - n_b * n_b.transpose()) / norm_b;

            // Force evaluation into concrete 1x3 matrices to prevent lazy-evaluation memory aliasing
            gtsam::Matrix13 de3_dna = n_b.transpose();
            gtsam::Matrix13 de3_dnb = n_a.transpose();

            gtsam::Matrix13 de3_dca = -de3_dna * P_a;
            gtsam::Matrix13 de3_dcb = -de3_dnb * P_b;

            if (H1) H1->row(2) = de3_dca * D_ca_pose;
            if (H2) H2->row(2) = de3_dcb * D_cb_pose;
            if (H3) H3->row(2) = -de3_dca - de3_dcb; // Note: dn_a/dp_c = P_a, so de3/dp_c = de3_dna*P_a = -de3_dca

            // Rows 3-4: e4, e5. With t1, t2 constant, de/dp_c = t^T and
            // de/dc_a = -t^T (v = p_c - c_a). Body B has no effect on v under
            // the fixed-C-frame approximation, so H2 rows 3-4 stay zero.
            if (H3) H3->row(3) = t1.transpose();
            if (H3) H3->row(4) = t2.transpose();
            if (H1) H1->row(3) = -t1.transpose() * D_ca_pose;
            if (H1) H1->row(4) = -t2.transpose() * D_ca_pose;
        }

        return (gtsam::Vector(5) << e1, e2, e3, e4, e5).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SphereWitnessContactFactor(*this)));
    }
};

} // namespace crest_sparse
