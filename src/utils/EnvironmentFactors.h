#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/constrained/NonlinearInequalityConstraint.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <Eigen/Core>

#include <optional>
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
    // collision_avoidance is the master on/off switch (default true). When
    // true, the planners add the AL inequality collision factors (finger-object
    // at every trajectory step, plus finger-finger in the hand). When false, no
    // collision factors are added at all.
    //
    // collision_node_is_proximal is a parallel vector to collision_node_indices
    // (1 = proximal, 0 = distal). It only matters for finger-finger collision in
    // the hand: a sphere pair is skipped iff BOTH spheres are proximal (the
    // metacarpal/base bones are rigidly attached to the shared hand base, so a
    // constraint between them is constant and useless). An empty vector means
    // every node is treated as non-proximal (no pair excluded).
    bool collision_avoidance = true;
    double collision_sigma   = 1e-3;
    std::vector<int>    collision_node_indices;
    std::vector<double> collision_node_radii;
    std::vector<int>    collision_node_is_proximal;

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
class SdfWitnessContactFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> {
private:
    double R_;
    openvdb::FloatGrid::Ptr sdf_grid_;

public:
    SdfWitnessContactFactor(gtsam::Key node_pose_key, gtsam::Key object_key, gtsam::Key point_key,
                            double radius, const openvdb::FloatGrid::Ptr& sdf_grid,
                            const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor3(noise_model, node_pose_key, object_key, point_key),
          R_(radius), sdf_grid_(sdf_grid) {}

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

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(5, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(5, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(5, 3);

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

            // Rows 3-4: e4, e5. With t1, t2 constant, de/dp_c = t^T and
            // de/dc_i = -t^T (v = p_c - c_i). Object pose has no effect on v
            // under the fixed-C-frame approximation, so H2 rows 3-4 stay zero.
            if (H3) H3->row(3) = t1.transpose();
            if (H3) H3->row(4) = t2.transpose();
            if (H1) H1->row(3) = -t1.transpose() * D_center_pose;
            if (H1) H1->row(4) = -t2.transpose() * D_center_pose;
        }

        return (gtsam::Vector(5) << e1, e2, e3, e4, e5).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SdfWitnessContactFactor(*this)));
    }
};


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
