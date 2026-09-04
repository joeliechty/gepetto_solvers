#pragma once

// Net virtual-wrench equilibrium over a set of sphere CENTERS, against the
// analytic ellipsoid set (Eq h_grasp,E).

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "gepetto_solvers/environment/EnvironmentConfig.h"
#include "gepetto_solvers/environment/EllipsoidDistance.h"

namespace gepetto_solvers {

// Approximate geometric grasp alignment: the contacts must SURROUND the object
// during the approximation phase, before any exact SDF contact exists.
//
//   h_grasp,E({c_i}, E_obj) = sum_i [        -n_i                ]  = 0  (Vector6)
//                                   [ -(c_i - t_obj) x n_i       ]
//
// with n_i = R_obj * normalize(grad_x d_geom(x, E_obj)) at x = T_obj^{-1} c_i --
// the object's outward surface normal of the ELLIPSOID SET, evaluated at the
// sphere center and pushed into the world frame.
//
// THE RADIUS DROPS OUT, and that is the whole reason this factor can exist.
// A unit virtual force is applied along the inward normal -n_i at the surface
// contact point p_i = c_i - r_i n_i, so the torque about the object origin is
//
//   -(p_i - t_obj) x n_i = -(c_i - t_obj) x n_i + r_i (n_i x n_i)
//                        = -(c_i - t_obj) x n_i
//
// because the applied force is collinear with the radius vector. No r_i survives,
// so the constraint needs no witness variable and keys directly off the sphere
// center. That is exactly what the SDF sibling cannot do.
//
// RELATION TO GraspAlignmentFactor. Same 6-vector, same purely KINEMATIC
// equilibrium: every contact is credited with one unit "virtual force", and the
// constraint says those forces and their torques cancel. No mass, no friction
// cone, no force magnitudes. The differences are all about WHERE the normal comes
// from and WHAT the factor keys off:
//
//   GraspAlignmentFactor        this factor
//   ------------------------    -------------------------------------------
//   Point3 witness keys         Pose3 site-pose keys (the sphere centers)
//   baked OpenVDB SDF grid      analytic ellipsoid set (LogSumExp smooth min)
//   normals by FD on the grid   normals from the closed-form distance gradient
//   refuses center-direct       IS the center-direct form
//
// The SDF sibling throws on any digit using the center-direct contact form,
// because a witness point is the variable it keys off. This one is built for
// precisely that case -- the approximation phase, where the hand is still being
// steered by the smooth ellipsoid proxy and no witness exists yet. The two are
// independent constraints and may both be enabled.
//
// Because the forces are unit and the normals are unit, the residual is
// dimensionless in its top three rows and has units of length in its bottom
// three (a moment arm). Size the noise model accordingly: for a centimetre-scale
// object the two halves are within an order of magnitude of each other, but a
// large object wants the torque rows loosened relative to the force rows.
//
// Variable arity: |C| is runtime-determined, so -- like GraspAlignmentFactor and
// PreGraspHandCenteringFactor -- this derives from gtsam::NoiseModelFactor
// directly (not NoiseModelFactorN) and hand-builds its KeyVector.
//
// Keys: [node_pose_key_0, ..., node_pose_key_{|C|-1}, object_key]. Residual:
// Vector6.
//
// Wrap in gtsam::ZeroCostConstraint to hand it to the AL optimizer as the
// equality h_grasp,E = 0, exactly as the sibling is wrapped.
//
// NORMAL DERIVATIVES ARE *NOT* DROPPED HERE, for the same reason the sibling
// gives. The contact and gap factors hold the surface normal constant within a
// Gauss-Newton step (the locally-constant-gradient convention), which is sound
// for them: their residuals are distances, which keep a first-order dependence
// on the point even with a frozen normal. It would be fatal here. With
// dn_i/dc_i == 0 the force rows have an IDENTICALLY ZERO Jacobian with respect
// to every sphere center, so the solver would be told that sliding a contact
// around the object cannot change the force balance, and the constraint would
// only ever act on the object pose. So this factor differentiates the normal
// field too, via the finite-difference shape operator dn/dc.
class EllipsoidGraspAlignmentFactor : public gtsam::NoiseModelFactor {
private:
    size_t num_contacts_;
    double beta_;
    double curvature_step_;
    std::vector<EllipsoidDistance> metric_;   // per-k signed distance field
    std::vector<gtsam::Matrix3> Rk_T_;        // per-k R_k^T, precomputed
    std::vector<gtsam::Vector3> tk_;          // per-k t_k

public:
    using NoiseModelFactor::unwhitenedError;

    // `curvature_step` is the central-difference stencil for dn/dc, and its 0.0
    // sentinel means 1e-5 m -- three orders of magnitude tighter than the SDF
    // sibling's half-a-voxel default, deliberately. That factor differences a
    // TRILINEAR INTERPOLANT: inside a voxel the interpolant's gradient is
    // constant and its second derivative identically zero, so a sub-voxel
    // stencil measures that cell's own linear fit rather than the geometry, and
    // the voxel size is a hard floor on the usable step. Here the field is a
    // CLOSED FORM with no such floor, so the step is the ordinary truncation
    // (O(h^2)) against round-off (O(eps/h)) trade, and 1e-5 m sits near its
    // optimum for double precision on centimetre-scale geometry. Raise it only
    // to smooth a near-degenerate (very eccentric) member; as always for a
    // constraint Jacobian, smoothing is the safe direction to err in.
    EllipsoidGraspAlignmentFactor(const std::vector<gtsam::Key>& node_pose_keys,
                                  gtsam::Key object_key,
                                  const std::vector<EllipsoidPrimitive>& ellipsoids,
                                  double beta,
                                  const gtsam::SharedNoiseModel& noise_model,
                                  bool taubin = false,
                                  double curvature_step = 0.0)
        : NoiseModelFactor(noise_model, /* keys: */ [&]() {
              gtsam::KeyVector keys(node_pose_keys.begin(), node_pose_keys.end());
              keys.push_back(object_key);
              return keys;
          }()),
          num_contacts_(node_pose_keys.size()),
          beta_(beta),
          curvature_step_(curvature_step)
    {
        // Silent-garbage cases, not recoverable ones -- the same three the set
        // gap factor rejects, plus the arity check the sibling makes. An empty
        // set has no normal to report; a non-positive beta flips the smooth min
        // into a smooth MAX; no contacts is a wrench over nothing.
        if (num_contacts_ == 0)
            throw std::invalid_argument(
                "EllipsoidGraspAlignmentFactor: no contact points");
        if (ellipsoids.empty())
            throw std::invalid_argument(
                "EllipsoidGraspAlignmentFactor: the ellipsoid set is empty");
        if (!(beta > 0.0))
            throw std::invalid_argument(
                "EllipsoidGraspAlignmentFactor: beta must be > 0 (got " +
                std::to_string(beta) + ")");
        if (curvature_step_ <= 0.0) curvature_step_ = 1e-5;

        metric_.reserve(ellipsoids.size());
        Rk_T_.reserve(ellipsoids.size());
        tk_.reserve(ellipsoids.size());
        for (size_t k = 0; k < ellipsoids.size(); ++k) {
            const gtsam::Vector3& a = ellipsoids[k].semi_axes;
            if (!(a.x() > 0.0 && a.y() > 0.0 && a.z() > 0.0))
                throw std::invalid_argument(
                    "EllipsoidGraspAlignmentFactor: ellipsoid " + std::to_string(k) +
                    " has a non-positive semi-axis");
            metric_.emplace_back(a, taubin);
            // x_k = T_k^{-1} p_obj = R_k^T (p_obj - t_k); T_k is constant, so
            // cache the two pieces and skip a Pose3 inverse per member per
            // evaluation. Same cache as EllipsoidSetCollisionGapFactor.
            Rk_T_.push_back(ellipsoids[k].local_pose.rotation().matrix().transpose());
            tk_.push_back(ellipsoids[k].local_pose.translation());
        }
    }

    gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                  gtsam::OptionalMatrixVecType H = nullptr) const override
    {
        // Key layout: [0 .. |C|-1] = sphere-center site poses, [|C|] = object.
        const gtsam::Pose3& object_pose = x.at<gtsam::Pose3>(keys()[num_contacts_]);
        gtsam::Matrix36 D_tobj_pose;
        gtsam::Point3 t_obj = object_pose.translation(H ? &D_tobj_pose : nullptr);
        const gtsam::Matrix3 R_obj = object_pose.rotation().matrix();

        gtsam::Vector6 e = gtsam::Vector6::Zero();
        if (H) {
            H->resize(num_contacts_ + 1);
            (*H)[num_contacts_] = gtsam::Matrix::Zero(6, 6);
        }

        for (size_t i = 0; i < num_contacts_; ++i) {
            // The sphere CENTER, read off the site pose exactly as the gap
            // factors read it.
            const gtsam::Pose3& node_pose = x.at<gtsam::Pose3>(keys()[i]);
            gtsam::Matrix36 D_c_pose;
            gtsam::Point3 c_i = node_pose.translation(H ? &D_c_pose : nullptr);

            // c_local = T_obj^{-1} c_i, with d(c_local)/d(xi_obj) and
            // d(c_local)/d(c_i) = R_obj^T.
            gtsam::Matrix36 D_clocal_obj;
            gtsam::Matrix3 D_clocal_point;
            gtsam::Point3 c_local = object_pose.transformTo(
                c_i, H ? &D_clocal_obj : nullptr, H ? &D_clocal_point : nullptr);

            const gtsam::Vector3 n_local = normal_local(c_local);
            const gtsam::Vector3 n_i = R_obj * n_local;   // world outward normal
            const gtsam::Vector3 a_i = c_i - t_obj;       // moment arm

            // Unit inward virtual force, and its torque about the object origin.
            // The radius has already cancelled analytically (see the header), so
            // this is the surface wrench even though c_i is not on the surface.
            e.head<3>() += -n_i;
            e.tail<3>() += -a_i.cross(n_i);

            if (!H) continue;

            // Shape operator in the object-local frame, pushed to the world:
            //   M = d(n_i)/d(c_i) = R_obj * (dn_local/dc_local) * R_obj^T.
            const gtsam::Matrix3 G = dnormal_dlocal(c_local);
            const gtsam::Matrix3 M = R_obj * G * D_clocal_point;

            // r_top = -n,  r_bot = -(a x n) = skew(n) a = -skew(a) n
            //   d(r_bot) = skew(n) da - skew(a) dn,  with da/dc = I.
            // Both chain through d(c_i)/d(pose), which is what makes these
            // blocks 6x6 rather than the sibling's 6x3.
            const gtsam::Matrix3 skew_n = gtsam::skewSymmetric(n_i);
            const gtsam::Matrix3 skew_a = gtsam::skewSymmetric(a_i);

            gtsam::Matrix H_i = gtsam::Matrix::Zero(6, 6);
            H_i.block<3, 6>(0, 0) = -M * D_c_pose;
            H_i.block<3, 6>(3, 0) = (skew_n - skew_a * M) * D_c_pose;
            (*H)[i] = H_i;

            // d(n_i)/d(xi_obj), GTSAM Pose3 tangent order [omega(3), upsilon(3)]:
            // the normal turns with the object -- d(R v)/d(omega) = -R skew(v) --
            // AND the sample point slides in the local frame, which the shape
            // operator picks up through d(c_local)/d(xi_obj).
            gtsam::Matrix36 dn_dxi = gtsam::Matrix36::Zero();
            dn_dxi.block<3, 3>(0, 0) = -R_obj * gtsam::skewSymmetric(n_local);
            dn_dxi += R_obj * G * D_clocal_obj;

            // d(a_i)/d(xi_obj) = -d(t_obj)/d(xi_obj).
            (*H)[num_contacts_].block<3, 6>(0, 0) += -dn_dxi;
            (*H)[num_contacts_].block<3, 6>(3, 0) +=
                -skew_n * D_tobj_pose - skew_a * dn_dxi;
        }

        return e;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(
                new EllipsoidGraspAlignmentFactor(*this)));
    }

private:
    // n_local(c) = normalize(grad_c d_E(c)), the outward normal of the smooth-min
    // union at a point in the OBJECT frame.
    //
    // d_E = -(1/beta) ln sum_k exp(-beta d_k) (Eq 1.11), so its gradient is the
    // softmin-weighted blend of the per-member gradients, exactly as
    // EllipsoidSetCollisionGapFactor assembles it:
    //     grad d_E = sum_k (w_k / s) * grad d_k,   w_k = exp(-beta (d_k - d_min))
    // and the shift by d_min keeps every exponent <= 0, so exp() cannot overflow,
    // s >= 1, and an underflowing far member simply contributes weight 0.
    //
    // NORMALIZED UNCONDITIONALLY. The exact metric's per-member gradient is
    // already a unit vector everywhere (that is the conditioning argument for it
    // over Taubin), but a convex blend of unit vectors is NOT unit wherever two
    // members carry comparable weight -- i.e. exactly at the seams a sliding
    // finger spends its time -- and it shrinks toward zero where two members
    // pull in opposite directions. The residual here IS the normal, so an
    // unnormalized blend would silently under-credit a contact near a seam.
    //
    // Degenerate blends (opposed members cancelling, or a point where every
    // gradient underflows) fall back to +Z rather than exploding, matching
    // GraspAlignmentFactor and EllipsoidDistance::algebraic_normal.
    gtsam::Vector3 normal_local(const gtsam::Vector3& c) const {
        const size_t K = metric_.size();

        std::vector<double> d(K);
        std::vector<gtsam::Vector3> grad(K);
        double d_min = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < K; ++k) {
            gtsam::Vector3 xk = Rk_T_[k] * (c - tk_[k]);   // x_k = T_k^{-1} c
            gtsam::Matrix13 dd_dx;
            d[k] = metric_[k].signed_distance(xk, &dd_dx);
            // Into the OBJECT frame through the constant rotation: dx/dc = R_k^T.
            grad[k] = (dd_dx * Rk_T_[k]).transpose();
            if (d[k] < d_min) d_min = d[k];
        }

        double s = 0.0;
        gtsam::Vector3 g = gtsam::Vector3::Zero();
        for (size_t k = 0; k < K; ++k) {
            const double w = std::exp(-beta_ * (d[k] - d_min));
            s += w;
            g += w * grad[k];
        }
        g /= s;

        const double norm = g.norm();
        if (norm > 1e-8) return g / norm;
        return gtsam::Vector3(0.0, 0.0, 1.0);
    }

    // dn_local/dc_local by central differences of the already-normalized field,
    // rather than by assembling the Hessian of d_E and projecting. Same
    // arithmetic to first order, but differencing the normalized field keeps the
    // (I - n n^T) projection exact -- and it is the only tractable route through
    // the LogSumExp, whose second derivative couples every member's gradient to
    // every other member's weight.
    gtsam::Matrix3 dnormal_dlocal(const gtsam::Vector3& c) const {
        const double h = curvature_step_;
        gtsam::Matrix3 G;
        for (int j = 0; j < 3; ++j) {
            gtsam::Vector3 dc = gtsam::Vector3::Zero();
            dc(j) = h;
            G.col(j) = (normal_local(c + dc) - normal_local(c - dc)) / (2.0 * h);
        }
        return G;
    }
};

}  // namespace gepetto_solvers
