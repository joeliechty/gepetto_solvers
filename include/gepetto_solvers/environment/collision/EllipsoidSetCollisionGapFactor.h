#pragma once

// Clearance against a smooth-min union of ellipsoids.

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
#include "gepetto_solvers/environment/EnvironmentConfig.h"
#include "gepetto_solvers/environment/ConstraintWrappers.h"

namespace gepetto_solvers {

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
//     single-ellipsoid precedent is HandModel::build_graph, which already wraps
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

}  // namespace gepetto_solvers
