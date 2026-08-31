#pragma once

// Fingertip on an ellipsoid, witness form (Eq 1.89-1.90).

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
#include "gepetto_solvers/environment/TangentBasis.h"

namespace gepetto_solvers {

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

}  // namespace gepetto_solvers
