#pragma once

// Finger-object clearance against one ellipsoid (Eq 1.91).

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

}  // namespace gepetto_solvers
