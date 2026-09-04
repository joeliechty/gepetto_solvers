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
#include "gepetto_solvers/environment/EllipsoidDistance.h"

namespace gepetto_solvers {

// Finger-ellipsoid (sphere-to-analytic-ellipsoid) penetration gap
// (Section 1.6.3, Eq 1.91). The analytic analog of SdfCollisionGapFactor for a
// hyper-ellipsoid object surface with shape matrix M = diag(a^-2, b^-2, c^-2).
// The raw algebraic value x^T M x - 1 warps space non-uniformly (it is not a
// Euclidean distance), so the residual is built on a real signed distance:
//   x       = T_obj^{-1} p_i          (object-local sphere center)
//   dist    = EllipsoidDistance(semi_axes, taubin).signed_distance(x)
//   c_pen   = r_i - dist              (> 0 <=> penetration)
// `taubin` picks WHICH distance -- exact orthogonal (default) or the first-order
// algebraic approximation this factor used to hard-code; see the TWO METRICS
// note on EllipsoidDistance. Either way the gradient is fully analytic (no SDF
// sampling). Wrap an instance in a CollisionInequalityConstraint to enforce
// c_pen <= 0.
class EllipsoidCollisionGapFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    double radius_;
    EllipsoidDistance metric_;

public:
    EllipsoidCollisionGapFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                                double radius, const gtsam::Vector3& semi_axes,
                                const gtsam::SharedNoiseModel& noise_model,
                                bool taubin = false)
        : NoiseModelFactorN(noise_model, node_pose_key, object_key),
          radius_(radius),
          metric_(semi_axes, taubin) {}

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

        gtsam::Matrix13 ddist_dx;
        const double dist  = metric_.signed_distance(
            x, (H1 || H2) ? &ddist_dx : nullptr);
        const double c_pen = radius_ - dist;           // > 0 <=> penetration

        if (H1 || H2) {
            // c_pen = r - dist, so the whole row flips sign off the distance's.
            const gtsam::Matrix13 dcpen_dplocal = -ddist_dx;
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
