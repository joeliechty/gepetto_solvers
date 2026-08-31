#pragma once

// Finger-finger clearance (Eq 1.58).

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

}  // namespace gepetto_solvers
