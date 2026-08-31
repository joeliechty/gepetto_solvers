#pragma once

// Finger-table clearance (Eq 1.59).

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

}  // namespace gepetto_solvers
