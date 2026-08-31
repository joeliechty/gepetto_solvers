#pragma once

// The opposition half-space, constant Jacobian (Eq 1.99).

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

}  // namespace gepetto_solvers
