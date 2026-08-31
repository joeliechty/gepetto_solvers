#pragma once

// Finger-object clearance against a baked SDF (Eq 1.57).

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

}  // namespace gepetto_solvers
