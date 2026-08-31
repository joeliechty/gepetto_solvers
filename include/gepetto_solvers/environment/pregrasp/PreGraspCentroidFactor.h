#pragma once

// Pinch-centroid equality on the wrist alone.

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

namespace gepetto_solvers {

// Pre-grasp PINCH-CENTROID centering: put a point that is FIXED in the wrist
// frame onto the object (raised by a clearance along n_hat).
//
//   c_world(T_wrist) = T_wrist * c_local
//   target(T_obj)    = p_obj + h_clear * n_hat
//   c_centroid       = c_world - target = 0        (Vector3)
//
// The hardcoded-point counterpart of PreGraspHandCenteringFactor: that factor
// averages the thumb's and the opposing fingers' ACHIEVED sphere centers, so
// it only says something once the fingers are already near the grasp. c_local
// here is instead the offline-measured point where a given finger combination
// meets (HAND_PINCH_POSES, keyed by which fingers are checked), so this
// constrains where the HAND must be for that pinch to land on the object --
// a statement about the wrist alone, true whatever the fingers are doing.
//
// Because c_local is constant, only two variables enter and the arity is fixed
// -- so unlike its two siblings this is a plain NoiseModelFactorN and gets
// evaluateError() rather than a hand-built KeyVector and unwhitenedError().
//
// It also means the factor references the wrist variable DIRECTLY
// (TendonHandModel::wrist_key), sidestepping the root-reparameterization trap
// that node-0 of a finger has no pose key of its own when uses_root() -- there
// is no finger node here to remap.
//
// Keys: [wrist_pose_key, object_key].  Residual: Vector3.
class PreGraspCentroidFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    gtsam::Point3 c_local_;
    double h_clear_;
    gtsam::Vector3 n_hat_;

public:
    PreGraspCentroidFactor(gtsam::Key wrist_key,
                           gtsam::Key object_key,
                           const gtsam::Point3& centroid_local,
                           double h_clear,
                           const gtsam::Vector3& n_hat,
                           const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, wrist_key, object_key),
          c_local_(centroid_local),
          h_clear_(h_clear),
          n_hat_(n_hat.normalized()) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& wrist_pose,
                                const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H_wrist,
                                gtsam::OptionalMatrixType H_object) const override
    {
        // Constant body-frame point pushed to the world, with its 3x6
        // Jacobian -- the same primitive TendonLengthFactor uses for a
        // routing hole.
        gtsam::Matrix36 D_cworld_wrist;
        gtsam::Point3 c_world =
            wrist_pose.transformFrom(c_local_, H_wrist ? &D_cworld_wrist : nullptr);

        gtsam::Matrix36 D_pobj_pose;
        gtsam::Point3 p_obj =
            object_pose.translation(H_object ? &D_pobj_pose : nullptr);

        if (H_wrist) *H_wrist = D_cworld_wrist;
        if (H_object) *H_object = -D_pobj_pose;

        return gtsam::Vector3(c_world - (p_obj + h_clear_ * n_hat_));
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PreGraspCentroidFactor(*this)));
    }
};

}  // namespace gepetto_solvers
