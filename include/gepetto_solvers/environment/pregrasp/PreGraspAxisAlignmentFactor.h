#pragma once

// Short-axis alignment (companion to Eq 2.16-2.17).

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

// Pre-grasp short-axis alignment (companion to the opposition half-space,
// Eq 2.16-2.17): the vector between the thumb's and the opposing fingers'
// contact centroids should align with the PERPENDICULAR to the half-space
// split plane -- the same in-plane axis m_hat the opposition split itself
// uses (perpendicular to the object's longest in-plane axis, hence
// implicitly its shortest in-plane axis) -- NOT the object's raw 3D shortest
// axis, which can be entirely out of the table plane (e.g. a coin's
// thickness). Direction-agnostic: squaring the cosine means it does not
// matter which of the two antiparallel directions m_hat happens to point.
//
//   v = c_thumb - mean_{i in F} c_i
//   v_hat = v / |v|
//   c_align(v, m_hat) = 1 - (v_hat . m_hat)^2 = 0
//
// m_hat is a FROZEN constant, supplied by the caller (Python computes it once
// via config.opposition_axis_from_object, the same helper the opposition
// half-space itself uses) -- matching HalfSpaceGapFactor's own convention for
// this axis: it is not re-derived from a live object Pose3 each iteration, so
// this factor carries no object key at all.
//
// Variable arity like PreGraspHandCenteringFactor -- spans the thumb and an
// arbitrary number of opposing fingers, so it derives from
// gtsam::NoiseModelFactor directly (not NoiseModelFactorN) with a hand-built
// KeyVector, following the same TendonLengthFactor-style pattern.
//
// Keys: [thumb_pose_key, finger_pose_key_0, ..., finger_pose_key_{|F|-1}]
// Residual: scalar.
class PreGraspAxisAlignmentFactor : public gtsam::NoiseModelFactor {
private:
    size_t num_fingers_;
    gtsam::Vector3 m_hat_;

public:
    using NoiseModelFactor::unwhitenedError;

    PreGraspAxisAlignmentFactor(gtsam::Key thumb_pose_key,
                                const std::vector<gtsam::Key>& finger_pose_keys,
                                const gtsam::Vector3& target_axis,
                                const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor(noise_model, /* keys: */ [&]() {
              gtsam::KeyVector keys;
              keys.push_back(thumb_pose_key);
              keys.insert(keys.end(), finger_pose_keys.begin(), finger_pose_keys.end());
              return keys;
          }()),
          num_fingers_(finger_pose_keys.size()),
          m_hat_(target_axis.normalized()) {}

    gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                  gtsam::OptionalMatrixVecType H = nullptr) const override
    {
        // Key layout: [0] = thumb, [1 .. num_fingers_] = opposing fingers.
        gtsam::Matrix36 D_cthumb_pose;
        const gtsam::Pose3& thumb_pose = x.at<gtsam::Pose3>(keys()[0]);
        gtsam::Point3 c_thumb = thumb_pose.translation(H ? &D_cthumb_pose : nullptr);

        std::vector<gtsam::Matrix36> D_ci_pose(num_fingers_);
        gtsam::Vector3 c_sum = gtsam::Vector3::Zero();
        for (size_t j = 0; j < num_fingers_; ++j) {
            const gtsam::Pose3& finger_pose = x.at<gtsam::Pose3>(keys()[1 + j]);
            gtsam::Point3 c_i = finger_pose.translation(H ? &D_ci_pose[j] : nullptr);
            c_sum += c_i;
        }
        gtsam::Vector3 c_mean = (num_fingers_ > 0)
            ? gtsam::Vector3(c_sum / double(num_fingers_))
            : gtsam::Vector3::Zero();

        gtsam::Vector3 v = c_thumb - c_mean;
        double vn = v.norm();
        if (vn < 1e-9) vn = 1e-9;
        gtsam::Vector3 v_hat = v / vn;

        double d = v_hat.dot(m_hat_);
        double e = 1.0 - d * d;   // 0 <=> colinear with m_hat, either direction

        if (H) {
            H->resize(1 + num_fingers_);
            // de/dv_hat = -2 d * m_hat^T ; dv_hat/dv = (I - v_hat v_hat^T)/|v|
            const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
            gtsam::Matrix3 P = (I3 - v_hat * v_hat.transpose()) / vn;
            gtsam::Matrix13 de_dv = (-2.0 * d * m_hat_.transpose()) * P;

            (*H)[0] = de_dv * D_cthumb_pose;   // d(e)/d(thumb), 1x6
            double w = (num_fingers_ > 0) ? -1.0 / double(num_fingers_) : 0.0;
            for (size_t j = 0; j < num_fingers_; ++j)
                (*H)[1 + j] = (w * de_dv) * D_ci_pose[j];   // d(e)/d(finger_j), 1x6
        }

        return gtsam::Vector1(e);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PreGraspAxisAlignmentFactor(*this)));
    }
};

}  // namespace gepetto_solvers
