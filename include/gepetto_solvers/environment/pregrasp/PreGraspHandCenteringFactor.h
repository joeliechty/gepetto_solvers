#pragma once

// Thumb/finger midpoint onto the object (Eq 2.18-2.19).

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

// Pre-grasp hand-centering constraint (Section 2.2.1, Eq 1.92 numbering ->
// paper Eq 2.18-2.19). Centers the hand over the object prior to initiating
// surface contact by aligning the midpoint of the thumb's contact-sphere
// center and the mean of the opposing fingers' contact-sphere centers with
// the object centroid, raised by a fixed clearance offset along the support
// surface normal:
//
//   c_hand = (1/2) * ( c_thumb + (1/|F|) sum_{i in F} c_i )
//   c_center(c_thumb, F, p_obj) = c_hand - (p_obj + h_clear * n_hat) = 0
//
// where c_thumb and each c_i are world-frame sphere centers (the translations
// of their node Pose3 variables -- same convention as HalfSpaceGapFactor and
// PlaneCollisionGapFactor), p_obj is the object pose's translation, and
// h_clear / n_hat are constructor-supplied constants (n_hat need not be the
// table normal specifically -- whatever clearance axis the caller wants).
//
// Variable arity: the number of opposing fingers |F| is runtime-determined,
// so -- like TendonLengthFactor -- this derives from gtsam::NoiseModelFactor
// directly (not NoiseModelFactorN) and hand-builds its KeyVector in the
// initializer list.
//
// Keys: [thumb_pose_key, finger_pose_key_0, ..., finger_pose_key_{|F|-1}, object_key]
// Residual: Vector3.
class PreGraspHandCenteringFactor : public gtsam::NoiseModelFactor {
private:
    size_t num_fingers_;
    double h_clear_;
    gtsam::Vector3 n_hat_;

public:
    using NoiseModelFactor::unwhitenedError;

    PreGraspHandCenteringFactor(gtsam::Key thumb_pose_key,
                                const std::vector<gtsam::Key>& finger_pose_keys,
                                gtsam::Key object_key,
                                double h_clear,
                                const gtsam::Vector3& n_hat,
                                const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor(noise_model, /* keys: */ [&]() {
              gtsam::KeyVector keys;
              keys.push_back(thumb_pose_key);
              keys.insert(keys.end(), finger_pose_keys.begin(), finger_pose_keys.end());
              keys.push_back(object_key);
              return keys;
          }()),
          num_fingers_(finger_pose_keys.size()),
          h_clear_(h_clear),
          n_hat_(n_hat.normalized()) {}

    gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                  gtsam::OptionalMatrixVecType H = nullptr) const override
    {
        // Key layout: [0] = thumb, [1 .. num_fingers_] = fingers, [1+num_fingers_] = object.
        gtsam::Matrix36 D_cthumb_pose;
        const gtsam::Pose3& thumb_pose = x.at<gtsam::Pose3>(keys()[0]);
        gtsam::Point3 c_thumb = thumb_pose.translation(H ? &D_cthumb_pose : nullptr);

        std::vector<gtsam::Matrix36> D_ci_pose(num_fingers_);
        gtsam::Vector3 c_fingers_sum = gtsam::Vector3::Zero();
        for (size_t j = 0; j < num_fingers_; ++j) {
            const gtsam::Pose3& finger_pose = x.at<gtsam::Pose3>(keys()[1 + j]);
            gtsam::Point3 c_i = finger_pose.translation(H ? &D_ci_pose[j] : nullptr);
            c_fingers_sum += c_i;
        }
        gtsam::Vector3 c_fingers_mean = (num_fingers_ > 0)
            ? gtsam::Vector3(c_fingers_sum / double(num_fingers_))
            : gtsam::Vector3::Zero();

        gtsam::Matrix36 D_pobj_pose;
        const gtsam::Pose3& object_pose = x.at<gtsam::Pose3>(keys()[1 + num_fingers_]);
        gtsam::Point3 p_obj = object_pose.translation(H ? &D_pobj_pose : nullptr);

        gtsam::Vector3 c_hand = 0.5 * (c_thumb + c_fingers_mean);
        gtsam::Vector3 target = p_obj + h_clear_ * n_hat_;
        gtsam::Vector3 e = c_hand - target;

        if (H) {
            H->resize(2 + num_fingers_);
            (*H)[0] = 0.5 * D_cthumb_pose;   // d(c_hand)/d(thumb), 3x6
            double w = (num_fingers_ > 0) ? 0.5 / double(num_fingers_) : 0.0;
            for (size_t j = 0; j < num_fingers_; ++j)
                (*H)[1 + j] = w * D_ci_pose[j];   // d(c_hand)/d(finger_j), 3x6
            (*H)[1 + num_fingers_] = -D_pobj_pose; // d(-target)/d(object), 3x6
        }

        return e;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new PreGraspHandCenteringFactor(*this)));
    }
};

}  // namespace gepetto_solvers
