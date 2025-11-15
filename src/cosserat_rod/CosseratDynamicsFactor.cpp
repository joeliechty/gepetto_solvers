#include "CosseratDynamicsFactor.h"

#include <gtsam/base/numericalDerivative.h>

#include "utils/WrenchTransforms.h"

using namespace gtsam;


CosseratDynamicsFactor::CosseratDynamicsFactor(
    Key pose_prev_key,
    Key pose_key,
    Key pose_next_key,
    Key wrench_key,
    const SharedNoiseModel& model,
    double dt,
    double linear_damping,
    double rotational_damping)
:
    NoiseModelFactorN(model, pose_prev_key, pose_key, pose_next_key, wrench_key), 
    dt_(dt),
    linear_damping_(linear_damping),
    rotational_damping_(rotational_damping) {}


Vector CosseratDynamicsFactor::evaluateError(
    const Pose3& pose_prev, 
    const Pose3& pose,
    const Pose3& pose_next, 
    const Vector6& wrench,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const
{
    Vector6 velocity = Pose3::Logmap(pose.inverse() * pose_next) / dt_;
    // Vector6 vel_next = Pose3::Logmap(pose.inverse() * pose_next) / dt_;
    // Vector6 velocity = (vel_next + vel_prev) / 2.0;

    Vector6 damping_wrench;
    damping_wrench.head<3>() = -rotational_damping_  * velocity.head<3>();
    damping_wrench.tail<3>() = -linear_damping_ * velocity.tail<3>();
    
    Vector6 wrench_body = spatial_to_body_wrench(wrench, pose);

    Vector6 wrench_error = damping_wrench - wrench_body;

    if (H1) {
        *H1 = numericalDerivative11<Vector6, Pose3>(
            [&](const Pose3& p) {
                return this->evaluateError(p, pose, pose_next, wrench, nullptr, nullptr, nullptr, nullptr);
            }, pose_prev);
    }

    if (H2) {
        *H2 = numericalDerivative11<Vector6, Pose3>(
            [&](const Pose3& p) {
                return this->evaluateError(pose_prev, p, pose_next, wrench, nullptr, nullptr, nullptr, nullptr);
            }, pose);
    }

    if (H3) {
        *H3 = numericalDerivative11<Vector6, Pose3>(
            [&](const Pose3& p) {
                return this->evaluateError(pose_prev, pose, p, wrench, nullptr, nullptr, nullptr, nullptr);
            }, pose_next);
    }

    if (H4) {
        *H4 = numericalDerivative11<Vector6, Vector6>(
            [&](const Vector6& w) {
                return this->evaluateError(pose_prev, pose, pose_next, w, nullptr, nullptr, nullptr, nullptr);
            }, wrench);
    }

    return wrench_error;
}