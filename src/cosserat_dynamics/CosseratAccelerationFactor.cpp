#include "CosseratAccelerationFactor.h"

#include <gtsam/base/Lie.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>

#include "utils/WrenchTransforms.h"

using namespace gtsam;

CosseratAccelerationFactor::CosseratAccelerationFactor(
    Key v_prev_key,
    Key pose_key,
    Key v_key,
    Key wrench_key,
    const SharedNoiseModel& model,
    double dt,
    double linear_damping,
    double rotational_damping,
    double linear_inertia,
    double rotational_inertia)
:
    NoiseModelFactorN(model, v_prev_key, pose_key, v_key, wrench_key), 
    dt_(dt)
{
    Vector6 damping;
    damping << 
        rotational_damping, rotational_damping, rotational_damping,
        linear_damping, linear_damping, linear_damping;
    
    damping_ = damping.asDiagonal();

    Vector6 inertia;
    inertia <<
        rotational_inertia, rotational_inertia, rotational_inertia,
        linear_inertia, linear_inertia, linear_inertia;

    inertia_ = inertia.asDiagonal();
    inertia_inverse_ = inertia_.inverse();
}


Vector CosseratAccelerationFactor::evaluateError(
    const Vector6& v0,
    const Pose3& p1,
    const Vector6& v1,
    const Vector6& wrench,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const
{
    Vector6 a = (v1 - v0) / dt_;
    Vector6 inertial_wrench = -inertia_ * a;
    Vector6 damping_wrench = -damping_ * v1;

    Matrix6 d_wrench_body_d_wrench, d_wrench_body_d_p2;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, p1, d_wrench_body_d_wrench, d_wrench_body_d_p2);

    // This is a bit hacky. Basically, it is easier to set acceleration by setting the wrench to be -m * a
    Vector6 accel_error = inertia_inverse_ * (inertial_wrench + damping_wrench - wrench_body);

    double fd_step = 1e-5;

    if (H1) {
        auto f = [&](const Vector6& v0_var) -> Vector6 {
        return evaluateError(
            v0_var, p1, v1, wrench,
            nullptr, nullptr, nullptr, nullptr);
        };

        *H1 = numericalDerivative11<Vector6, Vector6>(f, v0, fd_step);
    }

    if (H2) {
        auto f = [&](const Pose3& p1_var) -> Vector6 {
        return evaluateError(
            v0, p1_var, v1, wrench,
            nullptr, nullptr, nullptr, nullptr);
        };

        *H2 = numericalDerivative11<Vector6, Pose3>(f, p1, fd_step);
    }

    if (H3) {
        auto f = [&](const Vector6& v1_var) -> Vector6 {
        return evaluateError(
            v0, p1, v1_var, wrench,
            nullptr, nullptr, nullptr, nullptr);
        };

        *H3 = numericalDerivative11<Vector6, Vector6>(f, v1, fd_step);
    }

    if (H4) {
        auto f = [&](const Vector6& wrench_var) -> Vector6 {
        return evaluateError(
            v0, p1, v1, wrench_var,
            nullptr, nullptr, nullptr, nullptr);
        };

        *H4 = numericalDerivative11<Vector6, Vector6>(f, wrench, fd_step);
    }

    return accel_error;
    // Matrix6 d_v0_d_p0, d_v0_d_p1;
    // Vector6 v0 = get_velocity_from_poses(p0, p1, dt_, d_v0_d_p0, d_v0_d_p1);

    // Matrix6 d_v1_d_p1, d_v1_d_p2;
    // Vector6 v1 = get_velocity_from_poses(p1, p2, dt_, d_v1_d_p1, d_v1_d_p2);

    // Vector6 a = (v1 - v0) / dt_;

    // Vector6 damping_wrench = -damping_ * v1;
    // Vector6 inertial_wrench = -inertia_ * a;

    // Matrix6 d_wrench_body_d_wrench, d_wrench_body_d_p2;
    // Vector6 wrench_body = spatial_to_body_wrench(wrench, p2, d_wrench_body_d_wrench, d_wrench_body_d_p2);

    // Vector6 wrench_error = inertial_wrench + damping_wrench - wrench_body;

    // if (H1) {
    //     *H1 = -inertia_ * (1 / dt_) * (-d_v0_d_p0);
    // }

    // if (H2) {
    //     *H2 = -inertia_ * (1 / dt_) * (d_v1_d_p1 - d_v0_d_p1) - damping_ * d_v1_d_p1;
    // }

    // if (H3) {
    //     *H3 = -inertia_ * (1/ dt_) * d_v1_d_p2 - damping_ * d_v1_d_p2 - d_wrench_body_d_p2;
    // }

    // if (H4) { *H4 = -d_wrench_body_d_wrench; }

    // return wrench_error;
}