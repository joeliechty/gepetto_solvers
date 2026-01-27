#include "CosseratDynamicsFactor.h"

#include <gtsam/base/Lie.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>

#include "utils/WrenchTransforms.h"

using namespace gtsam;


CosseratDynamicsFactor::CosseratDynamicsFactor(
    Key pose_prev_key,
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
    NoiseModelFactorN(model, pose_prev_key, v_prev_key, pose_key, v_key, wrench_key), 
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
}


Vector6 get_velocity_from_poses(
    Pose3 p0, 
    Pose3 p1,
    double dt,
    Matrix6& d_v_d_p0, 
    Matrix6& d_v_d_p1)
{
    Matrix6 d_p0_inv_d_p0;
    Pose3 p0_inv = p0.inverse(d_p0_inv_d_p0);

    Matrix6 d_comp_d_p0_inv, d_comp_d_p1;
    Pose3 comp = p0_inv.compose(p1, d_comp_d_p0_inv, d_comp_d_p1);

    Matrix6 d_delta_d_comp;
    Vector6 delta = Pose3::Logmap(comp, d_delta_d_comp);

    Vector6 v = delta / dt;

    d_v_d_p0 = (1.0 / dt) * d_delta_d_comp * d_comp_d_p0_inv * d_p0_inv_d_p0;
    d_v_d_p1 = (1.0 / dt) * d_delta_d_comp * d_comp_d_p1;

    return v;
}


Matrix6 ad(const Vector6& xi) {
    Matrix6 ad_xi = Matrix6::Zero();

    // xi = [omega; v]
    Vector3 omega = xi.head<3>();
    Vector3 v = xi.tail<3>();

    // Adjoint operator for se(3)
    ad_xi.block<3,3>(0,0) = skewSymmetric(omega);
    ad_xi.block<3,3>(0,3) = Matrix3::Zero();
    ad_xi.block<3,3>(3,0) = skewSymmetric(v);
    ad_xi.block<3,3>(3,3) = skewSymmetric(omega);

    return ad_xi;
}


Vector CosseratDynamicsFactor::evaluateError(
    const Pose3& p0, 
    const Vector6& v0,
    const Pose3& p1,
    const Vector6& v1,
    const Vector6& wrench,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4,
    OptionalMatrixType H5) const
{
    Matrix6 d_wrench_body_d_wrench, d_wrench_body_d_p2;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, p1, d_wrench_body_d_wrench, d_wrench_body_d_p2);
    
    // Vector6 coriolis = ad(v0).transpose() * (inertia_ * v0);

    Vector6 v1_pred = v0 + inertia_.inverse() * (wrench_body - damping_ * v0) * dt_;

    Pose3 p1_pred = p0.expmap(v1_pred * dt_);

    Vector6 v_error = v1 - v1_pred;
    Pose3 p_error = p1.between(p1_pred);

    Vector12 error;
    error.head<6>() = Pose3::Logmap(p_error);
    error.tail<6>() = v_error;

    if (H1) {
        auto f = [&](const Pose3& p0_var) -> Vector12 {
        return evaluateError(
            p0_var, v0, p1, v1, wrench,
            nullptr, nullptr, nullptr, nullptr, nullptr);
        };

        *H1 = numericalDerivative11<Vector12, Pose3>(f, p0, 1e-3);
    }

    if (H2) {
        auto f = [&](const Vector6& v0_var) -> Vector12 {
        return evaluateError(
            p0, v0_var, p1, v1, wrench,
            nullptr, nullptr, nullptr, nullptr, nullptr);
        };

        *H2 = numericalDerivative11<Vector12, Vector6>(f, v0, 1e-3);
    }

    if (H3) {
        auto f = [&](const Pose3& p1_var) -> Vector12 {
        return evaluateError(
            p0, v0, p1_var, v1, wrench,
            nullptr, nullptr, nullptr, nullptr, nullptr);
        };

        *H3 = numericalDerivative11<Vector12, Pose3>(f, p1, 1e-3);
    }

    if (H4) {
        auto f = [&](const Vector6& v1_var) -> Vector12 {
        return evaluateError(
            p0, v0, p1, v1_var, wrench,
            nullptr, nullptr, nullptr, nullptr, nullptr);
        };

        *H4 = numericalDerivative11<Vector12, Vector6>(f, v1, 1e-3);
    }

    if (H5) {
        auto f = [&](const Vector6& wrench_var) -> Vector12 {
        return evaluateError(
            p0, v0, p1, v1, wrench_var,
            nullptr, nullptr, nullptr, nullptr, nullptr);
        };

        *H5 = numericalDerivative11<Vector12, Vector6>(f, wrench, 1e-3);
    }

    return error;
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