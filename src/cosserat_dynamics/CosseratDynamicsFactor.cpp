#include "CosseratDynamicsFactor.h"

#include <gtsam/base/Lie.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>

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
    double rotational_damping,
    double linear_inertia,
    double rotational_inertia)
:
    NoiseModelFactorN(model, pose_prev_key, pose_key, pose_next_key, wrench_key), 
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


Vector CosseratDynamicsFactor::evaluateError(
    const Pose3& p0, 
    const Pose3& p1,
    const Pose3& p2, 
    const Vector6& wrench,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const
{
    Matrix6 d_v_d_p0, d_v_d_p2;
    Vector6 v = get_velocity_from_poses(p0, p2, 2.0 * dt_, d_v_d_p0, d_v_d_p2); // central difference
    
    Matrix6 d_v0_d_p0, d_v0_d_p1;
    Vector6 v0 = get_velocity_from_poses(p0, p1, dt_, d_v0_d_p0, d_v0_d_p1);

    Matrix6 d_v1_d_p1, d_v1_d_p2;
    Vector6 v1 = get_velocity_from_poses(p1, p2, dt_, d_v1_d_p1, d_v1_d_p2);

    Vector6 a = (v1 - v0) / dt_;

    Vector6 damping_wrench = -damping_ * v;
    Vector6 inertial_wrench = -inertia_ * a;

    Matrix6 d_wrench_body_d_wrench, d_wrench_body_d_p1;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, p1, d_wrench_body_d_wrench, d_wrench_body_d_p1);

    Vector6 wrench_error = inertial_wrench + damping_wrench - wrench_body;

    if (H1) {
        *H1 = -inertia_ * (1 / dt_) * (-d_v0_d_p0) - damping_ * d_v_d_p0;
    }

    if (H2) {
        *H2 = -inertia_ * (1 / dt_) * (d_v1_d_p1 - d_v0_d_p1) - d_wrench_body_d_p1;
    }

    if (H3) {
        *H3 = -inertia_ * (1/ dt_) * d_v1_d_p2 - damping_ * d_v_d_p2;
    }

    if (H4) { *H4 = -d_wrench_body_d_wrench; }

    return wrench_error;
}


// Vector6 get_velocity_from_poses(
//     Pose3 p0, 
//     Pose3 p1,
//     double dt,
//     Matrix6& d_v_d_p0, 
//     Matrix6& d_v_d_p1)
// {
//     Matrix6 d_p0_inv_d_p0;
//     Pose3 p0_inv = p0.inverse(d_p0_inv_d_p0);

//     Matrix6 d_comp_d_p0_inv, d_comp_d_p1;
//     Pose3 comp = p0_inv.compose(p1, d_comp_d_p0_inv, d_comp_d_p1);

//     Matrix6 d_delta_d_comp;
//     Vector6 delta = Pose3::Logmap(comp, d_delta_d_comp);

//     Vector6 v = delta / dt;

//     d_v_d_p0 = (1.0 / dt) * d_delta_d_comp * d_comp_d_p0_inv * d_p0_inv_d_p0;
//     d_v_d_p1 = (1.0 / dt) * d_delta_d_comp * d_comp_d_p1;

//     return v;
// }


// Vector CosseratDynamicsFactor::evaluateError(
//     const Pose3& p0, 
//     const Pose3& p1,
//     const Pose3& p2, 
//     const Vector6& wrench,
//     OptionalMatrixType H1, 
//     OptionalMatrixType H2, 
//     OptionalMatrixType H3, 
//     OptionalMatrixType H4) const
// {
//     Matrix6 d_v_d_p0, d_v_d_p2;
//     Vector6 v = get_velocity_from_poses(p0, p2, 2.0 * dt_, d_v_d_p0, d_v_d_p2); // central difference
    
//     Matrix6 d_v0_d_p0, d_v0_d_p1;
//     Vector6 v0 = get_velocity_from_poses(p0, p1, dt_, d_v0_d_p0, d_v0_d_p1);

//     Matrix6 d_v1_d_p1, d_v1_d_p2;
//     Vector6 v1 = get_velocity_from_poses(p1, p2, dt_, d_v1_d_p1, d_v1_d_p2);

//     Vector6 a = (v1 - v0) / dt_;

//     Vector6 damping_wrench = -damping_ * v;

//     Matrix6 d_wrench_body_d_wrench, d_wrench_body_d_p1;
//     Vector6 wrench_body = spatial_to_body_wrench(wrench, p1, d_wrench_body_d_wrench, d_wrench_body_d_p1);

//     Vector6 inertial_wrench = wrench_body - damping_wrench;
//     Vector6 a_pred = -inertia_inverse_ * inertial_wrench;

//     Vector6 error = a - a_pred;

//     double d_a_d_v1 = 1.0 / dt_;
//     double d_a_d_v0 = -1.0 / dt_;

//     if (H1) {
//         Matrix6 d_a_d_p0 = d_a_d_v0 * d_v0_d_p0;
//         Matrix6 d_inertial_wrench_d_p0 = damping_ * d_v_d_p0;
//         Matrix6 d_a_pred_d_p0 = -inertia_inverse_ * d_inertial_wrench_d_p0;

//         *H1 = d_a_d_p0 - d_a_pred_d_p0;
//     }

//     if (H2) {
//         Matrix6 d_a_d_p1 = d_a_d_v0 * d_v0_d_p1 + d_a_d_v1 * d_v1_d_p1;
//         Matrix6 d_inertial_wrench_d_p1 = d_wrench_body_d_p1;
//         Matrix6 d_a_pred_d_p1 = -inertia_inverse_ * d_inertial_wrench_d_p1;

//         *H2 = d_a_d_p1 - d_a_pred_d_p1;
//     }

//     if (H3) {
//         Matrix6 d_a_d_p2 = d_a_d_v1 * d_v1_d_p2;
//         Matrix6 d_inertial_wrench_d_p2 = damping_ * d_v_d_p2;
//         Matrix6 d_a_pred_d_p2 = -inertia_inverse_ * d_inertial_wrench_d_p2;

//         *H3 = d_a_d_p2 - d_a_pred_d_p2;
//     }

//     if (H4) { 
//         *H4 = inertia_inverse_ * d_wrench_body_d_wrench;
//     }

//     return error;
// }