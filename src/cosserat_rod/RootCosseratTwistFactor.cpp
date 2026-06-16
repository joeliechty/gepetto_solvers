#include "RootCosseratTwistFactor.h"
#include <gtsam/base/Matrix.h>

using namespace gtsam;


RootCosseratTwistFactor::RootCosseratTwistFactor(
    Key pose_base_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    const Pose3& offset,
    double ds,
    const Vector6& nominal_strain,
    const Matrix6& K_inv,
    const SharedNoiseModel& model,
    bool use_midpoint)
:
    RootCosseratTwistBase(model, pose_base_key, pose_1_key, stress_0_key, stress_1_key),
    offset_(offset),
    ds_(ds),
    use_midpoint_(use_midpoint),
    nominal_strain_(nominal_strain),
    K_inv_(K_inv)
{}


Vector RootCosseratTwistFactor::evaluateError(
    const Pose3& pose_base,
    const Pose3& p1,
    const Vector6& s0,
    const Vector6& s1,
    OptionalMatrixType H_base,
    OptionalMatrixType H2,
    OptionalMatrixType H3,
    OptionalMatrixType H4) const
{
    // Deterministic node-0 pose and the SE(3) composition Jacobian (Eq. 43).
    Matrix6 H_compose;
    Pose3 p0 = pose_base.compose(offset_, H_compose);

    // Get delta in of p1 relative to p0
    Matrix6 d_delta_d_p0, d_delta_d_p1;
    Pose3 delta = p0.between(p1, d_delta_d_p0, d_delta_d_p1);

    // Twist based on poses in frame p0
    Matrix6 d_twist_d_delta;
    Vector6 twist = Pose3::Logmap(delta, d_twist_d_delta);

    // Stress causes a twist predicted by rod mechanics scaled by ds
    Vector6 s = use_midpoint_ ?  0.5 * (s0 + s1) : s0;
    Vector6 twist_pred = ds_ * (K_inv_ * s + nominal_strain_);

    Vector6 twist_error = twist_pred - twist;

    // Chain rule back to the hand base (Eq. 44).
    if (H_base) { *H_base = (-d_twist_d_delta * d_delta_d_p0) * H_compose; }

    if (H2) { *H2 = -d_twist_d_delta * d_delta_d_p1; }

    if (H3) {
        *H3 = use_midpoint_ ? 0.5 * ds_ * K_inv_ : ds_ * K_inv_;
     }

    if (H4) {
        Matrix6 zero = Matrix6::Zero();
        *H4 = use_midpoint_ ? 0.5 * ds_ * K_inv_ : zero;
    }

    return twist_error;
}
