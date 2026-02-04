#include "VectorCosseratTwistFactor.h"
#include <gtsam/base/Lie.h>
#include <gtsam/geometry/Pose3.h>

using namespace gtsam;


VectorCosseratTwistFactor::VectorCosseratTwistFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    double ds,
    const Vector6& nominal_strain,
    const Matrix6& K_inv,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key),
    ds_(ds),
    nominal_strain_(nominal_strain),
    K_inv_(K_inv) {}


Vector VectorCosseratTwistFactor::evaluateError(
    const Vector6& twist_0, 
    const Vector6& twist_1, 
    const Vector6& stress_0, 
    const Vector6& stress_1, 
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const 
{
    Matrix6 d_pose_0_d_twist_0;
    Pose3 pose_0 = Pose3::Expmap(twist_0, d_pose_0_d_twist_0);

    Matrix6 d_pose_1_d_twist_1;
    Pose3 pose_1 = Pose3::Expmap(twist_1, d_pose_1_d_twist_1);

    Matrix66 d_delta_d_pose_0, d_delta_d_pose_1;
    Pose3 delta = pose_0.between(pose_1, &d_delta_d_pose_0, &d_delta_d_pose_1);

    Matrix66 d_twist_d_delta;
    Vector6 twist = Pose3::Logmap(delta, &d_twist_d_delta);
    
    Vector6 stress_mid = 0.5 * (stress_0 + stress_1);

    Vector6 twist_p = ds_ * (K_inv_ * stress_mid + nominal_strain_);
    
    Vector6 twist_error = twist_p - twist;

    if (H1) { *H1 = -d_twist_d_delta * d_delta_d_pose_0 * d_pose_0_d_twist_0; }

    if (H2) { *H2 = -d_twist_d_delta * d_delta_d_pose_1 * d_pose_1_d_twist_1; }

    if (H3) { *H3 = 0.5 * ds_ * K_inv_; }
    
    if (H4) { *H4 = 0.5 * ds_ * K_inv_; }

    return twist_error;
}
