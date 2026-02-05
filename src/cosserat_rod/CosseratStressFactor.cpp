#include "CosseratStressFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;


CosseratStressFactor::CosseratStressFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    Key wrench_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key) {}


Vector CosseratStressFactor::evaluateError(
    const Pose3& pose_0, 
    const Pose3& pose_1, 
    const Vector6& stress_0, 
    const Vector6& stress_1, 
    const Vector6& wrench,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4,
    OptionalMatrixType H5) const 
{
    // This factor assumes wrench is in spatial frame, must convert coordinates to body (pose_0) frame
    Matrix6 d_wrench_body_d_pose_0, d_wrench_body_d_wrench;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, pose_0, d_wrench_body_d_wrench, d_wrench_body_d_pose_0);

    // We transform stress_1 to pose_0 frame for summation with wrench_body
    Matrix6 d_stress_pred_d_pose_0, d_stress_pred_d_pose_1, d_stress_pred_d_stress_1;
    Vector6 stress_pred = transform_wrench_adjoint(
        stress_1, 
        pose_1, 
        pose_0, 
        &d_stress_pred_d_stress_1,
        &d_stress_pred_d_pose_1,
        &d_stress_pred_d_pose_0) + wrench_body;
    
    Vector6 stress_error = stress_pred - stress_0;

    if (H1) { *H1 = d_stress_pred_d_pose_0 + d_wrench_body_d_pose_0; }

    if (H2) { *H2 = d_stress_pred_d_pose_1; }

    if (H3) { *H3 = -Matrix6::Identity(); }
    
    if (H4) { *H4 = d_stress_pred_d_stress_1; }

    if (H5) { *H5 = d_wrench_body_d_wrench; }

    return stress_error;
}
