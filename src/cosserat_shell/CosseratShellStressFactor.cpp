#include "CosseratShellStressFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;


CosseratShellStressFactor::CosseratShellStressFactor(
    Key pose_key,
    Key pose_1_key,
    Key pose_2_key,
    Key stress_key,
    Key stress_1_key,
    Key stress_2_key,
    Key wrench_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(
        model, 
        pose_key, 
        pose_1_key, 
        pose_2_key, 
        stress_key, 
        stress_1_key, 
        stress_2_key, 
        wrench_key) {}


Vector CosseratShellStressFactor::evaluateError(
    const Pose3& pose, 
    const Pose3& pose_1, 
    const Pose3& pose_2, 
    const Vector6& stress, 
    const Vector6& stress_1, 
    const Vector6& stress_2, 
    const Vector6& wrench,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4,
    OptionalMatrixType H5,
    OptionalMatrixType H6,
    OptionalMatrixType H7) const 
{
    Matrix6 d_wrench_body_d_pose, d_wrench_body_d_wrench;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, pose, d_wrench_body_d_wrench, d_wrench_body_d_pose);

    Matrix6 d_stress_1p_d_pose, d_stress_1p_d_pose_1, d_stress_1p_d_stress_1;
    Vector6 stress_1p = transform_wrench_adjoint(
        stress_1, 
        pose_1, 
        pose, 
        &d_stress_1p_d_stress_1,
        &d_stress_1p_d_pose_1,
        &d_stress_1p_d_pose);
    
    Matrix6 d_stress_2p_d_pose, d_stress_2p_d_pose_2, d_stress_2p_d_stress_2;
    Vector6 stress_2p = transform_wrench_adjoint(
        stress_2, 
        pose_2, 
        pose, 
        &d_stress_2p_d_stress_2,
        &d_stress_2p_d_pose_2,
        &d_stress_2p_d_pose);

    Vector6 stress_error = stress_1p + stress_2p + wrench_body - stress; // Result should equal stress_0

    if (H1) { *H1 = d_stress_1p_d_pose + d_stress_2p_d_pose + d_wrench_body_d_pose; }

    if (H2) { *H2 = d_stress_1p_d_pose_1; }
    
    if (H3) { *H3 = d_stress_2p_d_pose_2; }

    if (H4) { *H4 = -Matrix6::Identity(); }
    
    if (H5) { *H5 = d_stress_1p_d_stress_1; }
    
    if (H6) { *H6 = d_stress_2p_d_stress_2; }

    if (H7) { *H7 = d_wrench_body_d_wrench; }

    return stress_error;
}
