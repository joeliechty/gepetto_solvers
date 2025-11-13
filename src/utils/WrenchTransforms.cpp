#include "WrenchTransforms.h"

using namespace gtsam;


Vector6 transform_wrench_adjoint(
    const Vector6& wrench_0,
    const Pose3& pose_0,
    const Pose3& pose,
    OptionalJacobian<6, 6> H_wrench_0,
    OptionalJacobian<6, 6> H_pose_0,
    OptionalJacobian<6, 6> H_pose) 
{   
    Matrix6 d_pose_0_inv_d_pose_0;
    Pose3 pose_0_inv = pose_0.inverse(&d_pose_0_inv_d_pose_0);
    
    Matrix6 d_delta_d_pose_0_inv, d_delta_d_pose;
    Pose3 delta = pose_0_inv.compose(pose, &d_delta_d_pose_0_inv, &d_delta_d_pose);
    
    Matrix6 d_wrench_d_delta, d_wrench_d_wrench_0;
    Vector6 wrench = delta.AdjointTranspose(wrench_0, &d_wrench_d_delta, &d_wrench_d_wrench_0);

    if (H_wrench_0) { *H_wrench_0 = d_wrench_d_wrench_0; }

    if (H_pose_0) {
        *H_pose_0 = d_wrench_d_delta * d_delta_d_pose_0_inv * d_pose_0_inv_d_pose_0;
    }

    if (H_pose) { *H_pose = d_wrench_d_delta * d_delta_d_pose; }
    
    return wrench;
}


Vector6 spatial_to_body_wrench(
    const Vector6& wrench_spatial, 
    const Pose3& pose, 
    OptionalJacobian<6, 6> H_wrench,
    OptionalJacobian<6, 6> H_pose)
{
    Matrix3 d_moment_d_rotation, d_force_d_rotation, d_moment_d_moment, d_force_d_force;
    Matrix36 d_rotation_d_pose;

    Vector6 wrench_body;

    Rot3 rot = pose.rotation(d_rotation_d_pose);
    
    wrench_body.head<3>() = rot.unrotate(wrench_spatial.head<3>(),
        H_pose ? &d_moment_d_rotation : 0,
        H_wrench ? &d_moment_d_moment : 0);
    
    wrench_body.tail<3>() = rot.unrotate(wrench_spatial.tail<3>(),
        H_pose ? &d_force_d_rotation : 0,
        H_wrench ? &d_force_d_force : 0);
    
    if (H_pose) {
        H_pose->setZero();
        H_pose->block<3,3>(0,0) = d_moment_d_rotation;
        H_pose->block<3,3>(3,0) = d_force_d_rotation;
    }

    if (H_wrench) {
        H_wrench->setZero();
        H_wrench->block<3,3>(0,0) = d_moment_d_moment;
        H_wrench->block<3,3>(3,3) = d_force_d_force;
    }
    
    return wrench_body;
}
