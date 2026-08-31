#include "gepetto_solvers/cosserat_rod/RootBoundaryStressFactor.h"

#include "gepetto_solvers/utils/WrenchTransforms.h"

using namespace gtsam;


RootBoundaryStressFactor::RootBoundaryStressFactor(
    Key stress_key,
    Key wrench_key,
    Key pose_base_key,
    const Pose3& offset,
    const SharedNoiseModel& model,
    bool is_base)
:
    RootBoundaryStressBase(model, stress_key, wrench_key, pose_base_key),
    offset_(offset),
    is_base_(is_base) {}


Vector RootBoundaryStressFactor::evaluateError(
    const Vector6& stress,
    const Vector6& wrench,
    const Pose3& pose_base,
    OptionalMatrixType H1,
    OptionalMatrixType H2,
    OptionalMatrixType H_base) const
{
    // Deterministic node-0 pose and the SE(3) composition Jacobian (Eq. 43).
    Matrix6 H_compose;
    Pose3 pose = pose_base.compose(offset_, H_compose);

    // This factor assumes wrench is in spatial frame, must convert coordinates to body (pose_0) frame
    Matrix6 d_wrench_body_d_pose, d_wrench_body_d_wrench;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, pose, d_wrench_body_d_wrench, d_wrench_body_d_pose);

    // At the base, the stress is negative wrench, since it flows out of the rod
    double sign = is_base_ ? 1.0 : -1.0;

    Vector6 stress_error = stress + sign * wrench_body;

    if (H1) { *H1 = Matrix6::Identity(); }

    if (H2) { *H2 = sign * d_wrench_body_d_wrench; }

    // Chain rule back to the hand base (Eq. 44).
    if (H_base) { *H_base = sign * d_wrench_body_d_pose * H_compose; }

    return stress_error;
}
