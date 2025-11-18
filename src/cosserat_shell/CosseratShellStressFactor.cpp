#include "CosseratShellStressFactor.h"

#include "utils/WrenchTransforms.h"

using namespace gtsam;


CosseratShellStressFactor::CosseratShellStressFactor(
    Key pose_key,
    Key pose_1_key,
    Key pose_2_key,
    Key stress_x_key,
    Key stress_y_key,
    Key stress_1x_key,
    Key stress_2y_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(
        model, 
        pose_key, 
        pose_1_key, 
        pose_2_key, 
        stress_x_key, 
        stress_y_key,
        stress_1x_key, 
        stress_2y_key) {}


Vector CosseratShellStressFactor::evaluateError(
    const Pose3& p, 
    const Pose3& p1, 
    const Pose3& p2, 
    const Vector6& sx,
    const Vector6& sy, 
    const Vector6& s1x, 
    const Vector6& s2y,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4,
    OptionalMatrixType H5,
    OptionalMatrixType H6,
    OptionalMatrixType H7) const 
{
    Matrix6 d_s1p_d_s1x, d_s1p_d_p1, d_s1p_d_p;
    Vector6 s1p = transform_wrench_adjoint(
        s1x, p1, p, &d_s1p_d_s1x, &d_s1p_d_p1, &d_s1p_d_p);
    
    Matrix6 d_s2p_d_s2y, d_s2p_d_p2, d_s2p_d_p;
    Vector6 s2p = transform_wrench_adjoint(
        s2y, p2, p, &d_s2p_d_s2y, &d_s2p_d_p2, &d_s2p_d_p);

    Vector6 stress_error = s1p + s2p - sx - sy; 

    if (H1) { *H1 = d_s1p_d_p + d_s2p_d_p; }

    if (H2) { *H2 = d_s1p_d_p1; }
    
    if (H3) { *H3 = d_s2p_d_p2; }

    if (H4) { *H4 = -Matrix6::Identity(); }
    
    if (H5) { *H5 = -Matrix6::Identity(); }

    if (H6) { *H6 = d_s1p_d_s1x; }
    
    if (H7) { *H7 = d_s2p_d_s2y; }

    return stress_error;
}
