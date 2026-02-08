#include "CosseratTwistFactor.h"
#include <gtsam/base/Matrix.h>

using namespace gtsam;


CosseratTwistFactor::CosseratTwistFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    double ds,
    const Vector6& nominal_strain,
    const Matrix6& K_inv,
    const SharedNoiseModel& model,
    bool use_midpoint)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key),
    ds_(ds),
    nominal_strain_(nominal_strain),
    use_midpoint_(use_midpoint),
    K_inv_(K_inv) 
{}


Vector CosseratTwistFactor::evaluateError(
    const Pose3& p0, 
    const Pose3& p1, 
    const Vector6& s0, 
    const Vector6& s1, 
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const 
{
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

    if (H1) { *H1 = -d_twist_d_delta * d_delta_d_p0; }

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
