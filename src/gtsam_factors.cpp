#include "gtsam_factors.h"
#include <gtsam/base/numericalDerivative.h>

using namespace gtsam;


CosseratRodTwistFactor::CosseratRodTwistFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    double ds,
    const Matrix66& K_inv,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key),
    ds_(ds), 
    K_inv_(K_inv) {}


Vector CosseratRodTwistFactor::evaluateError(
    const Pose3& pose_0, 
    const Pose3& pose_1, 
    const Vector6& stress_0, 
    const Vector6& stress_1, 
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4) const 
{
    Matrix66 d_delta_d_pose_0, d_delta_d_pose_1;
    Pose3 delta = pose_0.between(pose_1, &d_delta_d_pose_0, &d_delta_d_pose_1);

    Matrix66 d_twist_d_delta;
    Vector6 twist = Pose3::Logmap(delta, &d_twist_d_delta);
    
    Vector6 stress_mid = 0.5 * (stress_0 + stress_1);

    Vector6 nominal_strain = Vector6::Zero();
    nominal_strain[5] = 1.0;  // Straight rod: linear velocity in z direction only
    Vector6 twist_p = ds_ * (K_inv_ * stress_mid + nominal_strain);
    
    Vector6 twist_error = twist_p - twist;

    if (H1) { *H1 = -d_twist_d_delta * d_delta_d_pose_0; }

    if (H2) { *H2 = -d_twist_d_delta * d_delta_d_pose_1; }

    if (H3) { *H3 = 0.5 * ds_ * K_inv_; }
    
    if (H4) { *H4 = 0.5 * ds_ * K_inv_; }

    return twist_error;
}


CosseratRodStressFactor::CosseratRodStressFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    Key wrench_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key) {}


Vector CosseratRodStressFactor::evaluateError(
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
    
    Vector6 stress_error = stress_pred - stress_0; // Result should equal stress_0

    if (H1) { *H1 = d_stress_pred_d_pose_0 + d_wrench_body_d_pose_0; }

    if (H2) { *H2 = d_stress_pred_d_pose_1; }

    if (H3) { *H3 = -Matrix6::Identity(); }
    
    if (H4) { *H4 = d_stress_pred_d_stress_1; }

    if (H5) { *H5 = d_wrench_body_d_wrench; }

    return stress_error;
}


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


BoundaryStressWrenchFactor::BoundaryStressWrenchFactor(
    Key stress_key,
    Key wrench_key,
    Key pose_key,
    const SharedNoiseModel& model,
    bool is_base)
:
    NoiseModelFactorN(model, stress_key, wrench_key, pose_key),
    is_base_(is_base) {}


Vector BoundaryStressWrenchFactor::evaluateError(
    const Vector6& stress, 
    const Vector6& wrench,
    const Pose3& pose,
    OptionalMatrixType H1, 
    OptionalMatrixType H2,
    OptionalMatrixType H3) const 
{
    // This factor assumes wrench is in spatial frame, must convert coordinates to body (pose_0) frame
    Matrix6 d_wrench_body_d_pose, d_wrench_body_d_wrench;
    Vector6 wrench_body = spatial_to_body_wrench(wrench, pose, d_wrench_body_d_wrench, d_wrench_body_d_pose);

    // At the base, the stress is negative wrench, since it flows out of the rod
    double sign = is_base_ ? 1.0 : -1.0;

    Vector6 stress_error = stress + sign * wrench_body;

    if (H1) { *H1 = Matrix6::Identity(); }

    if (H2) { *H2 = sign * d_wrench_body_d_wrench; }

    if (H3) { *H3 = sign * d_wrench_body_d_pose; }

    return stress_error;
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


PlatformWrenchBalanceFactor::PlatformWrenchBalanceFactor(
    Key stress_key_0, Key pose_key_0,
    Key stress_key_1, Key pose_key_1,
    Key stress_key_2, Key pose_key_2,
    Key stress_key_3, Key pose_key_3,
    Key stress_key_4, Key pose_key_4,
    Key stress_key_5, Key pose_key_5,
    Key platform_stress_key, Key platform_pose_key,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, 
        stress_key_0, pose_key_0,
        stress_key_1, pose_key_1,
        stress_key_2, pose_key_2,
        stress_key_3, pose_key_3,
        stress_key_4, pose_key_4,
        stress_key_5, pose_key_5,
        platform_stress_key, platform_pose_key) {}

Vector PlatformWrenchBalanceFactor::evaluateError(
    const Vector6& stress_0, const Pose3& pose_0,
    const Vector6& stress_1, const Pose3& pose_1,
    const Vector6& stress_2, const Pose3& pose_2,
    const Vector6& stress_3, const Pose3& pose_3,
    const Vector6& stress_4, const Pose3& pose_4,
    const Vector6& stress_5, const Pose3& pose_5,
    const Vector6& platform_stress, const Pose3& platform_pose,
    OptionalMatrixType H1, OptionalMatrixType H2,
    OptionalMatrixType H3, OptionalMatrixType H4, 
    OptionalMatrixType H5, OptionalMatrixType H6,
    OptionalMatrixType H7, OptionalMatrixType H8,
    OptionalMatrixType H9, OptionalMatrixType H10, 
    OptionalMatrixType H11, OptionalMatrixType H12,
    OptionalMatrixType H13, OptionalMatrixType H14) const 
{
    Vector6 stress_0_p = transform_wrench_adjoint(stress_0, pose_0, platform_pose);
    Vector6 stress_1_p = transform_wrench_adjoint(stress_1, pose_1, platform_pose);
    Vector6 stress_2_p = transform_wrench_adjoint(stress_2, pose_2, platform_pose);
    Vector6 stress_3_p = transform_wrench_adjoint(stress_3, pose_3, platform_pose);
    Vector6 stress_4_p = transform_wrench_adjoint(stress_4, pose_4, platform_pose);
    Vector6 stress_5_p = transform_wrench_adjoint(stress_5, pose_5, platform_pose);

    Vector6 stress_error = stress_0_p + stress_1_p + stress_2_p + stress_3_p + stress_4_p + stress_5_p - platform_stress;


    if (H1) { 
        *H1 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                s, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, stress_0, 1e-5);
    }

    if (H2) { 
        *H2 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, p,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, pose_0, 1e-5);
    }

    if (H3) { 
        *H3 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                stress_0, pose_0,
                s, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, stress_1, 1e-5);
    }

    if (H4) { 
        *H4 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, p,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, pose_1, 1e-5);
    }

    if (H5) { 
        *H5 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                s, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, stress_2, 1e-5);
    }

    if (H6) { 
        *H6 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, p,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, pose_2, 1e-5);
    }

    if (H7) { 
        *H7 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                s, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, stress_3, 1e-5);
    }

    if (H8) { 
        *H8 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, p,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, pose_3, 1e-5);
    }

    if (H9) { 
        *H9 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                s, pose_4,
                stress_5, pose_5,
                platform_stress, platform_pose);}, stress_4, 1e-5);
    }

    if (H10) { 
        *H10 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, p,
                stress_5, pose_5,
                platform_stress, platform_pose);}, pose_4, 1e-5);
    }

    if (H11) { 
        *H11 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                s, pose_5,
                platform_stress, platform_pose);}, stress_5, 1e-5);
    }

    if (H12) { 
        *H12 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, p,
                platform_stress, platform_pose);}, pose_5, 1e-5);
    }

    if (H13) { 
        *H13 = gtsam::numericalDerivative11<Vector6, Vector6>(
            [&](const gtsam::Vector6& s) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                s, platform_pose);}, platform_stress, 1e-5);
    }

    if (H14) { 
        *H14 = gtsam::numericalDerivative11<Vector6, Pose3>(
            [&](const gtsam::Pose3& p) {return this->evaluateError(
                stress_0, pose_0,
                stress_1, pose_1,
                stress_2, pose_2,
                stress_3, pose_3,
                stress_4, pose_4,
                stress_5, pose_5,
                platform_stress, p);}, platform_pose, 1e-5);
    }


    return stress_error;
}


// TendonDiscWrenchFactor::TendonDiscWrenchFactor(
//     Key pose_prev_key,
//     Key pose_key,
//     Key pose_next_key, // Set to dummy key if we are at the tip
//     Key wrench_key,
//     Key tensions_key,
//     Key external_wrench_key,
//     const bool is_tip,
//     const std::vector<Point3>& holes_prev,
//     const std::vector<Point3>& holes,
//     const std::vector<Point3>& holes_next, // Not used if we are at the tip
//     const SharedNoiseModel& model)
// : 
//     NoiseModelFactorN(model, pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key, external_wrench_key),
//     is_tip_(is_tip),
//     holes_prev_(holes_prev),
//     holes_(holes),
//     holes_next_(holes_next) {}


// Vector TendonDiscWrenchFactor::evaluateError(
//     const Pose3& pose_prev, 
//     const Pose3& pose, 
//     const Pose3& pose_next, 
//     const Vector6& wrench, 
//     const Vector4& tensions,
//     const Vector6& wrench_external,
//     OptionalMatrixType H1, 
//     OptionalMatrixType H2, 
//     OptionalMatrixType H3, 
//     OptionalMatrixType H4, 
//     OptionalMatrixType H5,
//     OptionalMatrixType H6) const 
// {
//     Vector6 wrench_tendons = Vector6::Zero();
    
//     Matrix64 d_wrench_d_tensions = Matrix64::Zero();
//     Matrix66 d_wrench_d_pose = Matrix66::Zero();
//     Matrix66 d_wrench_d_pose_prev = Matrix66::Zero();
//     Matrix66 d_wrench_d_pose_next = Matrix66::Zero();

//     // Sum up all tendon wrenches on this disc
//     for (int tendon_idx = 0; tendon_idx < tensions.size(); ++tendon_idx) {
//         // Wrench from previous disc
//         Vector6 d_wrench_prev_d_tension;
//         Matrix6 d_wrench_prev_d_pose, d_wrench_prev_d_pose_prev;

//         Vector6 wrench_prev = get_single_tendon_wrench(
//             tensions[tendon_idx],
//             pose,
//             pose_prev,
//             holes_[tendon_idx],
//             holes_prev_[tendon_idx],
//             H5 ? &d_wrench_prev_d_tension : 0,
//             H2 ? &d_wrench_prev_d_pose : 0,
//             H1 ? &d_wrench_prev_d_pose_prev : 0);
        
//         wrench_tendons += wrench_prev;
//         Vector6 d_wrench_d_tension = d_wrench_prev_d_tension;
//         d_wrench_d_pose += d_wrench_prev_d_pose;
//         d_wrench_d_pose_prev += d_wrench_prev_d_pose_prev;
        
//         // Wrench from next disc. Ignore if we are at the tip
//         if (!is_tip_){
//             Vector6 d_wrench_next_d_tension;
//             Matrix6 d_wrench_next_d_pose, d_wrench_next_d_pose_next;

//             Vector6 wrench_next = get_single_tendon_wrench(
//                 tensions[tendon_idx], 
//                 pose,
//                 pose_next, 
//                 holes_[tendon_idx],
//                 holes_next_[tendon_idx],
//                 H5 ? &d_wrench_next_d_tension : 0,
//                 H2 ? &d_wrench_next_d_pose : 0,
//                 H3 ? &d_wrench_next_d_pose_next : 0);
            
//             wrench_tendons += wrench_next;
//             d_wrench_d_tension += d_wrench_next_d_tension;
//             d_wrench_d_pose += d_wrench_next_d_pose;
//             d_wrench_d_pose_next += d_wrench_next_d_pose_next;
//         }

//         d_wrench_d_tensions.col(tendon_idx) = d_wrench_d_tension;
//     }

//     Matrix6 d_wrench_external, d_wrench_external_d_pose;
//     Vector6 wrench_external_body = spatial_to_body_wrench(wrench_external, pose, d_wrench_external, d_wrench_external_d_pose);

//     Vector6 wrench_error = wrench - wrench_tendons - wrench_external_body;

//     if (H1) { *H1 = -d_wrench_d_pose_prev; }

//     if (H2) { *H2 = -d_wrench_d_pose - d_wrench_external_d_pose; }

//     if (H3) { *H3 = -d_wrench_d_pose_next; }

//     if (H4) { *H4 = Matrix6::Identity(); }

//     if (H5) { *H5 = -d_wrench_d_tensions; }

//     if (H6) { *H6 = -d_wrench_external; }

//     return wrench_error;
// }

// Vector6 TendonDiscWrenchFactor::get_single_tendon_wrench(
//     const double tension, 
//     const Pose3& pose, 
//     const Pose3& pose_other, 
//     const Point3& hole, 
//     const Point3& hole_other,
//     OptionalJacobian<6, 1> H_tension,
//     OptionalJacobian<6, 6> H_pose,
//     OptionalJacobian<6, 6> H_pose_other) const
// {
//     Matrix36 d_hole_other_world_d_pose_other;
//     Point3 hole_other_world = pose_other.transformFrom(hole_other, 
//         H_pose_other ? &d_hole_other_world_d_pose_other : 0);
    
//     Matrix36 d_hole_other_local_d_pose;
//     Matrix3 d_hole_other_local_d_hole_other_world;
//     Point3 hole_other_local = pose.transformTo(hole_other_world,
//         H_pose? &d_hole_other_local_d_pose : 0,
//         d_hole_other_local_d_hole_other_world);

//     Vector3 hole_diff = hole_other_local - hole;
//     double norm = hole_diff.norm();

//     Vector3 force_dir;
//     Matrix3 d_force_dir_d_hole_diff = Matrix3::Zero();
    
//     bool valid = hole_diff.allFinite() && norm > 1e-3;
    
//     if (valid) {
//         force_dir = normalize(hole_diff, H_pose || H_pose_other ? &d_force_dir_d_hole_diff : 0);
//     } else {
//         force_dir = Vector3::Zero();
//     }

//     Vector3 force = tension * force_dir;
//     Matrix31 d_force_d_tension = force_dir;
//     Matrix33 d_force_d_force_dir = tension * Matrix3::Identity();

//     Matrix33 d_moment_d_force;
//     Vector3 moment = cross(hole, force, nullptr, 
//          H_tension || H_pose || H_pose_other ? &d_moment_d_force : 0);

//     Vector6 wrench;
//     wrench << moment, force;

//     if (H_tension) {
//         H_tension->head<3>() = d_moment_d_force * d_force_d_tension;
//         H_tension->tail<3>() = d_force_d_tension;
//     }

//     if (H_pose) {
//         Matrix36 d_force_dir_d_pose = d_force_dir_d_hole_diff * d_hole_other_local_d_pose;
//         Matrix36 d_force_d_pose = d_force_d_force_dir * d_force_dir_d_pose;
//         Matrix36 d_moment_d_pose = d_moment_d_force * d_force_d_pose;

//         H_pose->block<3,6>(0,0) = d_moment_d_pose;
//         H_pose->block<3,6>(3,0) = d_force_d_pose;
//     }

//     if (H_pose_other) {
//         Matrix36 d_force_dir_d_pose_other =
//             d_force_dir_d_hole_diff *
//             d_hole_other_local_d_hole_other_world *
//             d_hole_other_world_d_pose_other;

//         Matrix36 d_force_d_pose_other = d_force_d_force_dir * d_force_dir_d_pose_other;
//         Matrix36 d_moment_d_pose_other = d_moment_d_force * d_force_d_pose_other;

//         H_pose_other->block<3,6>(0,0) = d_moment_d_pose_other;
//         H_pose_other->block<3,6>(3,0) = d_force_d_pose_other;
//     }

//     return wrench;
// }






// PositionMeasurementFactor::PositionMeasurementFactor(
//     Key pose_key,
//     Vector3 position_meas,
//     const SharedNoiseModel& model)
// : 
//     NoiseModelFactor4(model, pose_key), position_meas_(position_meas) {}


// Vector PositionMeasurementFactor::evaluateError(const Pose3& pose, OptionalMatrixType H1) const {  
//     Matrix36 d_position_d_pose;
//     Vector3 error = pose.translation(d_position_d_pose) - position_meas_;

//     if (H1) { *H1 = d_position_d_pose; }

//     return error;
// }