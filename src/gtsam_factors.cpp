#include "gtsam_factors.h"

using namespace gtsam;


CosseratRodTwistFactor::CosseratRodTwistFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    double ds,
    const Matrix66& K_inv,
    bool use_midpoint, 
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key),
    ds_(ds), 
    K_inv_(K_inv), 
    use_midpoint_(use_midpoint) {}


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
    Pose3 delta = pose_0.between(pose_1,
        H1 ? &d_delta_d_pose_0 : 0,
        H2 ? &d_delta_d_pose_1 : 0);

    Matrix66 d_twist_d_delta;
    Vector6 twist = Pose3::Logmap(delta, 
        H1 || H2 ? &d_twist_d_delta : 0);
    
    Vector6 stress_mid = 0.5 * (stress_0 + stress_1);
    Vector6 stress = use_midpoint_ ? stress_mid : stress_0;

    Vector6 nominal_strain = Vector6::Zero();
    nominal_strain[5] = 1.0;  // Straight rod: linear velocity in z direction only
    Vector6 twist_p = ds_ * (K_inv_ * stress + nominal_strain);
    
    Vector6 twist_error = twist_p - twist;

    if (H1) {
        *H1 = -d_twist_d_delta * d_delta_d_pose_0;
    }

    if (H2) {
        *H2 = -d_twist_d_delta * d_delta_d_pose_1;
    }

    if (H3) {
        *H3 = ds_ * K_inv_;

        if (use_midpoint_) {
            *H3 *= 0.5;
        }
    }
    
    if (H4) {
        if (use_midpoint_) {
            *H4 = Matrix6::Zero();
        } else {
            *H4 = 0.5 * ds_ * K_inv_;
        }
    }

    return twist_error;
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


CosseratRodStressFactor::CosseratRodStressFactor(
    Key pose_0_key,
    Key pose_1_key,
    Key stress_0_key,
    Key stress_1_key,
    Key wrench_key,
    const bool is_wrench_spatial,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_0_key, pose_1_key, stress_0_key, stress_1_key, wrench_key), 
    is_wrench_spatial_(is_wrench_spatial) {}


Vector CosseratRodStressFactor::evaluateError(
    const Pose3& pose_0, 
    const Pose3& pose_1, 
    const Vector6& stress_0, 
    const Vector6& stress_1, 
    const Vector6& wrench_1,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4,
    OptionalMatrixType H5) const {
    
    Matrix6 d_stress_p_d_pose_0, d_stress_p_d_pose_1, d_stress_p_d_wrench_sum;
    Matrix6 d_wrench_body_d_pose_1, d_wrench_body_d_wrench;

    Vector6 wrench_body;

    if (is_wrench_spatial_) {
        wrench_body = spatial_to_body_wrench(wrench_1, pose_1, d_wrench_body_d_wrench, d_wrench_body_d_pose_1);
    } else {
        wrench_body = wrench_1;
        d_wrench_body_d_wrench.setIdentity();
        d_wrench_body_d_pose_1.setZero();
    }

    Vector6 stress_p = transform_wrench_adjoint(pose_0, pose_1, wrench_body + stress_1, 
        H1 ? &d_stress_p_d_pose_0 : 0,
        H2 ? &d_stress_p_d_pose_1 : 0,
        H4 || H5 ? &d_stress_p_d_wrench_sum : 0);
    
    Vector6 stress_error = stress_p - stress_0;

    if (H1) { *H1 = d_stress_p_d_pose_0; }

    if (H2) {
        *H2 = d_stress_p_d_pose_1 + d_stress_p_d_wrench_sum * d_wrench_body_d_pose_1;
    }

    if (H3) { *H3 = -Matrix6::Identity(); }
    
    if (H4) { *H4 = d_stress_p_d_wrench_sum; }

    if (H5) {
        *H5 = d_stress_p_d_wrench_sum * d_wrench_body_d_wrench;
    }

    return stress_error;
}


Vector6 CosseratRodStressFactor::transform_wrench_adjoint(
    const Pose3& pose,
    const Pose3& tip_pose,
    const Vector6& tip_wrench,
    OptionalJacobian<6, 6> H_pose,
    OptionalJacobian<6, 6> H_tip_pose,
    OptionalJacobian<6, 6> H_tip_wrench) const
{
    Matrix66 d_tip_pose_inv_d_tip_pose;
    Matrix66 d_delta_d_tip_pose_inv, d_delta_d_pose;
    Matrix66 d_wrench_d_delta, d_wrench_d_tip_wrench_;

    Pose3 tip_pose_inv = tip_pose.inverse(
        (H_pose ? &d_tip_pose_inv_d_tip_pose : 0));

    Pose3 delta = tip_pose_inv.compose(pose,
        (H_tip_pose ? &d_delta_d_tip_pose_inv : 0),
        (H_pose ? &d_delta_d_pose : 0));

    Vector6 wrench = delta.AdjointTranspose(
        tip_wrench,
        (H_pose || H_tip_pose ? &d_wrench_d_delta : 0),
        (H_tip_wrench ? &d_wrench_d_tip_wrench_ : 0));

    // Assign Jacobians if needed
    if (H_pose) {
        *H_pose = d_wrench_d_delta * d_delta_d_pose;
    }
    if (H_tip_pose) {
        *H_tip_pose = d_wrench_d_delta * d_delta_d_tip_pose_inv * d_tip_pose_inv_d_tip_pose;
    }
    if (H_tip_wrench) {
        *H_tip_wrench = d_wrench_d_tip_wrench_;
    }

    return wrench;
}


TendonDiscWrenchFactor::TendonDiscWrenchFactor(
    Key pose_prev_key,
    Key pose_key,
    Key pose_next_key, // Set to dummy key if we are at the tip
    Key wrench_key,
    Key tensions_key,
    Key external_wrench_key,
    const bool is_tip,
    const std::vector<Point3>& holes_prev,
    const std::vector<Point3>& holes,
    const std::vector<Point3>& holes_next, // Not used if we are at the tip
    const SharedNoiseModel& model)
: 
    NoiseModelFactorN(model, pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key, external_wrench_key),
    is_tip_(is_tip),
    holes_prev_(holes_prev),
    holes_(holes),
    holes_next_(holes_next) {}


Vector TendonDiscWrenchFactor::evaluateError(
    const Pose3& pose_prev, 
    const Pose3& pose, 
    const Pose3& pose_next, 
    const Vector6& wrench, 
    const Vector4& tensions,
    const Vector6& wrench_external,
    OptionalMatrixType H1, 
    OptionalMatrixType H2, 
    OptionalMatrixType H3, 
    OptionalMatrixType H4, 
    OptionalMatrixType H5,
    OptionalMatrixType H6) const 
{
    Vector6 wrench_tendons = Vector6::Zero();
    
    Matrix64 d_wrench_d_tensions = Matrix64::Zero();
    Matrix66 d_wrench_d_pose = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_prev = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_next = Matrix66::Zero();

    // Sum up all tendon wrenches on this disc
    for (int tendon_idx = 0; tendon_idx < tensions.size(); ++tendon_idx) {
        // Wrench from previous disc
        Vector6 d_wrench_prev_d_tension;
        Matrix6 d_wrench_prev_d_pose, d_wrench_prev_d_pose_prev;

        Vector6 wrench_prev = get_single_tendon_wrench(
            tensions[tendon_idx],
            pose,
            pose_prev,
            holes_[tendon_idx],
            holes_prev_[tendon_idx],
            H5 ? &d_wrench_prev_d_tension : 0,
            H2 ? &d_wrench_prev_d_pose : 0,
            H1 ? &d_wrench_prev_d_pose_prev : 0);
        
        wrench_tendons += wrench_prev;
        Vector6 d_wrench_d_tension = d_wrench_prev_d_tension;
        d_wrench_d_pose += d_wrench_prev_d_pose;
        d_wrench_d_pose_prev += d_wrench_prev_d_pose_prev;
        
        // Wrench from next disc. Ignore if we are at the tip
        if (!is_tip_){
            Vector6 d_wrench_next_d_tension;
            Matrix6 d_wrench_next_d_pose, d_wrench_next_d_pose_next;

            Vector6 wrench_next = get_single_tendon_wrench(
                tensions[tendon_idx], 
                pose,
                pose_next, 
                holes_[tendon_idx],
                holes_next_[tendon_idx],
                H5 ? &d_wrench_next_d_tension : 0,
                H2 ? &d_wrench_next_d_pose : 0,
                H3 ? &d_wrench_next_d_pose_next : 0);
            
            wrench_tendons += wrench_next;
            d_wrench_d_tension += d_wrench_next_d_tension;
            d_wrench_d_pose += d_wrench_next_d_pose;
            d_wrench_d_pose_next += d_wrench_next_d_pose_next;
        }

        d_wrench_d_tensions.col(tendon_idx) = d_wrench_d_tension;
    }

    Matrix6 d_wrench_external, d_wrench_external_d_pose;
    Vector6 wrench_external_body = spatial_to_body_wrench(wrench_external, pose, d_wrench_external, d_wrench_external_d_pose);

    Vector6 wrench_error = wrench - wrench_tendons - wrench_external_body;

    if (H1) { *H1 = -d_wrench_d_pose_prev; }

    if (H2) { *H2 = -d_wrench_d_pose - d_wrench_external_d_pose; }

    if (H3) { *H3 = -d_wrench_d_pose_next; }

    if (H4) { *H4 = Matrix6::Identity(); }

    if (H5) { *H5 = -d_wrench_d_tensions; }

    if (H6) { *H6 = -d_wrench_external; }

    return wrench_error;
}

Vector6 TendonDiscWrenchFactor::get_single_tendon_wrench(
    const double tension, 
    const Pose3& pose, 
    const Pose3& pose_other, 
    const Point3& hole, 
    const Point3& hole_other,
    OptionalJacobian<6, 1> H_tension,
    OptionalJacobian<6, 6> H_pose,
    OptionalJacobian<6, 6> H_pose_other) const
{
    Matrix36 d_hole_other_world_d_pose_other;
    Point3 hole_other_world = pose_other.transformFrom(hole_other, 
        H_pose_other ? &d_hole_other_world_d_pose_other : 0);
    
    Matrix36 d_hole_other_local_d_pose;
    Matrix3 d_hole_other_local_d_hole_other_world;
    Point3 hole_other_local = pose.transformTo(hole_other_world,
        H_pose? &d_hole_other_local_d_pose : 0,
        d_hole_other_local_d_hole_other_world);

    Vector3 hole_diff = hole_other_local - hole;
    double norm = hole_diff.norm();

    Vector3 force_dir;
    Matrix3 d_force_dir_d_hole_diff = Matrix3::Zero();
    
    bool valid = hole_diff.allFinite() && norm > 1e-3;
    
    if (valid) {
        force_dir = normalize(hole_diff, H_pose || H_pose_other ? &d_force_dir_d_hole_diff : 0);
    } else {
        force_dir = Vector3::Zero();
    }

    Vector3 force = tension * force_dir;
    Matrix31 d_force_d_tension = force_dir;
    Matrix33 d_force_d_force_dir = tension * Matrix3::Identity();

    Matrix33 d_moment_d_force;
    Vector3 moment = cross(hole, force, nullptr, 
         H_tension || H_pose || H_pose_other ? &d_moment_d_force : 0);

    Vector6 wrench;
    wrench << moment, force;

    if (H_tension) {
        H_tension->head<3>() = d_moment_d_force * d_force_d_tension;
        H_tension->tail<3>() = d_force_d_tension;
    }

    if (H_pose) {
        Matrix36 d_force_dir_d_pose = d_force_dir_d_hole_diff * d_hole_other_local_d_pose;
        Matrix36 d_force_d_pose = d_force_d_force_dir * d_force_dir_d_pose;
        Matrix36 d_moment_d_pose = d_moment_d_force * d_force_d_pose;

        H_pose->block<3,6>(0,0) = d_moment_d_pose;
        H_pose->block<3,6>(3,0) = d_force_d_pose;
    }

    if (H_pose_other) {
        Matrix36 d_force_dir_d_pose_other =
            d_force_dir_d_hole_diff *
            d_hole_other_local_d_hole_other_world *
            d_hole_other_world_d_pose_other;

        Matrix36 d_force_d_pose_other = d_force_d_force_dir * d_force_dir_d_pose_other;
        Matrix36 d_moment_d_pose_other = d_moment_d_force * d_force_d_pose_other;

        H_pose_other->block<3,6>(0,0) = d_moment_d_pose_other;
        H_pose_other->block<3,6>(3,0) = d_force_d_pose_other;
    }

    return wrench;
}






PositionMeasurementFactor::PositionMeasurementFactor(
    Key pose_key,
    Vector3 position_meas,
    const SharedNoiseModel& model)
: 
    NoiseModelFactor4(model, pose_key), position_meas_(position_meas) {}


Vector PositionMeasurementFactor::evaluateError(const Pose3& pose, OptionalMatrixType H1) const {  
    Matrix36 d_position_d_pose;
    Vector3 error = pose.translation(d_position_d_pose) - position_meas_;

    if (H1) { *H1 = d_position_d_pose; }

    return error;
}