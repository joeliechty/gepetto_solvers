// #include "gtsam_factors.h"
// #include <gtsam/base/numericalDerivative.h>

// using namespace gtsam;





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



