#include "gepetto_solvers/digits/tendon/TendonDiscWrenchFactor.h"

#include <gtsam/base/Matrix.h>

#include "gepetto_solvers/utils/WrenchTransforms.h"


using namespace gtsam;


template<int N>
TendonDiscWrenchFactor<N>::TendonDiscWrenchFactor(
    Key pose_prev_key,
    Key pose_key,
    Key pose_next_key, // Set to dummy key if we are at the tip
    Key wrench_key,
    Key tensions_key,
    Key external_wrench_key,
    const bool is_tip,
    const std::array<Point3, N>& holes_prev,
    const std::array<Point3, N>& holes,
    const std::array<Point3, N>& holes_next, // Not used if we are at the tip
    const std::array<bool, N>& active,
    const std::array<bool, N>& active_prev,
    const std::array<bool, N>& active_next,
    const SharedNoiseModel& model,
    std::optional<Pose3> pose_prev_offset)
:
    TendonWrenchBase<N>(model, pose_prev_key, pose_key, pose_next_key, wrench_key, tensions_key, external_wrench_key),
    is_tip_(is_tip),
    holes_prev_(holes_prev),
    holes_(holes),
    holes_next_(holes_next),
    active_(active),
    active_prev_(active_prev),
    active_next_(active_next),
    pose_prev_offset_(pose_prev_offset) {}


template<int N>
Vector TendonDiscWrenchFactor<N>::evaluateError(
    const Pose3& pose_prev,
    const Pose3& pose,
    const Pose3& pose_next,
    const Vector6& wrench,
    const Eigen::Vector<double, N>& tensions,
    const Vector6& wrench_external,
    OptionalMatrixType H1,
    OptionalMatrixType H2,
    OptionalMatrixType H3,
    OptionalMatrixType H4,
    OptionalMatrixType H5,
    OptionalMatrixType H6) const
{
    // Hand-base reparameterization: when the previous disc is node 0, the passed
    // pose_prev is the hand base; reconstruct the disc pose as pose_prev o offset
    // and keep the composition Jacobian so H1 maps back to the hand base.
    Pose3 pose_prev_eff = pose_prev;
    Matrix6 H_prev_compose = Matrix6::Identity();
    if (pose_prev_offset_) {
        pose_prev_eff = pose_prev.compose(*pose_prev_offset_, H_prev_compose);
    }

    Vector6 wrench_tendons = Vector6::Zero();

    Eigen::Matrix<double, 6, N> d_wrench_d_tensions = Eigen::Matrix<double, 6, N>::Zero();
    Matrix66 d_wrench_d_pose = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_prev = Matrix66::Zero();
    Matrix66 d_wrench_d_pose_next = Matrix66::Zero();

    // Sum up all tendon wrenches on this disc
    for (int tendon_idx = 0; tendon_idx < N; ++tendon_idx) {
        // Skip tendon entirely if it has no hole at this disc
        if (!active_[tendon_idx]) {
            continue;
        }

        Vector6 d_wrench_d_tension_t = Vector6::Zero();

        // Wrench from previous disc (only if tendon has hole at prev disc)
        if (active_prev_[tendon_idx]) {
            Vector6 d_wrench_prev_d_tension;
            Matrix6 d_wrench_prev_d_pose, d_wrench_prev_d_pose_prev;

            // Get wrench from prev disc on current disc in spatial coords
            Vector6 wrench_prev = get_single_tendon_wrench(
                tensions[tendon_idx],
                pose,
                pose_prev_eff,
                holes_[tendon_idx],
                holes_prev_[tendon_idx],
                d_wrench_prev_d_tension,
                d_wrench_prev_d_pose,
                d_wrench_prev_d_pose_prev);

            wrench_tendons += wrench_prev;
            d_wrench_d_tension_t += d_wrench_prev_d_tension;
            d_wrench_d_pose += d_wrench_prev_d_pose;
            d_wrench_d_pose_prev += d_wrench_prev_d_pose_prev;
        }

        // Wrench from next disc. Ignore if we are at the tip or tendon has no hole at next disc
        if (!is_tip_ && active_next_[tendon_idx]) {
            Vector6 d_wrench_next_d_tension;
            Matrix6 d_wrench_next_d_pose, d_wrench_next_d_pose_next;

            Vector6 wrench_next = get_single_tendon_wrench(
                tensions[tendon_idx],
                pose,
                pose_next,
                holes_[tendon_idx],
                holes_next_[tendon_idx],
                d_wrench_next_d_tension,
                d_wrench_next_d_pose,
                d_wrench_next_d_pose_next);

            wrench_tendons += wrench_next;
            d_wrench_d_tension_t += d_wrench_next_d_tension;
            d_wrench_d_pose += d_wrench_next_d_pose;
            d_wrench_d_pose_next += d_wrench_next_d_pose_next;
        }

        d_wrench_d_tensions.col(tendon_idx) = d_wrench_d_tension_t;
    }

    // Error between total wrench and sum of applied, all in spatial coords
    Vector6 wrench_error = wrench - wrench_tendons - wrench_external;

    if (H1) { *H1 = -d_wrench_d_pose_prev * H_prev_compose; }

    if (H2) { *H2 = -d_wrench_d_pose; }

    if (H3) { *H3 = -d_wrench_d_pose_next; }

    if (H4) { *H4 = Matrix6::Identity(); }

    if (H5) { *H5 = -d_wrench_d_tensions; }

    if (H6) { *H6 = -Matrix6::Identity(); }

    return wrench_error;
}


template<int N>
Vector6 TendonDiscWrenchFactor<N>::get_single_tendon_wrench(
    const double tension,
    const Pose3& p0,
    const Pose3& p1,
    const Point3& h0,
    const Point3& h1,
    OptionalJacobian<6, 1> H_tension,
    OptionalJacobian<6, 6> H_p0,
    OptionalJacobian<6, 6> H_p1) const
{
    // TF body hole 1 location to frame 0 for differencing
    Matrix36 d_h1w_d_p1;
    Point3 h1w = p1.transformFrom(h1, d_h1w_d_p1);

    Matrix36 d_h10_d_p0;
    Matrix3 d_h10_d_h1w;
    Point3 h10 = p0.transformTo(h1w, d_h10_d_p0, d_h10_d_h1w); // Hole 1 in frame 0

    // Difference between two holes is direction of force
    Vector3 diff = h10 - h0; // Both holes in frame 0
    Matrix3 d_diff_d_h10 = Matrix3::Identity();

    Matrix3 d_dir_d_diff;
    Vector3 dir = normalize(diff, &d_dir_d_diff); // Frame 0

    // Force is tension in that direction
    Vector3 force = tension * dir;  // Frame 0
    Matrix31 d_force_d_tension = dir;
    Matrix33 d_force_d_dir = tension * Matrix3::Identity();

    // Compute moment about frame 0 origin and combine to wrench
    Matrix3 d_moment_d_force;
    Vector3 moment = cross(h0, force, std::nullopt, d_moment_d_force); // Frame 0

    Vector6 body;
    body << moment, force;
    Matrix63 d_body_d_moment = Matrix63::Zero();
    d_body_d_moment.topRows(3) = Matrix3::Identity();
    Matrix63 d_body_d_force = Matrix63::Zero();
    d_body_d_force.bottomRows(3) = Matrix3::Identity();

    // Our wrenches are all defined in spatial coordinates, so rotate
    Matrix6 d_spatial_d_body, d_spatial_d_p0;
    Vector6 spatial = body_to_spatial_wrench(body, p0, d_spatial_d_body, d_spatial_d_p0);

    if (H_tension) {
        *H_tension = d_spatial_d_body * d_body_d_force * d_force_d_tension +
            d_spatial_d_body * d_body_d_moment * d_moment_d_force * d_force_d_tension;
    }

    if (H_p0) {
        Matrix36 d_force_d_p0 = d_force_d_dir * d_dir_d_diff * d_diff_d_h10 * d_h10_d_p0;
        *H_p0 = d_spatial_d_p0 +
            d_spatial_d_body * d_body_d_force * d_force_d_p0 +
            d_spatial_d_body * d_body_d_moment * d_moment_d_force * d_force_d_p0;
    }

    if (H_p1) {
        Matrix36 d_force_d_p1 = d_force_d_dir * d_dir_d_diff * d_diff_d_h10 * d_h10_d_h1w * d_h1w_d_p1;
        *H_p1 = d_spatial_d_body * d_body_d_force * d_force_d_p1 +
            d_spatial_d_body * d_body_d_moment * d_moment_d_force * d_force_d_p1;
    }

    return spatial;
}


// Explicit instantiations
template class TendonDiscWrenchFactor<1>;
template class TendonDiscWrenchFactor<2>;
template class TendonDiscWrenchFactor<3>;
template class TendonDiscWrenchFactor<4>;
template class TendonDiscWrenchFactor<5>;
template class TendonDiscWrenchFactor<6>;
template class TendonDiscWrenchFactor<7>;
template class TendonDiscWrenchFactor<8>;
template class TendonDiscWrenchFactor<9>;
template class TendonDiscWrenchFactor<10>;
