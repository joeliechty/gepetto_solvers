#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "tendon_finger/TendonFingerModel.h"

#include <array>


template<int N>
using TendonWrenchBase = gtsam::NoiseModelFactorN<
    gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, Eigen::Vector<double, N>, gtsam::Vector6>;

template<int N>
class TendonDiscWrenchFactor: public TendonWrenchBase<N> {
    using TendonWrenchBase<N>::evaluateError;

public:
    TendonDiscWrenchFactor(
        gtsam::Key pose_prev_key,
        gtsam::Key pose_key,
        gtsam::Key pose_next_key, // Set to dummy key if we are at the tip
        gtsam::Key wrench_key,
        gtsam::Key tensions_key,
        gtsam::Key external_wrench_key,
        const bool is_tip,
        const std::array<gtsam::Point3, N>& holes_prev,
        const std::array<gtsam::Point3, N>& holes,
        const std::array<gtsam::Point3, N>& holes_next, // Not used if we are at the tip
        const std::array<bool, N>& active,          // Has hole at current disc
        const std::array<bool, N>& active_prev,     // Has hole at prev disc
        const std::array<bool, N>& active_next,     // Has hole at next disc
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_prev,
        const gtsam::Pose3& pose,
        const gtsam::Pose3& pose_next,
        const gtsam::Vector6& wrench,
        const Eigen::Vector<double, N>& tensions,
        const gtsam::Vector6& wrench_external,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3,
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5,
        gtsam::OptionalMatrixType H6) const override;

private:
    gtsam::Vector6 get_single_tendon_wrench(
        const double tension,
        const gtsam::Pose3& pose,
        const gtsam::Pose3& pose_other,
        const gtsam::Point3& hole,
        const gtsam::Point3& hole_other,
        gtsam::OptionalJacobian<6, 1> H_tension = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {},
        gtsam::OptionalJacobian<6, 6> H_pose_other = {}) const;

    bool is_tip_;
    std::array<gtsam::Point3, N> holes_prev_;
    std::array<gtsam::Point3, N> holes_;
    std::array<gtsam::Point3, N> holes_next_;
    std::array<bool, N> active_;         // tendon has hole at THIS disc
    std::array<bool, N> active_prev_;    // tendon has hole at PREV disc
    std::array<bool, N> active_next_;    // tendon has hole at NEXT disc
};
