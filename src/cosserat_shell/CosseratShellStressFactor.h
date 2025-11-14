#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using CosseratShellStressBase = gtsam::NoiseModelFactorN<
    gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, 
    gtsam::Vector6, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6>;

class CosseratShellStressFactor: public CosseratShellStressBase {
    using CosseratShellStressBase::evaluateError;

public:
    CosseratShellStressFactor(
        gtsam::Key pose_key,
        gtsam::Key pose_1_key,
        gtsam::Key pose_2_key,
        gtsam::Key stress_key,
        gtsam::Key stress_1_key,
        gtsam::Key stress_2_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose,
        const gtsam::Pose3& pose_1,
        const gtsam::Pose3& pose_2,
        const gtsam::Vector6& stress, 
        const gtsam::Vector6& stress_1,
        const gtsam::Vector6& stress_2,
        const gtsam::Vector6& wrench_1,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5,
        gtsam::OptionalMatrixType H6,
        gtsam::OptionalMatrixType H7) const override;
};
