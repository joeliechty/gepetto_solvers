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
        gtsam::Key stress_x_key,
        gtsam::Key stress_y_key,
        gtsam::Key stress_1x_key,
        gtsam::Key stress_2y_key,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& p,
        const gtsam::Pose3& p1,
        const gtsam::Pose3& p2,
        const gtsam::Vector6& sx,
        const gtsam::Vector6& sy, 
        const gtsam::Vector6& s1x,
        const gtsam::Vector6& s2y,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5,
        gtsam::OptionalMatrixType H6,
        gtsam::OptionalMatrixType H7) const override;
};
