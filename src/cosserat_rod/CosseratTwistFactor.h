#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using CosseratTwistBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector6>;

class CosseratTwistFactor: public CosseratTwistBase {
    using CosseratTwistBase::evaluateError;

public:
    CosseratTwistFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        double segment_length,
        const gtsam::Matrix6& K_inv,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_0, 
        const gtsam::Pose3& pose_1, 
        const gtsam::Vector6& stress_0, 
        const gtsam::Vector6& stress_1, 
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4) const override;

private:
    double ds_;
    gtsam::Matrix66 K_inv_;
};