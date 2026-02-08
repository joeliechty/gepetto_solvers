#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using PlatformWrenchBase = gtsam::NoiseModelFactorN<
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3,
    gtsam::Vector6, gtsam::Pose3>;

class PlatformWrenchBalanceFactor: public PlatformWrenchBase {
    using PlatformWrenchBase::evaluateError;

public:
    PlatformWrenchBalanceFactor(
        gtsam::Key w0_key, gtsam::Key p0_key,
        gtsam::Key w1_key, gtsam::Key p1_key,
        gtsam::Key w2_key, gtsam::Key p2_key,
        gtsam::Key w3_key, gtsam::Key p3_key,
        gtsam::Key w4_key, gtsam::Key p4_key,
        gtsam::Key w5_key, gtsam::Key p5_key,
        gtsam::Key w_platform_key, gtsam::Key p_platform_key,
        const gtsam::SharedNoiseModel& model);
        
    gtsam::Vector evaluateError(
        const gtsam::Vector6& w0, const gtsam::Pose3& p0,
        const gtsam::Vector6& w1, const gtsam::Pose3& p1,
        const gtsam::Vector6& w2, const gtsam::Pose3& p2,
        const gtsam::Vector6& w3, const gtsam::Pose3& p3,
        const gtsam::Vector6& w4, const gtsam::Pose3& p4,
        const gtsam::Vector6& w5, const gtsam::Pose3& p5,
        const gtsam::Vector6& w_platform, const gtsam::Pose3& p_platform,
        gtsam::OptionalMatrixType H1, gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3, gtsam::OptionalMatrixType H4, 
        gtsam::OptionalMatrixType H5, gtsam::OptionalMatrixType H6,
        gtsam::OptionalMatrixType H7, gtsam::OptionalMatrixType H8,
        gtsam::OptionalMatrixType H9, gtsam::OptionalMatrixType H10, 
        gtsam::OptionalMatrixType H11, gtsam::OptionalMatrixType H12,
        gtsam::OptionalMatrixType H13, gtsam::OptionalMatrixType H14) const override;
};
