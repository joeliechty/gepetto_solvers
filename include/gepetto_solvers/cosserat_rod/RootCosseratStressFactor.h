#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


// Root variant of CosseratStressFactor (paper Section 4, Eq. 43-44).
//
// The first pose argument is the shared hand-base pose rather than the finger's
// node-0 pose. The node-0 pose used in the stress residual is reconstructed as
// T_0 = T_base o T_offset, and the Jacobian w.r.t. node-0 is mapped back to the
// hand base via the SE(3) composition Jacobian.
using RootCosseratStressBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6>;

class RootCosseratStressFactor: public RootCosseratStressBase {
    using RootCosseratStressBase::evaluateError;

public:
    RootCosseratStressFactor(
        gtsam::Key pose_base_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        gtsam::Key wrench_key,
        const gtsam::Pose3& offset,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_base,
        const gtsam::Pose3& pose_1,
        const gtsam::Vector6& stress_0,
        const gtsam::Vector6& stress_1,
        const gtsam::Vector6& wrench_1,
        gtsam::OptionalMatrixType H_base,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3,
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5) const override;

private:
    const gtsam::Pose3 offset_;
};
