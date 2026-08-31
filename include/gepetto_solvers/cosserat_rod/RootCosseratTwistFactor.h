#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


// Root variant of CosseratTwistFactor (paper Section 4, Eq. 43-44).
//
// The first pose argument is the shared hand-base pose rather than the finger's
// node-0 pose. The node-0 pose used in the twist residual is reconstructed as
// T_0 = T_base o T_offset, and the Jacobian w.r.t. node-0 is mapped back to the
// hand base via the SE(3) composition Jacobian.
using RootCosseratTwistBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector6>;

class RootCosseratTwistFactor: public RootCosseratTwistBase {
    using RootCosseratTwistBase::evaluateError;

public:
    RootCosseratTwistFactor(
        gtsam::Key pose_base_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        const gtsam::Pose3& offset,
        double ds,
        const gtsam::Vector6& nominal_strain,
        const gtsam::Matrix6& K_inv,
        const gtsam::SharedNoiseModel& model,
        bool use_midpoint = true);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_base,
        const gtsam::Pose3& pose_1,
        const gtsam::Vector6& stress_0,
        const gtsam::Vector6& stress_1,
        gtsam::OptionalMatrixType H_base,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3,
        gtsam::OptionalMatrixType H4) const override;

private:
    const gtsam::Pose3 offset_;
    const double ds_;
    const bool use_midpoint_;
    const gtsam::Vector6 nominal_strain_;
    const gtsam::Matrix6 K_inv_;
};
