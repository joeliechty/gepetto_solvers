#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>


// Root variant of BoundaryStressFactor (paper Section 4, Eq. 43-44).
//
// Instead of taking the finger's node-0 pose key, this factor takes the shared
// hand-base pose key and a fixed structural offset T_offset. The node-0 pose is
// reconstructed deterministically as T_0 = T_base o T_offset, and the Jacobian
// w.r.t. the original node-0 pose is mapped back to the hand base via the SE(3)
// composition Jacobian (chain rule).
using RootBoundaryStressBase = gtsam::NoiseModelFactorN<gtsam::Vector6, gtsam::Vector6, gtsam::Pose3>;

class RootBoundaryStressFactor: public RootBoundaryStressBase {
    using RootBoundaryStressBase::evaluateError;

public:
    RootBoundaryStressFactor(
        gtsam::Key stress_key,
        gtsam::Key wrench_key,
        gtsam::Key pose_base_key,
        const gtsam::Pose3& offset,
        const gtsam::SharedNoiseModel& model,
        bool is_base);

    gtsam::Vector evaluateError(
        const gtsam::Vector6& stress,
        const gtsam::Vector6& wrench,
        const gtsam::Pose3& pose_base,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H_base) const override;

private:
    gtsam::Pose3 offset_;
    bool is_base_;
};
