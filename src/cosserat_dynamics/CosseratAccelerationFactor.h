#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using CosseratAccelBase = gtsam::NoiseModelFactorN<
    gtsam::Vector6, gtsam::Pose3, gtsam::Vector6,gtsam::Vector6>;

class CosseratAccelerationFactor: public CosseratAccelBase {
    using CosseratAccelBase::evaluateError;

public:
    CosseratAccelerationFactor(
        gtsam::Key v_prev_key,
        gtsam::Key pose_key,
        gtsam::Key v_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model,
        double dt,
        double linear_damping,
        double rotational_damping,
        double linear_inertia,
        double rotational_inertia);

    gtsam::Vector evaluateError(
        const gtsam::Vector6& v_prev,
        const gtsam::Pose3& pose, 
        const gtsam::Vector6& v,
        const gtsam::Vector6& wrench,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4) const override;

    private:
        const double dt_;
        gtsam::Matrix6 damping_;
        gtsam::Matrix6 inertia_;
        gtsam::Matrix6 inertia_inverse_;
};
