#pragma once

#include <gtsam/base/types.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using CosseratDynamicsBase = gtsam::NoiseModelFactorN<
    gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6>;

class CosseratDynamicsFactor: public CosseratDynamicsBase {
    using CosseratDynamicsBase::evaluateError;  
public:
    CosseratDynamicsFactor(
        gtsam::Key p_im2_key,
        gtsam::Key p_im1_key,
        gtsam::Key p_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model,
        double dt,
        double linear_damping,
        double rotational_damping,
        double linear_inertia,
        double rotational_inertia);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& p_im2,
        const gtsam::Pose3& p_im1,
        const gtsam::Pose3& p,
        const gtsam::Vector6& wrench,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4) const override;

    private:
        const double dt_;
        gtsam::Matrix6 damping_;
        gtsam::Matrix6 inertia_;
};
