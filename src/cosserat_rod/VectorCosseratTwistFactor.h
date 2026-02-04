#pragma once

#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using VectorCosseratTwistBase = gtsam::NoiseModelFactorN<gtsam::Vector6, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6>;

class VectorCosseratTwistFactor: public VectorCosseratTwistBase {
    using VectorCosseratTwistBase::evaluateError;

public:
    VectorCosseratTwistFactor(
        gtsam::Key twist_0_key,
        gtsam::Key twist_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        double ds,
        const gtsam::Vector6& nominal_strain,
        const gtsam::Matrix6& K_inv,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Vector6& twist_0, 
        const gtsam::Vector6& twist_1, 
        const gtsam::Vector6& stress_0, 
        const gtsam::Vector6& stress_1, 
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4) const override;

private:
    const double ds_;
    const gtsam::Vector6 nominal_strain_;
    const gtsam::Matrix6 K_inv_;
};