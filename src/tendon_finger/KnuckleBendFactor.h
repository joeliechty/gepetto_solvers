#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

class KnuckleBendFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
    double z_bend;

public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

    /**
     * Constructor
     * @param key_d1   Key for the proximal disk pose (T_{d1})
     * @param key_d2   Key for the distal disk pose (T_{d2})
     * @param z_bend   The measured bend angle in radians
     * @param model    The noise model (usually derived from sensor datasheet covariance)
     */
    KnuckleBendFactor(
        gtsam::Key key_d1,
        gtsam::Key key_d2,
        double z_bend,
        const gtsam::SharedNoiseModel& model);
    
    /**
     * Error function: h(T_d1, T_d2) - z
     * Calcualte error and optionally Jacobians
     */   
     gtsam::Vector evaluateError(
        const gtsam::Pose3& T_d1,
        const gtsam::Pose3& T_d2,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2) const override;

};