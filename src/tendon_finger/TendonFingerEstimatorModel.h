// TendonFingerEstimatorModel.h
#pragma once

#include "TendonFingerModel.h"
#include "KnuckleBendFactor.h"
#include <vector>

template<int N>
class TendonFingerEstimatorModel : public TendonFingerModel<N> {
public:

    using TendonFingerModel<N>::TendonFingerModel;  // Inherit constructors

    /**
     * Builds the standard tendon graph and adds the sensor factors like the KnuckleBendFactor, etc.
     * @param tensions The Gaussian prior for tendon tensions
     * @param measured_bend The bend angle on the first knuckle (disk_1 to disk_2 where disk_0 is the base) measured by the bend sensor
     * @param bend_noise The noise model for the bend measurements
     */
    gtsam::NonlinearFactorGraph build_estimation_graph(
        const VectorNGaussian<N>& tensions,
        double measured_bend,
        const gtsam::SharedNoiseModel& bend_noise) const;
    };