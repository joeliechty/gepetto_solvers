#pragma once

#include "TendonFingerEstimatorModel.h"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <optional>
#include <vector>

// Config for cont. time GP estimator
struct TendonFingerEstimatorConfig {
    TendonFingerSolverConfig base_config;

    
    Eigen::MatrixXd gp_tense_Qc;    // Size NxN. Empty to disable.
    Eigen::MatrixXd gp_len_Qc;      // Size NxN. Empty to disable.
    gtsam::Matrix6 gp_pose_Qc;      // Size 6x6. Identity to disable.
};

template<int N>
class TendonFingerIterativeSolver {
public:
    TendonFingerIterativeSolver(
        const TendonFingerEstimatorConfig& config,
        gtsam::SharedNoiseModel bend_noise);
    
        /**
         * Event driven step function. Call this whenver sensor reading arrives.
         * @param timestamp_sec Continuous timestamp of new measurement
         * @param tension_meas Optional motor encoder / tension reading
         * @param bend_meas Optional knuckle bend sensor reading
         */
    void step(
        double timestamp_sec,
        const std::optional<VectorNGaussian<N>>& tensions_meas,
        const std::optional<double>& measured_bend);
    
        gtsam::Values get_current_estimate() const {return current_estimate_; }

        // TODO later: Helper to extract the state in a clean way for planner
        // (Assuming have a struct to hold physical state data)
        // TendonFingerState extract_state_for_planner() const;

    private:
        TendonFingerEstimatorConfig config_;
        gtsam::SharedNoiseModel bend_noise_;

        gtsam::ISAM2 isam_;
        gtsam::Values current_estimate_;

        std::optional<double> prev_timestamp_;
        std::optional<gtsam::Key> prev_tensions_key_;
        std::optional<gtsam::Key> prev_lengths_key_;
        std::vector<gtsam::Key> prev_pose_keys_;
};

