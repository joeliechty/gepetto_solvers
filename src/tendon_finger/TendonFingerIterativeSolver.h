#pragma once

#include "TendonFingerSolver.h"
#include "TendonFingerEstimatorModel.h"
#include "utils/Gaussians.h"
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

// Config for cont. time GP estimator
struct TendonFingerEstimatorConfig {
    TendonFingerSolverConfig base_config;

    // Background tension prior — mirrors planner's background_tensions_mean/sigmas.
    // Applied every step as the base prior before any sensor measurements.
    Eigen::VectorXd background_tensions_mean;  // N-dim; empty → zero mean
    Eigen::MatrixXd background_tensions_cov;   // NxN;   empty → 1e6 * I (unconstrained)

    Eigen::MatrixXd gp_tense_Qc;    // Size NxN. Empty to disable.
    Eigen::MatrixXd gp_len_Qc;      // Size NxN. Empty to disable.
    gtsam::Matrix6 gp_pose_Qc;      // Size 6x6. Identity to disable.
};

template<int N>
class TendonFingerIterativeSolver {
public:
    static constexpr int NumTendons = N;

    TendonFingerIterativeSolver(
        const TendonFingerEstimatorConfig& config,
        gtsam::SharedNoiseModel bend_noise);

        /**
         * Event driven step function. Call this whenver sensor reading arrives.
         * @param timestamp_sec    Continuous timestamp of new measurement
         * @param tensions_meas    Optional motor encoder / tension reading
         * @param lengths_meas     Optional commanded/measured tendon-length reading
         * @param measured_bend    Optional knuckle bend sensor reading
         * @param tip_wrench_meas  Optional tip load cell (6-DOF wrench)
         * @param tip_position_meas Optional tip position (e.g. mocap)
         */
    void step(
        double timestamp_sec,
        const std::optional<VectorNGaussian<N>>& tensions_meas,
        const std::optional<VectorNGaussian<N>>& lengths_meas,
        const std::optional<double>& measured_bend,
        const std::optional<Vector6Gaussian>& tip_wrench_meas = std::nullopt,
        const std::optional<Vector3Gaussian>& tip_position_meas = std::nullopt);

        gtsam::Values get_current_estimate() const {return current_estimate_; }

        /**
         * Extract TendonFingerMarginals from the current ISAM2 estimate using
         * the most recently stepped model's keys. Requires at least one successful step().
         */
        TendonFingerMarginals get_current_marginals() const;

    private:
        TendonFingerEstimatorConfig config_;
        gtsam::SharedNoiseModel bend_noise_;
        gtsam::SharedDiagonal stress_noise_;

        gtsam::ISAM2 isam_;
        gtsam::Values current_estimate_;

        std::optional<double> prev_timestamp_;
        std::optional<gtsam::Key> prev_tensions_key_;
        std::optional<gtsam::Key> prev_lengths_key_;
        std::vector<gtsam::Key> prev_pose_keys_;

        // Model from the most recent step. Kept alive so get_current_marginals()
        // can query keys and call TendonFingerModel::get_marginals().
        std::unique_ptr<TendonFingerEstimatorModel<N>> latest_model_;
};


// Runtime dispatch wrapper that selects the correct template specialization
// based on config.base_config.num_tendons. This is the class exposed to Python.
class TendonFingerIterativeSolverDispatch {
public:
    TendonFingerIterativeSolverDispatch(
        const TendonFingerEstimatorConfig& config,
        double bend_sigma);

    void step(
        double timestamp_sec,
        const std::optional<VectorXGaussian>& tensions_meas,
        const std::optional<VectorXGaussian>& lengths_meas,
        const std::optional<double>& measured_bend,
        const std::optional<Vector6Gaussian>& tip_wrench_meas = std::nullopt,
        const std::optional<Vector3Gaussian>& tip_position_meas = std::nullopt);

    TendonFingerMarginals get_current_marginals() const;

    int num_tendons() const { return num_tendons_; }

private:
    int num_tendons_;

    using SolverVariant = std::variant<
        std::unique_ptr<TendonFingerIterativeSolver<1>>,
        std::unique_ptr<TendonFingerIterativeSolver<2>>,
        std::unique_ptr<TendonFingerIterativeSolver<3>>,
        std::unique_ptr<TendonFingerIterativeSolver<4>>,
        std::unique_ptr<TendonFingerIterativeSolver<5>>,
        std::unique_ptr<TendonFingerIterativeSolver<6>>,
        std::unique_ptr<TendonFingerIterativeSolver<7>>,
        std::unique_ptr<TendonFingerIterativeSolver<8>>,
        std::unique_ptr<TendonFingerIterativeSolver<9>>,
        std::unique_ptr<TendonFingerIterativeSolver<10>>
    >;
    SolverVariant solver_;
};

