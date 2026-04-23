#include "TendonFingerIterativeSolver.h"
#include <gtsam/nonlinear/PriorFactor.h>
#include <iostream>

template<int N>
TendonFingerIterativeSolver<N>::TendonFingerIterativeSolver(
    const TendonFingerEstimatorConfig& config,
    gtsam::SharedNoiseModel bend_noise)
    : config_(config), bend_noise_(bend_noise)
{
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01; // Tighter threshold for cont. tracking
    parameters.relinearizeSkip = 1;         // Check every iteration
    parameters.optimizationParams = gtsam::ISAM2DoglegParams();
    isam_ = gtsam::ISAM2(parameters);
}


template<int N>
void TendonFingerIterativeSolver<N>::step(
    double timestamp_sec,
    const std::optional<VectorNGaussian<N>>& tensions_meas,
    const std::optional<double>& measured_bend)
{
    // 1. Instantiate the model to generate the kinematics for a specific event
    const auto& bc = config_.base_config;
    auto twist_noise  = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(bc.sigma_twist_rot));
    auto stress_noise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(bc.sigma_stress_force));
    auto base_noise   = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector6::Constant(bc.sigma_base_pos));
    gtsam::Pose3 base_pose(bc.base_pose);

    std::unique_ptr<TendonFingerEstimatorModel<N>> current_model_ptr;
    if (bc.per_disc_tendon_input.is_populated()) {
        current_model_ptr = std::make_unique<TendonFingerEstimatorModel<N>>(
            bc.rod_length, bc.num_discs, bc.num_between_nodes,
            bc.per_disc_tendon_input, bc.K_inv,
            twist_noise, stress_noise, base_pose, base_noise,
            bc.disc_positions_normalized);
    } else {
        current_model_ptr = std::make_unique<TendonFingerEstimatorModel<N>>(
            bc.rod_length, bc.num_discs, bc.num_between_nodes,
            bc.tendon_input, bc.K_inv,
            twist_noise, stress_noise, base_pose, base_noise,
            bc.disc_positions_normalized);
    }
    auto& current_model = *current_model_ptr;

    // Build the pure kinematic/static base graph (no sensor priors yet)
    // Pass in empty tensions here, will add measurement below if it exists
    VectorNGaussian<N> empty_tensions;
    empty_tensions.mean = Eigen::Vector<double, N>::Zero();
    empty_tensions.cov = Eigen::Matrix<double, N, N>::Identity() * 1e6; // Large covariance to effectively ignore this prior

    gtsam::NonlinearFactorGraph new_factors = current_model.build_graph(empty_tensions);
    gtsam::Values new_initial_values = current_model.get_initial_values();

    // ========================================================================
    // A. CALCULATE DYNAMIC TIME STEP
    // ========================================================================
    double dt = 0.0;
    if (prev_timestamp_.has_value()){
        dt = timestamp_sec - prev_timestamp_.value();
    }
    
    // avoid backwards/really small time steps which can cause numerical issues, if sensors 
    // are that close to each other then we can stick them in the same state
    if (prev_timestamp_.has_value() && dt > 1e-6){

        // GP on Tendon Lengths
        if (prev_lengths_key_.has_value() && config_.gp_len_Qc.size() > 0) {
            Eigen::Matrix<double, N, N> Qc_len = config_.gp_len_Qc.topLeftCorner<N, N>();
            auto gp_len_noise = gtsam::noiseModel::Gaussian::Covariance(Qc_len * dt);
            
            new_factors.add(gtsam::BetweenFactor<Eigen::Vector<double, N>>(
                prev_lengths_key_.value(),
                current_model.get_lengths_key(),
                Eigen::Vector<double, N>::Zero(),
                gp_len_noise));
        }

        // GP on Poses
        if (!prev_pose_keys_.empty() && config_.gp_pose_Qc.size() > 0) {
            auto gp_pose_noise = gtsam::noiseModel::Gaussian::Covariance(config_.gp_pose_Qc * dt);
            const auto& current_pose_keys = current_model.rod_->get_pose_keys();
            
            for (size_t i = 0; i < prev_pose_keys_.size(); ++i) {
                new_factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    prev_pose_keys_[i],
                    current_pose_keys[i],
                    gtsam::Pose3::Identity(),
                    gp_pose_noise));
            }
        }

        // GP on Tensions (If configured, though as discussed, maybe omit for position control)
        if (prev_tensions_key_.has_value() && config_.gp_tense_Qc.size() > 0) {
            Eigen::Matrix<double, N, N> Qc_tens = config_.gp_tense_Qc.topLeftCorner<N, N>();
            auto gp_tens_noise = gtsam::noiseModel::Gaussian::Covariance(Qc_tens * dt);
            
            new_factors.add(gtsam::BetweenFactor<Eigen::Vector<double, N>>(
                prev_tensions_key_.value(),
                current_model.get_tensions_key(),
                Eigen::Vector<double, N>::Zero(),
                gp_tens_noise));
        }

        // --- WARM START ---
        // Overwrite the straight-rod initial guesses with the converged state 
        // from the previous timestep. This is critical for real-time performance.
        if (!current_estimate_.empty()) {
            const auto& curr_keys = current_model.rod_->get_pose_keys();
            for (size_t i = 0; i < curr_keys.size(); ++i) {
                new_initial_values.update(curr_keys[i], current_estimate_.at<gtsam::Pose3>(prev_pose_keys_[i]));
            }
            new_initial_values.update(current_model.get_lengths_key(), current_estimate_.at<Eigen::Vector<double, N>>(prev_lengths_key_.value()));
            new_initial_values.update(current_model.get_tensions_key(), current_estimate_.at<Eigen::Vector<double, N>>(prev_tensions_key_.value()));
        }
    }

    // ========================================================================
    // C. ADD ASYNCHRONOUS SENSOR MEASUREMENTS
    // ========================================================================

    // Did the knuckle bend sensor trigger this step?
    if (measured_bend.has_value() && current_model.get_num_nodes() >=3) {
        int pose_idx_proximal = current_model.get_tendon_config().disc_pose_idx[1];
        int pose_idx_distal = current_model.get_tendon_config().disc_pose_idx[2];

        new_factors.add(KnuckleBendFactor(
            current_model.rod_->get_pose_key(pose_idx_proximal),
            current_model.rod_->get_pose_key(pose_idx_distal),
            measured_bend.value(),
            bend_noise_
        ));
    }

    // Did the motor encoders trigger this step?
    if (tensions_meas.has_value()) {
        new_factors.add(gtsam::PriorFactor<Eigen::Vector<double, N>>(
            current_model.get_tensions_key(),
            tensions_meas.value().mean,
            gtsam::noiseModel::Gaussian::Covariance(tensions_meas.value().cov)
        ));
    }

    // ========================================================================
    // D. UPDATE iSAM2
    // ========================================================================

    // only update if dt > 1e-6 OR if it's the very first initialization (no previous timestamp)
    if (dt > 1e-6 || !prev_timestamp_.has_value()) {
        isam_.update(new_factors, new_initial_values);
        isam_.update(); // Force update to ensure convergence before next step

        current_estimate_ = isam_.calculateEstimate();

        // Save keys and time for the next step
        prev_timestamp_ = timestamp_sec;
        prev_lengths_key_ = current_model.get_lengths_key();
        prev_tensions_key_ = current_model.get_tensions_key();
        prev_pose_keys_ = current_model.rod_->get_pose_keys();
    }


}

// Explicit instantiations
template class TendonFingerIterativeSolver<1>;
template class TendonFingerIterativeSolver<2>;
template class TendonFingerIterativeSolver<3>;
template class TendonFingerIterativeSolver<4>;
template class TendonFingerIterativeSolver<5>;
template class TendonFingerIterativeSolver<6>;
template class TendonFingerIterativeSolver<7>;
template class TendonFingerIterativeSolver<8>;
template class TendonFingerIterativeSolver<9>;
template class TendonFingerIterativeSolver<10>;
