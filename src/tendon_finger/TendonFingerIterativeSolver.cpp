#include "TendonFingerIterativeSolver.h"
#include "measurement/PositionPriorFactor.h"
#include "utils/MiscInline.h"
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <iostream>

template<int N>
TendonFingerIterativeSolver<N>::TendonFingerIterativeSolver(
    const TendonFingerEstimatorConfig& config,
    gtsam::SharedNoiseModel bend_noise)
    : config_(config), bend_noise_(bend_noise)
{
    stress_noise_ = get_noise_model_rot_pos(
        config.base_config.sigma_stress_moment, config.base_config.sigma_stress_force);

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01; // Tighter threshold for cont. tracking
    parameters.relinearizeSkip = 1;         // Check every iteration
    parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams();
    isam_ = gtsam::ISAM2(parameters);
}


template<int N>
void TendonFingerIterativeSolver<N>::step(
    double timestamp_sec,
    const std::optional<VectorNGaussian<N>>& tensions_meas,
    const std::optional<VectorNGaussian<N>>& lengths_meas,
    const std::optional<double>& measured_bend,
    const std::optional<Vector6Gaussian>& tip_wrench_meas,
    const std::optional<Vector3Gaussian>& tip_position_meas)
{
    // 1. Instantiate the model to generate the kinematics for a specific event
    const auto& bc = config_.base_config;
    auto twist_noise = get_noise_model_rot_pos(bc.sigma_twist_rot, bc.sigma_twist_pos);
    auto base_noise  = get_noise_model_rot_pos(bc.sigma_base_rot,  bc.sigma_base_pos);

    // Mirror TendonFingerSolver: apply the canonical base rotation when base_pose is unset.
    gtsam::Pose3 base_pose;
    if (bc.base_pose.isZero()) {
        gtsam::Rot3 base_rot = gtsam::Rot3::Rx(-M_PI / 2).compose(gtsam::Rot3::Rz(M_PI));
        base_pose = gtsam::Pose3(base_rot, gtsam::Point3::Zero());
    } else {
        base_pose = gtsam::Pose3(bc.base_pose);
    }

    std::unique_ptr<TendonFingerEstimatorModel<N>> current_model_ptr;
    if (bc.per_disc_tendon_input.is_populated()) {
        if (bc.K_inv_per_segment.empty()) {
            current_model_ptr = std::make_unique<TendonFingerEstimatorModel<N>>(
                bc.rod_length, bc.num_discs, bc.num_between_nodes,
                bc.per_disc_tendon_input, bc.K_inv,
                twist_noise, stress_noise_, base_pose, base_noise,
                bc.disc_positions_normalized);
        } else {
            current_model_ptr = std::make_unique<TendonFingerEstimatorModel<N>>(
                bc.rod_length, bc.num_discs, bc.num_between_nodes,
                bc.per_disc_tendon_input, bc.K_inv_per_segment,
                twist_noise, stress_noise_, base_pose, base_noise,
                bc.disc_positions_normalized);
        }
    } else {
        if (bc.K_inv_per_segment.empty()) {
            current_model_ptr = std::make_unique<TendonFingerEstimatorModel<N>>(
                bc.rod_length, bc.num_discs, bc.num_between_nodes,
                bc.tendon_input, bc.K_inv,
                twist_noise, stress_noise_, base_pose, base_noise,
                bc.disc_positions_normalized);
        } else {
            current_model_ptr = std::make_unique<TendonFingerEstimatorModel<N>>(
                bc.rod_length, bc.num_discs, bc.num_between_nodes,
                bc.tendon_input, bc.K_inv_per_segment,
                twist_noise, stress_noise_, base_pose, base_noise,
                bc.disc_positions_normalized);
        }
    }
    auto& current_model = *current_model_ptr;

    // Build the pure kinematic/static base graph using background tension prior.
    // Falls back to zero mean / 1e6*I cov if not configured.
    VectorNGaussian<N> bg_tensions;
    if (config_.background_tensions_mean.size() == N) {
        bg_tensions.mean = config_.background_tensions_mean;
    } else {
        bg_tensions.mean = Eigen::Vector<double, N>::Zero();
    }
    if (config_.background_tensions_cov.rows() == N && config_.background_tensions_cov.cols() == N) {
        bg_tensions.cov = config_.background_tensions_cov;
    } else {
        bg_tensions.cov = Eigen::Matrix<double, N, N>::Identity() * 1e6;
    }

    gtsam::NonlinearFactorGraph new_factors = current_model.build_graph(bg_tensions);
    gtsam::Values new_initial_values = current_model.get_initial_values();

    // Constrain external wrenches — mirrors TendonFingerSolver::build_graph() exactly.
    // Without these priors the wrench variables are unconstrained and the system is singular.
    int num_nodes = current_model.get_num_nodes();
    for (int i = 1; i + 1 < num_nodes; ++i) {
        new_factors.add(gtsam::PriorFactor<gtsam::Vector6>(
            current_model.get_external_wrench_key(i),
            gtsam::Vector6::Zero(),
            stress_noise_));
    }

    // Tip wrench: use measurement if provided, otherwise pin to zero.
    gtsam::Vector6 tip_wrench_mean = gtsam::Vector6::Zero();
    gtsam::SharedNoiseModel tip_wrench_noise =
        gtsam::noiseModel::Gaussian::Covariance(stress_noise_->covariance());
    if (tip_wrench_meas.has_value()) {
        tip_wrench_mean = tip_wrench_meas->mean;
        tip_wrench_noise = gtsam::noiseModel::Gaussian::Covariance(tip_wrench_meas->cov);
    }
    new_factors.add(gtsam::PriorFactor<gtsam::Vector6>(
        current_model.get_external_wrench_key(num_nodes - 1),
        tip_wrench_mean,
        tip_wrench_noise));

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

    // Did a commanded/measured tendon length reading trigger this step?
    if (lengths_meas.has_value()) {
        new_factors.add(gtsam::PriorFactor<Eigen::Vector<double, N>>(
            current_model.get_lengths_key(),
            lengths_meas.value().mean,
            gtsam::noiseModel::Gaussian::Covariance(lengths_meas.value().cov)
        ));
    }

    // Tip position measurement (e.g. mocap / vision)
    if (tip_position_meas.has_value()) {
        new_factors.add(PositionPriorFactor(
            current_model.rod_->get_pose_key(-1),
            tip_position_meas->mean,
            gtsam::noiseModel::Gaussian::Covariance(tip_position_meas->cov)));
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

        // Retain the model so get_current_marginals() can use its keys.
        latest_model_ = std::move(current_model_ptr);
    }


}


template<int N>
TendonFingerMarginals TendonFingerIterativeSolver<N>::get_current_marginals() const
{
    if (!latest_model_) {
        throw std::runtime_error(
            "TendonFingerIterativeSolver::get_current_marginals() called before any successful step().");
    }
    gtsam::Marginals marginals(
        isam_.getFactorsUnsafe(), current_estimate_,
        gtsam::Marginals::Factorization::CHOLESKY);
    return latest_model_->get_marginals(current_estimate_, marginals);
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


// --- TendonFingerIterativeSolverDispatch (runtime dispatch wrapper) ---

TendonFingerIterativeSolverDispatch::TendonFingerIterativeSolverDispatch(
    const TendonFingerEstimatorConfig& config,
    double bend_sigma)
    : num_tendons_(config.base_config.num_tendons)
{
    auto bend_noise = gtsam::noiseModel::Diagonal::Sigmas(
        gtsam::Vector1::Constant(bend_sigma));

    switch (num_tendons_) {
        case 1:  solver_ = std::make_unique<TendonFingerIterativeSolver<1>>(config, bend_noise); break;
        case 2:  solver_ = std::make_unique<TendonFingerIterativeSolver<2>>(config, bend_noise); break;
        case 3:  solver_ = std::make_unique<TendonFingerIterativeSolver<3>>(config, bend_noise); break;
        case 4:  solver_ = std::make_unique<TendonFingerIterativeSolver<4>>(config, bend_noise); break;
        case 5:  solver_ = std::make_unique<TendonFingerIterativeSolver<5>>(config, bend_noise); break;
        case 6:  solver_ = std::make_unique<TendonFingerIterativeSolver<6>>(config, bend_noise); break;
        case 7:  solver_ = std::make_unique<TendonFingerIterativeSolver<7>>(config, bend_noise); break;
        case 8:  solver_ = std::make_unique<TendonFingerIterativeSolver<8>>(config, bend_noise); break;
        case 9:  solver_ = std::make_unique<TendonFingerIterativeSolver<9>>(config, bend_noise); break;
        case 10: solver_ = std::make_unique<TendonFingerIterativeSolver<10>>(config, bend_noise); break;
        default: throw std::invalid_argument(
            "num_tendons must be between 1 and 10, got " + std::to_string(num_tendons_));
    }
}


void TendonFingerIterativeSolverDispatch::step(
    double timestamp_sec,
    const std::optional<VectorXGaussian>& tensions_meas,
    const std::optional<VectorXGaussian>& lengths_meas,
    const std::optional<double>& measured_bend,
    const std::optional<Vector6Gaussian>& tip_wrench_meas,
    const std::optional<Vector3Gaussian>& tip_position_meas)
{
    if (tensions_meas.has_value() && tensions_meas->mean.size() != num_tendons_) {
        throw std::invalid_argument(
            "tensions_meas size (" + std::to_string(tensions_meas->mean.size()) +
            ") does not match num_tendons (" + std::to_string(num_tendons_) + ")");
    }
    if (lengths_meas.has_value() && lengths_meas->mean.size() != num_tendons_) {
        throw std::invalid_argument(
            "lengths_meas size (" + std::to_string(lengths_meas->mean.size()) +
            ") does not match num_tendons (" + std::to_string(num_tendons_) + ")");
    }

    std::visit([&](auto& solver_ptr) {
        using SolverType = typename std::remove_reference_t<decltype(*solver_ptr)>;
        constexpr int M = SolverType::NumTendons;

        std::optional<VectorNGaussian<M>> t_fixed;
        if (tensions_meas.has_value()) {
            VectorNGaussian<M> t;
            t.mean = tensions_meas->mean;
            t.cov = tensions_meas->cov;
            t_fixed = t;
        }

        std::optional<VectorNGaussian<M>> l_fixed;
        if (lengths_meas.has_value()) {
            VectorNGaussian<M> l;
            l.mean = lengths_meas->mean;
            l.cov = lengths_meas->cov;
            l_fixed = l;
        }

        solver_ptr->step(timestamp_sec, t_fixed, l_fixed, measured_bend,
                         tip_wrench_meas, tip_position_meas);
    }, solver_);
}


TendonFingerMarginals TendonFingerIterativeSolverDispatch::get_current_marginals() const
{
    return std::visit([](const auto& solver_ptr) -> TendonFingerMarginals {
        return solver_ptr->get_current_marginals();
    }, solver_);
}
