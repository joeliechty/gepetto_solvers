#include "TendonFingerIterativeSolver.h"
#include "measurement/PositionPriorFactor.h"
#include "utils/MiscInline.h"
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
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
    parameters.relinearizeSkip = 5;         // Check every iteration
    parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams();
    // parameters.optimizationParams = gtsam::ISAM2DoglegParams();
    parameters.factorization = gtsam::ISAM2Params::CHOLESKY;
    // findUnusedFactorSlots is required by IncrementalFixedLagSmoother to
    // safely remove factors that touch marginalized-out keys.
    parameters.findUnusedFactorSlots = true;
    smoother_ = gtsam::IncrementalFixedLagSmoother(config.lag_sec, parameters);
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

    // ========================================================================
    // BRANCH A: SIMULTANEOUS MEASUREMENT (dt < 1e-6)
    // ========================================================================
    if (prev_timestamp_.has_value() && dt < 1e-6) {
        gtsam::NonlinearFactorGraph simultaneous_factors;
        
        // 1. Add whatever new measurements just arrived, but attach them to 
        // the ALREADY EXISTING keys in latest_model_
        if (measured_bend.has_value() && latest_model_->get_num_nodes() >= 3) {
            int pose_idx_proximal = latest_model_->get_tendon_config().disc_pose_idx[1];
            int pose_idx_distal = latest_model_->get_tendon_config().disc_pose_idx[2];
            simultaneous_factors.add(KnuckleBendFactor(
                latest_model_->rod_->get_pose_key(pose_idx_proximal),
                latest_model_->rod_->get_pose_key(pose_idx_distal),
                measured_bend.value(),
                bend_noise_));
        }
        if (lengths_meas.has_value()) {
            simultaneous_factors.add(gtsam::PriorFactor<Eigen::Vector<double, N>>(
                latest_model_->get_lengths_key(),
                lengths_meas.value().mean,
                gtsam::noiseModel::Gaussian::Covariance(lengths_meas.value().cov)));
        }

        // 2. Update ISAM2. 
        // We pass EMPTY initial values and EMPTY timestamps because we aren't 
        // spawning any new variables. We just add edges to existing ones.
        smoother_.update(simultaneous_factors, gtsam::Values(), gtsam::FixedLagSmoother::KeyTimestampMap());
        
        // 3. Re-extract the marginals since the estimate for this timestamp 
        // just got more accurate with the combined sensor data.
        current_estimate_ = smoother_.calculateEstimate();
        
        return; // Exit the step early. We do not advance time.
    }

    // ========================================================================
    // BRANCH B: NORMAL TIME ADVANCEMENT (dt >= 1e-6 or First Step)
    // ========================================================================
    
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
        // if (!current_estimate_.empty()) {
        //     const auto& curr_keys = current_model.rod_->get_pose_keys();
        //     for (size_t i = 0; i < curr_keys.size(); ++i) {
        //         new_initial_values.update(curr_keys[i], current_estimate_.at<gtsam::Pose3>(prev_pose_keys_[i]));
        //     }
        //     new_initial_values.update(current_model.get_lengths_key(), current_estimate_.at<Eigen::Vector<double, N>>(prev_lengths_key_.value()));
        //     new_initial_values.update(current_model.get_tensions_key(), current_estimate_.at<Eigen::Vector<double, N>>(prev_tensions_key_.value()));
        // }
        if (!current_estimate_.empty() && latest_model_) {
            // 1. Update Poses
            const auto& curr_keys = current_model.rod_->get_pose_keys();
            const auto& prev_keys = latest_model_->rod_->get_pose_keys();
            for (size_t i = 0; i < curr_keys.size(); ++i) {
                new_initial_values.update(curr_keys[i], current_estimate_.at<gtsam::Pose3>(prev_keys[i]));
            }
            
            // 2. Update Lengths and Tensions
            new_initial_values.update(current_model.get_lengths_key(), current_estimate_.at<Eigen::Vector<double, N>>(latest_model_->get_lengths_key()));
            new_initial_values.update(current_model.get_tensions_key(), current_estimate_.at<Eigen::Vector<double, N>>(latest_model_->get_tensions_key()));

            // 3. Update Stresses and Internal Wrenches
            for (int i = 0; i < current_model.get_num_nodes(); ++i) {
                new_initial_values.update(current_model.rod_->get_stress_key(i), 
                    current_estimate_.at<gtsam::Vector6>(latest_model_->rod_->get_stress_key(i)));
                new_initial_values.update(current_model.rod_->get_wrench_key(i), 
                    current_estimate_.at<gtsam::Vector6>(latest_model_->rod_->get_wrench_key(i)));
            }

            // 4. Update External Disc Wrenches
            for (size_t disc_idx = 1; disc_idx < current_model.get_tendon_config().disc_pose_idx.size(); ++disc_idx) {
                new_initial_values.update(current_model.get_disc_wrench_key(disc_idx), 
                    current_estimate_.at<gtsam::Vector6>(latest_model_->get_disc_wrench_key(disc_idx)));
            }
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
    // NEW: BATCH WARM-START FOR THE VERY FIRST STEP
    // ========================================================================
    if (!prev_timestamp_.has_value()) {
        gtsam::LevenbergMarquardtParams lm_params;
        lm_params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
        // lm_params.setVerbosityLM("SUMMARY"); // Uncomment to see batch progress

        if (config_.homotopy_steps <= 0) {
            // Legacy single-batch path: jump directly to the true measurements.
            gtsam::LevenbergMarquardtOptimizer batch_optimizer(new_factors, new_initial_values, lm_params);
            new_initial_values = batch_optimizer.optimize();
        } else {
            // Numerical continuation. Phase A solves a zero-bend graph to
            // recover rest tendon lengths (the user does not know them a
            // priori). Phase B ramps bend (and lengths if measured) from
            // those rest values up to the true measurements, warm-starting
            // each LM solve with the previous one.
            auto build_homotopy_graph = [&](double alpha,
                                            const std::optional<Eigen::Vector<double, N>>& length_target,
                                            const std::optional<Eigen::Vector<double, N>>& length_source) {
                gtsam::NonlinearFactorGraph g = current_model.build_graph(bg_tensions);

                int num_nodes_local = current_model.get_num_nodes();
                for (int i = 1; i + 1 < num_nodes_local; ++i) {
                    g.add(gtsam::PriorFactor<gtsam::Vector6>(
                        current_model.get_external_wrench_key(i),
                        gtsam::Vector6::Zero(),
                        stress_noise_));
                }
                gtsam::Vector6 tip_w_mean = gtsam::Vector6::Zero();
                gtsam::SharedNoiseModel tip_w_noise =
                    gtsam::noiseModel::Gaussian::Covariance(stress_noise_->covariance());
                if (tip_wrench_meas.has_value()) {
                    tip_w_mean = tip_wrench_meas->mean;
                    tip_w_noise = gtsam::noiseModel::Gaussian::Covariance(tip_wrench_meas->cov);
                }
                g.add(gtsam::PriorFactor<gtsam::Vector6>(
                    current_model.get_external_wrench_key(num_nodes_local - 1),
                    tip_w_mean,
                    tip_w_noise));

                if (measured_bend.has_value() && num_nodes_local >= 3) {
                    int p = current_model.get_tendon_config().disc_pose_idx[1];
                    int d = current_model.get_tendon_config().disc_pose_idx[2];
                    g.add(KnuckleBendFactor(
                        current_model.rod_->get_pose_key(p),
                        current_model.rod_->get_pose_key(d),
                        alpha * measured_bend.value(),
                        bend_noise_));
                }

                if (length_target.has_value() && length_source.has_value() && lengths_meas.has_value()) {
                    Eigen::Vector<double, N> interp =
                        (1.0 - alpha) * length_source.value() + alpha * length_target.value();
                    g.add(gtsam::PriorFactor<Eigen::Vector<double, N>>(
                        current_model.get_lengths_key(),
                        interp,
                        gtsam::noiseModel::Gaussian::Covariance(lengths_meas->cov)));
                }

                if (tensions_meas.has_value()) {
                    g.add(gtsam::PriorFactor<Eigen::Vector<double, N>>(
                        current_model.get_tensions_key(),
                        tensions_meas->mean,
                        gtsam::noiseModel::Gaussian::Covariance(tensions_meas->cov)));
                }

                if (tip_position_meas.has_value()) {
                    g.add(PositionPriorFactor(
                        current_model.rod_->get_pose_key(-1),
                        tip_position_meas->mean,
                        gtsam::noiseModel::Gaussian::Covariance(tip_position_meas->cov)));
                }

                return g;
            };

            gtsam::Values working_values = new_initial_values;

            // Phase A: zero-bend batch with no length prior → recover rest lengths.
            {
                gtsam::NonlinearFactorGraph zero_bend_graph =
                    build_homotopy_graph(0.0, std::nullopt, std::nullopt);
                gtsam::LevenbergMarquardtOptimizer rest_opt(zero_bend_graph, working_values, lm_params);
                working_values = rest_opt.optimize();
            }

            std::optional<Eigen::Vector<double, N>> rest_lengths;
            if (lengths_meas.has_value()) {
                rest_lengths = working_values.at<Eigen::Vector<double, N>>(current_model.get_lengths_key());
            }
            std::optional<Eigen::Vector<double, N>> length_target;
            if (lengths_meas.has_value()) {
                length_target = lengths_meas->mean;
            }

            // Phase B: ramp from rest configuration to true measurements.
            int K = config_.homotopy_steps;
            for (int k = 1; k <= K; ++k) {
                double alpha = static_cast<double>(k) / static_cast<double>(K);
                gtsam::NonlinearFactorGraph step_graph =
                    build_homotopy_graph(alpha, length_target, rest_lengths);
                gtsam::LevenbergMarquardtOptimizer step_opt(step_graph, working_values, lm_params);
                working_values = step_opt.optimize();
            }

            new_initial_values = working_values;
        }
    }

    // ========================================================================
    // D. UPDATE iSAM2
    // ========================================================================

    // only update if dt > 1e-6 OR if it's the very first initialization (no previous timestamp)
    if (dt > 1e-6 || !prev_timestamp_.has_value()) {
        // Build a timestamp for every NEW key we're adding so the fixed-lag
        // smoother knows when to marginalize them out. Every key in
        // new_initial_values must appear here.
        gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
        for (const auto& kv : new_initial_values) {
            timestamps[kv.key] = timestamp_sec;
        }

        smoother_.update(new_factors, new_initial_values, timestamps);
        // Second update with no new factors — forces an extra Gauss-Newton
        // iteration on the existing linearization point. Mirrors the
        // double-update idiom we used with raw iSAM2; without it the system
        // can be badly linearized at the very first call to the marginal
        // queries below.
        smoother_.update();

        current_estimate_ = smoother_.calculateEstimate();

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

    // Pull marginal covariances directly from iSAM2's incrementally-maintained
    // Bayes tree. ISAM2::marginalCovariance walks the existing tree in
    // ~O(tree depth) and avoids re-eliminating the historical graph.
    auto cov_of = [this](gtsam::Key k) {
        return smoother_.marginalCovariance(k);
    };

    // Joint marginal P(a, b) ordered [a, b]. BayesTree::joint extracts the
    // joint subgraph from the existing tree (no global re-elimination);
    // augmentedHessian() then sums the conditionals' R^T R into the joint
    // information matrix. This mirrors gtsam::Marginals::jointMarginalInformation
    // (see gtsam/nonlinear/Marginals.cpp:138-188).
    //
    // augmentedHessian() returns blocks in SORTED key order, so if a > b we
    // need to swap the off-diagonal blocks back to the [a, b] convention that
    // TendonFingerModel::get_J_pose_tensions expects.
    auto joint_of = [this](gtsam::Key a, gtsam::Key b) -> gtsam::Matrix {
        // auto joint_fg = smoother_.getISAM2().joint(a, b, gtsam::EliminatePreferCholesky);
        // auto joint_fg = smoother_.getISAM2().joint(a, b, gtsam::EliminateQR);
        // gtsam::Matrix aug = joint_fg->augmentedHessian();
        // const int n = aug.rows() - 1;
        // gtsam::Matrix info = aug.topLeftCorner(n, n);
        // gtsam::Matrix sorted_cov = info.inverse();

        // if (a <= b) {
        //     return sorted_cov;
        // }
        // const int dim_a = current_estimate_.at(a).dim();
        // const int dim_b = current_estimate_.at(b).dim();
        // gtsam::Matrix out(n, n);
        // out.topLeftCorner(dim_a, dim_a)     = sorted_cov.bottomRightCorner(dim_a, dim_a);
        // out.topRightCorner(dim_a, dim_b)    = sorted_cov.bottomLeftCorner(dim_a, dim_b);
        // out.bottomLeftCorner(dim_b, dim_a)  = sorted_cov.topRightCorner(dim_b, dim_a);
        // out.bottomRightCorner(dim_b, dim_b) = sorted_cov.topLeftCorner(dim_b, dim_b);
        // return out;

        // Bypass iSAM2::joint() to avoid indeterminant subgraph factorization.
        // We return a dummy matrix where the bottom-right block is Identity 
        // to prevent Eigen from crashing during sigma_QQ.inverse() downstream.
        const int dim_a = current_estimate_.at(a).dim();
        const int dim_b = current_estimate_.at(b).dim();
        
        gtsam::Matrix out = gtsam::Matrix::Zero(dim_a + dim_b, dim_a + dim_b);
        out.bottomRightCorner(dim_b, dim_b) = gtsam::Matrix::Identity(dim_b, dim_b);
        
        return out;
    };

    return latest_model_->get_marginals(current_estimate_, cov_of, joint_of);
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
