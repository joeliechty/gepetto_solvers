#include "TendonFingerTrajectoryPlanner.h"

#include "utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using namespace gtsam;


// --- Helper: create a TendonFingerModel<N> from config ---

template<int N>
static std::unique_ptr<TendonFingerModel<N>> make_model(
    const TendonFingerSolverConfig& config,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise)
{
    if (config.per_disc_tendon_input.is_populated()) {
        if (config.K_inv_per_segment.empty()) {
            return std::make_unique<TendonFingerModel<N>>(
                config.rod_length, config.num_discs, config.num_between_nodes,
                config.per_disc_tendon_input, config.K_inv,
                twist_noise, stress_noise, base_pose_mean, base_pose_noise,
                config.disc_positions_normalized);
        } else {
            return std::make_unique<TendonFingerModel<N>>(
                config.rod_length, config.num_discs, config.num_between_nodes,
                config.per_disc_tendon_input, config.K_inv_per_segment,
                twist_noise, stress_noise, base_pose_mean, base_pose_noise,
                config.disc_positions_normalized);
        }
    } else {
        if (config.K_inv_per_segment.empty()) {
            return std::make_unique<TendonFingerModel<N>>(
                config.rod_length, config.num_discs, config.num_between_nodes,
                config.tendon_input, config.K_inv,
                twist_noise, stress_noise, base_pose_mean, base_pose_noise,
                config.disc_positions_normalized);
        } else {
            return std::make_unique<TendonFingerModel<N>>(
                config.rod_length, config.num_discs, config.num_between_nodes,
                config.tendon_input, config.K_inv_per_segment,
                twist_noise, stress_noise, base_pose_mean, base_pose_noise,
                config.disc_positions_normalized);
        }
    }
}


// --- TendonFingerTrajectoryPlanner<N> ---

template<int N>
TendonFingerTrajectoryPlanner<N>::TendonFingerTrajectoryPlanner(
    const TrajectoryPlannerConfig& config)
:
    SolverBase(config.model_config.base),
    config_(config)
{
    const auto& mc = config.model_config;

    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        mc.sigma_twist_rot, mc.sigma_twist_pos);
    SharedDiagonal stress_noise = get_noise_model_rot_pos(
        mc.sigma_stress_moment, mc.sigma_stress_force);

    Pose3 base_pose_mean;
    if (mc.base_pose.isZero()) {
        Rot3 base_rot = Rot3::Rx(-M_PI / 2).compose(Rot3::Rz(M_PI));
        base_pose_mean = Pose3(base_rot, Point3::Zero());
    } else {
        base_pose_mean = Pose3(mc.base_pose);
    }
    SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
        mc.sigma_base_rot, mc.sigma_base_pos);

    ext_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_ext_wrench_moment, config.sigma_ext_wrench_force);

    // Create K+1 models (one per time step)
    models_.reserve(config.K + 1);
    for (int k = 0; k <= config.K; ++k) {
        models_.push_back(make_model<N>(
            mc, twist_noise, stress_noise, base_pose_mean, base_pose_noise));
    }

    get_initial_values();
}


template<int N>
void TendonFingerTrajectoryPlanner<N>::get_initial_values() {
    values_.clear();

    Eigen::Vector<double, N> t_init;
    if (config_.start_tensions.has_value()) {
        t_init = config_.start_tensions->head<N>();
    } else {
        t_init = config_.background_tensions_mean.head<N>();
    }

    for (int k = 0; k <= config_.K; ++k) {
        values_.insert(models_[k]->get_initial_values());
        // Override tension initial values
        values_.update(models_[k]->get_tensions_key(), t_init);
    }
}


template<int N>
void TendonFingerTrajectoryPlanner<N>::build_graph() {
    graph_.resize(0);

    int K = config_.K;

    // Convert config vectors to fixed-size types
    Eigen::Vector<double, N> bg_mean = config_.background_tensions_mean.head<N>();
    Eigen::Vector<double, N> bg_sigmas = config_.background_tensions_sigmas.head<N>();
    Eigen::Matrix<double, N, N> bg_cov =
        bg_sigmas.array().square().matrix().asDiagonal();

    Eigen::Matrix<double, N, N> Qc = config_.gp_Qc.topLeftCorner<N, N>();

    // build the graph for each time step
    for (int k = 0; k <= K; ++k) {
        // 1. Kinematic factors (rod mechanics + base pose + tendon disc wrenches)
        graph_.add(models_[k]->build_graph_kinematic());

        // 2. Zero external wrench priors on all non-base nodes
        int num_nodes = models_[k]->get_num_nodes();
        for (int i = 1; i < num_nodes; ++i) {
            graph_.add(PriorFactor<Vector6>(
                models_[k]->get_external_wrench_key(i),
                Vector6::Zero(),
                ext_wrench_noise_));
        }

        // 3. Background tension prior (p_bg) — skipped at k=0 if start_tensions is set,
        //    and at k=K if goal_tensions is set (those priors replace it)
        bool skip_bg = (k == 0 && config_.start_tensions.has_value()) ||
                       (k == K && config_.goal_tensions.has_value());
        if (!skip_bg) {
            graph_.add(PriorFactor<Eigen::Vector<double, N>>(
                models_[k]->get_tensions_key(),
                bg_mean,
                noiseModel::Gaussian::Covariance(bg_cov)));
        }

        // 4. Tension limit barrier (p_lim)
        if (!config_.active_tendon_indices.empty()) {
            int num_active = config_.active_tendon_indices.size();
            auto limit_noise = noiseModel::Isotropic::Sigma(num_active, 1.0);
            graph_.add(TensionLimitFactor<N>(
                models_[k]->get_tensions_key(),
                config_.tension_limit_alpha,
                config_.tension_limit_q_min,
                config_.active_tendon_indices,
                limit_noise));
        }

        // 5. GP temporal prior between consecutive time steps
        if (k < K) {
            auto gp_noise = noiseModel::Gaussian::Covariance(Qc * config_.dt);
            graph_.add(BetweenFactor<Eigen::Vector<double, N>>(
                models_[k]->get_tensions_key(),
                models_[k + 1]->get_tensions_key(),
                Eigen::Vector<double, N>::Zero(),
                gp_noise));
        }
    }

    // 6. Start boundary conditions at k=0
    if (config_.start_tensions.has_value()) {
        Eigen::Matrix<double, N, N> st_cov =
            config_.start_tensions_cov.topLeftCorner<N, N>();
        graph_.add(PriorFactor<Eigen::Vector<double, N>>(
            models_[0]->get_tensions_key(),
            config_.start_tensions->head<N>(),
            noiseModel::Gaussian::Covariance(st_cov)));
    }
    if (config_.start_pose.has_value()) {
        graph_.add(PriorFactor<Pose3>(
            models_[0]->rod_->get_pose_key(-1),
            Pose3(*config_.start_pose),
            noiseModel::Gaussian::Covariance(config_.start_pose_cov)));
    }
    if (config_.start_position.has_value()) {
        graph_.add(PositionPriorFactor(
            models_[0]->rod_->get_pose_key(-1),
            *config_.start_position,
            noiseModel::Gaussian::Covariance(config_.start_position_cov)));
    }

    // 7. Goal boundary conditions at k=K
    if (config_.goal_pose.has_value()) {
        graph_.add(PriorFactor<Pose3>(
            models_[K]->rod_->get_pose_key(-1),
            Pose3(*config_.goal_pose),
            noiseModel::Gaussian::Covariance(config_.goal_pose_cov)));
    }
    if (config_.goal_position.has_value()) {
        graph_.add(PositionPriorFactor(
            models_[K]->rod_->get_pose_key(-1),
            *config_.goal_position,
            noiseModel::Gaussian::Covariance(config_.goal_position_cov)));
    }
    if (config_.goal_tensions.has_value()) {
        Eigen::Matrix<double, N, N> gt_cov =
            config_.goal_tensions_cov.topLeftCorner<N, N>();
        graph_.add(PriorFactor<Eigen::Vector<double, N>>(
            models_[K]->get_tensions_key(),
            config_.goal_tensions->head<N>(),
            noiseModel::Gaussian::Covariance(gt_cov)));
    }
}


template<int N>
void TendonFingerTrajectoryPlanner<N>::extract_solution() {
    result_.trajectory.clear();
    result_.trajectory.reserve(config_.K + 1);
    for (int k = 0; k <= config_.K; ++k) {
        result_.trajectory.push_back(
            models_[k]->get_marginals(values_, marginals_));
    }
}


template<int N>
TrajectoryPlannerResult TendonFingerTrajectoryPlanner<N>::plan() {
    result_ = TrajectoryPlannerResult{};
    result_.meta = optimize();
    return result_;
}


// Explicit instantiations
template class TendonFingerTrajectoryPlanner<1>;
template class TendonFingerTrajectoryPlanner<2>;
template class TendonFingerTrajectoryPlanner<3>;
template class TendonFingerTrajectoryPlanner<4>;
template class TendonFingerTrajectoryPlanner<5>;
template class TendonFingerTrajectoryPlanner<6>;
template class TendonFingerTrajectoryPlanner<7>;
template class TendonFingerTrajectoryPlanner<8>;
template class TendonFingerTrajectoryPlanner<9>;
template class TendonFingerTrajectoryPlanner<10>;


// --- TendonFingerTrajectoryPlannerDispatch ---

TendonFingerTrajectoryPlannerDispatch::TendonFingerTrajectoryPlannerDispatch(
    const TrajectoryPlannerConfig& config)
    : num_tendons_(config.model_config.num_tendons)
{
    switch (config.model_config.num_tendons) {
        case 1:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<1>>(config); break;
        case 2:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<2>>(config); break;
        case 3:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<3>>(config); break;
        case 4:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<4>>(config); break;
        case 5:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<5>>(config); break;
        case 6:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<6>>(config); break;
        case 7:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<7>>(config); break;
        case 8:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<8>>(config); break;
        case 9:  planner_ = std::make_unique<TendonFingerTrajectoryPlanner<9>>(config); break;
        case 10: planner_ = std::make_unique<TendonFingerTrajectoryPlanner<10>>(config); break;
        default: throw std::invalid_argument(
            "num_tendons must be between 1 and 10, got " + std::to_string(config.model_config.num_tendons));
    }
}


TrajectoryPlannerResult TendonFingerTrajectoryPlannerDispatch::plan() {
    return std::visit([](auto& p) { return p->plan(); }, planner_);
}
