#include "TendonFingerTrajectoryPlanner.h"

#include "utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>

using namespace gtsam;
using crest_sparse::SdfCollisionFactor;
using crest_sparse::SdfContactFactor;


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

    if (!config_.environment) return;

    const auto& env = *config_.environment;
    Pose3 obj_mean(env.object_pose_mean);

    if (env.object_pose_per_step) {
        for (int k = 0; k <= config_.K; ++k) {
            values_.insert(object_key(k), obj_mean);
        }
    } else {
        values_.insert(object_key(0), obj_mean);
    }

    if (env.target_contact_node.has_value()) {
        // Seed p_c on the object's surface, on the side facing the tip, so the
        // contact factor's SDF row has a non-zero gradient at iteration 0.
        // We query the SDF at the object's local-frame origin to get the
        // distance from that origin to the nearest surface (negative inside);
        // |sdf| is the step length from the object center toward the tip that
        // lands on the surface for convex shapes whose local origin sits
        // inside.
        int i_node = *env.target_contact_node;
        const Pose3 tip_pose = values_.at<Pose3>(
            models_[config_.K]->rod_->get_pose_key(i_node));
        Point3 tip   = tip_pose.translation();
        Point3 obj_c = obj_mean.translation();
        Point3 diff  = tip - obj_c;
        double norm  = diff.norm();
        Point3 dir   = (norm > 1e-8) ? Point3(diff / norm)
                                     : Point3(0.0, 0.0, 1.0);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>
            sampler(*env.sdf_grid);
        double r_obj = std::abs(sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0)));

        Point3 seed = obj_c + r_obj * dir;
        values_.insert(dummy_point_key(), seed);
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

    Eigen::Matrix<double, N, N> Qc = config_.gp_tense_Qc.topLeftCorner<N, N>();

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
        // 5.1 GP on tensions (commented out - using GP on lengths instead)
        if (k < K) {
            auto gp_noise = noiseModel::Gaussian::Covariance(Qc * config_.dt);
            graph_.add(BetweenFactor<Eigen::Vector<double, N>>(
                models_[k]->get_tensions_key(),
                models_[k + 1]->get_tensions_key(),
                Eigen::Vector<double, N>::Zero(),
                gp_noise));
        }

        // 5.2 GP on tendon lengths (identity state transition)
        if (k < K && config_.gp_len_Qc.size() > 0) {
            Eigen::Matrix<double, N, N> Qc_len = config_.gp_len_Qc.topLeftCorner<N, N>();
            auto gp_len_noise = noiseModel::Gaussian::Covariance(Qc_len * config_.dt);
            graph_.add(BetweenFactor<Eigen::Vector<double, N>>(
                models_[k]->get_lengths_key(),
                models_[k + 1]->get_lengths_key(),
                Eigen::Vector<double, N>::Zero(),
                gp_len_noise));
        }

        // 5.3 GP on poses: apply between factor for every node i: T_{i,k} -> T_{i,k+1}
        // if (k < K) {
        //     auto gp_pose_noise = noiseModel::Gaussian::Covariance(config_.gp_pose_Qc * config_.dt);
        //     const auto& pose_keys_k  = models_[k]->rod_->get_pose_keys();
        //     const auto& pose_keys_k1 = models_[k + 1]->rod_->get_pose_keys();
        //     for (int i = 0; i < static_cast<int>(pose_keys_k.size()); ++i) {
        //         graph_.add(BetweenFactor<Pose3>(
        //             pose_keys_k[i],
        //             pose_keys_k1[i],
        //             Pose3::Identity(),
        //             gp_pose_noise));
        //     }
        // }

        // 5.4 SDF collision running cost (Eq 28/29). Cubic barrier on each
        // configured node sphere against the object SDF.
        if (config_.environment && config_.environment->sdf_grid &&
            !config_.environment->collision_node_indices.empty()) {
            const auto& env = *config_.environment;
            auto col_noise = noiseModel::Isotropic::Sigma(1, env.collision_sigma);
            for (size_t j = 0; j < env.collision_node_indices.size(); ++j) {
                int i_node = env.collision_node_indices[j];
                double r   = env.collision_node_radii[j];
                graph_.add(SdfCollisionFactor(
                    models_[k]->rod_->get_pose_key(i_node),
                    object_key(k),
                    r, env.collision_epsilon, env.sdf_grid, col_noise));
            }
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
    // In contact-as-goal mode (Eq 30) the terminal contact factor replaces
    // the goal pose/position priors.
    const bool contact_mode = config_.environment &&
                              config_.environment->target_contact_node.has_value();

    if (contact_mode &&
        (config_.goal_pose.has_value() || config_.goal_position.has_value())) {
        std::cerr << "[TendonFingerTrajectoryPlanner] target_contact_node is set; "
                  << "suppressing goal_pose/goal_position priors (Eq 30)." << std::endl;
    }

    if (!contact_mode && config_.goal_pose.has_value()) {
        graph_.add(PriorFactor<Pose3>(
            models_[K]->rod_->get_pose_key(-1),
            Pose3(*config_.goal_pose),
            noiseModel::Gaussian::Covariance(config_.goal_pose_cov)));
    }
    if (!contact_mode && config_.goal_position.has_value()) {
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

    // 8. Environment factors: object pose prior(s), optional per-step GP,
    //    and the terminal surface-contact factor (Eq 26).
    if (config_.environment) {
        const auto& env = *config_.environment;
        auto obj_prior = noiseModel::Gaussian::Covariance(env.object_pose_cov);
        Pose3 obj_mean(env.object_pose_mean);

        if (env.object_pose_per_step) {
            for (int k = 0; k <= K; ++k) {
                graph_.add(PriorFactor<Pose3>(object_key(k), obj_mean, obj_prior));
            }
            if (config_.gp_pose_Qc.size() > 0) {
                auto gp = noiseModel::Gaussian::Covariance(config_.gp_pose_Qc * config_.dt);
                for (int k = 0; k < K; ++k) {
                    graph_.add(BetweenFactor<Pose3>(
                        object_key(k), object_key(k + 1), Pose3::Identity(), gp));
                }
            }
        } else {
            graph_.add(PriorFactor<Pose3>(object_key(0), obj_mean, obj_prior));
        }

        if (contact_mode) {
            int i_node = *env.target_contact_node;
            auto cn = noiseModel::Gaussian::Covariance(env.contact_cov);
            graph_.add(SdfContactFactor(
                models_[K]->rod_->get_pose_key(i_node),
                object_key(K),
                dummy_point_key(),
                env.contact_node_radius,
                env.sdf_grid, cn));

            // Tikhonov regularizer on the dummy contact point. SdfContactFactor
            // only constrains p_c along two of three DoF (sphere surface +
            // object surface), leaving a 1D sliding manifold that yields an
            // indeterminate linear system. A weak prior toward the seed picks
            // a unique minimum without overpowering the tight contact factor.
            Point3 p_seed = values_.at<Point3>(dummy_point_key());
            auto weak_prior_noise = noiseModel::Isotropic::Sigma(3, 10.0);
            graph_.add(PriorFactor<Point3>(
                dummy_point_key(), p_seed, weak_prior_noise));
        }
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
