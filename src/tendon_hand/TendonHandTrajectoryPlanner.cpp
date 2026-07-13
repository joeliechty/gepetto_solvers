#include "TendonHandTrajectoryPlanner.h"

#include "utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>

using namespace gtsam;


TendonHandTrajectoryPlanner::TendonHandTrajectoryPlanner(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    const TendonHandTrajectoryPlannerConfig& config)
:
    SolverBase(config.base),
    config_(config)
{
    const int K = config_.K;
    const Pose3 wrist_pose(config_.wrist_pose);

    // Loose hand-pose prior (Eq 1.40) applied only at the start step.
    SharedDiagonal start_wrist_noise = get_noise_model_rot_pos(
        config_.sigma_wrist_rot, config_.sigma_wrist_pos);

    // Contact lives only at the terminal step, so strip the per-finger contact
    // configs for every other step. Everything else about each finger is shared.
    auto free_space_configs = finger_configs;
    for (auto& [name, c] : free_space_configs) {
        c.sdf_contact.reset();
        c.sphere_contact.reset();
    }

    models_.reserve(K + 1);
    for (int k = 0; k <= K; ++k) {
        const bool is_start = (k == 0);
        const bool is_goal  = (k == K);
        const auto& step_configs = is_goal ? finger_configs : free_space_configs;
        models_.push_back(std::make_unique<TendonHandModel>(
            step_configs, wrist_pose, start_wrist_noise,
            /*step=*/k, /*emit_wrist_prior=*/is_start));
    }

    // A configured contact at k=K is a hard equality constraint => AL path.
    use_augmented_lagrangian_ = models_[K]->has_contact();

    get_initial_values();
}


void TendonHandTrajectoryPlanner::get_initial_values() {
    values_.clear();
    // Each model owns a distinct step-indexed wrist variable and (only at k=K)
    // the shared object + witness seeds, so the per-step Values are disjoint.
    for (const auto& model : models_)
        values_.insert(model->get_initial_values());
}


void TendonHandTrajectoryPlanner::build_graph() {
    graph_.resize(0);

    const int K = config_.K;

    for (int k = 0; k <= K; ++k) {
        // Per-step kinematics + wrench priors + tension priors, plus the wrist
        // prior at k=0 and the contact constraints at k=K (all handled inside
        // TendonHandModel::build_graph based on how each model was constructed).
        // At k=0 use the measured start-tension prior when provided (it replaces
        // the background tension prior, pinning the hand to its known opening);
        // every other step uses the background/target tensions.
        const auto& step_tensions =
            (k == 0 && !start_tensions_.empty()) ? start_tensions_ : tensions_;
        graph_.add(models_[k]->build_graph(step_tensions, tip_wrenches_));

        if (k < K) {
            // Wrist-pose GP (Eq 1.41/1.42): identity transition, zero twist mean.
            auto wrist_gp_noise = noiseModel::Gaussian::Covariance(
                config_.gp_wrist_Qc * config_.dt);
            graph_.add(BetweenFactor<Pose3>(
                models_[k]->wrist_key_instance(),
                models_[k + 1]->wrist_key_instance(),
                Pose3::Identity(),
                wrist_gp_noise));

            // Per-finger tension GP (Eq 1.11) and optional length GP (Eq 1.13).
            models_[k]->add_temporal_gp(
                graph_, *models_[k + 1],
                config_.gp_tense_Qc, config_.gp_len_Qc, config_.dt);
        }
    }
}


void TendonHandTrajectoryPlanner::extract_solution() {
    result_.trajectory.clear();
    result_.trajectory.reserve(models_.size());
    for (const auto& model : models_)
        result_.trajectory.push_back(model->get_marginals(values_, marginals_));
}


TendonHandTrajectoryResult TendonHandTrajectoryPlanner::plan(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches,
    const std::vector<VectorXGaussian>& start_tensions)
{
    tensions_ = tensions;
    tip_wrenches_ = tip_wrenches;
    start_tensions_ = start_tensions;

    result_ = TendonHandTrajectoryResult{};
    result_.meta = optimize();
    return result_;
}
