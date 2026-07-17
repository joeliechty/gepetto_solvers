#include "TendonHandTrajectoryPlanner.h"

#include "measurement/PositionPriorFactor.h"
#include "utils/MiscInline.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>

#include <stdexcept>

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

    // Contact-as-goal lives only at the terminal step, so strip the terminal
    // contact from every other step. Collision avoidance (Section 1.5), however,
    // applies at every *plannable* step, so preserve the sdf env's collision
    // fields (sdf_grid, object_pose_*, collision_*) and clear only the contact
    // aspect (target_contact_node / witness_point_seed). A contact-only sdf env
    // becomes inert after stripping (no grid factors) — handled by the
    // build_graph / get_initial_values guards in TendonHandModel.
    auto free_space_configs = finger_configs;
    for (auto& [name, c] : free_space_configs) {
        if (c.sdf_contact.has_value()) {
            c.sdf_contact->target_contact_node.reset();
            c.sdf_contact->witness_point_seed.reset();
        }
        c.sphere_contact.reset();
    }

    // The start step k=0 additionally gets NO collision constraints: it is a
    // measurement pinned by tight start priors (start tensions, wrist prior).
    // If the measured start is already in collision, a k=0 constraint is
    // infeasible by construction — the AL outer loop grinds mu up against the
    // measurement priors and distorts the whole solve instead of planning a
    // way out from where the hand actually is.
    auto start_configs = free_space_configs;
    for (auto& [name, c] : start_configs) {
        if (c.sdf_contact.has_value())
            c.sdf_contact->collision_avoidance = false;
    }

    models_.reserve(K + 1);
    for (int k = 0; k <= K; ++k) {
        const bool is_start = (k == 0);
        const bool is_goal  = (k == K);
        const auto& step_configs = is_goal ? finger_configs
                                 : is_start ? start_configs
                                 : free_space_configs;
        models_.push_back(std::make_unique<TendonHandModel>(
            step_configs, wrist_pose, start_wrist_noise,
            /*step=*/k, /*emit_wrist_prior=*/is_start));
    }

    // A configured contact at k=K (hard equality) or collision avoidance at any
    // step (hard inequality, Section 1.5) routes the solve through the AL path.
    use_augmented_lagrangian_ = models_[K]->has_contact() || models_[K]->has_collision();

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

        // Terminal per-finger tip-position goals (point-to-point). Soft priors on
        // each finger's tip node; only added when goal_positions is non-empty, so
        // legacy contact-as-goal / no-goal runs are unaffected.
        if (k == K && !config_.goal_positions.empty()) {
            const int num_fingers = models_[K]->num_fingers();
            if (static_cast<int>(config_.goal_positions.size()) != num_fingers) {
                throw std::runtime_error(
                    "TendonHandTrajectoryPlannerConfig::goal_positions size must "
                    "equal the number of fingers when non-empty.");
            }
            auto goal_noise =
                noiseModel::Gaussian::Covariance(config_.goal_position_cov);
            for (int i = 0; i < num_fingers; ++i) {
                graph_.add(PositionPriorFactor(
                    models_[K]->finger_tip_pose_key(i),
                    config_.goal_positions[i],
                    goal_noise));
            }
        }

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


TendonHandTrajectoryResult
TendonHandTrajectoryPlanner::extract_trajectory_means_only(const Values& values) const {
    TendonHandTrajectoryResult r;
    r.trajectory.reserve(models_.size());
    for (const auto& model : models_)
        r.trajectory.push_back(model->get_marginals_means_only(values));
    return r;
}


std::vector<TendonHandTrajectoryResult>
TendonHandTrajectoryPlanner::get_intermediate_solutions() const {
    std::vector<TendonHandTrajectoryResult> out;
    out.reserve(intermediate_values_.size());
    for (const auto& vals : intermediate_values_)
        out.push_back(extract_trajectory_means_only(vals));
    return out;
}


TendonHandTrajectoryResult
TendonHandTrajectoryPlanner::get_initial_solution() const {
    return extract_trajectory_means_only(initial_values_);
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
