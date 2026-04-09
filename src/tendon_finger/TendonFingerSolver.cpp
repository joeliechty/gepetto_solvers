#include "TendonFingerSolver.h"

#include "measurement/PositionPriorFactor.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"
#include "utils/SolverBase.h"
#include <gtsam/linear/NoiseModel.h>

using namespace gtsam;


// --- Templated TendonFingerSolver<N> ---

template<int N>
TendonFingerSolver<N>::TendonFingerSolver(const TendonFingerSolverConfig& config)
:
    SolverBase(config.base)
{
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos);

    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force);

    Pose3 base_pose_mean;
    if (config.base_pose.isZero()) {
        Rot3 base_rot = Rot3::Rx(-M_PI / 2).compose(Rot3::Rz(M_PI));
        base_pose_mean = Pose3(base_rot, Point3::Zero());
    } else {
        base_pose_mean = Pose3(config.base_pose);
    }
    SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
        config.sigma_base_rot, config.sigma_base_pos);

    if (config.per_disc_tendon_input.is_populated()) {
        // Per-disc routing path
        if (config.K_inv_per_segment.empty()) {
            robot_ = std::make_unique<TendonFingerModel<N>>(
                config.rod_length,
                config.num_discs,
                config.num_between_nodes,
                config.per_disc_tendon_input,
                config.K_inv,
                twist_noise,
                small_wrench_noise_,
                base_pose_mean,
                base_pose_noise,
                config.disc_positions_normalized);
        } else {
            robot_ = std::make_unique<TendonFingerModel<N>>(
                config.rod_length,
                config.num_discs,
                config.num_between_nodes,
                config.per_disc_tendon_input,
                config.K_inv_per_segment,
                twist_noise,
                small_wrench_noise_,
                base_pose_mean,
                base_pose_noise,
                config.disc_positions_normalized);
        }
    } else {
        // Simple TendonInput path (backward-compatible)
        if (config.K_inv_per_segment.empty()) {
            robot_ = std::make_unique<TendonFingerModel<N>>(
                config.rod_length,
                config.num_discs,
                config.num_between_nodes,
                config.tendon_input,
                config.K_inv,
                twist_noise,
                small_wrench_noise_,
                base_pose_mean,
                base_pose_noise,
                config.disc_positions_normalized);
        } else {
            robot_ = std::make_unique<TendonFingerModel<N>>(
                config.rod_length,
                config.num_discs,
                config.num_between_nodes,
                config.tendon_input,
                config.K_inv_per_segment,
                twist_noise,
                small_wrench_noise_,
                base_pose_mean,
                base_pose_noise,
                config.disc_positions_normalized);
        }
    }

    get_initial_values();
}


template<int N>
Solution<TendonFingerMarginals> TendonFingerSolver<N>::solve(
    const VectorNGaussian<N>& tensions,
    const std::optional<Vector6Gaussian>& tip_wrench,
    const std::optional<Vector3Gaussian>& tip_position_meas)
{
    tensions_ = tensions;
    tip_wrench_ = tip_wrench;
    tip_position_meas_ = tip_position_meas;

    Solution<TendonFingerMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}


template<int N>
void TendonFingerSolver<N>::build_graph() {
    // Build base robot graph
    graph_ = robot_->build_graph(tensions_);

    // Constrain all external load wrenches (except base and tip)
    int num_nodes = robot_->get_num_nodes();

    for (int i = 1; i + 1 < num_nodes; ++i) {
        graph_.add(PriorFactor<Vector6>(
            robot_->get_external_wrench_key(i),
            Vector6::Zero(),
            small_wrench_noise_));
    }

    // If we have tip force input, use it, otherwise default to tight zero
    Vector6 tip_wrench_mean = Vector6::Zero();
    auto tip_wrench_noise = noiseModel::Gaussian::Covariance(small_wrench_noise_->covariance());
    if (tip_wrench_) {
        tip_wrench_mean = tip_wrench_->mean;
        tip_wrench_noise = noiseModel::Gaussian::Covariance(tip_wrench_->cov);
    }

    graph_.add(PriorFactor<Vector6>(
        robot_->get_external_wrench_key(num_nodes - 1),
        tip_wrench_mean,
        tip_wrench_noise));

    // If we have a tip pose measurement, then use it
    if (tip_position_meas_) {
        graph_.add(PositionPriorFactor(
            robot_->rod_->get_pose_key(-1),
            tip_position_meas_->mean,
            noiseModel::Gaussian::Covariance(tip_position_meas_->cov)));
    }
}


template<int N>
void TendonFingerSolver<N>::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
}

template<int N>
void TendonFingerSolver<N>::get_initial_values() {
    values_ = robot_->get_initial_values();
}


// Explicit instantiations
template class TendonFingerSolver<1>;
template class TendonFingerSolver<2>;
template class TendonFingerSolver<3>;
template class TendonFingerSolver<4>;
template class TendonFingerSolver<5>;
template class TendonFingerSolver<6>;
template class TendonFingerSolver<7>;
template class TendonFingerSolver<8>;
template class TendonFingerSolver<9>;
template class TendonFingerSolver<10>;


// --- TendonFingerSolverDispatch (runtime dispatch wrapper) ---

TendonFingerSolverDispatch::TendonFingerSolverDispatch(const TendonFingerSolverConfig& config)
    : num_tendons_(config.num_tendons)
{
    switch (config.num_tendons) {
        case 1:  solver_ = std::make_unique<TendonFingerSolver<1>>(config); break;
        case 2:  solver_ = std::make_unique<TendonFingerSolver<2>>(config); break;
        case 3:  solver_ = std::make_unique<TendonFingerSolver<3>>(config); break;
        case 4:  solver_ = std::make_unique<TendonFingerSolver<4>>(config); break;
        case 5:  solver_ = std::make_unique<TendonFingerSolver<5>>(config); break;
        case 6:  solver_ = std::make_unique<TendonFingerSolver<6>>(config); break;
        case 7:  solver_ = std::make_unique<TendonFingerSolver<7>>(config); break;
        case 8:  solver_ = std::make_unique<TendonFingerSolver<8>>(config); break;
        case 9:  solver_ = std::make_unique<TendonFingerSolver<9>>(config); break;
        case 10: solver_ = std::make_unique<TendonFingerSolver<10>>(config); break;
        default: throw std::invalid_argument(
            "num_tendons must be between 1 and 10, got " + std::to_string(config.num_tendons));
    }
}

Solution<TendonFingerMarginals> TendonFingerSolverDispatch::solve(
    const VectorXGaussian& tensions,
    const std::optional<Vector6Gaussian>& tip_wrench,
    const std::optional<Vector3Gaussian>& tip_position_meas)
{
    if (tensions.mean.size() != num_tendons_)
        throw std::invalid_argument(
            "tensions size (" + std::to_string(tensions.mean.size()) +
            ") does not match num_tendons (" + std::to_string(num_tendons_) + ")");

    return std::visit([&](auto& solver_ptr) -> Solution<TendonFingerMarginals> {
        using SolverType = typename std::remove_reference_t<decltype(*solver_ptr)>;
        constexpr int M = SolverType::NumTendons;

        VectorNGaussian<M> t_fixed;
        t_fixed.mean = tensions.mean;
        t_fixed.cov = tensions.cov;

        return solver_ptr->solve(t_fixed, tip_wrench, tip_position_meas);
    }, solver_);
}
