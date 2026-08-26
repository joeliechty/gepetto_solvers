#include "TendonFingerSolver.h"

#include "measurement/PositionPriorFactor.h"
#include "utils/EnvironmentFactors.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"
#include "utils/SolverBase.h"
#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/PriorFactor.h>

#include <cmath>

using namespace gtsam;


// --- Templated TendonFingerSolver<N> ---

template<int N>
TendonFingerSolver<N>::TendonFingerSolver(const TendonFingerSolverConfig& config)
:
    SolverBase(config.base)
{
    sphere_contact_ = config.sphere_contact;
    sdf_contact_    = config.sdf_contact;

    // A configured contact (sphere-sphere or SDF surface) is a hard equality
    // constraint, so route this solve through SolverBase's Augmented Lagrangian
    // path. Without contact the solver stays on the legacy free-space
    // Dogleg/LM path.
    use_augmented_lagrangian_ =
        sphere_contact_.has_value() || sdf_contact_.has_value();

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

    // Hand-base reparameterization (Section 4). Must precede get_initial_values()
    // so node 0 is seeded via the hand-base variable instead of pose_keys_[0].
    if (config.use_hand_base) {
        robot_->set_hand_base(Pose3(config.hand_base_offset));
    }

    // Planar-bending approximation. Also before get_initial_values(), for the same
    // reason: it changes what build_graph() emits.
    if (config.planar_bending) {
        robot_->set_planar_bending(config.sigma_planar_bend, config.sigma_planar_twist);
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

    // Optional sphere-sphere tip contact: anchor a sphere primitive in the
    // world with a tight PriorFactor<Pose3>, then connect a rod node to it via
    // the 1-residual SphereSphereContactFactor (signed surface gap). The factor
    // is wrapped in a gtsam::ZeroCostConstraint so the AL optimizer enforces the
    // gap == 0 as a hard equality constraint; the factor's unit noise model is
    // only the source of the per-row constraint scaling.
    if (sphere_contact_) {
        const auto& sc = *sphere_contact_;
        Pose3 sphere_pose(Rot3(), sc.sphere_center);
        graph_.add(PriorFactor<Pose3>(
            sphere_object_key(), sphere_pose,
            noiseModel::Gaussian::Covariance(sc.sphere_pose_cov)));
        Key finger_key = robot_->rod_->get_pose_key(sc.finger_node_index);
        if (sc.witness) {
            // 5-residual witness-point form ([c_R, c_O, c_N, c_T1, c_T2]) --
            // the analytic counterpart of the SDF contact below. An explicit
            // dummy point p_c is driven onto both sphere surfaces with
            // antiparallel normals; the two C-frame tangent residuals pin its
            // remaining gauge DOF, so no stabilizing prior on p_c is needed
            // (the factor's gauge-fixing residuals make the system full rank).
            auto contact = std::make_shared<crest_sparse::SphereWitnessContactFactor>(
                finger_key,
                sphere_object_key(),
                dummy_point_key(),
                sc.finger_node_radius, sc.sphere_radius,
                noiseModel::Isotropic::Sigma(5, 1.0));
            graph_.add(gtsam::ZeroCostConstraint(contact));
        } else {
            auto contact = std::make_shared<crest_sparse::SphereSphereContactFactor>(
                finger_key,
                sphere_object_key(),
                sc.finger_node_radius, sc.sphere_radius,
                noiseModel::Isotropic::Sigma(1, 1.0));
            graph_.add(gtsam::ZeroCostConstraint(contact));
        }
    }

    // Optional SDF surface contact: anchor the object pose in the world with a
    // tight PriorFactor<Pose3>, then connect a rod node to it via the 5-residual
    // witness-point SdfWitnessContactFactor (Section 3, [c_R, c_O, c_N, c_T1, c_T2]).
    // The factor is wrapped in a gtsam::ZeroCostConstraint so the AL optimizer
    // drives all five residuals exactly to zero; the unit noise model is only the
    // per-row constraint scaling. Mirrors TendonFingerTrajectoryPlanner's contact mode.
    else if (sdf_contact_) {
        const auto& env = *sdf_contact_;
        Key tip_key = robot_->rod_->get_pose_key(*env.target_contact_node);
        graph_.add(PriorFactor<Pose3>(
            sphere_object_key(), Pose3(env.object_pose_mean),
            noiseModel::Gaussian::Covariance(env.object_pose_cov)));
        gtsam::NoiseModelFactor::shared_ptr contact;
        if (env.ellipsoid_semi_axes.norm() > 0.0) {
            // Analytic hyper-ellipsoid surface (Section 1.6.3).
            contact = std::make_shared<crest_sparse::EllipsoidWitnessContactFactor>(
                tip_key,
                sphere_object_key(),
                dummy_point_key(),
                env.contact_node_radius,
                env.ellipsoid_semi_axes,
                noiseModel::Isotropic::Sigma(5, 1.0));
        } else {
            contact = std::make_shared<crest_sparse::SdfWitnessContactFactor>(
                tip_key,
                sphere_object_key(),
                dummy_point_key(),
                env.contact_node_radius,
                env.sdf_grid,
                noiseModel::Isotropic::Sigma(5, 1.0));
        }
        graph_.add(gtsam::ZeroCostConstraint(contact));
        // The factor's C-frame tangent residuals ([c_T1, c_T2]) pin p_c's gauge
        // DOF, so the system is full rank without a stabilizing prior on the
        // dummy point. The ray-march seed in get_initial_values() supplies its
        // initial Value.
    }
}


template<int N>
void TendonFingerSolver<N>::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
}

template<int N>
void TendonFingerSolver<N>::get_initial_values() {
    values_ = robot_->get_initial_values();

    if (sphere_contact_) {
        const auto& sc = *sphere_contact_;
        values_.insert(sphere_object_key(),
            Pose3(Rot3(), sc.sphere_center));

        if (sc.witness) {
            // Seed the witness point on sphere B's surface, along the line from
            // the sphere center toward the finger node (the contact point lies
            // on this line at tangency). Mirrors the SDF path's surface seed.
            Point3 finger_pos = values_
                .at<Pose3>(robot_->rod_->get_pose_key(sc.finger_node_index))
                .translation();
            Vector3 d = finger_pos - sc.sphere_center;
            double dn = d.norm();
            Vector3 dir = (dn > 1e-8) ? Vector3(d / dn) : Vector3(0.0, 0.0, 1.0);
            values_.insert(dummy_point_key(),
                Point3(sc.sphere_center + sc.sphere_radius * dir));
        }
    }
    else if (sdf_contact_) {
        const auto& env = *sdf_contact_;
        Pose3 obj_mean(env.object_pose_mean);
        values_.insert(sphere_object_key(), obj_mean);

        // Seed the witness/dummy point on the surface. Ray-march in the
        // object's local frame from the local origin toward the contact node
        // until the SDF crosses zero; this keeps the seed in-plane with the tip
        // and object center. Mirrors TendonFingerTrajectoryPlanner's seed.
        int i_node = *env.target_contact_node;
        const Pose3 tip_pose = values_.at<Pose3>(robot_->rod_->get_pose_key(i_node));
        Point3 tip_world = tip_pose.translation();
        Point3 tip_local = obj_mean.transformTo(tip_world);

        // Analytic ellipsoid (Section 1.6.3): project the tip radially onto the
        // surface x^T M x = 1. seed = tip_local / sqrt(tip^T M tip).
        if (env.ellipsoid_semi_axes.norm() > 0.0) {
            const Vector3& a = env.ellipsoid_semi_axes;
            Vector3 m_diag(1.0 / (a.x() * a.x()), 1.0 / (a.y() * a.y()),
                           1.0 / (a.z() * a.z()));
            double q = tip_local.cwiseProduct(m_diag.cwiseProduct(tip_local)).sum();
            Point3 seed_local = (q > 1e-12) ? Point3(tip_local / std::sqrt(q))
                                            : Point3(a.x(), 0.0, 0.0);
            values_.insert(dummy_point_key(), obj_mean.transformFrom(seed_local));
            return;
        }

        double tip_local_norm = tip_local.norm();
        Point3 dir_local = (tip_local_norm > 1e-8)
                               ? Point3(tip_local / tip_local_norm)
                               : Point3(0.0, 0.0, 1.0);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>
            sampler(*env.sdf_grid);

        const double step   = 5e-4;   // 0.5 mm per step
        const int    max_it = 4000;   // up to 2 m along the ray
        double t = 0.0;
        double prev_sdf = sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0));
        double t_surface = -1.0;
        for (int it = 1; it <= max_it; ++it) {
            double tt = it * step;
            Point3 q = tt * dir_local;
            double sdf = sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z()));
            if (std::isfinite(prev_sdf) && std::isfinite(sdf)
                    && prev_sdf * sdf < 0.0) {
                double alpha = prev_sdf / (prev_sdf - sdf);
                t_surface = t + alpha * step;
                break;
            }
            prev_sdf = sdf;
            t = tt;
        }
        if (t_surface < 0.0) {
            t_surface = std::abs(sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0)));
        }
        Point3 seed_local = t_surface * dir_local;
        Point3 seed_world = obj_mean.transformFrom(seed_local);
        values_.insert(dummy_point_key(), seed_world);
    }
}


template<int N>
std::vector<Solution<TendonFingerMarginals>>
TendonFingerSolver<N>::get_intermediate_solutions() const {
    std::vector<Solution<TendonFingerMarginals>> results;
    results.reserve(intermediate_values_.size());

    // Zero-returning functors: extract means only without expensive Marginals computation.
    // joint_of must return a (6+N)x(6+N) block (pose dim + tensions dim), since
    // TendonFingerModel::get_J_pose_tensions reads block<6,N>(0,6) and block<N,N>(6,6).
    auto zero_cov   = [](gtsam::Key) { return gtsam::Matrix::Zero(6, 6); };
    auto zero_joint = [](gtsam::Key, gtsam::Key) { return gtsam::Matrix::Zero(6 + N, 6 + N); };

    for (const auto& vals : intermediate_values_) {
        Solution<TendonFingerMarginals> sol;
        sol.marginals = robot_->get_marginals(vals, zero_cov, zero_joint);
        results.push_back(std::move(sol));
    }
    return results;
}


template<int N>
Solution<TendonFingerMarginals>
TendonFingerSolver<N>::get_initial_solution() const {
    auto zero_cov   = [](gtsam::Key) { return gtsam::Matrix::Zero(6, 6); };
    auto zero_joint = [](gtsam::Key, gtsam::Key) { return gtsam::Matrix::Zero(6 + N, 6 + N); };
    Solution<TendonFingerMarginals> sol;
    sol.marginals = robot_->get_marginals(initial_values_, zero_cov, zero_joint);
    return sol;
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

std::vector<std::tuple<std::string, int, double>>
TendonFingerSolverDispatch::get_factor_error_summary() const {
    return std::visit([](const auto& solver_ptr) {
        return solver_ptr->get_factor_error_summary();
    }, solver_);
}

std::vector<std::pair<std::string, std::vector<double>>>
TendonFingerSolverDispatch::get_factor_errors_by_type() const {
    return std::visit([](const auto& solver_ptr) {
        return solver_ptr->get_factor_errors_by_type();
    }, solver_);
}

std::vector<std::tuple<std::string, int, double>>
TendonFingerSolverDispatch::get_initial_factor_error_summary() const {
    return std::visit([](const auto& solver_ptr) {
        return solver_ptr->get_initial_factor_error_summary();
    }, solver_);
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd>
TendonFingerSolverDispatch::get_hessian_and_gradient() const {
    return std::visit([](const auto& solver_ptr) {
        return solver_ptr->get_hessian_and_gradient();
    }, solver_);
}

std::vector<Solution<TendonFingerMarginals>>
TendonFingerSolverDispatch::get_intermediate_solutions() const {
    return std::visit([](const auto& solver_ptr) {
        return solver_ptr->get_intermediate_solutions();
    }, solver_);
}

Solution<TendonFingerMarginals>
TendonFingerSolverDispatch::get_initial_solution() const {
    return std::visit([](const auto& solver_ptr) {
        return solver_ptr->get_initial_solution();
    }, solver_);
}
