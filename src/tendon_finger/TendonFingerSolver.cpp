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
            // 3-residual witness-point form ([c_R, c_O, c_N]) -- the analytic
            // counterpart of the SDF contact below. An explicit dummy point p_c
            // is driven onto both sphere surfaces with antiparallel normals.
            auto contact = std::make_shared<crest_sparse::SphereSphereWitnessContactFactor>(
                finger_key,
                sphere_object_key(),
                dummy_point_key(),
                sc.finger_node_radius, sc.sphere_radius,
                noiseModel::Isotropic::Sigma(3, 1.0));
            graph_.add(gtsam::ZeroCostConstraint(contact));

            // The dummy point appears only in the hard constraint and no cost
            // factor, leaving the AL cost graph short one variable. A very weak
            // prior (1 m sigma) anchored at the finger node puts it in the cost
            // graph and makes the system full rank without biasing the contact
            // solution.
            //
            // NOTE: for the pure sphere-sphere witness case this stabilizing
            // prior is NOT sufficient. The witness point has a genuine 1-DOF
            // gauge freedom -- rotating it about the axis joining the two sphere
            // centers leaves all three residuals ([c_R, c_O, c_N]) invariant --
            // so the contact constraint pins only 2 of its 3 DOF. No fixed-sigma
            // prior can stabilize this: the AL penalty weight grows like
            // sqrt(mu) toward ~1e6, so a prior loose enough not to bias the
            // contact in its constrained directions at low mu is swamped in the
            // gauge direction at high mu (-> IndeterminantLinearSystem), while a
            // prior tight enough to survive high mu overpowers the contact early
            // and stalls short of the surface. Stabilizing the sphere-sphere
            // witness form would require a gauge-fixing residual in the factor
            // itself (or a mu-coupled prior). The SDF path below has no such
            // gauge -- a general surface normal is unique -- so the 1 m prior is
            // sufficient there.
            Point3 finger_pos = values_.at<Pose3>(finger_key).translation();
            graph_.add(PriorFactor<Point3>(
                dummy_point_key(), finger_pos,
                noiseModel::Isotropic::Sigma(3, 1.0)));
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
    // tight PriorFactor<Pose3>, then connect a rod node to it via the 3-residual
    // witness-point SdfContactFactor (Section 3, [c_R, c_O, c_N]). The factor is
    // wrapped in a gtsam::ZeroCostConstraint so the AL optimizer drives all three
    // residuals exactly to zero; the unit noise model is only the per-row
    // constraint scaling. Mirrors TendonFingerTrajectoryPlanner's contact mode.
    else if (sdf_contact_) {
        const auto& env = *sdf_contact_;
        Key tip_key = robot_->rod_->get_pose_key(*env.target_contact_node);
        graph_.add(PriorFactor<Pose3>(
            sphere_object_key(), Pose3(env.object_pose_mean),
            noiseModel::Gaussian::Covariance(env.object_pose_cov)));
        auto contact = std::make_shared<crest_sparse::SdfContactFactor>(
            tip_key,
            sphere_object_key(),
            dummy_point_key(),
            env.contact_node_radius,
            env.sdf_grid,
            noiseModel::Isotropic::Sigma(3, 1.0));
        graph_.add(gtsam::ZeroCostConstraint(contact));

        // Without this the dummy point appears only in the hard-constraint
        // factor and in no cost factor, leaving the AL optimizer's cost-graph
        // variable set short one variable -- which corrupts the heap / yields
        // an underconstrained linear system. A very weak prior (1 m sigma)
        // anchored at the tip puts the dummy point in the cost graph and makes
        // the system full rank without biasing the contact solution. Mirrors
        // a stabilizing prior for DummyPointContactFactor.
        Point3 tip_pos = values_.at<Pose3>(tip_key).translation();
        graph_.add(PriorFactor<Point3>(
            dummy_point_key(), tip_pos,
            noiseModel::Isotropic::Sigma(3, 1.0)));
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
