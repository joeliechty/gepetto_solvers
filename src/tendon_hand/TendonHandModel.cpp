#include "TendonHandModel.h"

#include "utils/MiscInline.h"

#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <openvdb/tools/Interpolation.h>

#include <cmath>
#include <stdexcept>

using namespace gtsam;


namespace {

// Construct a TendonFingerModel<N> from a TendonFingerSolverConfig, choosing the
// per-disc vs. simple routing path and the K_inv vs. per-segment path exactly as
// TendonFingerSolver does. base_pose_mean is supplied by the caller (= wrist o offset).
template<int N>
std::unique_ptr<TendonFingerModel<N>> make_finger_impl(
    const TendonFingerSolverConfig& c,
    const Pose3& base_pose_mean,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    SharedDiagonal base_pose_noise)
{
    if (c.per_disc_tendon_input.is_populated()) {
        if (c.K_inv_per_segment.empty()) {
            return std::make_unique<TendonFingerModel<N>>(
                c.rod_length, c.num_discs, c.num_between_nodes,
                c.per_disc_tendon_input, c.K_inv, twist_noise, stress_noise,
                base_pose_mean, base_pose_noise, c.disc_positions_normalized);
        }
        return std::make_unique<TendonFingerModel<N>>(
            c.rod_length, c.num_discs, c.num_between_nodes,
            c.per_disc_tendon_input, c.K_inv_per_segment, twist_noise, stress_noise,
            base_pose_mean, base_pose_noise, c.disc_positions_normalized);
    }
    if (c.K_inv_per_segment.empty()) {
        return std::make_unique<TendonFingerModel<N>>(
            c.rod_length, c.num_discs, c.num_between_nodes,
            c.tendon_input, c.K_inv, twist_noise, stress_noise,
            base_pose_mean, base_pose_noise, c.disc_positions_normalized);
    }
    return std::make_unique<TendonFingerModel<N>>(
        c.rod_length, c.num_discs, c.num_between_nodes,
        c.tendon_input, c.K_inv_per_segment, twist_noise, stress_noise,
        base_pose_mean, base_pose_noise, c.disc_positions_normalized);
}

}  // namespace


TendonHandModel::TendonHandModel(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    const Pose3& wrist_pose,
    SharedDiagonal wrist_noise,
    int step,
    bool emit_wrist_prior)
:
    wrist_pose_(wrist_pose),
    wrist_noise_(wrist_noise),
    step_(step),
    emit_wrist_prior_(emit_wrist_prior)
{
    const Key shared_wrist_key = wrist_key(step_);

    fingers_.reserve(finger_configs.size());
    finger_names_.reserve(finger_configs.size());
    sdf_contacts_.reserve(finger_configs.size());
    sphere_contacts_.reserve(finger_configs.size());
    small_wrench_noises_.reserve(finger_configs.size());

    for (const auto& [name, c] : finger_configs) {
        finger_names_.push_back(name);
        sdf_contacts_.push_back(c.sdf_contact);
        sphere_contacts_.push_back(c.sphere_contact);
        if (c.sdf_contact.has_value() || c.sphere_contact.has_value())
            has_contact_ = true;

        SharedDiagonal twist_noise = get_noise_model_rot_pos(
            c.sigma_twist_rot, c.sigma_twist_pos);
        SharedDiagonal stress_noise = get_noise_model_rot_pos(
            c.sigma_stress_moment, c.sigma_stress_force);
        SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
            c.sigma_base_rot, c.sigma_base_pos);
        // Tight per-finger external-wrench prior noise, matching TendonFingerSolver.
        small_wrench_noises_.push_back(
            get_noise_model_rot_pos(c.sigma_stress_moment, c.sigma_stress_force));

        // This finger's fixed attachment to the wrist. Its node-0 pose mean is
        // T_wrist o T_offset, so the shared base variable seeds to T_wrist for
        // every finger (see get_initial_values / set_root_reparameterization).
        Pose3 offset(c.hand_base_offset);
        Pose3 base_pose_mean = wrist_pose_ * offset;

        int Nt = c.num_tendons;
        switch (Nt) {
            case 1:  fingers_.push_back(make_finger_impl<1>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 2:  fingers_.push_back(make_finger_impl<2>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 3:  fingers_.push_back(make_finger_impl<3>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 4:  fingers_.push_back(make_finger_impl<4>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 5:  fingers_.push_back(make_finger_impl<5>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 6:  fingers_.push_back(make_finger_impl<6>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 7:  fingers_.push_back(make_finger_impl<7>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 8:  fingers_.push_back(make_finger_impl<8>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 9:  fingers_.push_back(make_finger_impl<9>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            case 10: fingers_.push_back(make_finger_impl<10>(c, base_pose_mean, twist_noise, stress_noise, base_pose_noise)); break;
            default: throw std::invalid_argument(
                "num_tendons must be between 1 and 10, got " + std::to_string(Nt));
        }

        // Share the one wrist variable and let this model own the wrist prior.
        std::visit([&](auto& fp) {
            fp->set_hand_base(offset, shared_wrist_key);
            fp->set_emit_base_prior(false);
        }, fingers_.back());
    }
}


NonlinearFactorGraph TendonHandModel::build_graph(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    if (tensions.size() != fingers_.size())
        throw std::invalid_argument("tensions size must match number of fingers");
    if (tip_wrenches.size() != fingers_.size())
        throw std::invalid_argument("tip_wrenches size must match number of fingers");

    NonlinearFactorGraph graph;

    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            if (tensions[i].mean.size() != N)
                throw std::invalid_argument(
                    "Finger " + std::to_string(i) + " expects " + std::to_string(N) +
                    " tendons, got " + std::to_string(tensions[i].mean.size()));

            VectorNGaussian<N> t;
            t.mean = tensions[i].mean;
            t.cov  = tensions[i].cov;

            // Rod + tendon factors (base prior suppressed via set_emit_base_prior).
            graph.add(fp->build_graph(t));

            // Constrain interior external wrenches to zero; the tip wrench uses
            // the caller's value (mirrors TendonFingerSolver::build_graph).
            int num_nodes = fp->get_num_nodes();
            for (int j = 1; j + 1 < num_nodes; ++j) {
                graph.add(PriorFactor<Vector6>(
                    fp->get_external_wrench_key(j), Vector6::Zero(), small_wrench_noises_[i]));
            }
            graph.add(PriorFactor<Vector6>(
                fp->get_external_wrench_key(num_nodes - 1),
                tip_wrenches[i].mean,
                noiseModel::Gaussian::Covariance(tip_wrenches[i].cov)));
        }, fingers_[i]);
    }

    // Single shared floating-wrist prior (rigidly anchors the gauge). In a
    // trajectory only the start step emits it (as the loose hand-pose prior,
    // Eq 1.40); interior/terminal wrists are pinned by the GP chain + kinematics.
    if (emit_wrist_prior_)
        graph.add(PriorFactor<Pose3>(wrist_key(step_), wrist_pose_, wrist_noise_));

    // Per-finger tip contact against one shared object. The object pose is
    // anchored once; each finger drives its own witness point onto the surface.
    bool object_anchored = false;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i] && !sphere_contacts_[i]) continue;
        std::visit([&](auto& fp) {
            if (sdf_contacts_[i]) {
                const auto& env = *sdf_contacts_[i];
                Key tip_key = fp->rod_->get_pose_key(*env.target_contact_node);
                if (!object_anchored) {
                    graph.add(PriorFactor<Pose3>(
                        object_key(), Pose3(env.object_pose_mean),
                        noiseModel::Gaussian::Covariance(env.object_pose_cov)));
                    object_anchored = true;
                }
                auto contact = std::make_shared<crest_sparse::SdfWitnessContactFactor>(
                    tip_key, object_key(), witness_key(static_cast<int>(i)),
                    env.contact_node_radius, env.sdf_grid,
                    noiseModel::Isotropic::Sigma(5, 1.0));
                graph.add(gtsam::ZeroCostConstraint(contact));
            } else {
                const auto& sc = *sphere_contacts_[i];
                Key finger_key = fp->rod_->get_pose_key(sc.finger_node_index);
                if (!object_anchored) {
                    graph.add(PriorFactor<Pose3>(
                        object_key(), Pose3(Rot3(), sc.sphere_center),
                        noiseModel::Gaussian::Covariance(sc.sphere_pose_cov)));
                    object_anchored = true;
                }
                if (sc.witness) {
                    auto contact = std::make_shared<crest_sparse::SphereWitnessContactFactor>(
                        finger_key, object_key(), witness_key(static_cast<int>(i)),
                        sc.finger_node_radius, sc.sphere_radius,
                        noiseModel::Isotropic::Sigma(5, 1.0));
                    graph.add(gtsam::ZeroCostConstraint(contact));
                } else {
                    auto contact = std::make_shared<crest_sparse::SphereSphereContactFactor>(
                        finger_key, object_key(),
                        sc.finger_node_radius, sc.sphere_radius,
                        noiseModel::Isotropic::Sigma(1, 1.0));
                    graph.add(gtsam::ZeroCostConstraint(contact));
                }
            }
        }, fingers_[i]);
    }

    return graph;
}


Values TendonHandModel::get_initial_values() const {
    Values values;

    // Merge each finger's values; the shared wrist variable appears in every
    // finger's values (identical), so keep only the first and drop the rest.
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            Values fv = fp->get_initial_values();
            if (i > 0) fv.erase(wrist_key(step_));
            values.insert(fv);
        }, fingers_[i]);
    }

    // Contact object pose (once) + per-finger witness point seeds.
    bool object_seeded = false;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i] && !sphere_contacts_[i]) continue;
        std::visit([&](const auto& fp) {
            if (sdf_contacts_[i]) {
                const auto& env = *sdf_contacts_[i];
                Pose3 obj_mean(env.object_pose_mean);
                if (!object_seeded) { values.insert(object_key(), obj_mean); object_seeded = true; }

                Point3 seed_local;
                if (env.witness_point_seed) {
                    // Caller-provided seed (object-local frame); skip the march.
                    seed_local = *env.witness_point_seed;
                } else {
                    // Ray-march in the object-local frame from the local origin
                    // toward the tip until the SDF crosses zero (mirrors
                    // TendonFingerSolver).
                    int i_node = *env.target_contact_node;
                    Point3 tip_world = values.at<Pose3>(fp->rod_->get_pose_key(i_node)).translation();
                    Point3 tip_local = obj_mean.transformTo(tip_world);
                    double tip_local_norm = tip_local.norm();
                    Point3 dir_local = (tip_local_norm > 1e-8)
                                           ? Point3(tip_local / tip_local_norm)
                                           : Point3(0.0, 0.0, 1.0);

                    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>
                        sampler(*env.sdf_grid);
                    const double step = 5e-4;
                    const int max_it = 4000;
                    double t = 0.0;
                    double prev_sdf = sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0));
                    double t_surface = -1.0;
                    for (int it = 1; it <= max_it; ++it) {
                        double tt = it * step;
                        Point3 q = tt * dir_local;
                        double sdf = sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z()));
                        if (std::isfinite(prev_sdf) && std::isfinite(sdf) && prev_sdf * sdf < 0.0) {
                            double alpha = prev_sdf / (prev_sdf - sdf);
                            t_surface = t + alpha * step;
                            break;
                        }
                        prev_sdf = sdf;
                        t = tt;
                    }
                    if (t_surface < 0.0)
                        t_surface = std::abs(sampler.wsSample(openvdb::Vec3R(0.0, 0.0, 0.0)));
                    seed_local = Point3(t_surface * dir_local);
                }
                Point3 seed_world = obj_mean.transformFrom(seed_local);
                values.insert(witness_key(static_cast<int>(i)), seed_world);
            } else {
                const auto& sc = *sphere_contacts_[i];
                if (!object_seeded) {
                    values.insert(object_key(), Pose3(Rot3(), sc.sphere_center));
                    object_seeded = true;
                }
                if (sc.witness) {
                    Point3 finger_pos = values
                        .at<Pose3>(fp->rod_->get_pose_key(sc.finger_node_index)).translation();
                    Vector3 d = finger_pos - sc.sphere_center;
                    double dn = d.norm();
                    Vector3 dir = (dn > 1e-8) ? Vector3(d / dn) : Vector3(0.0, 0.0, 1.0);
                    values.insert(witness_key(static_cast<int>(i)),
                                  Point3(sc.sphere_center + sc.sphere_radius * dir));
                }
            }
        }, fingers_[i]);
    }

    return values;
}


Key TendonHandModel::finger_tension_key(int i) const {
    return std::visit(
        [](const auto& fp) { return fp->get_tensions_key(); }, fingers_.at(i));
}


Key TendonHandModel::finger_length_key(int i) const {
    return std::visit(
        [](const auto& fp) { return fp->get_lengths_key(); }, fingers_.at(i));
}


void TendonHandModel::add_temporal_gp(
    NonlinearFactorGraph& graph,
    const TendonHandModel& next,
    const Eigen::MatrixXd& gp_tense_Qc,
    const Eigen::MatrixXd& gp_len_Qc,
    double dt) const
{
    const bool has_len_gp = gp_len_Qc.size() > 0;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            // Tension GP (Eq 1.11): identity transition, zero-mean between factor.
            Eigen::Matrix<double, N, N> Qc = gp_tense_Qc.topLeftCorner<N, N>();
            graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                fp->get_tensions_key(),
                next.finger_tension_key(static_cast<int>(i)),
                Eigen::Vector<double, N>::Zero(),
                noiseModel::Gaussian::Covariance(Qc * dt)));

            // Length GP (Eq 1.13), optional.
            if (has_len_gp) {
                Eigen::Matrix<double, N, N> Qc_len = gp_len_Qc.topLeftCorner<N, N>();
                graph.add(BetweenFactor<Eigen::Vector<double, N>>(
                    fp->get_lengths_key(),
                    next.finger_length_key(static_cast<int>(i)),
                    Eigen::Vector<double, N>::Zero(),
                    noiseModel::Gaussian::Covariance(Qc_len * dt)));
            }
        }, fingers_[i]);
    }
}


TendonHandMarginals TendonHandModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    TendonHandMarginals out;
    out.fingers.reserve(fingers_.size());
    out.finger_names = finger_names_;
    for (const auto& finger : fingers_) {
        std::visit([&](const auto& fp) {
            out.fingers.push_back(fp->get_marginals(values, marginals));
        }, finger);
    }
    return out;
}
