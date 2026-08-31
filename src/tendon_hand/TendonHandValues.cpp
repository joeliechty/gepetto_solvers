// Initial values and state round-tripping.
//
// get_initial_values seeds witnesses AFTER merging warm-start poses, so they
// are projected from where the finger actually converged rather than from a
// straight hand. values_from_marginals is the reverse direction, and it also
// recovers the shared wrist -- which no finger carries directly, since node 0
// has no pose variable under the hand-base reparameterization.

#include "gepetto_solvers/tendon_hand/TendonHandModel.h"

#include "gepetto_solvers/utils/MiscInline.h"

#include <gtsam/constrained/NonlinearEqualityConstraint.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <openvdb/tools/Interpolation.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>

using namespace gtsam;


Values TendonHandModel::get_initial_values(const Values* warm) const {
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

    // Adopt any warm-start poses BEFORE the witness seeding below, so the
    // projections that derive each witness from its contact node start from
    // where the finger actually converged rather than from a straight hand.
    if (warm) {
        for (Key k : values.keys())
            if (warm->exists(k)) values.update(k, warm->at(k));
    }

    // Contact object pose (once) + per-finger witness point seeds.
    bool object_seeded = false;
    for (size_t i = 0; i < fingers_.size(); ++i) {
        if (!sdf_contacts_[i] && !sphere_contacts_[i]) continue;
        std::visit([&](const auto& fp) {
            if (sdf_contacts_[i]) {
                const auto& env = *sdf_contacts_[i];
                // Only seed the shared object when this env actually contributes
                // a factor in build_graph: a terminal contact and/or active
                // collision. An inert sdf env seeds nothing (avoids an orphan
                // object variable with no factors).
                bool has_col = env.collision_avoidance &&
                               gepetto_solvers::has_object_surface(env) &&
                               !env.collision_node_indices.empty();
                if (!env.target_contact_node.has_value() && !has_col) return;

                Pose3 obj_mean(env.object_pose_mean);
                if (!object_seeded) { values.insert(object_key(), obj_mean); object_seeded = true; }

                // Collision-only env (no target_contact_node): the object is
                // seeded above; there is no witness point to seed.
                if (!env.target_contact_node.has_value()) return;

                // Center-direct contact (Eq 1.101) constrains the sphere center
                // directly, so build_graph creates no witness variable here —
                // seeding one would leave an orphan value with no factors. Must
                // stay in lockstep with build_graph, hence the shared predicate.
                if (uses_center_direct_contact(env)) return;

                Point3 seed_local;
                if (env.witness_target) {
                    // Controller phase 3 (Eq 1.111): start the witness at its
                    // nominal grasp target (given in the WORLD frame) so the
                    // geodesic pull has a short way to travel.
                    seed_local = obj_mean.transformTo(*env.witness_target);
                } else if (env.witness_point_seed) {
                    // Caller-provided seed (object-local frame); skip the march.
                    seed_local = *env.witness_point_seed;
                } else if (env.ellipsoid_semi_axes.norm() > 0.0) {
                    // Analytic ellipsoid (Section 1.6.3): project the tip radially
                    // onto the surface x^T M x = 1. seed = tip_local / sqrt(tip^T M tip).
                    int i_node = *env.target_contact_node;
                    Point3 tip_world = values.at<Pose3>(fp->rod_->get_pose_key(i_node)).translation();
                    Point3 tip_local = obj_mean.transformTo(tip_world);
                    const Vector3& a = env.ellipsoid_semi_axes;
                    Vector3 m_diag(1.0 / (a.x() * a.x()), 1.0 / (a.y() * a.y()),
                                   1.0 / (a.z() * a.z()));
                    double q = tip_local.cwiseProduct(m_diag.cwiseProduct(tip_local)).sum();
                    if (q > 1e-12) seed_local = Point3(tip_local / std::sqrt(q));
                    else           seed_local = Point3(a.x(), 0.0, 0.0);
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

    // No support-plane seeding: the table contact equality constrains the contact
    // node's sphere CENTER directly (see build_graph), so there is no witness
    // variable to seed. Must stay in lockstep with build_graph -- seeding one
    // here would leave an orphan value with no factors, i.e. an indeterminate
    // system.

    return values;
}


Values TendonHandModel::values_from_marginals(const TendonHandMarginals& state) const {
    if (state.fingers.size() != fingers_.size())
        throw std::invalid_argument(
            "values_from_marginals: state has " +
            std::to_string(state.fingers.size()) + " fingers, this model has " +
            std::to_string(fingers_.size()));

    Values values;

    for (size_t i = 0; i < fingers_.size(); ++i) {
        const TendonFingerMarginals& fm = state.fingers[i];
        std::visit([&](const auto& fp) {
            using FingerType = typename std::remove_reference_t<decltype(*fp)>;
            constexpr int N = FingerType::NumTendons;

            const int num_nodes = fp->get_num_nodes();
            if (static_cast<int>(fm.rod.states.size()) != num_nodes)
                throw std::invalid_argument(
                    "values_from_marginals: finger " + std::to_string(i) +
                    " has " + std::to_string(fm.rod.states.size()) +
                    " rod states, this model has " + std::to_string(num_nodes) +
                    " nodes");
            if (fm.tensions.mean.size() != N)
                throw std::invalid_argument(
                    "values_from_marginals: finger " + std::to_string(i) +
                    " has " + std::to_string(fm.tensions.mean.size()) +
                    " tendons, this model has " + std::to_string(N));

            // Rod chain. Node 0's pose is NOT a variable under the hand-base
            // reparameterization (see the header note); its stress and wrench
            // still are, so only the pose insert is skipped.
            const bool skip_node0_pose = fp->rod_->uses_root();
            for (int j = 0; j < num_nodes; ++j) {
                const auto& s = fm.rod.states[j];
                if (!(skip_node0_pose && j == 0))
                    values.insert(fp->rod_->get_pose_key(j), Pose3(s.pose.mean));
                values.insert(fp->rod_->get_stress_key(j), Vector6(s.stress.mean));
                values.insert(fp->rod_->get_wrench_key(j), Vector6(s.wrench.mean));
            }

            // External disc wrenches. external_wrenches is indexed by NODE and
            // resolves through get_external_wrench_key, which aliases the rod's
            // own wrench key at non-disc nodes -- already written above. Only the
            // genuine Symbol('D', ...) variables are left, so walk the discs.
            const auto& disc_pose_idx = fp->get_tendon_config().disc_pose_idx;
            for (size_t d = 1; d < disc_pose_idx.size(); ++d) {
                const int node = disc_pose_idx[d];
                if (node < 0 || node >= static_cast<int>(fm.external_wrenches.size()))
                    continue;
                values.insert(fp->get_disc_wrench_key(static_cast<int>(d)),
                              Vector6(fm.external_wrenches[node].mean));
            }

            values.insert(fp->get_tensions_key(),
                          Eigen::Vector<double, N>(fm.tensions.mean));

            if (static_cast<int>(fm.tendon_lengths.size()) == N) {
                Eigen::Vector<double, N> L;
                for (int t = 0; t < N; ++t) L(t) = fm.tendon_lengths[t];
                values.insert(fp->get_lengths_key(), L);
            }
        }, fingers_[i]);
    }

    // The shared wrist. No finger carries it directly: under the hand-base
    // reparameterization node 0's pose is not a variable but the composition
    // T_0 = T_wrist o T_offset, so the loop above deliberately skipped it and
    // the wrist would otherwise be missing from the bundle entirely. A warm
    // start built from that would hold every rod pose from the state and the
    // wrist at whatever the receiving model was constructed with -- an
    // inconsistent guess that the Root factors and the wrist prior immediately
    // tear back apart, i.e. the hand snapping to the commanded base pose on the
    // first iteration. Invert the relation instead and carry it.
    const bool uses_root = !fingers_.empty() && std::visit(
        [](const auto& fp) { return fp->rod_->uses_root(); }, fingers_[0]);
    if (uses_root && !hand_base_offsets_.empty()) {
        const Pose3 T0(state.fingers[0].rod.states[0].pose.mean);
        values.insert(wrist_key(step_), T0 * hand_base_offsets_[0].inverse());
    }

    return values;
}
