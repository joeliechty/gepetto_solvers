// Initial values: the kinematics seeds its own variables, then this seeds the
// TASK ones -- the shared object pose and the per-digit witness points.
//
// Order matters. The kinematics adopts the warm-start poses as it goes, so the
// witness projections below start from where the digit actually converged rather
// than from a straight hand.

#include "gepetto_solvers/hand/HandModel.h"

#include "gepetto_solvers/utils/MiscInline.h"

#include <openvdb/tools/Interpolation.h>

#include <cmath>

using namespace gtsam;
using gepetto_solvers::HandSite;


Values HandModel::get_initial_values(const Values* warm) const {
    Values values;

    // Everything the mechanism owns, with the warm-start merge applied inside.
    kin_->insert_initial_values(values, warm);

    // Contact object pose (once) + per-digit witness point seeds.
    const int n_digits = num_digits();
    bool object_seeded = false;
    for (int i = 0; i < n_digits; ++i) {
        if (!env_[i] && !sphere_contacts_[i]) continue;

        if (env_[i]) {
            const auto& env = *env_[i];
            // Only seed the shared object when this env actually contributes a
            // factor in build_graph: a terminal contact and/or active collision.
            // An inert env seeds nothing (avoids an orphan object variable with
            // no factors).
            bool has_col = env.collision_avoidance &&
                           gepetto_solvers::has_object_surface(env) &&
                           !env.collision_node_indices.empty();
            if (!env.target_contact_node.has_value() && !has_col) continue;

            Pose3 obj_mean(env.object_pose_mean);
            if (!object_seeded) {
                values.insert(object_key(), obj_mean);
                object_seeded = true;
            }

            // Collision-only env (no target_contact_node): the object is seeded
            // above; there is no witness point to seed.
            if (!env.target_contact_node.has_value()) continue;

            // Center-direct contact (Eq 1.101) constrains the sphere center
            // directly, so build_graph creates no witness variable here --
            // seeding one would leave an orphan value with no factors. Must stay
            // in lockstep with build_graph, hence the shared predicate.
            if (uses_center_direct_contact(env)) continue;

            Point3 seed_local;
            if (env.witness_target) {
                // Controller phase 3 (Eq 1.111): start the witness at its nominal
                // grasp target (given in the WORLD frame) so the geodesic pull
                // has a short way to travel.
                seed_local = obj_mean.transformTo(*env.witness_target);
            } else if (env.witness_point_seed) {
                // Caller-provided seed (object-local frame); skip the march.
                seed_local = *env.witness_point_seed;
            } else if (env.ellipsoid_semi_axes.norm() > 0.0) {
                // Analytic ellipsoid (Section 1.6.3): project the tip radially
                // onto the surface x^T M x = 1. seed = tip_local / sqrt(tip^T M tip).
                Point3 tip_world = values.at<Pose3>(
                    kin_->site_pose_key({i, *env.target_contact_node})).translation();
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
                Point3 tip_world = values.at<Pose3>(
                    kin_->site_pose_key({i, *env.target_contact_node})).translation();
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
            values.insert(witness_key(i), seed_world);
        } else {
            const auto& sc = *sphere_contacts_[i];
            if (!object_seeded) {
                values.insert(object_key(), Pose3(Rot3(), sc.sphere_center));
                object_seeded = true;
            }
            if (sc.witness) {
                Point3 digit_pos = values.at<Pose3>(
                    kin_->site_pose_key({i, sc.finger_node_index})).translation();
                Vector3 d = digit_pos - sc.sphere_center;
                double dn = d.norm();
                Vector3 dir = (dn > 1e-8) ? Vector3(d / dn) : Vector3(0.0, 0.0, 1.0);
                values.insert(witness_key(i),
                              Point3(sc.sphere_center + sc.sphere_radius * dir));
            }
        }
    }

    // No support-plane seeding: the table contact equality constrains the contact
    // site's sphere CENTER directly (see build_graph), so there is no witness
    // variable to seed. Must stay in lockstep with build_graph -- seeding one
    // here would leave an orphan value with no factors, i.e. an indeterminate
    // system.

    return values;
}
