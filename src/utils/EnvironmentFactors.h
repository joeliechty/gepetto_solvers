#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <Eigen/Core>

#include <optional>
#include <vector>

namespace crest_sparse {

// Configuration for an OpenVDB-backed environment used by trajectory planners.
// Implements the math in Section 3 of the underactuated object manipulation
// formulation (cubic-polynomial barrier collision + dummy-point surface
// contact).
struct EnvironmentConfig {
    openvdb::FloatGrid::Ptr sdf_grid;
    gtsam::Matrix4 object_pose_mean = gtsam::Matrix4::Identity();
    gtsam::Matrix6 object_pose_cov  = 1e-8 * gtsam::Matrix6::Identity();
    bool object_pose_per_step = false;

    // Collision running cost (Eq 28/29): e = (eps - Phi_D)^3 if Phi_D <= eps, else 0.
    double collision_epsilon = 0.0;
    double collision_sigma   = 1e-3;
    std::vector<int>    collision_node_indices;
    std::vector<double> collision_node_radii;

    // Contact-as-goal terminal factor (Eq 26 / 30).
    // 3x3 covariance: rows 0-1 weight the dummy-point sphere/surface equality
    // (e1, e2 in SdfContactFactor); row 2 weights the tip-side non-penetration
    // hinge e3 = max(0, R - SDF(T_obj^-1 p_tip)).
    std::optional<int> target_contact_node;
    double contact_node_radius = 0.0;
    gtsam::Matrix3 contact_cov = (gtsam::Matrix3() << 1e-6, 0.0, 0.0,
                                                      0.0, 1e-6, 0.0,
                                                      0.0, 0.0, 1e-6).finished();
};


// C^2 cubic-polynomial barrier collision factor (Eq 27, 28, 29).
//   Phi_D(p, T_obj) = SDF(T_obj^{-1} p) - radius
//   e(Phi_D)        = (eps - Phi_D)^3   if Phi_D <= eps, else 0
//
// Inactive branch (Phi_D > eps) returns exactly zero with zero Jacobians.
// Because value, first, and second derivative all vanish at the boundary,
// the transition is smooth and Gauss-Newton-friendly.
class SdfCollisionFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    double radius_;
    double epsilon_;
    openvdb::FloatGrid::Ptr sdf_grid_;

public:
    SdfCollisionFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                       double radius, double epsilon,
                       const openvdb::FloatGrid::Ptr& sdf_grid,
                       const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, node_pose_key, object_key),
          radius_(radius), epsilon_(epsilon), sdf_grid_(sdf_grid) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_pworld_finger;
        gtsam::Point3 p_world = node_pose.translation(H1 ? &D_pworld_finger : nullptr);

        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_pworld;
        gtsam::Point3 p_local = object_pose.transformTo(p_world,
            H2 ? &D_plocal_obj    : nullptr,
            H1 ? &D_plocal_pworld : nullptr);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R q(p_local.x(), p_local.y(), p_local.z());
        double sdf = sampler.wsSample(q);
        double phi = sdf - radius_;
        double gap = epsilon_ - phi;

        if (gap <= 0.0) {
            if (H1) *H1 = gtsam::Matrix::Zero(1, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 6);
            return gtsam::Vector1(0.0);
        }

        double e = gap * gap * gap;

        if (H1 || H2) {
            double h = 1e-4;
            double dx = sampler.wsSample(openvdb::Vec3R(q.x() + h, q.y(), q.z())) -
                        sampler.wsSample(openvdb::Vec3R(q.x() - h, q.y(), q.z()));
            double dy = sampler.wsSample(openvdb::Vec3R(q.x(), q.y() + h, q.z())) -
                        sampler.wsSample(openvdb::Vec3R(q.x(), q.y() - h, q.z()));
            double dz = sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z() + h)) -
                        sampler.wsSample(openvdb::Vec3R(q.x(), q.y(), q.z() - h));
            gtsam::Vector3 grad(dx, dy, dz);
            double norm = grad.norm();
            if (norm > 1e-8) grad /= norm;

            // de/dphi = -3 * gap^2 ;  dphi/dp_local = grad^T
            gtsam::Matrix13 de_dplocal = (-3.0 * gap * gap) * grad.transpose();
            if (H1) *H1 = de_dplocal * D_plocal_pworld * D_pworld_finger;
            if (H2) *H2 = de_dplocal * D_plocal_obj;
        }

        return gtsam::Vector1(e);
    }
};


// Surface-to-surface contact factor (Eq 26) + tip-side non-penetration hinge.
// Connects:
//   - node_pose_key  (Pose3)   : finger node whose sphere should touch the surface
//   - object_key     (Pose3)   : object pose
//   - point_key      (Point3)  : dummy contact point p_c in world frame
//
// 3D residual = [ ||p_c - p_i||_2 - R,
//                 SDF(T_obj^{-1} p_c),
//                 max(0, R - SDF(T_obj^{-1} p_i)) ].
// Rows 0-1 drive p_c onto both the body sphere (radius R around the tip) and
// the object surface — tangential contact. Row 2 is a one-sided hinge: it
// penalises configurations where the tip center sits inside the object's
// R-offset surface (i.e. the tip-sphere penetrates the object). Adding row 2
// removes the side-symmetry of rows 0-1 — without it, a tip *inside* the
// object can satisfy e1 = e2 = 0 by placing p_c on the surface "behind" the
// tip, which is the penetration mode observed at the end of the trajectory.
class SdfContactFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> {
private:
    double R_;
    openvdb::FloatGrid::Ptr sdf_grid_;

public:
    SdfContactFactor(gtsam::Key node_pose_key, gtsam::Key object_key, gtsam::Key point_key,
                     double radius, const openvdb::FloatGrid::Ptr& sdf_grid,
                     const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor3(noise_model, node_pose_key, object_key, point_key),
          R_(radius), sdf_grid_(sdf_grid) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        gtsam::Matrix36 D_center_pose;
        gtsam::Point3 center = node_pose.translation(H1 ? &D_center_pose : nullptr);

        double dist_to_center = gtsam::distance3(center, dummy_point);
        if (dist_to_center < 1e-7) dist_to_center = 1e-7;
        double e1 = dist_to_center - R_;

        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_point;
        gtsam::Point3 p_local = object_pose.transformTo(dummy_point,
            H2 ? &D_plocal_obj   : nullptr,
            H3 ? &D_plocal_point : nullptr);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R vdb_pt(p_local.x(), p_local.y(), p_local.z());
        double e2 = sampler.wsSample(vdb_pt);

        // Tip in object-local frame for the non-penetration hinge e3.
        gtsam::Matrix36 D_tiplocal_obj;
        gtsam::Matrix33 D_tiplocal_center;
        gtsam::Point3 tip_local = object_pose.transformTo(center,
            (H1 || H2) ? &D_tiplocal_obj    : nullptr,
            (H1)       ? &D_tiplocal_center : nullptr);

        openvdb::Vec3R vdb_tip(tip_local.x(), tip_local.y(), tip_local.z());
        double sdf_tip = sampler.wsSample(vdb_tip);
        const bool e3_active = (sdf_tip < R_);
        double e3 = e3_active ? (R_ - sdf_tip) : 0.0;

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(3, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(3, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(3, 3);

            // Row 0: e1 = ||p_c - p_i|| - R
            gtsam::Vector3 n_sphere = (dummy_point - center) / dist_to_center;
            if (H1) H1->row(0) = -n_sphere.transpose() * D_center_pose;
            if (H3) H3->row(0) = n_sphere.transpose();

            // Row 1: e2 = SDF(T_obj^-1 p_c)  — FD gradient in object-local frame.
            double h = 1e-4;
            double dx = sampler.wsSample(openvdb::Vec3R(vdb_pt.x() + h, vdb_pt.y(), vdb_pt.z())) -
                        sampler.wsSample(openvdb::Vec3R(vdb_pt.x() - h, vdb_pt.y(), vdb_pt.z()));
            double dy = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() + h, vdb_pt.z())) -
                        sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() - h, vdb_pt.z()));
            double dz = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() + h)) -
                        sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() - h));
            gtsam::Vector3 n_local(dx, dy, dz);
            double norm = n_local.norm();
            if (norm > 1e-8) n_local /= norm;

            gtsam::Matrix13 de2_dplocal = n_local.transpose();
            if (H2) H2->row(1) = de2_dplocal * D_plocal_obj;
            if (H3) H3->row(1) = de2_dplocal * D_plocal_point;

            // Row 2: e3 = max(0, R - SDF(tip_local)). Zero Jacobian when inactive.
            if (e3_active) {
                double dx_t = sampler.wsSample(openvdb::Vec3R(vdb_tip.x() + h, vdb_tip.y(), vdb_tip.z())) -
                              sampler.wsSample(openvdb::Vec3R(vdb_tip.x() - h, vdb_tip.y(), vdb_tip.z()));
                double dy_t = sampler.wsSample(openvdb::Vec3R(vdb_tip.x(), vdb_tip.y() + h, vdb_tip.z())) -
                              sampler.wsSample(openvdb::Vec3R(vdb_tip.x(), vdb_tip.y() - h, vdb_tip.z()));
                double dz_t = sampler.wsSample(openvdb::Vec3R(vdb_tip.x(), vdb_tip.y(), vdb_tip.z() + h)) -
                              sampler.wsSample(openvdb::Vec3R(vdb_tip.x(), vdb_tip.y(), vdb_tip.z() - h));
                gtsam::Vector3 n_tip(dx_t, dy_t, dz_t);
                double norm_t = n_tip.norm();
                if (norm_t > 1e-8) n_tip /= norm_t;

                gtsam::Matrix13 de3_dtiplocal = -n_tip.transpose();
                if (H1) H1->row(2) = de3_dtiplocal * D_tiplocal_center * D_center_pose;
                if (H2) H2->row(2) = de3_dtiplocal * D_tiplocal_obj;
                // H3 row 2 stays zero: e3 does not depend on p_c.
            }
        }

        return gtsam::Vector3(e1, e2, e3);
    }
};

} // namespace crest_sparse
