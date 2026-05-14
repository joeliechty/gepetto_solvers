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
    std::optional<int> target_contact_node;
    double contact_node_radius = 0.0;
    gtsam::Matrix2 contact_cov = (gtsam::Matrix2() << 1e-6, 0.0, 0.0, 1e-6).finished();
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


// Surface-to-surface contact factor (Eq 26).
// Connects:
//   - node_pose_key  (Pose3)   : finger node whose sphere should touch the surface
//   - object_key     (Pose3)   : object pose
//   - point_key      (Point3)  : dummy contact point p_c in world frame
//
// 2D residual = [ ||p_c - p_i||_2 - r ,  SDF(T_obj^{-1} p_c) ].
// Driving both terms to zero places p_c on both the sphere surface and the
// object surface, i.e. tangential contact.
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

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(2, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(2, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(2, 3);

            gtsam::Vector3 n_sphere = (dummy_point - center) / dist_to_center;
            if (H1) H1->row(0) = -n_sphere.transpose() * D_center_pose;
            if (H3) H3->row(0) = n_sphere.transpose();

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
        }

        return gtsam::Vector2(e1, e2);
    }
};

} // namespace crest_sparse
