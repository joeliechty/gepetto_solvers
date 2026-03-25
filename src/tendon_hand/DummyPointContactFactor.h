#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

namespace crest_sparse {

class DummyPointContactFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> {
private:
    double R_; // sphere radius
    openvdb::FloatGrid::Ptr sdf_grid_; // SDF grid for the object

public:
    DummyPointContactFactor(gtsam::Key sphere_key, gtsam::Key object_key, gtsam::Key point_key, 
                            double radius, const openvdb::FloatGrid::Ptr& sdf_grid,
                            const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor3(noise_model, sphere_key, object_key, point_key), R_(radius), sdf_grid_(sdf_grid) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& sphere_pose, const gtsam::Pose3& object_pose, const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        // ==========================================
        // 1. Evaluate Sphere Error (e1)
        // ==========================================
        gtsam::Matrix36 D_center_pose;
        gtsam::Point3 center = sphere_pose.translation(H1 ? &D_center_pose : nullptr);

        double dist_to_center = gtsam::distance3(center, dummy_point);
        if (dist_to_center < 1e-7) dist_to_center = 1e-7; // Protect against divide-by-zero

        double e1 = dist_to_center - R_;

        // ==========================================
        // 2. Evaluate SDF Error (e2)
        // ==========================================
        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_point;
        // Transform dummy point to object's local frame
        gtsam::Point3 p_local = object_pose.transformTo(dummy_point,
            H2 ? &D_plocal_obj : nullptr,
            H3 ? &D_plocal_point : nullptr);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R vdb_pt(p_local.x(), p_local.y(), p_local.z());
        
        double e2 = sampler.wsSample(vdb_pt);

        // ==========================================
        // 3. Calculate Jacobians
        // ==========================================
        if (H1 || H2 || H3) {
            // Initialize with zeros (2 rows)
            if (H1) *H1 = gtsam::Matrix::Zero(2, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(2, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(2, 3);

            // -- Sphere Jacobians (Row 0) --
            gtsam::Vector3 n_sphere = (dummy_point - center) / dist_to_center; // 1x3 normal
            
            if (H1) H1->row(0) = -n_sphere.transpose() * D_center_pose;
            if (H3) H3->row(0) = n_sphere.transpose();

            // -- SDF Jacobians (Row 1) --
            double eps = 1e-4;
            double dx = sampler.wsSample(openvdb::Vec3R(vdb_pt.x() + eps, vdb_pt.y(), vdb_pt.z())) -
                        sampler.wsSample(openvdb::Vec3R(vdb_pt.x() - eps, vdb_pt.y(), vdb_pt.z()));
            double dy = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() + eps, vdb_pt.z())) -
                        sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() - eps, vdb_pt.z()));
            double dz = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() + eps)) -
                        sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() - eps));

            gtsam::Vector3 n_local(dx, dy, dz);
            double norm = n_local.norm();
            if (norm > 1e-8) n_local /= norm;

            // Chain rule: de2/dp_local = n_local^T (1x3)
            gtsam::Matrix13 de2_dplocal = n_local.transpose();

            if (H2) H2->row(1) = de2_dplocal * D_plocal_obj;
            if (H3) H3->row(1) = de2_dplocal * D_plocal_point;
        }

        return gtsam::Vector2(e1, e2);
    }
};

} // namespace crest_sparse