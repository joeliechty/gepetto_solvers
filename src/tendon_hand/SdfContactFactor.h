#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

namespace crest_sparse {

class SdfContactFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
private:
    double R_; // contact radius threshold
    openvdb::FloatGrid::Ptr sdf_grid_; // SDF grid for collision checking

public:
    SdfContactFactor(gtsam::Key finger_key, gtsam::Key object_key, double radius,
                     const openvdb::FloatGrid::Ptr& sdf_grid,
                     const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactorN(noise_model, finger_key, object_key), R_(radius), sdf_grid_(sdf_grid) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& finger_pose, const gtsam::Pose3& object_pose,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        // Get finger tip position in world frame
        gtsam::Matrix36 D_pworld_finger;
        gtsam::Point3 p_world = finger_pose.translation(H1 ? &D_pworld_finger : nullptr);

        // Transform finger point into object's local frame
        // transformTo: Hself is 3x6 (w.r.t. object pose), Hpoint is 3x3 (w.r.t. input point)
        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_pworld;
        gtsam::Point3 p_local = object_pose.transformTo(p_world,
            H2 ? &D_plocal_obj : nullptr,
            H1 ? &D_plocal_pworld : nullptr);

        // --- OpenVDB Trilinear Interpolation ---
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R vdb_pt(p_local.x(), p_local.y(), p_local.z());

        double d = sampler.wsSample(vdb_pt);
        if (d < 1e-7 && d > -1e-7) d = (d >= 0) ? 1e-7 : -1e-7; // Zero protection

        double alpha = 2000.0;
        double x = R_ - d;

        // Early exit if no contact (for sparsity)
        if (x < -0.002) {
            if (H1) *H1 = gtsam::Matrix::Zero(1, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(1, 6);
            return gtsam::Vector1(0.0);
        }

        double error_val, sigmoid;
        if (alpha * x > 50.0) {
            error_val = x;
            sigmoid = 1.0;
        } else {
            error_val = std::log(1.0 + std::exp(alpha * x)) / alpha;
            sigmoid = 1.0 / (1.0 + std::exp(-alpha * x));
        }

        if (H1 || H2) {
            // Calculate gradient (Normal Vector) using finite difference
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

            // Chain rule: de/dd = -sigmoid, dd/dp_local = n_local
            // de/dp_local = de/dd * dd/dp_local = -sigmoid * n_local^T  (1x3)
            double de_dd = -sigmoid;
            gtsam::Matrix13 de_dplocal = de_dd * n_local.transpose();

            // H1 (finger): de/dp_local * dp_local/dp_world * dp_world/d_finger
            //              (1x3)         (3x3)              (3x6)  = (1x6)
            if (H1) *H1 = de_dplocal * D_plocal_pworld * D_pworld_finger;

            // H2 (object): de/dp_local * dp_local/d_object
            //              (1x3)         (3x6)  = (1x6)
            if (H2) *H2 = de_dplocal * D_plocal_obj;
        }

        return gtsam::Vector1(error_val);
    }
};

}
