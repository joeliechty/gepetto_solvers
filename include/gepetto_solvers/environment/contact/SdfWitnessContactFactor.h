#pragma once

// Fingertip on an SDF surface, witness form (Eq 1.27-1.31).

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/constrained/NonlinearInequalityConstraint.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "gepetto_solvers/environment/EnvironmentConfig.h"
#include "gepetto_solvers/environment/TangentBasis.h"

namespace gepetto_solvers {

// Surface-to-surface witness-point contact factor (Eq 30-31).
// Connects:
//   - node_pose_key  (Pose3)   : finger node whose sphere should touch the surface
//   - object_key     (Pose3)   : object pose
//   - point_key      (Point3)  : dummy contact point p_c in world frame
//
// 5D residual = [ ||p_c - p_i||_2 - R,            (c_R)
//                 SDF(T_obj^{-1} p_c),            (c_O)
//                 1 + N_i . N_obj,                (c_N)
//                 (p_c - p_i) . t1(N_obj),        (c_T1)
//                 (p_c - p_i) . t2(N_obj) ].      (c_T2)
// Rows 0-1 drive p_c onto both the body sphere (radius R around the tip) and
// the object surface — tangential contact. Row 2 enforces antiparallel
// alignment of the body-sphere outward normal (N_i = (p_c - c_i)/||.||) and
// the object surface normal (N_obj = R_obj * normalize(grad SDF)). Rows 3-4
// are the C-frame gauge-fixing residuals: t1, t2 span the object surface's
// tangent plane (Frisvad basis of N_obj), so penalizing the projection of
// (p_c - p_i) onto them pins p_c strictly along the contact normal axis. This
// removes the residual gauge freedom of p_c and yields a full-rank gradient,
// so no Tikhonov regularizer / stabilizing prior on p_c is required.
//
// drop_normal_row (Section 1.8, Eq 1.107-1.110): when true the c_N row is omitted
// and the residual is the 4-vector [c_R, c_O, c_T1, c_T2]. Justified in §1.8:
// because the robot's collision geometry is modeled exclusively as spheres, the
// tangential-slip rows already force (p_c - p_i) -- and hence the outward sphere
// normal -- collinear with the object surface normal, making c_N redundant. The
// caller must size its noise model to match (Isotropic::Sigma(4, ...)).
class SdfWitnessContactFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3> {
private:
    double R_;
    openvdb::FloatGrid::Ptr sdf_grid_;
    bool drop_normal_row_;

public:
    SdfWitnessContactFactor(gtsam::Key node_pose_key, gtsam::Key object_key, gtsam::Key point_key,
                            double radius, const openvdb::FloatGrid::Ptr& sdf_grid,
                            const gtsam::SharedNoiseModel& noise_model,
                            bool drop_normal_row = false)
        : NoiseModelFactor3(noise_model, node_pose_key, object_key, point_key),
          R_(radius), sdf_grid_(sdf_grid), drop_normal_row_(drop_normal_row) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        // --- e1 = ||p_c - c_i|| - R --------------------------------------
        gtsam::Matrix36 D_center_pose;
        gtsam::Point3 center = node_pose.translation(H1 ? &D_center_pose : nullptr);

        gtsam::Vector3 diff = dummy_point - center;
        double d = diff.norm();
        if (d < 1e-7) d = 1e-7;
        double e1 = d - R_;
        gtsam::Vector3 n_i = diff / d;  // body-sphere outward normal (world frame)

        // --- e2 = SDF(T_obj^{-1} p_c) ------------------------------------
        gtsam::Matrix36 D_plocal_obj;
        gtsam::Matrix33 D_plocal_point;
        gtsam::Point3 p_local = object_pose.transformTo(dummy_point,
            (H2 || H3) ? &D_plocal_obj   : nullptr,
            (H2 || H3) ? &D_plocal_point : nullptr);

        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*sdf_grid_);
        openvdb::Vec3R vdb_pt(p_local.x(), p_local.y(), p_local.z());
        double e2 = sampler.wsSample(vdb_pt);

        // FD SDF gradient in object-local frame. Reused for e2 Jacobian and
        // to build N_obj for e3.
        double h = 1e-4;
        double dx = sampler.wsSample(openvdb::Vec3R(vdb_pt.x() + h, vdb_pt.y(), vdb_pt.z())) -
                    sampler.wsSample(openvdb::Vec3R(vdb_pt.x() - h, vdb_pt.y(), vdb_pt.z()));
        double dy = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() + h, vdb_pt.z())) -
                    sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y() - h, vdb_pt.z()));
        double dz = sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() + h)) -
                    sampler.wsSample(openvdb::Vec3R(vdb_pt.x(), vdb_pt.y(), vdb_pt.z() - h));
        gtsam::Vector3 n_obj_local(dx, dy, dz);
        double g_norm = n_obj_local.norm();
        if (g_norm > 1e-8) n_obj_local /= g_norm;
        else               n_obj_local = gtsam::Vector3(0.0, 0.0, 1.0);

        // --- e3 = 1 + N_i . N_obj_world ----------------------------------
        gtsam::Matrix3 R_obj = object_pose.rotation().matrix();
        gtsam::Vector3 n_obj_world = R_obj * n_obj_local;
        double e3 = 1.0 + n_i.dot(n_obj_world);

        // --- e4, e5 = C-frame gauge fixing (Eq 30-31) --------------------
        // Build a deterministic tangent basis (t1, t2) of the object surface
        // normal and penalize the projection of v = (p_c - c_i) onto it, so
        // p_c is pinned along the contact normal axis. t1, t2 are treated as
        // constant within the local Gauss-Newton step (C-frame held fixed --
        // standard SOTA contact convention), so their Jacobian contribution
        // reduces to the tangent vectors themselves.
        gtsam::Vector3 t1, t2;
        frisvad_tangent_basis(n_obj_world, t1, t2);
        gtsam::Vector3 v = diff;  // p_c - c_i
        double e4 = v.dot(t1);
        double e5 = v.dot(t2);

        // Row layout: [c_R, c_O, (c_N), c_T1, c_T2]. Dropping c_N shifts the two
        // tangent rows up by one and shrinks the residual to 4.
        const int dim = drop_normal_row_ ? 4 : 5;
        const int rT1 = drop_normal_row_ ? 2 : 3;

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(dim, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(dim, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(dim, 3);

            // Row 0: e1 -- only c_i (via node_pose) and p_c.
            if (H1) H1->row(0) = -n_i.transpose() * D_center_pose;
            if (H3) H3->row(0) =  n_i.transpose();

            // Row 1: e2 -- via p_local(object_pose, p_c).
            gtsam::Matrix13 de2_dplocal = n_obj_local.transpose();
            if (H2) H2->row(1) = de2_dplocal * D_plocal_obj;
            if (H3) H3->row(1) = de2_dplocal * D_plocal_point;

            // Row 2: e3. Chain rule with the locally-constant-gradient
            // approximation -- treat n_obj_local as p_c-independent (the
            // standard locally-constant-gradient contact convention).
            // n_i = (p_c - c_i)/d, projector P = (I - n_i n_i^T)/d.
            // dn_i/dc_i = -P,   dn_i/dp_c = P.
            if (!drop_normal_row_) {
                const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
                gtsam::Matrix3 P = (I3 - n_i * n_i.transpose()) / d;
                Eigen::RowVector3d nobjT = n_obj_world.transpose();

                if (H1) H1->row(2) = -(nobjT * P) * D_center_pose;
                if (H3) H3->row(2) =  nobjT * P;

                // d(n_obj_world)/d(xi_obj) under GTSAM's Pose3 tangent order
                // [omega(3), upsilon(3)]:  d(R v)/d(omega) = -R * skew(v).
                // Translation has no effect on the normal under the
                // locally-constant-gradient approximation.
                if (H2) {
                    gtsam::Matrix3 dRv_dxiR = -R_obj * gtsam::skewSymmetric(n_obj_local);
                    H2->block<1, 3>(2, 0) = n_i.transpose() * dRv_dxiR;
                    H2->block<1, 3>(2, 3) = Eigen::RowVector3d::Zero();
                }
            }

            // Tangent rows: e4, e5. With t1, t2 constant, de/dp_c = t^T and
            // de/dc_i = -t^T (v = p_c - c_i). Object pose has no effect on v
            // under the fixed-C-frame approximation, so those H2 rows stay zero.
            if (H3) H3->row(rT1)     = t1.transpose();
            if (H3) H3->row(rT1 + 1) = t2.transpose();
            if (H1) H1->row(rT1)     = -t1.transpose() * D_center_pose;
            if (H1) H1->row(rT1 + 1) = -t2.transpose() * D_center_pose;
        }

        if (drop_normal_row_)
            return (gtsam::Vector(4) << e1, e2, e4, e5).finished();
        return (gtsam::Vector(5) << e1, e2, e3, e4, e5).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SdfWitnessContactFactor(*this)));
    }
};

}  // namespace gepetto_solvers
