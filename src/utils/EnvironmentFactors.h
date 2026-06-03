#pragma once

#include <gtsam/base/Matrix.h>
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

    // Contact-as-goal terminal constraint (Eq 33-35). When target_contact_node
    // is set, the planner adds a hard equality constraint on that node — the
    // 3-residual SdfContactFactor [e1, e2, e3] wrapped in a
    // gtsam::ZeroCostConstraint — and solves with the Augmented Lagrangian
    // optimizer, which drives all three residuals exactly to zero. Convergence
    // is governed by the AL parameters on SolverBaseConfig, not a covariance.
    std::optional<int> target_contact_node;
    double contact_node_radius = 0.0;
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

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SdfCollisionFactor(*this)));
    }
};


// Surface-to-surface contact factor (Eq 30).
// Connects:
//   - node_pose_key  (Pose3)   : finger node whose sphere should touch the surface
//   - object_key     (Pose3)   : object pose
//   - point_key      (Point3)  : dummy contact point p_c in world frame
//
// 3D residual = [ ||p_c - p_i||_2 - R,
//                 SDF(T_obj^{-1} p_c),
//                 1 + N_i . N_obj ].
// Rows 0-1 drive p_c onto both the body sphere (radius R around the tip) and
// the object surface — tangential contact. Row 2 enforces antiparallel
// alignment of the body-sphere outward normal (N_i = (p_c - c_i)/||.||) and
// the object surface normal (N_obj = R_obj * normalize(grad SDF)). Together
// the three residuals fully constrain p_c, removing the 1-DoF sliding
// manifold that rows 0-1 alone leave behind — so no Tikhonov regularizer on
// p_c is required.
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

        if (H1 || H2 || H3) {
            if (H1) *H1 = gtsam::Matrix::Zero(3, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(3, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(3, 3);

            // Row 0: e1 -- only c_i (via node_pose) and p_c.
            if (H1) H1->row(0) = -n_i.transpose() * D_center_pose;
            if (H3) H3->row(0) =  n_i.transpose();

            // Row 1: e2 -- via p_local(object_pose, p_c).
            gtsam::Matrix13 de2_dplocal = n_obj_local.transpose();
            if (H2) H2->row(1) = de2_dplocal * D_plocal_obj;
            if (H3) H3->row(1) = de2_dplocal * D_plocal_point;

            // Row 2: e3. Chain rule with the locally-constant-gradient
            // approximation -- treat n_obj_local as p_c-independent (same
            // convention used by SdfCollisionFactor).
            // n_i = (p_c - c_i)/d, projector P = (I - n_i n_i^T)/d.
            // dn_i/dc_i = -P,   dn_i/dp_c = P.
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

        return gtsam::Vector3(e1, e2, e3);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SdfContactFactor(*this)));
    }
};


// Sphere-sphere contact factor (analytical, 1-residual gap form). Use when
// both bodies are spheres (e.g. finger vs. spherical primitive). Connects
// two Pose3 variables whose translations are the sphere centers:
//
//   e = ||c_a - c_b|| - (r_a + r_b)
//
// e == 0 means tangent contact; e > 0 separated; e < 0 inter-penetrating.
// Only the translations enter the residual; rotation Jacobian blocks are
// zero. Single-residual form avoids the rank-deficient slack subspace that
// the 3-residual (p_c-bearing) form introduces when both surfaces are
// analytic spheres -- p_c is uniquely determined by c_a, c_b, r_a, r_b and
// is not a real degree of freedom here.
class SphereSphereContactFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
private:
    double r_a_, r_b_;

public:
    SphereSphereContactFactor(gtsam::Key pose_a_key, gtsam::Key pose_b_key,
                              double r_a, double r_b,
                              const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor2(noise_model, pose_a_key, pose_b_key),
          r_a_(r_a), r_b_(r_b) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_a,
                                const gtsam::Pose3& pose_b,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2) const override
    {
        gtsam::Matrix36 D_ca_pose, D_cb_pose;
        gtsam::Point3 c_a = pose_a.translation(H1 ? &D_ca_pose : nullptr);
        gtsam::Point3 c_b = pose_b.translation(H2 ? &D_cb_pose : nullptr);

        gtsam::Vector3 d = c_a - c_b;
        double dn = d.norm();
        if (dn < 1e-7) dn = 1e-7;
        gtsam::Vector3 n = d / dn;       // unit vector from c_b toward c_a

        double e = dn - (r_a_ + r_b_);

        if (H1) *H1 =  n.transpose() * D_ca_pose;   // 1x6
        if (H2) *H2 = -n.transpose() * D_cb_pose;   // 1x6

        return (gtsam::Vector(1) << e).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SphereSphereContactFactor(*this)));
    }
};

} // namespace crest_sparse
