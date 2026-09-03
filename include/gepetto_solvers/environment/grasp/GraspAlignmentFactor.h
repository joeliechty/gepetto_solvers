#pragma once

// Net virtual-wrench equilibrium over a set of witness points (Eq h_grasp).

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

namespace gepetto_solvers {

// Geometric grasp alignment: the contacts must surround the object, not merely
// touch it.
//
//   h_grasp({p_i}, T_obj) = sum_i [        -n_i               ]  = 0   (Vector6)
//                                 [ -(p_i - t_obj) x n_i      ]
//
// with n_i = R_obj * normalize(grad Phi_obj(T_obj^{-1} p_i)) -- the object's
// outward surface normal at witness point p_i, pushed into the world frame.
//
// This is a purely KINEMATIC equilibrium. Every contact is credited with one
// UNIT "virtual force" along the inward normal (-n_i); the constraint says those
// unit forces, and the torques they generate about the object origin t_obj, sum
// to zero. No mass, no friction cone, no force magnitudes: driving this 6-vector
// to zero is a statement about where the contacts SIT on the surface, and it is
// satisfied only when the normals geometrically oppose one another -- which is
// what makes the posture a grasp rather than |C| independent touches. It is the
// piece the per-contact witness factors cannot express, because each of those
// sees exactly one contact.
//
// Because the forces are unit and the normals are unit, the residual is
// dimensionless in its top three rows and has units of length in its bottom
// three (a moment arm). Size the noise model accordingly: for a centimetre-scale
// object the two halves are within an order of magnitude of each other, but a
// large object wants the torque rows loosened relative to the force rows.
//
// Variable arity: |C| is runtime-determined, so -- like PreGraspHandCenteringFactor
// and TendonLengthFactor -- this derives from gtsam::NoiseModelFactor directly
// (not NoiseModelFactorN) and hand-builds its KeyVector.
//
// Keys: [point_key_0, ..., point_key_{|C|-1}, object_key].  Residual: Vector6.
//
// Wrap in gtsam::ZeroCostConstraint to hand it to the AL optimizer as the
// equality h_grasp = 0, exactly as the witness contact factors are wrapped.
//
// NORMAL DERIVATIVES ARE *NOT* DROPPED HERE. The sibling contact factors hold
// the surface normal constant within a Gauss-Newton step (the locally-constant-
// gradient convention). That is sound for them -- their residuals are distances,
// which have a first-order dependence on the point that survives the frozen
// normal. It would be fatal here: with dn_i/dp_i == 0 the force rows have an
// IDENTICALLY ZERO Jacobian wrt every witness point, so the solver would be
// told that sliding a contact around the object cannot change the force
// balance, and the constraint would only ever act on the object pose. So this
// factor differentiates the normal field too, via the finite-difference shape
// operator dn/dp (see curvature_step below).
class GraspAlignmentFactor : public gtsam::NoiseModelFactor {
private:
    using Sampler = openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>;

    openvdb::FloatGrid::Ptr sdf_grid_;
    size_t num_contacts_;
    double grad_step_;       // inner FD step: grad Phi   (default 0.5 * voxel)
    double curvature_step_;  // outer FD step: dn/dp      (default 0.5 * voxel)

public:
    using NoiseModelFactor::unwhitenedError;

    // BOTH finite-difference steps default (0.0 sentinel) to HALF THE GRID
    // VOXEL SIZE, and the default matters more than it looks like it should: a
    // sub-voxel stencil on a voxel grid measures interpolation artefact rather
    // than geometry. The trilinear interpolant is (multi)linear inside a voxel
    // -- its gradient is constant there and discontinuous across voxel faces,
    // and its second derivative inside a voxel is identically zero -- so a
    // stencil narrower than a voxel resolves that cell's own linear fit instead
    // of the surface. Stepping wider than a voxel instead pays the usual
    // truncation error, quadratically in h.
    //
    // Both measured on a 5 cm level-set sphere against the closed-form normal
    // field, over three grid resolutions:
    //
    //   gradient_step (error in the 6-vector residual)
    //     voxel     h = 1e-4 (sub-voxel)     h = 0.5 * voxel
    //     5.0 mm         6.1e-2                  1.1e-3
    //     2.5 mm         3.8e-2                  1.4e-4
    //     1.0 mm         8.2e-3                  3.1e-5
    //
    //   curvature_step (max relative error in the Jacobian blocks, 2.5 mm voxel)
    //     0.25 vox   0.5 vox    1 vox     2 vox     4 vox     8 vox
    //     1.1e-2     4.7e-4     1.7e-3    5.3e-3    1.9e-2    7.2e-2
    //
    // Two to three orders of magnitude on the residual, and this factor is far
    // more sensitive to it than the witness contact factors are: there the
    // gradient only orients a tangent basis, here the normals ARE the residual.
    // Raise either step on a noisy grid; the cost is a smoothed normal /
    // curvature, which for a constraint Jacobian is the safe direction to err in.
    GraspAlignmentFactor(const std::vector<gtsam::Key>& point_keys,
                         gtsam::Key object_key,
                         const openvdb::FloatGrid::Ptr& sdf_grid,
                         const gtsam::SharedNoiseModel& noise_model,
                         double curvature_step = 0.0,
                         double gradient_step = 0.0)
        : NoiseModelFactor(noise_model, /* keys: */ [&]() {
              gtsam::KeyVector keys(point_keys.begin(), point_keys.end());
              keys.push_back(object_key);
              return keys;
          }()),
          sdf_grid_(sdf_grid),
          num_contacts_(point_keys.size()),
          grad_step_(gradient_step),
          curvature_step_(curvature_step)
    {
        if (!sdf_grid_)
            throw std::invalid_argument("GraspAlignmentFactor: null SDF grid");
        if (num_contacts_ == 0)
            throw std::invalid_argument("GraspAlignmentFactor: no contact points");
        const double voxel = sdf_grid_->voxelSize()[0];
        if (grad_step_ <= 0.0)
            grad_step_ = (voxel > 0.0) ? 0.5 * voxel : 1e-3;
        if (curvature_step_ <= 0.0)
            curvature_step_ = (voxel > 0.0) ? 0.5 * voxel : 1e-3;
    }

    gtsam::Vector unwhitenedError(const gtsam::Values& x,
                                  gtsam::OptionalMatrixVecType H = nullptr) const override
    {
        Sampler sampler(*sdf_grid_);

        // Key layout: [0 .. |C|-1] = witness points, [|C|] = object.
        const gtsam::Pose3& object_pose = x.at<gtsam::Pose3>(keys()[num_contacts_]);
        gtsam::Matrix36 D_tobj_pose;
        gtsam::Point3 t_obj = object_pose.translation(H ? &D_tobj_pose : nullptr);
        const gtsam::Matrix3 R_obj = object_pose.rotation().matrix();

        gtsam::Vector6 e = gtsam::Vector6::Zero();
        if (H) {
            H->resize(num_contacts_ + 1);
            (*H)[num_contacts_] = gtsam::Matrix::Zero(6, 6);
        }

        for (size_t i = 0; i < num_contacts_; ++i) {
            const gtsam::Point3& p_i = x.at<gtsam::Point3>(keys()[i]);

            // p_local = T_obj^{-1} p_i, with d(p_local)/d(xi_obj) and
            // d(p_local)/d(p_i) = R_obj^T.
            gtsam::Matrix36 D_plocal_obj;
            gtsam::Matrix3 D_plocal_point;
            gtsam::Point3 p_local = object_pose.transformTo(
                p_i, H ? &D_plocal_obj : nullptr, H ? &D_plocal_point : nullptr);

            const gtsam::Vector3 n_local = normal_local(sampler, p_local);
            const gtsam::Vector3 n_i = R_obj * n_local;   // world outward normal
            const gtsam::Vector3 a_i = p_i - t_obj;       // moment arm

            // Unit inward virtual force, and its torque about the object origin.
            e.head<3>() += -n_i;
            e.tail<3>() += -a_i.cross(n_i);

            if (!H) continue;

            // Shape operator in the object-local frame, pushed to the world:
            //   M = d(n_i)/d(p_i) = R_obj * (dn_local/dp_local) * R_obj^T.
            const gtsam::Matrix3 G = dnormal_dlocal(sampler, p_local);
            const gtsam::Matrix3 M = R_obj * G * D_plocal_point;

            // r_top = -n,  r_bot = -(a x n) = skew(n) a = -skew(a) n
            //   d(r_bot) = skew(n) da - skew(a) dn.
            const gtsam::Matrix3 skew_n = gtsam::skewSymmetric(n_i);
            const gtsam::Matrix3 skew_a = gtsam::skewSymmetric(a_i);

            gtsam::Matrix H_i = gtsam::Matrix::Zero(6, 3);
            H_i.block<3, 3>(0, 0) = -M;                    // da/dp = I
            H_i.block<3, 3>(3, 0) = skew_n - skew_a * M;
            (*H)[i] = H_i;

            // d(n_i)/d(xi_obj), GTSAM Pose3 tangent order [omega(3), upsilon(3)]:
            // the normal turns with the object -- d(R v)/d(omega) = -R skew(v) --
            // AND the sample point slides in the local frame, which the shape
            // operator picks up through d(p_local)/d(xi_obj).
            gtsam::Matrix36 dn_dxi = gtsam::Matrix36::Zero();
            dn_dxi.block<3, 3>(0, 0) = -R_obj * gtsam::skewSymmetric(n_local);
            dn_dxi += R_obj * G * D_plocal_obj;

            // d(a_i)/d(xi_obj) = -d(t_obj)/d(xi_obj).
            (*H)[num_contacts_].block<3, 6>(0, 0) += -dn_dxi;
            (*H)[num_contacts_].block<3, 6>(3, 0) +=
                -skew_n * D_tobj_pose - skew_a * dn_dxi;
        }

        return e;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new GraspAlignmentFactor(*this)));
    }

private:
    // n_local(p) = normalize(grad Phi(p)), central differences in the object
    // frame -- the same gradient the witness contact factors build their
    // C-frame from. Degenerate gradients (a flat region of the grid, or a point
    // outside the narrow band) fall back to +Z rather than exploding, matching
    // SdfWitnessContactFactor.
    gtsam::Vector3 normal_local(const Sampler& sampler, const gtsam::Vector3& p) const {
        const double h = grad_step_;
        const double dx = sampler.wsSample(openvdb::Vec3R(p.x() + h, p.y(), p.z())) -
                          sampler.wsSample(openvdb::Vec3R(p.x() - h, p.y(), p.z()));
        const double dy = sampler.wsSample(openvdb::Vec3R(p.x(), p.y() + h, p.z())) -
                          sampler.wsSample(openvdb::Vec3R(p.x(), p.y() - h, p.z()));
        const double dz = sampler.wsSample(openvdb::Vec3R(p.x(), p.y(), p.z() + h)) -
                          sampler.wsSample(openvdb::Vec3R(p.x(), p.y(), p.z() - h));
        gtsam::Vector3 g(dx, dy, dz);
        const double norm = g.norm();
        if (norm > 1e-8) return g / norm;
        return gtsam::Vector3(0.0, 0.0, 1.0);
    }

    // dn_local/dp_local by central differences of the normal field itself,
    // rather than by assembling grad^2 Phi and projecting. Same arithmetic to
    // first order, but differencing the already-normalized field keeps the
    // (I - n n^T) projection exact instead of relying on |grad Phi| == 1, which
    // a resampled or clipped grid does not honour.
    gtsam::Matrix3 dnormal_dlocal(const Sampler& sampler, const gtsam::Vector3& p) const {
        const double h = curvature_step_;
        gtsam::Matrix3 G;
        for (int j = 0; j < 3; ++j) {
            gtsam::Vector3 dp = gtsam::Vector3::Zero();
            dp(j) = h;
            G.col(j) = (normal_local(sampler, p + dp) - normal_local(sampler, p - dp)) / (2.0 * h);
        }
        return G;
    }
};

}  // namespace gepetto_solvers
