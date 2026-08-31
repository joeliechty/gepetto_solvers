#pragma once

// Analytic sphere-sphere contact, 5-residual form.

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

// 5-residual sphere-to-sphere witness-point contact factor (Eq 30-31).
// Serves as an analytical counterpart of the 1-DoF SphereSphereContactFactor
// and the SDF-backed SdfWitnessContactFactor.
//
// Connects:
//   - pose_a_key  (Pose3)  : Body A (e.g., finger node)
//   - pose_b_key  (Pose3)  : Body B (e.g., primitive sphere -- the contacted object)
//   - point_key   (Point3) : Dummy contact point p_c in world frame
//
// 5D residual = [ ||p_c - c_a|| - r_a,        (c_R)
//                 ||p_c - c_b|| - r_b,        (c_O)
//                 1 + N_a . N_b,              (c_N)
//                 (p_c - c_a) . t1(N_b),      (c_T1)
//                 (p_c - c_a) . t2(N_b) ].    (c_T2)
// Rows 3-4 are the C-frame gauge-fixing residuals: t1, t2 span the tangent
// plane of body B's outward normal N_b (the contacted object), so penalizing
// the projection of (p_c - c_a) onto them pins p_c along the contact normal
// axis. This removes the genuine 1-DoF gauge freedom that rows 0-2 alone leave
// behind (rotating p_c about the center-to-center axis is invariant to them),
// so no stabilizing prior on p_c is required.
class SphereWitnessContactFactor
    : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
{
private:
    double r_a_, r_b_;

public:
    SphereWitnessContactFactor(gtsam::Key pose_a_key, gtsam::Key pose_b_key, gtsam::Key point_key,
                               double r_a, double r_b,
                               const gtsam::SharedNoiseModel& noise_model)
        : NoiseModelFactor3(noise_model, pose_a_key, pose_b_key, point_key),
          r_a_(r_a), r_b_(r_b) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose_a,
                                const gtsam::Pose3& pose_b,
                                const gtsam::Point3& dummy_point,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        // --- Get centers and their Jacobians wrt Pose ---
        gtsam::Matrix36 D_ca_pose, D_cb_pose;
        gtsam::Point3 c_a = pose_a.translation(H1 ? &D_ca_pose : nullptr);
        gtsam::Point3 c_b = pose_b.translation(H2 ? &D_cb_pose : nullptr);

        // --- Sphere A Geometry ---
        gtsam::Vector3 d_a = dummy_point - c_a;
        double norm_a = d_a.norm();
        if (norm_a < 1e-7) norm_a = 1e-7;
        gtsam::Vector3 n_a = d_a / norm_a; // Outward unit normal from A

        // --- Sphere B Geometry ---
        gtsam::Vector3 d_b = dummy_point - c_b;
        double norm_b = d_b.norm();
        if (norm_b < 1e-7) norm_b = 1e-7;
        gtsam::Vector3 n_b = d_b / norm_b; // Outward unit normal from B

        // --- Residuals ---
        double e1 = norm_a - r_a_;
        double e2 = norm_b - r_b_;
        double e3 = 1.0 + n_a.dot(n_b);

        // --- e4, e5 = C-frame gauge fixing (Eq 30-31) --------------------
        // Tangent basis of body B's normal (the contacted object), used to pin
        // p_c along the contact normal axis. t1, t2 held constant within the
        // local Gauss-Newton step (C-frame fixed), so their Jacobian reduces to
        // the tangent vectors themselves.
        gtsam::Vector3 t1, t2;
        frisvad_tangent_basis(n_b, t1, t2);
        gtsam::Vector3 v = d_a;  // p_c - c_a
        double e4 = v.dot(t1);
        double e5 = v.dot(t2);

        // --- Jacobians ---
        if (H1 || H2 || H3) {
            // GTSAM passes these in as default-constructed 0x0 matrices
            // (see NoiseModelFactor::linearize: `std::vector<Matrix> A(size())`).
            // We must ASSIGN a correctly-sized matrix to resize the storage --
            // H->setZero() does NOT resize a dynamic Eigen matrix, so the later
            // H->row()/H->block() writes would scribble past a 0-byte allocation
            // and corrupt the heap.
            if (H1) *H1 = gtsam::Matrix::Zero(5, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(5, 6);
            if (H3) *H3 = gtsam::Matrix::Zero(5, 3);

            // Row 0: e1 (Tangent to Sphere A)
            // de1/dc_a = -n_a^T, de1/dp_c = n_a^T
            if (H1) H1->row(0) = -n_a.transpose() * D_ca_pose;
            if (H3) H3->row(0) =  n_a.transpose();

            // Row 1: e2 (Tangent to Sphere B)
            // de2/dc_b = -n_b^T, de2/dp_c = n_b^T
            if (H2) H2->row(1) = -n_b.transpose() * D_cb_pose;
            if (H3) H3->row(1) =  n_b.transpose();

            // // Row 2: e3 (Normal alignment)
            // // Projectors for unit vectors: P = (I - n*n^T)/norm
            // const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
            // gtsam::Matrix3 P_a = (I3 - n_a * n_a.transpose()) / norm_a;
            // gtsam::Matrix3 P_b = (I3 - n_b * n_b.transpose()) / norm_b;

            // // dn_a/dp_c = P_a,  dn_a/dc_a = -P_a
            // // dn_b/dp_c = P_b,  dn_b/dc_b = -P_b
            // Eigen::RowVector3d naT = n_a.transpose();
            // Eigen::RowVector3d nbT = n_b.transpose();

            // // Chain rule: de3 = n_b^T * dn_a + n_a^T * dn_b
            // if (H1) H1->row(2) = (-nbT * P_a) * D_ca_pose;
            // if (H2) H2->row(2) = (-naT * P_b) * D_cb_pose;
            // if (H3) H3->row(2) =  nbT * P_a + naT * P_b;
            // Row 2: e3 (Normal alignment)
            const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();
            gtsam::Matrix3 P_a = (I3 - n_a * n_a.transpose()) / norm_a;
            gtsam::Matrix3 P_b = (I3 - n_b * n_b.transpose()) / norm_b;

            // Force evaluation into concrete 1x3 matrices to prevent lazy-evaluation memory aliasing
            gtsam::Matrix13 de3_dna = n_b.transpose();
            gtsam::Matrix13 de3_dnb = n_a.transpose();

            gtsam::Matrix13 de3_dca = -de3_dna * P_a;
            gtsam::Matrix13 de3_dcb = -de3_dnb * P_b;

            if (H1) H1->row(2) = de3_dca * D_ca_pose;
            if (H2) H2->row(2) = de3_dcb * D_cb_pose;
            if (H3) H3->row(2) = -de3_dca - de3_dcb; // Note: dn_a/dp_c = P_a, so de3/dp_c = de3_dna*P_a = -de3_dca

            // Rows 3-4: e4, e5. With t1, t2 constant, de/dp_c = t^T and
            // de/dc_a = -t^T (v = p_c - c_a). Body B has no effect on v under
            // the fixed-C-frame approximation, so H2 rows 3-4 stay zero.
            if (H3) H3->row(3) = t1.transpose();
            if (H3) H3->row(4) = t2.transpose();
            if (H1) H1->row(3) = -t1.transpose() * D_ca_pose;
            if (H1) H1->row(4) = -t2.transpose() * D_ca_pose;
        }

        return (gtsam::Vector(5) << e1, e2, e3, e4, e5).finished();
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new SphereWitnessContactFactor(*this)));
    }
};

}  // namespace gepetto_solvers
