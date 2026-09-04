#pragma once

// In-plane clearance against an ellipsoid union.

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
#include "gepetto_solvers/environment/ConstraintWrappers.h"
#include "gepetto_solvers/environment/EllipsoidDistance.h"

namespace gepetto_solvers {

// Finger-ellipsoid-SET distance measured IN THE TENDON-ALIGNED PULLING PLANE
// (Section III-A.2, Eq 11 and Eq 13).
//
// A tendon can only pull its fingertip within the finger's own actuation plane. A
// contact constraint written on the full 3D distance pulls the tip along the
// object's global surface normal instead, which asks the solver for out-of-plane
// torsion the hand cannot produce. Eq 13 therefore measures contact against the
// 2D CROSS-SECTION the pulling plane cuts out of the object,
// G_planar = G intersect P_pull, rather than against the 3D surface.
//
// THE PLANE (Eq 11) is spanned by three points: the finger's metacarpal base
// p_base, the LIVE fingertip p_tip, and the multi-finger pinch centroid p_centroid:
//
//   n_pull = (p_tip - p_base) x (p_centroid - p_base) / || . ||
//
// p_base and p_centroid are constants in the WRIST frame (the pinch centroid is
// the offline-measured meeting point PreGraspCentroidFactor already uses), so they
// enter through the wrist variable -- hence the third key. That is also why the
// finger's own node-0 pose is NOT used for the base: under root reparameterization
// node 0 has no pose key of its own, the trap PreGraspCentroidFactor documents.
// p_tip is the node variable, so the plane MOVES with the finger and its
// derivative is carried in H1/H3 rather than frozen.
//
// THE DISTANCE. Eq 13 names d_planar but does not give its formula; this is the
// restriction of Eq 9 to the plane. Writing points in the plane as
// p(u,v) = p_tip + u e1 + v e2, the cross-section curve is F(u,v) = f(p(u,v)) = 0
// and its Taubin distance is F / ||grad_2D F||, with
// grad_2D F = [e1 e2]^T grad f -- i.e. the SAME first-order distance, with the
// gradient projected into the plane. With grad f = 2 M x that is
//
//   d_planar = (x^T M x - 1) / (2 || P M x ||),   P = I - n n^T
//
// so d_planar is the Taubin ratio with the gradient projected into the plane. The
// query point needs no projection: the plane is built THROUGH p_tip, so the tip
// lies in it by construction.
//
// THE taubin FLAG REACHES d3 ONLY. The 3D distance goes through EllipsoidDistance
// like every other ellipsoid factor -- exact orthogonal by default, Taubin under
// taubin=true -- because the whole point of the flag is that all of them measure
// with one metric, and because weight 0 here MUST still reduce exactly to
// EllipsoidSetCollisionGapFactor. d_planar stays the projected Taubin ratio in
// both modes: the exact orthogonal distance to the plane's CROSS-SECTION is a
// second root-find on a 2D conic whose axes depend on the plane normal, so its
// derivative w.r.t. n_pull -- which this factor carries live, not frozen -- is a
// different derivation rather than a reuse of the 3D one. In exact mode the blend
// therefore mixes a metre with a first-order metre; they agree to first order at
// the surface, which is where the contact equality lives, and the blend already
// mixes two different distances (3D and in-plane) in either mode.
//
// Since ||P grad f|| <= ||grad f||, d_planar >= d_geom always. That inequality is
// exactly why Eq 12's COLLISION must keep the full 3D distance and is never
// projected: an inequality on the larger number would report clearance while the
// finger is really inside the object.
//
// TWO FALLBACKS, one mechanism. Both are smoothstep blends back to the plain 3D
// distance, and both carry their derivative (freezing it leaves the row wrong
// through the whole blend band, which is where the interesting postures are).
//
//  1. THE PLANE MISSES THE MEMBER. Then G_planar is empty for it and there is no
//     in-plane distance to report, so fall back to the 3D minimum distance. The
//     test is the support ratio
//
//       rho = |n . x| / sqrt(n^T M^-1 n)     ( >= 1  <=>  plane misses )
//
//     -- cheap because the plane passes through the query point, so the plane's
//     offset from the member's centre along n IS n . x, and the ellipsoid's
//     support in that direction is sqrt(n^T M^-1 n) with M^-1 = diag(a^2,b^2,c^2).
//     Blended over [rho_lo, rho_hi] rather than switched at rho = 1, so a finger
//     whose plane slides off the object keeps a continuous value and gradient.
//     This also tames d_planar's own singularity: ||P M x|| -> 0 (the query point
//     at the cross-section's centre) is approached as the section shrinks, and the
//     rho blend is already weighting that member out by then.
//
//  2. THE PLANE IS UNDEFINED. Eq 11's cross product vanishes when tip, base and
//     centroid go collinear -- which is precisely what a closing PINCH does, since
//     the tips converge on the centroid. The measure of it is the tip's standoff
//     from the base->centroid axis, ||n|| / ||p_centroid - p_base||, a length in
//     metres (the denominator is constant). Below gap_lo the normal is numerically
//     arbitrary, so the factor reports the ordinary 3D distance rather than a
//     confident number derived from a direction that means nothing.
//
// Per member: weight_k = mu * (1 - lambda_k), "the plane is well defined AND it
// cuts this member". weight 0 recovers EllipsoidSetCollisionGapFactor exactly.
//
// Keys: [node_pose_key, object_key, wrist_key].  Residual: Vector1,
//   c_pen = r_i - d_set     ( > 0 <=> penetration )
// -- the same sign as EllipsoidSetCollisionGapFactor, so the value can be wrapped
// in gtsam::ZeroCostConstraint to get Eq 13 (its zero set is d_set = r_i) exactly
// as the 3D form is today. K = 1 with an identity local_pose covers a single
// analytic ellipsoid (Eq 9), so there is no separate single-ellipsoid class.
class EllipsoidSetPlanarGapFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3> {
public:
    // Per-member breakdown of one evaluation, for visualization and tests. The
    // factor is the single definition of this geometry, so a caller that wants to
    // DRAW it asks the factor rather than re-deriving it and risking an overlay
    // that shows a plane the solve never used.
    struct Report {
        std::vector<double> d3;        // per member, full 3D Taubin distance
        std::vector<double> d_planar;  // per member, in-plane Taubin distance
        std::vector<double> rho;       // per member, support ratio (>=1 => missed)
        std::vector<double> lambda;    // per member, 3D blend weight from rho
        std::vector<double> weight;    // per member, planar weight mu*(1-lambda)
        double d_set = 0.0;            // the fused (LogSumExp) distance
        double mu = 0.0;               // how well defined the plane is, in [0,1]
        double axis_gap = 0.0;         // tip standoff from the base->centroid axis (m)
        gtsam::Vector3 normal = gtsam::Vector3::Zero();   // n_pull, world frame
    };

private:
    double radius_;
    double beta_;
    std::vector<EllipsoidDistance> metric_;   // per-k signed distance field
    std::vector<gtsam::Matrix3> Rk_T_;        // per-k R_k^T, precomputed
    std::vector<gtsam::Vector3> tk_;          // per-k t_k
    gtsam::Point3 base_local_;                // p_base,     WRIST frame
    gtsam::Point3 centroid_local_;            // p_centroid, WRIST frame
    double a_norm_;                           // ||p_centroid - p_base||, constant
    double rho_lo_, rho_hi_;
    double gap_lo_, gap_hi_;

public:
    EllipsoidSetPlanarGapFactor(gtsam::Key node_pose_key, gtsam::Key object_key,
                                gtsam::Key wrist_key,
                                double radius,
                                const std::vector<EllipsoidPrimitive>& ellipsoids,
                                double beta,
                                const gtsam::Point3& base_local,
                                const gtsam::Point3& centroid_local,
                                const gtsam::SharedNoiseModel& noise_model,
                                double rho_lo = 0.90, double rho_hi = 1.00,
                                double gap_lo = 0.002, double gap_hi = 0.010,
                                bool taubin = false)
        : NoiseModelFactorN(noise_model, node_pose_key, object_key, wrist_key),
          radius_(radius), beta_(beta),
          base_local_(base_local), centroid_local_(centroid_local),
          rho_lo_(rho_lo), rho_hi_(rho_hi), gap_lo_(gap_lo), gap_hi_(gap_hi)
    {
        // Same silent-garbage cases EllipsoidSetCollisionGapFactor rejects, plus the
        // two blend bands and the plane's own axis -- an inverted band would run the
        // blend backwards, and a zero-length axis has no plane at any posture.
        if (ellipsoids.empty())
            throw std::invalid_argument(
                "EllipsoidSetPlanarGapFactor: the ellipsoid set is empty");
        if (!(beta > 0.0))
            throw std::invalid_argument(
                "EllipsoidSetPlanarGapFactor: beta must be > 0 (got " +
                std::to_string(beta) + ")");
        if (!(rho_hi > rho_lo))
            throw std::invalid_argument(
                "EllipsoidSetPlanarGapFactor: need rho_hi > rho_lo");
        if (!(gap_hi > gap_lo) || !(gap_lo > 0.0))
            throw std::invalid_argument(
                "EllipsoidSetPlanarGapFactor: need 0 < gap_lo < gap_hi");

        a_norm_ = (centroid_local_ - base_local_).norm();
        if (!(a_norm_ > 0.0))
            throw std::invalid_argument(
                "EllipsoidSetPlanarGapFactor: base_local and centroid_local coincide, "
                "so Eq 11 has no plane axis");

        metric_.reserve(ellipsoids.size());
        Rk_T_.reserve(ellipsoids.size());
        tk_.reserve(ellipsoids.size());
        for (size_t k = 0; k < ellipsoids.size(); ++k) {
            const gtsam::Vector3& a = ellipsoids[k].semi_axes;
            if (!(a.x() > 0.0 && a.y() > 0.0 && a.z() > 0.0))
                throw std::invalid_argument(
                    "EllipsoidSetPlanarGapFactor: ellipsoid " + std::to_string(k) +
                    " has a non-positive semi-axis");
            metric_.emplace_back(a, taubin);
            Rk_T_.push_back(ellipsoids[k].local_pose.rotation().matrix().transpose());
            tk_.push_back(ellipsoids[k].local_pose.translation());
        }
    }

    // The fused distance and, optionally, its Jacobians and the per-member
    // breakdown. evaluateError() and report() both route through here so the
    // number that gets drawn cannot drift from the number the solver would see.
    double distance(const gtsam::Pose3& node_pose,
                    const gtsam::Pose3& object_pose,
                    const gtsam::Pose3& wrist_pose,
                    gtsam::Matrix16* H_node = nullptr,
                    gtsam::Matrix16* H_object = nullptr,
                    gtsam::Matrix16* H_wrist = nullptr,
                    Report* report = nullptr) const
    {
        const bool need_jac = (H_node || H_object || H_wrist);
        const gtsam::Matrix3 I3 = gtsam::Matrix3::Identity();

        // --- The pulling plane (Eq 11) ---------------------------------------
        gtsam::Matrix36 D_pw_node;
        gtsam::Point3 p_w = node_pose.translation(need_jac ? &D_pw_node : nullptr);

        gtsam::Matrix36 D_bw_wrist, D_cw_wrist;
        gtsam::Point3 b_w = wrist_pose.transformFrom(
            base_local_, need_jac ? &D_bw_wrist : nullptr);
        gtsam::Point3 c_w = wrist_pose.transformFrom(
            centroid_local_, need_jac ? &D_cw_wrist : nullptr);

        const gtsam::Vector3 u = p_w - b_w;      // p_tip - p_base
        const gtsam::Vector3 a = c_w - b_w;      // p_centroid - p_base, ||a|| constant
        const gtsam::Vector3 n_vec = u.cross(a);
        const double nrm = n_vec.norm();
        // Floored only to keep the arithmetic finite where the plane is undefined;
        // mu is exactly 0 there, so every term this direction feeds is multiplied
        // by zero and the floor cannot leak into the result.
        const double nrm_safe = (nrm > 1e-12) ? nrm : 1e-12;
        const gtsam::Vector3 nhat = n_vec / nrm_safe;

        const double axis_gap = nrm / a_norm_;
        double dS_dt = 0.0;
        const double mu = smoothstep01((axis_gap - gap_lo_) / (gap_hi_ - gap_lo_), &dS_dt);
        const double dmu_dgap = dS_dt / (gap_hi_ - gap_lo_);

        // --- Into the object, and the plane normal with it ---------------------
        gtsam::Matrix36 D_pobj_obj;
        gtsam::Matrix33 D_pobj_pw;
        gtsam::Point3 p_obj = object_pose.transformTo(p_w,
            need_jac ? &D_pobj_obj : nullptr,
            need_jac ? &D_pobj_pw  : nullptr);

        gtsam::Matrix36 D_R_obj;
        const gtsam::Rot3& R_obj = object_pose.rotation(need_jac ? &D_R_obj : nullptr);
        gtsam::Matrix33 D_nobj_R, D_nobj_nhat;
        gtsam::Vector3 n_obj = R_obj.unrotate(nhat,
            need_jac ? &D_nobj_R    : nullptr,
            need_jac ? &D_nobj_nhat : nullptr);

        const size_t K = metric_.size();
        std::vector<double> d(K);
        std::vector<gtsam::Matrix13> dd_dpobj(need_jac ? K : 0);
        std::vector<gtsam::Matrix13> dd_dnobj(need_jac ? K : 0);
        std::vector<double> dd_dmu(need_jac ? K : 0);
        if (report) {
            report->d3.assign(K, 0.0);
            report->d_planar.assign(K, 0.0);
            report->rho.assign(K, 0.0);
            report->lambda.assign(K, 0.0);
            report->weight.assign(K, 0.0);
        }

        double d_min = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < K; ++k) {
            const gtsam::Vector3& m_diag    = metric_[k].m_diag();
            const gtsam::Vector3& minv_diag = metric_[k].minv_diag();

            const gtsam::Vector3 x  = Rk_T_[k] * (p_obj - tk_[k]);   // x_k
            const gtsam::Vector3 nk = Rk_T_[k] * n_obj;              // plane normal, member frame
            const gtsam::Vector3 Mx = m_diag.cwiseProduct(x);

            const double f  = x.dot(Mx) - 1.0;

            // Eq 9, through the shared metric: exact orthogonal, or Taubin.
            gtsam::Matrix13 dd3_dx;
            const double d3 = metric_[k].signed_distance(
                x, need_jac ? &dd3_dx : nullptr);

            // In-plane: the Taubin ratio with the gradient projected into the plane.
            const double sn = nk.dot(Mx);
            const gtsam::Vector3 q = Mx - sn * nk;                   // P M x
            double gP = q.norm();
            if (gP < 1e-9) gP = 1e-9;
            const double dP = f / (2.0 * gP);                        // d_planar

            // Support test: does the plane reach this member at all?
            const double s = nk.dot(x);
            const gtsam::Vector3 Minv_nk = minv_diag.cwiseProduct(nk);
            double den = std::sqrt(nk.dot(Minv_nk));
            if (den < 1e-12) den = 1e-12;
            const double rho = std::abs(s) / den;

            double dlam_dt = 0.0;
            const double lam =
                smoothstep01((rho - rho_lo_) / (rho_hi_ - rho_lo_), &dlam_dt);
            const double dlam_drho = dlam_dt / (rho_hi_ - rho_lo_);

            const double w = mu * (1.0 - lam);     // planar weight
            d[k] = (1.0 - w) * d3 + w * dP;
            if (d[k] < d_min) d_min = d[k];

            if (report) {
                report->d3[k] = d3;
                report->d_planar[k] = dP;
                report->rho[k] = rho;
                report->lambda[k] = lam;
                report->weight[k] = w;
            }

            if (need_jac) {
                // d d_planar / dx. dd3_dx came back from the metric above; this is
                // the same Taubin form with ||M x|| replaced by the projected ||q||
                // and M(M x) by M q.
                const gtsam::Vector3 mq = m_diag.cwiseProduct(q);
                const gtsam::Matrix13 ddP_dx =
                      (Mx.transpose() / gP)
                    - (f / (2.0 * gP * gP * gP)) * mq.transpose();

                // d dP / d nk. dq/dnk = -(nk (Mx)^T + sn I), and q . nk = 0 (q is the
                // in-plane part of Mx), which kills the first term:
                //   dgP/dnk = -sn q^T / gP
                const gtsam::Matrix13 ddP_dnk =
                    (f * sn / (2.0 * gP * gP * gP)) * q.transpose();

                const double sgn = (s >= 0.0) ? 1.0 : -1.0;
                const gtsam::Matrix13 drho_dx  = (sgn / den) * nk.transpose();
                const gtsam::Matrix13 drho_dnk =
                      (sgn / den) * x.transpose()
                    - (rho / (den * den)) * Minv_nk.transpose();

                // w = mu (1 - lambda(rho)), so dw = -mu dlambda -- the derivative the
                // earlier attempt dropped, at the cost of ~100% row error in the band.
                const gtsam::Matrix13 dw_dx  = (-mu * dlam_drho) * drho_dx;
                const gtsam::Matrix13 dw_dnk = (-mu * dlam_drho) * drho_dnk;

                dd_dpobj[k] = ((1.0 - w) * dd3_dx + w * ddP_dx
                               + (dP - d3) * dw_dx) * Rk_T_[k];
                dd_dnobj[k] = (w * ddP_dnk + (dP - d3) * dw_dnk) * Rk_T_[k];
                dd_dmu[k]   = (dP - d3) * (1.0 - lam);
            }
        }

        // --- LogSumExp smooth min (Eq 10), shifted by d_min --------------------
        double s_sum = 0.0;
        std::vector<double> sm(need_jac ? K : 0);
        for (size_t k = 0; k < K; ++k) {
            const double e = std::exp(-beta_ * (d[k] - d_min));
            if (need_jac) sm[k] = e;
            s_sum += e;
        }
        const double d_set = d_min - std::log(s_sum) / beta_;

        if (report) {
            report->d_set = d_set;
            report->mu = mu;
            report->axis_gap = axis_gap;
            report->normal = (mu > 0.0) ? nhat : gtsam::Vector3::Zero();
        }

        if (need_jac) {
            gtsam::Matrix13 A_x = gtsam::Matrix13::Zero();
            gtsam::Matrix13 A_n = gtsam::Matrix13::Zero();
            double A_mu = 0.0;
            for (size_t k = 0; k < K; ++k) {
                const double wk = sm[k] / s_sum;    // softmin weights, sum to 1
                A_x  += wk * dd_dpobj[k];
                A_n  += wk * dd_dnobj[k];
                A_mu += wk * dd_dmu[k];
            }

            // The plane's own chain: everything that reaches the residual through the
            // normal, collected as one row in n's space.
            const gtsam::Matrix13 dgap_dn = nhat.transpose() / a_norm_;
            const gtsam::Matrix33 dnhat_dn = (I3 - nhat * nhat.transpose()) / nrm_safe;
            const gtsam::Matrix13 B_n =
                A_n * D_nobj_nhat * dnhat_dn + (A_mu * dmu_dgap) * dgap_dn;

            // n = u x a, with du/dp_tip = I, du/dwrist = -D_bw, da/dwrist = D_cw - D_bw.
            const gtsam::Matrix3 skew_a = gtsam::skewSymmetric(a);
            const gtsam::Matrix3 skew_u = gtsam::skewSymmetric(u);
            const gtsam::Matrix33 dn_dpw = -skew_a;
            const gtsam::Matrix36 dn_dwrist =
                skew_a * D_bw_wrist + skew_u * (D_cw_wrist - D_bw_wrist);

            if (H_node)
                *H_node = (A_x * D_pobj_pw + B_n * dn_dpw) * D_pw_node;
            if (H_object)
                *H_object = A_x * D_pobj_obj + A_n * (D_nobj_R * D_R_obj);
            if (H_wrist)
                *H_wrist = B_n * dn_dwrist;
        }

        return d_set;
    }

    // The per-member breakdown at one state, with no Jacobians -- what the
    // visualizer overlay reads.
    Report report(const gtsam::Pose3& node_pose,
                  const gtsam::Pose3& object_pose,
                  const gtsam::Pose3& wrist_pose) const {
        Report r;
        distance(node_pose, object_pose, wrist_pose, nullptr, nullptr, nullptr, &r);
        return r;
    }

    gtsam::Vector evaluateError(const gtsam::Pose3& node_pose,
                                const gtsam::Pose3& object_pose,
                                const gtsam::Pose3& wrist_pose,
                                gtsam::OptionalMatrixType H1,
                                gtsam::OptionalMatrixType H2,
                                gtsam::OptionalMatrixType H3) const override
    {
        gtsam::Matrix16 Hd_node, Hd_obj, Hd_wrist;
        const double d_set = distance(node_pose, object_pose, wrist_pose,
                                      H1 ? &Hd_node  : nullptr,
                                      H2 ? &Hd_obj   : nullptr,
                                      H3 ? &Hd_wrist : nullptr);
        // c_pen = r - d_set, so every block flips sign off the distance's.
        if (H1) *H1 = -Hd_node;
        if (H2) *H2 = -Hd_obj;
        if (H3) *H3 = -Hd_wrist;
        return gtsam::Vector1(radius_ - d_set);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new EllipsoidSetPlanarGapFactor(*this)));
    }
};

}  // namespace gepetto_solvers
