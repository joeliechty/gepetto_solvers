#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

/**
 * Planar-bending approximation for one rod segment.
 *
 * The Cosserat model treats the backbone as a rod that can bend in any direction
 * and twist freely. The physical finger cannot: its discs are keyed to the
 * backbone and its joints are flexures, so it bends about its local +y axis
 * (flexion -- the axis KnuckleBendFactor measures) and structurally reacts both
 * out-of-plane bending and torsion rather than deflecting.
 *
 * This factor states that. With xi = Log(R_i^T R_j) = [wx, wy, wz]:
 *
 *   wx  out-of-plane bending (about local x)  -- penalised
 *   wy  flexion              (about local y)  -- LEFT FREE
 *   wz  torsion              (about local z)  -- penalised
 *
 *   e = [ wx / ds, wz / ds ]
 *
 * Driving both to zero makes the centreline a planar curve (a curve whose
 * binormal never rotates cannot leave its osculating plane) and the material
 * frame twist-free, while leaving flexion entirely to the tendons. It is the
 * exact orthogonal complement of KnuckleBendFactor, which reads only wy.
 *
 * The two rows are given SEPARATE sigmas because they are not equally useful.
 * Measured across four grasp scenes, torsion is the cause and out-of-plane
 * bending is the symptom: the spiral-routed lateral tendons inject twist, twist
 * rotates the material frame, and the following segment's flexion then lands out
 * of plane. Tightening wz alone collapses wx with it at no cost in reach;
 * tightening wx directly buys the same planarity but fights the reach the rod
 * needs. See TendonFingerSolverConfig::sigma_planar_twist for the numbers.
 *
 * Dividing by the segment length ds makes the residual a CURVATURE (rad/m), the
 * same units as CosseratTwistFactor's rotation rows -- so the sigmas passed here
 * are directly comparable to sigma_twist_rot, and segments of different length
 * are weighted consistently.
 *
 * Relationship to the stiffness route: CosseratTwistFactor already carries
 * Log(T_i^-1 T_j)/ds - (K_inv S + nominal), so an anisotropic K_inv with
 * K_inv[0,0] -> 0 turns its x row into this same penalty, but pinned at weight
 * 1/sigma_twist_rot. This factor exists to make that weight independently
 * tunable without disturbing flexion accuracy. See
 * CosseratRodModel::set_planar_bending and get_K_inv()'s lateral_stiffness_scale.
 */
class PlanarBendFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
    double ds_;

public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

    /**
     * @param key_i  Key for the proximal node pose (T_i)
     * @param key_j  Key for the distal node pose (T_j)
     * @param ds     Length of the segment between them, metres
     * @param model  2-dimensional noise model: [sigma_bend, sigma_twist], rad/m
     */
    PlanarBendFactor(
        gtsam::Key key_i,
        gtsam::Key key_j,
        double ds,
        const gtsam::SharedNoiseModel& model);

    /** Error function: [wx, wz] / ds, and optionally the 2x6 Jacobians. */
    gtsam::Vector evaluateError(
        const gtsam::Pose3& T_i,
        const gtsam::Pose3& T_j,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2) const override;
};
