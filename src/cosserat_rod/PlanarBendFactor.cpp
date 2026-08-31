#include "gepetto_solvers/cosserat_rod/PlanarBendFactor.h"

#include <gtsam/base/Matrix.h>
#include <stdexcept>

using namespace gtsam;


PlanarBendFactor::PlanarBendFactor(
    Key key_i,
    Key key_j,
    double ds,
    const SharedNoiseModel& model)
:
    Base(model, key_i, key_j),
    ds_(ds)
{
    if (ds <= 0.0)
        throw std::runtime_error("PlanarBendFactor: ds must be positive");
}


Vector PlanarBendFactor::evaluateError(
    const Pose3& T_i,
    const Pose3& T_j,
    OptionalMatrixType H1,
    OptionalMatrixType H2) const
{
    // Chain: T -> R -> R_rel -> xi. Each step's Jacobian comes from GTSAM rather
    // than being written out by hand, so the manifold conventions stay GTSAM's.
    Matrix36 d_Ri_d_Ti, d_Rj_d_Tj;
    const Rot3 R_i = T_i.rotation(d_Ri_d_Ti);
    const Rot3 R_j = T_j.rotation(d_Rj_d_Tj);

    Matrix3 d_rel_d_Ri, d_rel_d_Rj;
    const Rot3 R_rel = R_i.between(R_j, d_rel_d_Ri, d_rel_d_Rj);

    // Logmap is singular at |xi| = pi. Per-segment relative rotations are small
    // here (tens of segments over a ~13 cm finger), so no guard is warranted --
    // a segment that close to a half turn is a diverged solve, not a pose.
    Matrix3 d_xi_d_rel;
    const Vector3 xi = Rot3::Logmap(R_rel, d_xi_d_rel);

    // Rows 0 and 2 only: out-of-plane bending and torsion. Row 1 (flexion) is
    // deliberately absent -- that is the DOF the tendons drive.
    Vector2 e;
    e << xi(0) / ds_, xi(2) / ds_;

    // The translation columns of d_R_d_T are supplied by GTSAM and are generally
    // non-zero (Pose3's retraction couples the two), so they are kept rather than
    // zeroed the way KnuckleBendFactor does.
    if (H1) {
        const Matrix36 d_xi_d_Ti = d_xi_d_rel * d_rel_d_Ri * d_Ri_d_Ti;
        Matrix26 J;
        J.row(0) = d_xi_d_Ti.row(0) / ds_;
        J.row(1) = d_xi_d_Ti.row(2) / ds_;
        *H1 = J;
    }

    if (H2) {
        const Matrix36 d_xi_d_Tj = d_xi_d_rel * d_rel_d_Rj * d_Rj_d_Tj;
        Matrix26 J;
        J.row(0) = d_xi_d_Tj.row(0) / ds_;
        J.row(1) = d_xi_d_Tj.row(2) / ds_;
        *H2 = J;
    }

    return e;
}
