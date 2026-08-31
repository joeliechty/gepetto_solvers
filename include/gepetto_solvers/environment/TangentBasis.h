#pragma once

// Local frames and smooth blends shared by the environment factors.

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

namespace gepetto_solvers {

// Frisvad/Hughes-Moller Householder basis: maps +Z onto the unit normal n and
// returns the two orthonormal tangent vectors spanning n's tangent plane. This
// is the explicitly-unrolled Householder reflection -- a few lines of arithmetic
// with no matrix allocation, suitable for a factor evaluated thousands of times
// per second. The single singularity at the south pole (n ~ -Z) is handled
// explicitly. Used by the witness-point contact factors to deterministically
// build the local Contact Frame (Section 3, Eq 30-31).
inline void frisvad_tangent_basis(const gtsam::Vector3& n,
                                  gtsam::Vector3& t1, gtsam::Vector3& t2) {
    if (n.z() < -0.9999999) {
        t1 = gtsam::Vector3( 0.0, -1.0,  0.0);
        t2 = gtsam::Vector3(-1.0,  0.0,  0.0);
    } else {
        const double a = 1.0 / (1.0 + n.z());
        const double b = -n.x() * n.y() * a;
        t1 = gtsam::Vector3(1.0 - n.x() * n.x() * a, b, -n.x());
        t2 = gtsam::Vector3(b, 1.0 - n.y() * n.y() * a, -n.y());
    }
}

// Cubic smoothstep on [0, 1] with its derivative: S(0)=0, S(1)=1, S'(0)=S'(1)=0,
// clamped outside. Used to blend between two distance definitions without giving
// the solver a step to chatter on -- a hard switch is discontinuous in the value
// AND the gradient, which an Augmented Lagrangian inner loop sees as a cliff.
inline double smoothstep01(double t, double* dS_dt = nullptr) {
    if (t <= 0.0) { if (dS_dt) *dS_dt = 0.0; return 0.0; }
    if (t >= 1.0) { if (dS_dt) *dS_dt = 0.0; return 1.0; }
    if (dS_dt) *dS_dt = 6.0 * t * (1.0 - t);
    return t * t * (3.0 - 2.0 * t);
}

}  // namespace gepetto_solvers
