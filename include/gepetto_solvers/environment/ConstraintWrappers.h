#pragma once

// Wrapping a plain factor as a hard constraint for the AL optimizer.

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

// ---------------------------------------------------------------------------
// Collision-avoidance factors (Section 1.5).
//
// Collision is modeled as a hard inequality constraint c_pen(x) <= 0 handled
// natively by GTSAM's AugmentedLagrangianOptimizer -- NOT as a soft cubic /
// quadratic penalty (the old SdfCollisionFactor). Each geometry pair is split
// into two pieces:
//   * a "gap factor" (a plain NoiseModelFactor) whose *unwhitened* error is the
//     raw signed penetration depth c_pen(x) with analytical Jacobians, and
//   * a CollisionInequalityConstraint wrapper that presents that gap factor to
//     the AL solver as the one-sided constraint c_pen(x) <= 0.
//
// Under the AL two-loop formulation:
//   Free space  (c_pen <  0): inactive -- the constraint ramps to zero error
//                             and zero Jacobian, adding nothing to the linear
//                             system and preserving graph sparsity.
//   Collision   (c_pen >= 0): active -- the inner loop applies a smooth
//                             quadratic penalty pushing the sphere back to the
//                             surface; mu / lambda are updated in the outer loop.
// This mirrors gtsam::ZeroCostConstraint (which does the equality analogue for
// the terminal contact factors) but for the one-sided collision case.
// ---------------------------------------------------------------------------

// Wraps a scalar "gap factor" -- any NoiseModelFactor whose unwhitened error is
// a penetration depth c_pen(x) -- into the inequality constraint c_pen(x) <= 0
// consumed by GTSAM's AugmentedLagrangianOptimizer. The constraint's sigma and
// keys are inherited from the wrapped factor, exactly as gtsam::ZeroCostConstraint
// does for equality constraints. The base class supplies the ramp (inactive
// branch), the active() test, and the L2 penalty; we additionally expose the
// g(x)=0 equality form used to build the Lagrange-multiplier term.
class CollisionInequalityConstraint : public gtsam::NonlinearInequalityConstraint {
private:
    gtsam::NoiseModelFactor::shared_ptr gap_factor_;

public:
    explicit CollisionInequalityConstraint(const gtsam::NoiseModelFactor::shared_ptr& gap_factor)
        : gtsam::NonlinearInequalityConstraint(
              constrainedNoise(gap_factor->noiseModel()->sigmas()), gap_factor->keys()),
          gap_factor_(gap_factor) {}

    // g(x) = raw penetration depth; the base class ramps this to enforce <= 0.
    gtsam::Vector unwhitenedExpr(const gtsam::Values& x,
                                 gtsam::OptionalMatrixVecType H = nullptr) const override {
        return gap_factor_->unwhitenedError(x, H);
    }

    // The corresponding g(x) = 0 equality constraint, used by the AL optimizer
    // to build the Lagrange-multiplier (linear) term for this inequality.
    gtsam::NonlinearEqualityConstraint::shared_ptr createEqualityConstraint() const override {
        return std::make_shared<gtsam::ZeroCostConstraint>(gap_factor_);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new CollisionInequalityConstraint(*this)));
    }
};

}  // namespace gepetto_solvers
