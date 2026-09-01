#pragma once

// Analytic sphere-sphere contact, 1-residual form.

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

// NOTE the Section 1.6 five-residual PlaneWitnessContactFactor (Eq 1.60-1.64)
// that used to live here has been REMOVED. The table sliding equality now
// constrains the contact sphere's CENTER directly, as the single residual
// PlaneCollisionGapFactor wrapped in a gtsam::ZeroCostConstraint (see
// HandModel::build_graph and the support_contact_node notes above).
// Four of the witness form's five rows existed only to pin the gauge of the
// free contact point it introduced; for a PLANE that point buys nothing, since
// a scalar residual on the center leaves no rotational freedom to brick the
// solver and still lets the tip slide laterally.


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

}  // namespace gepetto_solvers
