#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace crest_sparse {

    class SphereContactFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3> {
    private:
        double R_; // combined radius threshold for contact (r1 + r2)

    public:
        SphereContactFactor(gtsam::Key key1, gtsam::Key key2, double r1, double r2, gtsam::SharedNoiseModel noiseModel)
            : NoiseModelFactorN(noiseModel, key1, key2), R_(r1 + r2) {}

        gtsam::Vector evaluateError(const gtsam::Pose3& pose1, const gtsam::Pose3& pose2,
                                    gtsam::OptionalMatrixType H1,
                                    gtsam::OptionalMatrixType H2) const override
        {
            gtsam::Matrix36 D_p1_pose1, D_p2_pose2;
            gtsam::Point3 p1 = pose1.translation(H1 ? &D_p1_pose1 : nullptr);
            gtsam::Point3 p2 = pose2.translation(H2 ? &D_p2_pose2 : nullptr);

            double d = gtsam::distance3(p1, p2);
            if (d < 1e-7) d = 1e-7; // Protect against divide-by-zero on perfect overlap

            double alpha = 2000.0; // Stiff transition: cushion is ~2mm thick
            double x = R_ - d;     // Positive when penetrating

            // --- EARLY EXIT (Maintains Matrix Sparsity) ---
            // If distance is safely outside the 2mm soft cushion, return exactly zero.
            // This allows GTSAM's sparse solvers to bypass unnecessary math.
            if (x < -0.002) {
                if (H1) *H1 = gtsam::Matrix::Zero(1, 6);
                if (H2) *H2 = gtsam::Matrix::Zero(1, 6);
                return gtsam::Vector1::Zero();
            }

            double error_val;
            double sigmoid;

            // --- OVERFLOW PROTECTION ---
            // std::exp(alpha * x) overflows easily. For large x, softplus(x) asymptotes
            // exactly to x, and the sigmoid derivative asymptotes exactly to 1.0.
            if (alpha * x > 50.0) {
                error_val = x;
                sigmoid = 1.0;
            } else {
                error_val = std::log(1.0 + std::exp(alpha * x)) / alpha;
                sigmoid = 1.0 / (1.0 + std::exp(-alpha * x));
            }

            if (H1 || H2) {
                gtsam::Vector3 n = (p1 - p2) / d; // Unit vector from p2 to p1

                // Derivative of error with respect to distance d
                double de_dd = -sigmoid;

                if (H1) *H1 = de_dd * n.transpose() * D_p1_pose1;
                if (H2) *H2 = de_dd * (-n.transpose()) * D_p2_pose2;
            }

            return gtsam::Vector1(error_val);
        }

    };

}

