#include "KnuckleBendFactor.h"
#include <cmath>

using namespace gtsam;

KnuckleBendFactor::KnuckleBendFactor(
    gtsam::Key key_d1,
    gtsam::Key key_d2,
    double z_bend,
    const gtsam::SharedNoiseModel& model)
    : Base(model, key_d1, key_d2), z_bend(z_bend)
{}

Vector KnuckleBendFactor::evaluateError(
    const Pose3& T_d1,
    const Pose3& T_d2,
    OptionalMatrixType H1,
    OptionalMatrixType H2) const
    {
        // get rotation between poses
        Matrix3 R = T_d1.rotation().between(T_d2.rotation()).matrix();

        // get the cos and sin of the y rotation from the rotation matrix
        double R02 = R(0, 2);
        double R22 = R(2, 2);

        // compute the bend angle from the rotation matrix elements
        double h_bend = std::atan2(R02, R22);

        // compute the error
        double e_val = h_bend - z_bend;

        // Normalize angular error to be strictly within [-pi, pi]
        while (e_val > M_PI) e_val -= 2.0 * M_PI;
        while (e_val < -M_PI) e_val += 2.0 * M_PI;

        Vector1 e;
        e(0) = e_val;

        // Denominator for the chain rule of atan2: x^2 + y^2
        double denom = R02 * R02 + R22 * R22;

        // If requested, calculate the Jacobian H1 (1x6 matrix) with respect to pose_d1
        if (H1) {
            Matrix16 H1_mat = Matrix16::Zero();
            
            // Prevent division by zero at singularity where Y axis rotation points perfectly along Z
            if (denom > 1e-6) { 
                // Analytical Jacobians for the 3 rotational DOFs
                H1_mat(0, 0) =  (R02 * R(1, 2)) / denom;
                H1_mat(0, 1) = -1.0;
                H1_mat(0, 2) =  (R22 * R(1, 2)) / denom;
            }
            // Translational DOFs remain 0 because rotation doesn't depend on translation
            *H1 = H1_mat;
        }

        // If requested, calculate the Jacobian H2 (1x6 matrix) with respect to pose_d2
        if (H2) {
            Matrix16 H2_mat = Matrix16::Zero();
            
            if (denom > 1e-6) {
                // Analytical Jacobians for the 3 rotational DOFs
                H2_mat(0, 0) = (R02 * R(2, 1) - R22 * R(0, 1)) / denom;
                H2_mat(0, 1) = (R22 * R(0, 0) - R02 * R(2, 0)) / denom;
                H2_mat(0, 2) =  0.0; 
            }
            // Translational DOFs remain 0
            *H2 = H2_mat;
        }

        return e;

    }
    