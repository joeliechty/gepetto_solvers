#pragma once 

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using CosseratTwistBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector6>;

class CosseratRodTwistFactor: public CosseratTwistBase {
    using CosseratTwistBase::evaluateError;

public:
    CosseratRodTwistFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        double segment_length,
        const gtsam::Matrix6& K_inv,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_0, 
        const gtsam::Pose3& pose_1, 
        const gtsam::Vector6& stress_0, 
        const gtsam::Vector6& stress_1, 
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4) const override;

private:
    double ds_;
    gtsam::Matrix66 K_inv_;
};


gtsam::Vector6 transform_wrench_adjoint(
        const gtsam::Vector6& wrench_0,
        const gtsam::Pose3& pose_0,
        const gtsam::Pose3& pose,
        gtsam::OptionalJacobian<6, 6> H_wrench_0 = {},
        gtsam::OptionalJacobian<6, 6> H_pose_0 = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {});


gtsam::Vector6 spatial_to_body_wrench(
        const gtsam::Vector6& wrench_spatial, 
        const gtsam::Pose3& pose, 
        gtsam::OptionalJacobian<6, 6> H_wrench,
        gtsam::OptionalJacobian<6, 6> H_pose);


using CosseratStressBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector6, gtsam::Vector6>;

class CosseratRodStressFactor: public CosseratStressBase {
    using CosseratStressBase::evaluateError;

public:
    CosseratRodStressFactor(
        gtsam::Key pose_0_key,
        gtsam::Key pose_1_key,
        gtsam::Key stress_0_key,
        gtsam::Key stress_1_key,
        gtsam::Key wrench_key,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_0, 
        const gtsam::Pose3& pose_1, 
        const gtsam::Vector6& stress_0, 
        const gtsam::Vector6& stress_1, 
        const gtsam::Vector6& wrench_1,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4,
        gtsam::OptionalMatrixType H5) const override;
};


using TipStressBase = gtsam::NoiseModelFactorN<gtsam::Vector6, gtsam::Vector6, gtsam::Pose3>;

class TipStressWrenchFactor: public TipStressBase {
    using TipStressBase::evaluateError;

public:
    TipStressWrenchFactor(
        gtsam::Key tip_stress_key,
        gtsam::Key tip_wrench_key,
        gtsam::Key tip_pose_key,
        const gtsam::SharedNoiseModel& model);
        
    gtsam::Vector evaluateError(
        const gtsam::Vector6& stress, 
        const gtsam::Vector6& wrench,
        const gtsam::Pose3& pose,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3) const override;
};






using TendonWrenchBase = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Vector6, gtsam::Vector4, gtsam::Vector6>;

class TendonDiscWrenchFactor: public TendonWrenchBase {
    using TendonWrenchBase::evaluateError;

public:
    TendonDiscWrenchFactor(
        gtsam::Key pose_prev_key,
        gtsam::Key pose_key,
        gtsam::Key pose_next_key, // Set to dummy key if we are at the tip
        gtsam::Key wrench_key,
        gtsam::Key tensions_key,
        gtsam::Key external_wrench_key,
        const bool is_tip,
        const std::vector<gtsam::Point3>& holes_prev,
        const std::vector<gtsam::Point3>& holes,
        const std::vector<gtsam::Point3>& holes_next, // Not used if we are at the tip
        const gtsam::SharedNoiseModel& model);
        
    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose_prev, 
        const gtsam::Pose3& pose, 
        const gtsam::Pose3& pose_next, 
        const gtsam::Vector6& wrench, 
        const gtsam::Vector4& tensions,
        const gtsam::Vector6& wrench_external,
        gtsam::OptionalMatrixType H1, 
        gtsam::OptionalMatrixType H2, 
        gtsam::OptionalMatrixType H3, 
        gtsam::OptionalMatrixType H4, 
        gtsam::OptionalMatrixType H5,
        gtsam::OptionalMatrixType H6) const override;

private:
    gtsam::Vector6 get_single_tendon_wrench(
        const double tension, 
        const gtsam::Pose3& pose, 
        const gtsam::Pose3& pose_other, 
        const gtsam::Point3& hole, 
        const gtsam::Point3& hole_other,
        gtsam::OptionalJacobian<6, 1> H_tension = {},
        gtsam::OptionalJacobian<6, 6> H_pose = {},
        gtsam::OptionalJacobian<6, 6> H_pose_other = {}) const;

    bool is_tip_;
    std::vector<gtsam::Point3> holes_prev_;  // Previous disc hole location in local frame of previous disc, z = 0
    std::vector<gtsam::Point3> holes_;       // Tip disc hole locations in local frame of tip disc
    std::vector<gtsam::Point3> holes_next_;  // Disc hole locations in the next frame 
};




// Vector3 stress_to_fbg_signal(const Vector6& stress, const Matrix6& K_inv, const double rod_diameter, OptionalJacobian<3, 6> H_stress = {}) {
//     Vector6 strain = K_inv * stress;

//     Vector3 du = strain.head<3>();
//     Vector3 dv = strain.tail<3>();       // gamma_z

//     std::array<Vector3, 3> fbg_locations = {
//         rod_diameter * Point3(0, 1, 0),                                 // 0°
//         rod_diameter * Point3(std::sqrt(3)/2, -0.5, 0),                 // +120°
//         rod_diameter * Point3(-std::sqrt(3)/2, -0.5, 0)                 // -120°
//     };

//     Vector3 signal;
//     Matrix36 d_signal_d_strain;
//     d_signal_d_strain.setZero();

//     for (int i = 0; i < 3; ++i) {
//         Matrix3 d_strain_z_d_du;
//         Vector3 strain_z = dv + cross(du, fbg_locations[i], H_stress ? &d_strain_z_d_du : 0);

//         Matrix3 d_strain_z_d_dv = Matrix3::Identity();

//         signal(i) = strain_z.z();
//         d_signal_d_strain.block<1,3>(i,0) = d_strain_z_d_du.row(2); // w.r.t du
//         d_signal_d_strain.block<1,3>(i,3) = d_strain_z_d_dv.row(2); // w.r.t dv
//     }

//     if (H_stress) {
//         *H_stress = d_signal_d_strain * K_inv;
//     }

//     return signal;
// }

// class FbgMeasurementFactor: public NoiseModelFactorN<Vector6> {
//     Vector3 fbg_meas_;
//     Matrix6 K_inv_;
//     double rod_diameter_;
// public:

//     using NoiseModelFactorN<Vector6>::evaluateError;
  
//     FbgMeasurementFactor(Key stress_key,
//                          const Vector3& fbg_meas,
//                          const Matrix6& K_inv,
//                          const double rod_diameter, 
//                          const SharedNoiseModel& model): 
//         NoiseModelFactorN(model, stress_key), fbg_meas_(fbg_meas), K_inv_(K_inv), rod_diameter_(rod_diameter) {}

//     Vector evaluateError(const Vector6& stress, OptionalMatrixType H1) const override {
        
//         Matrix36 d_error_d_stress;
//         Vector3 signal_pred = stress_to_fbg_signal(stress, K_inv_, rod_diameter_, H1 ? &d_error_d_stress : 0);
//         Vector3 error = signal_pred - fbg_meas_;

//         if (H1) {
//             *H1 = d_error_d_stress;

//             // Matrix36 H1_check = numericalDerivative11<Vector3, Vector6>(
//             //     [&](const Vector6& stress_) {
//             //         return this->evaluateError(stress_, nullptr);
//             //     }, stress);

//             // std::cout << "H1 check: " << (*H1 - H1_check).cwiseAbs().maxCoeff() << std::endl;
//         }

//         return error;
//     }
// };

// class DistLoadSmoothingFactor: public NoiseModelFactorN<Vector6, Vector6, Vector6, Vector6> {
// public:
//     using NoiseModelFactorN<Vector6, Vector6, Vector6, Vector6>::evaluateError;
  
//     DistLoadSmoothingFactor(Key wrench_i_key,
//                             Key wrench_ip1_key,
//                             Key wrench_ip2_key,
//                             Key wrench_ip3_key,
//                             const SharedNoiseModel& model): 
//         NoiseModelFactor4(model, wrench_i_key, wrench_ip1_key, wrench_ip2_key, wrench_ip3_key) {}

//     Vector evaluateError(
//         const Vector6& wrench_i,
//         const Vector6& wrench_ip1,
//         const Vector6& wrench_ip2,
//         const Vector6& wrench_ip3,
//         OptionalMatrixType H1, 
//         OptionalMatrixType H2,
//         OptionalMatrixType H3,
//         OptionalMatrixType H4) const override 
//     {  
//         Vector3 f_i = wrench_i.tail<3>();
//         Vector3 f_ip1 = wrench_ip1.tail<3>();
//         Vector3 f_ip2 = wrench_ip2.tail<3>();
//         Vector3 f_ip3 = wrench_ip3.tail<3>();

//         Vector3 jerk = f_ip3 - 3.0 * f_ip2 + 3.0 * f_ip1 - f_i;

//         if (H1) {
//             *H1 = Matrix::Zero(3, 6);
//             H1->block<3,3>(0, 3) = -Matrix3::Identity();
//         }
//         if (H2) {
//             *H2 = Matrix::Zero(3, 6);
//             H2->block<3,3>(0, 3) = 3.0 * Matrix3::Identity();
//         }
//         if (H3) {
//             *H3 = Matrix::Zero(3, 6);
//             H3->block<3,3>(0, 3) = -3.0 * Matrix3::Identity();
//         }
//         if (H4) {
//             *H4 = Matrix::Zero(3, 6);
//             H4->block<3,3>(0, 3) = Matrix3::Identity();
//         }

//         return jerk;
//     }
// };

class PositionMeasurementFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3> {
    gtsam::Vector3 position_meas_;

public:
    using gtsam::NoiseModelFactorN<gtsam::Pose3>::evaluateError;
  
    PositionMeasurementFactor(
        gtsam::Key pose_key,
        gtsam::Vector3 position_meas,
        const gtsam::SharedNoiseModel& model);

    gtsam::Vector evaluateError(const gtsam::Pose3& pose, gtsam::OptionalMatrixType H1) const override;
};