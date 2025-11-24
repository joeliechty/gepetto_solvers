#pragma once

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