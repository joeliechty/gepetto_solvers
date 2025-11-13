#pragma once 

#include "nonlinear/NonlinearFactor.h"
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>







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
