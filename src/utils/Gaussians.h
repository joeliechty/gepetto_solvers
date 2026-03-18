#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>


template<int N>
struct VectorNGaussian {
    Eigen::Vector<double, N> mean;
    Eigen::Matrix<double, N, N> cov;
};

using Vector4Gaussian = VectorNGaussian<4>;


struct VectorXGaussian {
    Eigen::VectorXd mean;
    Eigen::MatrixXd cov;
};


struct Vector3Gaussian {
    gtsam::Vector3 mean;
    gtsam::Matrix3 cov;
};


struct Vector6Gaussian {
    gtsam::Vector6 mean;
    gtsam::Matrix6 cov;
};


struct Pose3Gaussian {
    gtsam::Matrix4 mean;
    gtsam::Matrix6 cov;
};
