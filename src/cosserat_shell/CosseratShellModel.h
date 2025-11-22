#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
    

enum StressDir {
    X, Y,
    NUM_DIR
};


struct CosseratShellMarginals {
    std::vector<std::vector<gtsam::Matrix4>> pose_mean;
    std::vector<std::vector<gtsam::Matrix6>> pose_cov;

    std::vector<std::vector<std::array<gtsam::Vector6, NUM_DIR>>> stress_mean;
    std::vector<std::vector<std::array<gtsam::Matrix6, NUM_DIR>>> stress_cov;
};


class CosseratShellModel {
public:
    CosseratShellModel(
        int num_nodes_x,
        int num_nodes_y,
        double element_size,
        const gtsam::Matrix6& K_inv, 
        gtsam::SharedDiagonal twist_cov,
        gtsam::SharedDiagonal stress_cov);

    gtsam::NonlinearFactorGraph build_graph(
        const gtsam::Matrix4& displacement_mean,
        const gtsam::Matrix6& displacement_cov) const;

    gtsam::Values get_initial_values() const;

    CosseratShellMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

private:
    const int num_nodes_x_;
    const int num_nodes_y_;
    const double element_size_;
    const gtsam::Matrix6 K_inv_;

    gtsam::SharedDiagonal twist_cov_;
    gtsam::SharedDiagonal stress_cov_;

    std::vector<std::vector<gtsam::Key>> pose_keys_;
    std::vector<std::vector<std::array<gtsam::Key, NUM_DIR>>> stress_keys_;
};