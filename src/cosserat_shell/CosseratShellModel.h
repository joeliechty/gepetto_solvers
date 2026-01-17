#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
    
#include "utils/Gaussians.h"


enum StressDir {
    X, Y,
    NUM_DIR
};


struct CosseratShellState {
    Pose3Gaussian pose;
    std::array<Vector6Gaussian, NUM_DIR> stress;
};


struct CosseratShellMarginals {
    std::vector<std::vector<CosseratShellState>> states;
    // Could add other things here later
};


class CosseratShellModel {
public:
    CosseratShellModel(
        int num_nodes_x,
        int num_nodes_y,
        double element_size,
        const gtsam::Matrix6& K_inv, 
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise);

    gtsam::NonlinearFactorGraph build_graph(
        const Pose3Gaussian& displacement) const;

    gtsam::Values get_initial_values() const;

    CosseratShellMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

private:
    const int num_nodes_x_;
    const int num_nodes_y_;
    const double element_size_;
    const gtsam::Matrix6 K_inv_;

    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    std::vector<std::vector<gtsam::Key>> pose_keys_;
    std::vector<std::vector<std::array<gtsam::Key, NUM_DIR>>> stress_keys_;
};