#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
    

struct CosseratShellMarginals {
    std::vector<std::vector<gtsam::Matrix4>> pose_mean;
    std::vector<std::vector<gtsam::Matrix6>> pose_cov;

    std::vector<std::vector<gtsam::Vector6>> stress_mean;
    std::vector<std::vector<gtsam::Matrix6>> stress_cov;

    std::vector<std::vector<gtsam::Vector6>> wrench_mean;
    std::vector<std::vector<gtsam::Matrix6>> wrench_cov;
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

    gtsam::NonlinearFactorGraph build_graph() const;

    gtsam::Values get_initial_values() const;

    CosseratShellMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

    gtsam::Key get_pose_key(int node_x_idx, int node_y_idx) const;

    gtsam::Key get_stress_key(int node_x_idx, int node_y_idx) const;
    
    gtsam::Key get_wrench_key(int node_x_idx, int node_y_idx) const;
    
    const std::vector<std::vector<gtsam::Key>>& get_wrench_keys() const;

private:
    int clamp_node_x_idx(int node_idx) const;

    int clamp_node_y_idx(int node_idx) const;

    const int num_nodes_x_;
    const int num_nodes_y_;
    const double element_size_;
    const gtsam::Matrix6 K_inv_;

    gtsam::SharedDiagonal twist_cov_;
    gtsam::SharedDiagonal stress_cov_;

    std::vector<std::vector<gtsam::Key>> pose_keys_;
    std::vector<std::vector<gtsam::Key>> stress_keys_;
    std::vector<std::vector<gtsam::Key>> wrench_keys_;
    gtsam::Key dummy_wrench_key_;
};
