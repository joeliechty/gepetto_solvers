#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
    

struct CosseratRodMarginals {
    std::vector<gtsam::Matrix4> pose_mean;
    std::vector<gtsam::Matrix6> pose_cov;

    std::vector<gtsam::Vector6> stress_mean;
    std::vector<gtsam::Matrix6> stress_cov;

    std::vector<gtsam::Vector6> wrench_mean;
    std::vector<gtsam::Matrix6> wrench_cov;
};


class CosseratRodModel {
public:
    CosseratRodModel(
        int num_nodes, 
        const gtsam::Matrix6& K_inv, 
        gtsam::SharedDiagonal twist_noise,
        gtsam::SharedDiagonal stress_noise);

    gtsam::NonlinearFactorGraph build_graph(
        double rod_length,
        const std::optional<gtsam::Vector6>& nominal_strain = std::nullopt) const;

    gtsam::Values get_initial_values() const;

    CosseratRodMarginals get_marginals(
        const gtsam::Values& values, 
        const gtsam::Marginals& marginals) const;

    gtsam::Key get_pose_key(int node_idx) const;

    gtsam::Key get_stress_key(int node_idx) const;
    
    gtsam::Key get_wrench_key(int node_idx) const;
    
    const std::vector<gtsam::Key>& get_wrench_keys() const;

    const std::vector<gtsam::Key>& get_pose_keys() const;

private:
    int clamp_node_idx(int node_idx) const;
    
    // We need unique rod IDs for unique Keys
    const int id_;
    inline static int next_id_ = 0;

    const int num_nodes_;
    std::vector<gtsam::Matrix6> K_inv_;

    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    std::vector<gtsam::Key> pose_keys_;
    std::vector<gtsam::Key> stress_keys_;
    std::vector<gtsam::Key> wrench_keys_;
    gtsam::Key dummy_wrench_key_;
};
