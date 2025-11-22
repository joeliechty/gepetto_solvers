#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>

#include "utils/SolverBase.h"


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


struct CosseratShellSolverConfig {
    double num_nodes_x;
    double num_nodes_y;
    double element_size;

    gtsam::Matrix6 K_inv;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_stress_force;
    double sigma_stress_moment;
};


class CosseratShellSolver : SolverBase {
public:
    CosseratShellSolver(const CosseratShellSolverConfig& config);

    Solution<CosseratShellMarginals> solve(
        const gtsam::Matrix4& displacement_mean,
        const gtsam::Matrix6& displacement_cov);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    const int num_nodes_x_;
    const int num_nodes_y_;
    const double element_size_;
    const gtsam::Matrix6 K_inv_;

    gtsam::Matrix4 displacement_mean_;
    gtsam::Matrix6 displacement_cov_;

    gtsam::SharedDiagonal twist_noise_;
    gtsam::SharedDiagonal stress_noise_;

    std::vector<std::vector<gtsam::Key>> pose_keys_;
    std::vector<std::vector<std::array<gtsam::Key, NUM_DIR>>> stress_keys_;

    CosseratShellMarginals extracted_;
};