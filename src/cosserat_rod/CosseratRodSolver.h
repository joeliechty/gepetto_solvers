#pragma once

#include <gtsam/base/Vector.h>

#include "cosserat_rod/CosseratRodModel.h"
#include "utils/SolverBase.h"


struct CosseratRodSolverConfig {
    double rod_length;
    int num_nodes;

    gtsam::Matrix6 K_inv;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_small_force;
    double sigma_small_moment;

    double sigma_base_pose_pos;
    double sigma_base_pose_rot;
};



class CosseratRodSolver : public SolverBase {
public:
    CosseratRodSolver(const CosseratRodSolverConfig& config);

    Solution<CosseratRodMarginals> solve(
        const std::optional<gtsam::Vector6>& tip_wrench_mean, 
        const std::optional<gtsam::Matrix6>& tip_wrench_cov,
        const std::optional<gtsam::Matrix4>& tip_pose_mean,
        const std::optional<gtsam::Matrix6>& tip_pose_cov,
        const std::optional<gtsam::Vector6>& nominal_strain);

private:
    void build_graph() override;

    void extract_solution() override;

    void get_initial_values() override;

    std::optional<gtsam::Vector6> tip_wrench_mean_; 
    std::optional<gtsam::Matrix6> tip_wrench_cov_;
    std::optional<gtsam::Matrix4> tip_pose_mean_;
    std::optional<gtsam::Matrix6> tip_pose_cov_;
    std::optional<gtsam::Vector6> nominal_strain_;

    double rod_length_;
    
    gtsam::SharedDiagonal small_wrench_noise_;
    gtsam::SharedDiagonal base_pose_noise_;

    std::unique_ptr<CosseratRodModel> rod_;
    CosseratRodMarginals extracted_;
};