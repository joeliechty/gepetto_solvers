#pragma once

#include <gtsam/base/Vector.h>
#include "cosserat_rod/CosseratRodModel.h"


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


struct CosseratRodSolution {
    SolutionMetadata meta;
    CosseratRodMarginals marginals;
};


class CosseratRodSolver {
public:
    CosseratRodSolver(const CosseratRodSolverConfig& config);

    CosseratRodSolution solve(
        const std::optional<gtsam::Vector6>& tip_wrench_mean, 
        const std::optional<gtsam::Matrix6>& tip_wrench_cov,
        const std::optional<gtsam::Matrix4>& tip_pose_mean,
        const std::optional<gtsam::Matrix6>& tip_pose_cov,
        const std::optional<gtsam::Vector6>& nominal_strain);

private:
    void add_prior_factors(
        const std::optional<gtsam::Vector6>& tip_wrench_mean, 
        const std::optional<gtsam::Matrix6>& tip_wrench_cov,
        const std::optional<gtsam::Matrix4>& tip_pose_mean,
        const std::optional<gtsam::Matrix6>& tip_pose_cov);
    
    double rod_length_;

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;
    
    // TODO need to change all of these to noise, its not actually a cov mat
    gtsam::SharedDiagonal small_wrench_cov_;
    gtsam::SharedDiagonal base_pose_cov_;

    std::unique_ptr<CosseratRodModel> rod_;
};