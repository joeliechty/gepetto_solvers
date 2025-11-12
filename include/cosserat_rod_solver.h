#include "cosserat_rod.h"


struct CosseratRodSolverConfig {
    double rod_length;
    int num_nodes;

    double k_bending;
    double k_torsion;
    double k_shear;
    double k_extension;

    double sigma_twist_pos;
    double sigma_twist_rot;

    double sigma_small_force;
    double sigma_small_moment;

    double sigma_base_pose_pos;
    double sigma_base_pose_rot;
};


struct SolutionMetadata {
    double solve_time_ms;
    double total_time_ms;
    int iterations;
    int error;
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
        const std::optional<gtsam::Matrix6>& tip_pose_cov);

private:
    void add_prior_factors(
        const std::optional<gtsam::Vector6>& tip_wrench_mean, 
        const std::optional<gtsam::Matrix6>& tip_wrench_cov,
        const std::optional<gtsam::Matrix4>& tip_pose_mean,
        const std::optional<gtsam::Matrix6>& tip_pose_cov);

    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
    gtsam::Marginals marginals_;

    gtsam::SharedDiagonal small_wrench_cov_;
    gtsam::SharedDiagonal base_pose_cov_;

    std::unique_ptr<CosseratRod> rod_;
};