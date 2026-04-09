#pragma once

#include "TendonFingerModel.h"
#include "TendonFingerSolver.h"
#include "TensionLimitFactor.h"
#include "measurement/PositionPriorFactor.h"
#include "utils/SolverBase.h"
#include "utils/Gaussians.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <memory>
#include <optional>
#include <variant>
#include <vector>


struct TrajectoryPlannerConfig {
    TendonFingerSolverConfig model_config;

    int K = 20;         // number of time steps (creates K+1 states: k=0..K)
    double dt = 0.1;    // time step duration

    // Start boundary conditions (each optional; if set, adds a prior factor at k=0)
    // start_tensions replaces the background tension prior at k=0 when set
    std::optional<gtsam::Matrix4>  start_pose;
    gtsam::Matrix6                 start_pose_cov;      // required when start_pose is set
    std::optional<gtsam::Vector3>  start_position;
    gtsam::Matrix3                 start_position_cov;  // required when start_position is set
    std::optional<Eigen::VectorXd> start_tensions;
    Eigen::MatrixXd                start_tensions_cov;  // required when start_tensions is set

    // Goal boundary conditions (each optional; if set, adds a prior factor at k=K)
    // goal_tensions replaces the background tension prior at k=K when set
    std::optional<gtsam::Matrix4>  goal_pose;
    gtsam::Matrix6                 goal_pose_cov;       // required when goal_pose is set
    std::optional<gtsam::Vector3>  goal_position;
    gtsam::Matrix3                 goal_position_cov;   // required when goal_position is set
    std::optional<Eigen::VectorXd> goal_tensions;
    Eigen::MatrixXd                goal_tensions_cov;   // required when goal_tensions is set

    // Background tension prior (p_bg) — applied at every k,
    // except k=0 when start_tensions is set, and k=K when goal_tensions is set
    Eigen::VectorXd background_tensions_mean;
    Eigen::VectorXd background_tensions_sigmas;  // tight for passive, loose for active

    // GP temporal prior between consecutive tensions
    Eigen::MatrixXd gp_Qc;  // N x N process noise covariance

    // Tension limit barrier
    double tension_limit_alpha = 10.0;
    double tension_limit_q_min = 0.0;
    std::vector<int> active_tendon_indices;  // which tendons are actively controlled

    // Noise for zero external wrench priors on interior nodes
    double sigma_ext_wrench_force = 1e-4;
    double sigma_ext_wrench_moment = 1e-5;
};


struct TrajectoryPlannerResult {
    std::vector<TendonFingerMarginals> trajectory;  // K+1 entries
    SolutionMetadata meta;
};


template<int N>
class TendonFingerTrajectoryPlanner : SolverBase {
public:
    static constexpr int NumTendons = N;

    TendonFingerTrajectoryPlanner(const TrajectoryPlannerConfig& config);

    TrajectoryPlannerResult plan();

private:
    void build_graph() override;
    void extract_solution() override;
    void get_initial_values() override;

    TrajectoryPlannerConfig config_;

    gtsam::SharedDiagonal ext_wrench_noise_;

    std::vector<std::unique_ptr<TendonFingerModel<N>>> models_;  // K+1 models

    TrajectoryPlannerResult result_;
};


// Runtime dispatch wrapper (same pattern as TendonFingerSolverDispatch)
class TendonFingerTrajectoryPlannerDispatch {
public:
    TendonFingerTrajectoryPlannerDispatch(const TrajectoryPlannerConfig& config);

    TrajectoryPlannerResult plan();

    int num_tendons() const { return num_tendons_; }

private:
    int num_tendons_;

    using PlannerVariant = std::variant<
        std::unique_ptr<TendonFingerTrajectoryPlanner<1>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<2>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<3>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<4>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<5>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<6>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<7>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<8>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<9>>,
        std::unique_ptr<TendonFingerTrajectoryPlanner<10>>
    >;
    PlannerVariant planner_;
};
