#include "CosseratShellSolver.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include "cosserat_rod/CosseratTwistFactor.h"
#include "CosseratShellStressFactor.h"
#include "utils/MiscInline.h"

using namespace gtsam;


CosseratShellSolver::CosseratShellSolver(const CosseratShellSolverConfig& config) 
:
    num_nodes_x_(config.num_nodes_x),
    num_nodes_y_(config.num_nodes_y),
    element_size_(config.element_size),
    K_inv_(config.K_inv)
{
    // We need to build one more node in each direction, since there is one more edge than elements
    pose_keys_.resize(num_nodes_x_ + 1);
    stress_keys_.resize(num_nodes_x_ + 1);

    for (int i = 0; i <= num_nodes_x_; i++) {
        pose_keys_[i].resize(num_nodes_y_ + 1);
        stress_keys_[i].resize(num_nodes_y_ + 1);
    }

    for (int i = 0; i <= num_nodes_x_; i++) {
        for (int j = 0; j <= num_nodes_y_; j++) {

            int id = i * (num_nodes_y_ + 1) + j;

            pose_keys_[i][j]   = Symbol('T', id);
            stress_keys_[i][j][X] = Symbol('X', id);
            stress_keys_[i][j][Y] = Symbol('Y', id);
        }
    }

    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    SharedDiagonal stress_cov = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force); 

    get_initial_values();
}


void CosseratShellSolver::get_initial_values() {
    values_.clear();
    
    for (int i = 0; i <= num_nodes_x_; ++i) {
        for (int j = 0; j <= num_nodes_y_; ++j) {
            values_.insert(pose_keys_[i][j], Pose3(Rot3::Identity(), Point3(i * element_size_, j * element_size_, 0)));
            values_.insert(stress_keys_[i][j][X], Vector6(Vector6::Zero()));
            values_.insert(stress_keys_[i][j][Y], Vector6(Vector6::Zero()));
        }
    }
}


void CosseratShellSolver::build_graph() {
    graph_.resize(0);

    Vector6 nominal_strain_x = Vector6::Zero();
    nominal_strain_x[3] = 1.0;

    Vector6 nominal_strain_y = Vector6::Zero();
    nominal_strain_y[4] = 1.0;

    // Twist factors in X direction
    for (int j = 0; j < num_nodes_y_; j++) {
        for (int i = 0; i < num_nodes_x_; i++) {
            graph_.add(CosseratTwistFactor(
                pose_keys_[i][j],
                pose_keys_[i + 1][j], 
                stress_keys_[i][j][X], 
                stress_keys_[i + 1][j][X], 
                element_size_,
                nominal_strain_x,
                K_inv_,
                twist_noise_));
        }
    }

    // Twist factors in Y direction
    for (int i = 0; i < num_nodes_x_; i++) {
        for (int j = 0; j < num_nodes_y_; j++) {
            graph_.add(CosseratTwistFactor(
                pose_keys_[i][j],
                pose_keys_[i][j + 1], 
                stress_keys_[i][j][Y], 
                stress_keys_[i][j + 1][Y], 
                element_size_,
                nominal_strain_y,
                K_inv_,
                twist_noise_));
        }
    }

    // 2D Stress factors on interior of shell surface
    for (int i = 0; i < num_nodes_x_; i++) {
        for (int j = 0; j < num_nodes_y_; j++) {
            graph_.add(CosseratShellStressFactor(
                pose_keys_[i][j],
                pose_keys_[i + 1][j],
                pose_keys_[i][j + 1],
                stress_keys_[i][j][X],
                stress_keys_[i][j][Y],
                stress_keys_[i + 1][j][X],
                stress_keys_[i][j + 1][Y],
                stress_noise_));
        }
    }

    // BCs to make x axis fixed
    for (int i = 0; i < num_nodes_x_; i++) {
        graph_.add(PriorFactor<Pose3>(
            pose_keys_[i][0],
            Pose3(Rot3::Identity(), Point3(element_size_ * i, 0, 0)),
            twist_noise_));
    }

    // BCs to make side X stresses zero
    for (int j = 0; j < num_nodes_y_; j++) {
        graph_.add(PriorFactor<Vector6>(
            stress_keys_[0][j][X],
            Vector6::Zero(),
            stress_noise_));
        graph_.add(PriorFactor<Vector6>(
            stress_keys_[num_nodes_x_ - 1 + 1][j][X],
            Vector6::Zero(),
            stress_noise_));
    }

    // BCs to make top of shell have specified pose displacment
    double x_length = element_size_ * (num_nodes_x_ - 1);
    double y_length = element_size_ * (num_nodes_y_ - 1);

    Pose3 nominal_middle_top_pose = Pose3(Rot3::Identity(), Point3(x_length/ 2, y_length, 0));
    int middle_top_idx = num_nodes_x_ / 2;
    
    // Top middle pose is specified with prior factor
    graph_.add(PriorFactor<Pose3>(
        pose_keys_[middle_top_idx][num_nodes_y_ - 1],
        nominal_middle_top_pose.compose(Pose3(displacement_mean_)),
        noiseModel::Gaussian::Covariance(displacement_cov_)));
    
    // All other top poses are constrained relative to top middle pose
    for (int i = 0; i + 1 < num_nodes_x_; i++) {
        graph_.add(BetweenFactor<Pose3>(
            pose_keys_[i][num_nodes_y_ - 1],
            pose_keys_[i + 1][num_nodes_y_ - 1],
            Pose3(Rot3::Identity(), Point3(element_size_, 0, 0)),
            twist_noise_));
    }
}


void CosseratShellSolver::extract_solution() {
    CosseratShellMarginals m;

    m.pose_mean.resize(num_nodes_x_);
    m.pose_cov.resize(num_nodes_x_);
    m.stress_mean.resize(num_nodes_x_);
    m.stress_cov.resize(num_nodes_x_);

    for (int i = 0; i < num_nodes_x_; i++) {
        m.pose_mean[i].resize(num_nodes_y_);
        m.pose_cov[i].resize(num_nodes_y_);
        m.stress_mean[i].resize(num_nodes_y_);
        m.stress_cov[i].resize(num_nodes_y_);
    }

    for (int i = 0; i < num_nodes_x_; ++i) {
        for (int j = 0; j < num_nodes_y_; ++j) {
            m.pose_mean[i][j] = values_.at<Pose3>(pose_keys_[i][j]).matrix();
            m.pose_cov[i][j] = marginals_.marginalCovariance(pose_keys_[i][j]);

            m.stress_mean[i][j][X] = values_.at<Vector6>(stress_keys_[i][j][X]);
            m.stress_cov[i][j][X] = marginals_.marginalCovariance(stress_keys_[i][j][X]);

            m.stress_mean[i][j][Y] = values_.at<Vector6>(stress_keys_[i][j][Y]);
            m.stress_cov[i][j][Y] = marginals_.marginalCovariance(stress_keys_[i][j][Y]);
        }
    }
    
    extracted_ = m;
}


Solution<CosseratShellMarginals> CosseratShellSolver::solve(
    const Matrix4& displacement_mean,
    const Matrix6& displacement_cov) 
{
    displacement_mean_ = displacement_mean;
    displacement_cov_ = displacement_cov; 

    Solution<CosseratShellMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}
