#include "CosseratShellModel.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include "cosserat_rod/CosseratTwistFactor.h"
#include "CosseratShellStressFactor.h"


using namespace gtsam;


CosseratShellModel::CosseratShellModel (
    int num_nodes_x,
    int num_nodes_y,
    double element_size,
    const Matrix6& K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise) 
: 
    num_nodes_x_(num_nodes_x),
    num_nodes_y_(num_nodes_y),
    element_size_(element_size),
    K_inv_(K_inv),
    twist_noise_(twist_noise), 
    stress_noise_(stress_noise)
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
}


Values CosseratShellModel::get_initial_values() const {
    Values values;
    
    for (int i = 0; i <= num_nodes_x_; ++i) {
        for (int j = 0; j <= num_nodes_y_; ++j) {
            values.insert(pose_keys_[i][j], Pose3(Rot3::Identity(), Point3(i * element_size_, j * element_size_, 0)));
            values.insert(stress_keys_[i][j][X], Vector6(Vector6::Zero()));
            values.insert(stress_keys_[i][j][Y], Vector6(Vector6::Zero()));
        }
    }
    
    return values;
}


NonlinearFactorGraph CosseratShellModel::build_graph(
    const Pose3Gaussian& displacement) const
{
    NonlinearFactorGraph graph;

    Vector6 nominal_strain_x = Vector6::Zero();
    nominal_strain_x[3] = 1.0;

    Vector6 nominal_strain_y = Vector6::Zero();
    nominal_strain_y[4] = 1.0;

    // Twist factors in X direction
    for (int j = 0; j < num_nodes_y_; j++) {
        for (int i = 0; i < num_nodes_x_; i++) {
            graph.add(CosseratTwistFactor(
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
            graph.add(CosseratTwistFactor(
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

    // 2D Stress factors on interior of shell
    for (int i = 0; i < num_nodes_x_; i++) {
        for (int j = 0; j < num_nodes_y_; j++) {
            graph.add(CosseratShellStressFactor(
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

    // Make the x axis fixed by adding base pose constraints
    for (int i = 0; i < num_nodes_x_; i++) {
        graph.add(PriorFactor<Pose3>(
            pose_keys_[i][0],
            Pose3(Rot3::Identity(), Point3(element_size_ * i, 0, 0)),
            twist_noise_));
    }

    // Make side X stresses zero
    for (int j = 0; j < num_nodes_y_; j++) {
        graph.add(PriorFactor<Vector6>(
            stress_keys_[0][j][X],
            Vector6::Zero(),
            stress_noise_));
        graph.add(PriorFactor<Vector6>(
            stress_keys_[num_nodes_x_ - 1 + 1][j][X],
            Vector6::Zero(),
            stress_noise_));
    }

    double x_length = element_size_ * (num_nodes_x_ - 1);
    double y_length = element_size_ * (num_nodes_y_ - 1);

    Pose3 nominal_middle_top_pose = Pose3(Rot3::Identity(), Point3(x_length/ 2, y_length, 0));
    int middle_top_idx = num_nodes_x_ / 2;
    
    graph.add(PriorFactor<Pose3>(
        pose_keys_[middle_top_idx][num_nodes_y_ - 1],
        nominal_middle_top_pose.compose(Pose3(displacement.mean)),
        noiseModel::Gaussian::Covariance(displacement.cov)));
    
    for (int i = 0; i + 1 < num_nodes_x_; i++) {
        graph.add(BetweenFactor<Pose3>(
            pose_keys_[i][num_nodes_y_ - 1],
            pose_keys_[i + 1][num_nodes_y_ - 1],
            Pose3(Rot3::Identity(), Point3(element_size_, 0, 0)),
            twist_noise_));
    }
    
    return graph;
}


CosseratShellMarginals CosseratShellModel::get_marginals(
    const gtsam::Values& values, 
    const gtsam::Marginals& marginals) const 
{
    CosseratShellMarginals solution;

    solution.states.resize(num_nodes_x_);
    for (int i = 0; i < num_nodes_x_; i++) {
        solution.states[i].resize(num_nodes_y_);
    }

    for (int i = 0; i < num_nodes_x_; ++i) {
        for (int j = 0; j < num_nodes_y_; ++j) {
            auto& state = solution.states[i][j];
            state.pose.mean = values.at<Pose3>(pose_keys_[i][j]).matrix();
            state.pose.cov = marginals.marginalCovariance(pose_keys_[i][j]);

            state.stress[X].mean = values.at<Vector6>(stress_keys_[i][j][X]);
            state.stress[X].cov = marginals.marginalCovariance(stress_keys_[i][j][X]);

            state.stress[Y].mean = values.at<Vector6>(stress_keys_[i][j][Y]);
            state.stress[Y].cov = marginals.marginalCovariance(stress_keys_[i][j][Y]);
        }
    }
    
    return solution;
}