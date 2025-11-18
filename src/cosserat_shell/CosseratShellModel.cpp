#include "CosseratShellModel.h"

#include "cosserat_rod/CosseratTwistFactor.h"
#include "CosseratShellStressFactor.h"

using namespace gtsam;


CosseratShellModel::CosseratShellModel (
    int num_nodes_x,
    int num_nodes_y,
    double element_size,
    const Matrix6& K_inv,
    SharedDiagonal twist_cov,
    SharedDiagonal stress_cov) 
: 
    num_nodes_x_(num_nodes_x),
    num_nodes_y_(num_nodes_y),
    element_size_(element_size),
    K_inv_(K_inv),
    twist_cov_(twist_cov), 
    stress_cov_(stress_cov)
{
    pose_keys_.resize(num_nodes_x_);
    stress_keys_.resize(num_nodes_x_);

    for (int i = 0; i < num_nodes_x_; i++) {
        pose_keys_[i].resize(num_nodes_y_);
        stress_keys_[i].resize(num_nodes_y_);
    }

    for (int i = 0; i < num_nodes_x_; i++) {
        for (int j = 0; j < num_nodes_y_; j++) {

            int id = i * num_nodes_y_ + j;

            pose_keys_[i][j]   = Symbol('T', id);
            stress_keys_[i][j][X] = Symbol('X', id);
            stress_keys_[i][j][Y] = Symbol('Y', id);
        }
    }
}


int clamp_node_idx(int idx, int num_nodes) {
    if (idx == -1) 
        return num_nodes - 1;
    
    if (idx < 0 || idx >= num_nodes)
        throw std::out_of_range("CosseratShell: invalid node_idx");
    
    return idx;
}


int CosseratShellModel::clamp_node_x_idx(int node_idx) const {
    return clamp_node_idx(node_idx, num_nodes_x_);
}


int CosseratShellModel::clamp_node_y_idx(int node_idx) const {
    return clamp_node_idx(node_idx, num_nodes_y_);
}

    
Key CosseratShellModel::get_pose_key(int node_x_idx, int node_y_idx) const { 
    return pose_keys_[clamp_node_x_idx(node_x_idx)][clamp_node_y_idx(node_y_idx)]; 
}


Key CosseratShellModel::get_stress_key(int node_x_idx, int node_y_idx, StressDir dir) const { 
    return stress_keys_[clamp_node_x_idx(node_x_idx)][clamp_node_y_idx(node_y_idx)][dir]; 
}


Values CosseratShellModel::get_initial_values() const {
    Values values;
    
    for (int i = 0; i < num_nodes_x_; ++i) {
        for (int j = 0; j < num_nodes_y_; ++j) {
            values.insert(pose_keys_[i][j], Pose3::Identity());
            values.insert(stress_keys_[i][j][X], Vector6(Vector6::Zero()));
            values.insert(stress_keys_[i][j][Y], Vector6(Vector6::Zero()));
        }
    }

    return values;
}


NonlinearFactorGraph CosseratShellModel::build_graph() const 
{
    NonlinearFactorGraph graph;

    Vector6 nominal_strain_x = Vector6::Zero();
    nominal_strain_x[3] = 1.0;

    Vector6 nominal_strain_y = Vector6::Zero();
    nominal_strain_y[4] = 1.0;

    // Twist factors in X direction
    for (int j = 0; j < num_nodes_y_; j++) {
        for (int i = 0; i + 1 < num_nodes_x_; i++) {
            graph.add(CosseratTwistFactor(
                pose_keys_[i][j],
                pose_keys_[i + 1][j], 
                stress_keys_[i][j][X], 
                stress_keys_[i + 1][j][X], 
                element_size_,
                nominal_strain_x,
                K_inv_,
                twist_cov_));
        }
    }

    // Twist factors in Y direction
    for (int i = 0; i < num_nodes_x_; i++) {
        for (int j = 0; j + 1 < num_nodes_y_; j++) {
            graph.add(CosseratTwistFactor(
                pose_keys_[i][j],
                pose_keys_[i][j + 1], 
                stress_keys_[i][j][Y], 
                stress_keys_[i][j + 1][Y], 
                element_size_,
                nominal_strain_y,
                K_inv_,
                twist_cov_));
        }
    }

    // Cosserat stress factors
    for (int i = 0; i + 1 < num_nodes_x_; i++) {
        for (int j = 0; j + 1 < num_nodes_y_; j++) {
            graph.add(CosseratShellStressFactor(
                pose_keys_[i][j],
                pose_keys_[i + 1][j],
                pose_keys_[i][j + 1],
                stress_keys_[i][j][X],
                stress_keys_[i][j][Y],
                stress_keys_[i + 1][j][X],
                stress_keys_[i][j + 1][Y],
                stress_cov_));
        }
    }

    // Make the x axis fixed by adding base pose constraints
    for (int i = 0; i < num_nodes_x_; i++) {
        graph.add(PriorFactor<Pose3>(
            pose_keys_[i][0],
            Pose3(Rot3::Identity(), Point3(element_size_ * i, 0, 0)),
            twist_cov_));
    }

    // Make top Y stresses zero except for corner
    Vector6 s;
    s << 0, 1, 0, 0, 0, 0;
    for (int i = 0; i < num_nodes_x_; i++) {
        graph.add(PriorFactor<Vector6>(
            stress_keys_[i][num_nodes_y_ - 1][Y],
            s,
            stress_cov_));
    }

    // Make side X stresses zero
    for (int j = 0; j < num_nodes_y_; j++) {
        graph.add(PriorFactor<Vector6>(
            stress_keys_[0][j][X],
            Vector6::Zero(),
            stress_cov_));
        graph.add(PriorFactor<Vector6>(
            stress_keys_[num_nodes_x_ - 1][j][X],
            Vector6::Zero(),
            stress_cov_));
    }

    return graph;
}


CosseratShellMarginals CosseratShellModel::get_marginals(
    const gtsam::Values& values, 
    const gtsam::Marginals& marginals) const 
{
    CosseratShellMarginals solution;

    solution.pose_mean.resize(num_nodes_x_);
    solution.pose_cov.resize(num_nodes_x_);
    solution.stress_mean.resize(num_nodes_x_);
    solution.stress_cov.resize(num_nodes_x_);

    for (int i = 0; i < num_nodes_x_; i++) {
        solution.pose_mean[i].resize(num_nodes_y_);
        solution.pose_cov[i].resize(num_nodes_y_);
        solution.stress_mean[i].resize(num_nodes_y_);
        solution.stress_cov[i].resize(num_nodes_y_);
    }

    for (int i = 0; i < num_nodes_x_; ++i) {
        for (int j = 0; j < num_nodes_y_; ++j) {
            solution.pose_mean[i][j] = values.at<Pose3>(pose_keys_[i][j]).matrix();
            solution.pose_cov[i][j] = marginals.marginalCovariance(pose_keys_[i][j]);

            solution.stress_mean[i][j][X] = values.at<Vector6>(stress_keys_[i][j][X]);
            solution.stress_cov[i][j][X] = marginals.marginalCovariance(stress_keys_[i][j][X]);

            solution.stress_mean[i][j][Y] = values.at<Vector6>(stress_keys_[i][j][Y]);
            solution.stress_cov[i][j][Y] = marginals.marginalCovariance(stress_keys_[i][j][Y]);
        }
    }
    
    return solution;
}