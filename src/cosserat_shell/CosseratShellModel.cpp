#include "CosseratShellModel.h"

#include "cosserat_rod/CosseratTwistFactor.h"
#include "CosseratShellStressFactor.h"
// #include "BoundaryStressFactor.h"

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
    wrench_keys_.resize(num_nodes_x_);

    for (int i = 0; i < num_nodes_x_; i++) {
        pose_keys_[i].resize(num_nodes_y_);
        stress_keys_[i].resize(num_nodes_y_);
        wrench_keys_[i].resize(num_nodes_y_);
    }

    for (int i = 0; i < num_nodes_x_; i++) {
        for (int j = 0; j < num_nodes_y_; j++) {

            size_t id = i * num_nodes_y_ + j;

            pose_keys_[i][j]   = Symbol('T', id);
            stress_keys_[i][j] = Symbol('S', id);
            wrench_keys_[i][j] = Symbol('F', id);
        }
    }

    dummy_wrench_key_ = Symbol('F', 999); 
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


Key CosseratShellModel::get_stress_key(int node_x_idx, int node_y_idx) const { 
    return stress_keys_[clamp_node_x_idx(node_x_idx)][clamp_node_y_idx(node_y_idx)]; 
}


Key CosseratShellModel::get_wrench_key(int node_x_idx, int node_y_idx) const { 
    return wrench_keys_[clamp_node_x_idx(node_x_idx)][clamp_node_y_idx(node_y_idx)]; 
}


const std::vector<std::vector<Key>>& CosseratShellModel::get_wrench_keys() const {return wrench_keys_; }


Values CosseratShellModel::get_initial_values() const {
    Values values;
    
    for (int i = 0; i < num_nodes_x_; ++i) {
        for (int j = 0; j < num_nodes_y_; ++j) {
            values.insert(pose_keys_[i][j], Pose3::Identity());
            values.insert(stress_keys_[i][j], Vector6(Vector6::Zero()));
            values.insert(wrench_keys_[i][j], Vector6(Vector6::Zero()));
        }
    }

    values.insert(dummy_wrench_key_, Vector6(Vector6::Zero()));
    return values;
}


NonlinearFactorGraph CosseratShellModel::build_graph() const 
{
    NonlinearFactorGraph graph;

    Vector6 straight_rod_strain = Vector6::Zero();
    straight_rod_strain[5] = 1.0;

    // // Cosserat twist factors
    // for (int i = 0; i + 1 < num_nodes_; ++i) {
    //     graph.add(CosseratTwistFactor(
    //         pose_keys_[i], 
    //         pose_keys_[i + 1], 
    //         stress_keys_[i], 
    //         stress_keys_[i + 1], 
    //         ds[i], 
    //         nominal_strain ? *nominal_strain : straight_rod_strain,
    //         K_inv_[i], 
    //         twist_cov_));
    // }
        
    // // Cosserat stress factors
    // for (int i = 0; i + 1 < num_nodes_; ++i) {
    //     Key wrench_key = (i == 0) ? dummy_wrench_key_ : wrench_keys_[i];

    //     graph.add(CosseratStressFactor(
    //         pose_keys_[i], 
    //         pose_keys_[i + 1], 
    //         stress_keys_[i], 
    //         stress_keys_[i + 1],
    //         wrench_key,
    //         stress_cov_));
    // }

    // // Constrain tip stress to be equal to tip force
    // bool is_base = false;
    // graph.add(BoundaryStressFactor(
    //     stress_keys_.back(), 
    //     wrench_keys_.back(),
    //     pose_keys_.back(),
    //     stress_cov_,
    //     is_base));
    
    // // Makey dummy wrench zero
    // graph.add(PriorFactor<Vector6>(dummy_wrench_key_, Vector6::Zero(), stress_cov_));
    
    // is_base = true;
    // graph.add(BoundaryStressFactor(
    //     stress_keys_.front(), 
    //     wrench_keys_.front(),
    //     pose_keys_.front(),
    //     stress_cov_,
    //     is_base));

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
    solution.wrench_mean.resize(num_nodes_x_);
    solution.wrench_cov.resize(num_nodes_x_);

    for (int i = 0; i < num_nodes_x_; i++) {
        solution.pose_mean[i].resize(num_nodes_y_);
        solution.pose_cov[i].resize(num_nodes_y_);
        solution.stress_mean[i].resize(num_nodes_y_);
        solution.stress_cov[i].resize(num_nodes_y_);
        solution.wrench_mean[i].resize(num_nodes_y_);
        solution.wrench_cov[i].resize(num_nodes_y_);
    }

    for (int i = 0; i < num_nodes_x_; ++i) {
        for (int j = 0; j < num_nodes_y_; ++j) {
            solution.pose_mean[i][j] = values.at<Pose3>(pose_keys_[i][j]).matrix();
            solution.pose_cov[i][j] = marginals.marginalCovariance(pose_keys_[i][j]);

            solution.stress_mean[i][j] = values.at<Vector6>(stress_keys_[i][j]);
            solution.stress_cov[i][j] = marginals.marginalCovariance(stress_keys_[i][j]);

            solution.wrench_mean[i][j] = values.at<Vector6>(wrench_keys_[i][j]);
            solution.wrench_cov[i][j] = marginals.marginalCovariance(wrench_keys_[i][j]);
        }
    }
    
    return solution;
}