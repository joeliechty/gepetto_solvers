#include "CosseratRodModel.h"

#include "CosseratTwistFactor.h"
#include "CosseratStressFactor.h"
#include "BoundaryStressFactor.h"

using namespace gtsam;


CosseratRodModel::CosseratRodModel (
    int num_nodes,
    const Matrix6& K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise) 
: 
    id_(next_id_++),
    num_nodes_(num_nodes),
    twist_noise_(twist_noise), 
    stress_noise_(stress_noise)
{
    K_inv_ = std::vector<Matrix6>(num_nodes - 1, K_inv);

    pose_keys_.reserve(num_nodes_);
    stress_keys_.reserve(num_nodes_);
    wrench_keys_.reserve(num_nodes_);

    for (int i = 0; i < num_nodes_; i++) {
        pose_keys_.push_back(  Symbol('T', 1000 * id_ + i));
        stress_keys_.push_back(Symbol('S', 1000 * id_ + i)); 
        wrench_keys_.push_back(Symbol('F', 1000 * id_ + i)); 
    }

    dummy_wrench_key_ = Symbol('F', 1000 * id_ + 999); 
}


int CosseratRodModel::clamp_node_idx(int node_idx) const {
    if (node_idx == -1) 
        return num_nodes_ - 1;
    
    if (node_idx < 0 || node_idx >= num_nodes_)
        throw std::out_of_range("CosseratRod: invalid node_idx");
    
    return node_idx;
}

    
Key CosseratRodModel::get_pose_key(int node_idx) const { return pose_keys_[clamp_node_idx(node_idx)]; }


Key CosseratRodModel::get_stress_key(int node_idx) const { return stress_keys_[clamp_node_idx(node_idx)]; }


Key CosseratRodModel::get_wrench_key(int node_idx) const { return wrench_keys_[clamp_node_idx(node_idx)]; }


const std::vector<Key>& CosseratRodModel::get_wrench_keys() const {return wrench_keys_; }


const std::vector<Key>& CosseratRodModel::get_pose_keys() const { return pose_keys_; }


Values CosseratRodModel::get_initial_values() const {
    Values values;
    
    for (int i = 0; i < num_nodes_; ++i) {
        values.insert(pose_keys_[i], Pose3::Identity());
        values.insert(stress_keys_[i], Vector6(Vector6::Zero()));
        values.insert(wrench_keys_[i], Vector6(Vector6::Zero()));
    }

    values.insert(dummy_wrench_key_, Vector6(Vector6::Zero()));
    return values;
}


NonlinearFactorGraph CosseratRodModel::build_graph(
    double rod_length,
    const std::optional<Vector6>& nominal_strain) const 
{
    NonlinearFactorGraph graph;

    // We can overload build_graph later to support different ds per node
    std::vector<double> ds(num_nodes_ - 1, rod_length / (num_nodes_ - 1));
    
    Vector6 straight_rod_strain = Vector6::Zero();
    straight_rod_strain[5] = 1.0;

    // Cosserat twist factors
    for (int i = 0; i + 1 < num_nodes_; ++i) {
        graph.add(CosseratTwistFactor(
            pose_keys_[i], 
            pose_keys_[i + 1], 
            stress_keys_[i], 
            stress_keys_[i + 1], 
            ds[i], 
            nominal_strain ? *nominal_strain : straight_rod_strain,
            K_inv_[i], 
            twist_noise_));
    }
        
    // Cosserat stress factors
    for (int i = 0; i + 1 < num_nodes_; ++i) {
        Key wrench_key = (i == 0) ? dummy_wrench_key_ : wrench_keys_[i];

        graph.add(CosseratStressFactor(
            pose_keys_[i], 
            pose_keys_[i + 1], 
            stress_keys_[i], 
            stress_keys_[i + 1],
            wrench_key,
            stress_noise_));
    }

    // Constrain tip stress to be equal to tip force
    bool is_base = false;
    graph.add(BoundaryStressFactor(
        stress_keys_.back(), 
        wrench_keys_.back(),
        pose_keys_.back(),
        stress_noise_,
        is_base));
    
    // Makey dummy wrench zero
    graph.add(PriorFactor<Vector6>(dummy_wrench_key_, Vector6::Zero(), stress_noise_));
    
    is_base = true;
    graph.add(BoundaryStressFactor(
        stress_keys_.front(), 
        wrench_keys_.front(),
        pose_keys_.front(),
        stress_noise_,
        is_base));

    return graph;
}


CosseratRodMarginals CosseratRodModel::get_marginals(
    const gtsam::Values& values, 
    const gtsam::Marginals& marginals) const 
{
    CosseratRodMarginals solution;

    solution.states.resize(num_nodes_);

    for (int i = 0; i < num_nodes_; ++i) {
        solution.states[i].pose.mean = values.at<Pose3>(pose_keys_[i]).matrix();
        solution.states[i].pose.cov = marginals.marginalCovariance(pose_keys_[i]);

        solution.states[i].stress.mean = values.at<Vector6>(stress_keys_[i]);
        solution.states[i].stress.cov = marginals.marginalCovariance(stress_keys_[i]);
        
        solution.states[i].wrench.mean = values.at<Vector6>(wrench_keys_[i]);
        solution.states[i].wrench.cov = marginals.marginalCovariance(wrench_keys_[i]);
    }
    
    return solution;
}