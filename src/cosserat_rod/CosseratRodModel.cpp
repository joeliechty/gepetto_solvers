#include "CosseratRodModel.h"

#include "CosseratTwistFactor.h"
#include "CosseratStressFactor.h"
#include "BoundaryStressFactor.h"
#include "RootCosseratTwistFactor.h"
#include "RootCosseratStressFactor.h"
#include "RootBoundaryStressFactor.h"
#include <gtsam/base/Vector.h>

using namespace gtsam;


CosseratRodModel::CosseratRodModel (
    int num_nodes,
    const Matrix6& K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    bool use_midpoint) 
: 
    id_(next_id_++),
    num_nodes_(num_nodes),
    twist_noise_(twist_noise), 
    stress_noise_(stress_noise),
    use_midpoint_(use_midpoint)
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


CosseratRodModel::CosseratRodModel (
    int num_nodes,
    const std::vector<Matrix6>& K_inv_per_segment,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    bool use_midpoint)
:
    id_(next_id_++),
    num_nodes_(num_nodes),
    twist_noise_(twist_noise),
    stress_noise_(stress_noise),
    use_midpoint_(use_midpoint)
{
    if (static_cast<int>(K_inv_per_segment.size()) != num_nodes - 1)
        throw std::invalid_argument(
            "CosseratRodModel: K_inv_per_segment must have exactly num_nodes - 1 entries");

    K_inv_ = K_inv_per_segment;

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


void CosseratRodModel::set_root_reparameterization(const Pose3& offset) {
    use_root_ = true;
    root_offset_ = offset;
    // 'H' for Hand base; namespaced by this rod's unique id like the other keys.
    root_base_key_ = Symbol('H', 1000 * id_);
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


Values CosseratRodModel::get_initial_values(
    double rod_length,
    const Pose3& base_pose_init) const
{
    std::vector<double> ds(num_nodes_ - 1, rod_length / (num_nodes_ - 1));
    return get_initial_values(ds, base_pose_init);
}


Values CosseratRodModel::get_initial_values(
    const std::vector<double>& ds,
    const Pose3& base_pose_init) const
{
    if (static_cast<int>(ds.size()) != num_nodes_ - 1)
        throw std::runtime_error("CosseratRodModel: ds vector size mismatch");

    Values values;
    double current_z = 0.0;

    // When reparameterized, node 0 is not a free variable; instead seed the hand
    // base T_base such that T_base o offset = base_pose_init (the intended node-0
    // pose), i.e. T_base = base_pose_init o offset^{-1}.
    if (use_root_)
        values.insert(root_base_key_, base_pose_init * root_offset_.inverse());

    for (int i = 0; i < num_nodes_; ++i) {
        if (i > 0) current_z += ds[i - 1];
        Vector3 p = Vector3(0, 0, current_z);
        Pose3 pose = base_pose_init * Pose3(Rot3::Identity(), p);
        if (!(use_root_ && i == 0))
            values.insert(pose_keys_[i], pose);
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
    std::vector<double> ds(num_nodes_ - 1, rod_length / (num_nodes_ - 1));
    return build_graph(ds, nominal_strain);
}


NonlinearFactorGraph CosseratRodModel::build_graph(
    const std::vector<double>& ds,
    const std::optional<Vector6>& nominal_strain) const
{
    if (static_cast<int>(ds.size()) != num_nodes_ - 1)
        throw std::runtime_error("CosseratRodModel: ds vector size mismatch");

    NonlinearFactorGraph graph;

    // Nominally only strain "velocity" in the linear z direction
    Vector6 straight_rod_strain = Vector6::Zero();
    straight_rod_strain[5] = 1.0;

    // Cosserat kinematics and mechanics factors
    for (int i = 0; i + 1 < num_nodes_; ++i) {
        Key wrench_key = (i == num_nodes_ - 2) ? dummy_wrench_key_ : wrench_keys_[i + 1];

        // When reparameterized, node 0's pose is the hand base composed with the
        // fixed offset (T_0 = T_base o offset), so the factors that touch node 0
        // are the Root* variants keyed on root_base_key_ instead of pose_keys_[0].
        if (use_root_ && i == 0) {
            graph.add(RootCosseratTwistFactor(
                root_base_key_,
                pose_keys_[i + 1],
                stress_keys_[i],
                stress_keys_[i + 1],
                root_offset_,
                ds[i],
                nominal_strain ? *nominal_strain : straight_rod_strain,
                K_inv_[i],
                twist_noise_,
                use_midpoint_));

            graph.add(RootCosseratStressFactor(
                root_base_key_,
                pose_keys_[i + 1],
                stress_keys_[i],
                stress_keys_[i + 1],
                wrench_key,
                root_offset_,
                stress_noise_));
            continue;
        }

        // Poses integrate due to stresses in rod
        graph.add(CosseratTwistFactor(
            pose_keys_[i],
            pose_keys_[i + 1],
            stress_keys_[i],
            stress_keys_[i + 1],
            ds[i],
            nominal_strain ? *nominal_strain : straight_rod_strain,
            K_inv_[i],
            twist_noise_,
            use_midpoint_));

        // Stresses integrate due to wrenches on the rod
        graph.add(CosseratStressFactor(
            pose_keys_[i],
            pose_keys_[i + 1],
            stress_keys_[i],
            stress_keys_[i + 1],
            wrench_key,
            stress_noise_));
    }

    // Make dummy wrench zero
    graph.add(PriorFactor<Vector6>(dummy_wrench_key_, Vector6::Zero(), stress_noise_));

    // Constrain tip stress to be equal to tip force
    bool is_base = false;
    graph.add(BoundaryStressFactor(
        stress_keys_.back(),
        wrench_keys_.back(),
        pose_keys_.back(),
        stress_noise_,
        is_base));

    // Constrain base stress to equal base force
    is_base = true;
    if (use_root_) {
        graph.add(RootBoundaryStressFactor(
            stress_keys_.front(),
            wrench_keys_.front(),
            root_base_key_,
            root_offset_,
            stress_noise_,
            is_base));
    } else {
        graph.add(BoundaryStressFactor(
            stress_keys_.front(),
            wrench_keys_.front(),
            pose_keys_.front(),
            stress_noise_,
            is_base));
    }

    return graph;
}


CosseratRodMarginals CosseratRodModel::get_marginals(
    const gtsam::Values& values,
    const gtsam::Marginals& marginals) const
{
    return get_marginals(values, [&](gtsam::Key k) {
        return marginals.marginalCovariance(k);
    });
}


CosseratRodMarginals CosseratRodModel::get_marginals(
    const gtsam::Values& values,
    const CovFn& cov_of) const
{
    CosseratRodMarginals solution;

    solution.states.resize(num_nodes_);

    for (int i = 0; i < num_nodes_; ++i) {
        if (use_root_ && i == 0) {
            // Node 0 is derived: T_0 = T_base o offset. Report the hand-base
            // marginal covariance (for offset = Identity this is the node-0 frame).
            Pose3 pose0 = values.at<Pose3>(root_base_key_).compose(root_offset_);
            solution.states[i].pose.mean = pose0.matrix();
            solution.states[i].pose.cov = cov_of(root_base_key_);
        } else {
            solution.states[i].pose.mean = values.at<Pose3>(pose_keys_[i]).matrix();
            solution.states[i].pose.cov = cov_of(pose_keys_[i]);
        }

        solution.states[i].stress.mean = values.at<Vector6>(stress_keys_[i]);
        solution.states[i].stress.cov = cov_of(stress_keys_[i]);

        solution.states[i].wrench.mean = values.at<Vector6>(wrench_keys_[i]);
        solution.states[i].wrench.cov = cov_of(wrench_keys_[i]);
    }

    return solution;
}