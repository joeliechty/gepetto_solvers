#include "TendonRobotModel.h"
#include "cosserat_rod/CosseratRodModel.h"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <memory>
#include <unordered_set>

#include "TendonDiscWrenchFactor.h"
#include "utils/Gaussians.h"

using namespace gtsam;


TendonRobotModel::TendonRobotModel(
    double rod_length,
    int num_discs,
    int num_between_nodes,
    TendonInput tendon_input,
    const Matrix6& K_inv, 
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise)
:
    rod_length_(rod_length),
    num_discs_(num_discs),
    num_nodes_(num_discs + (num_discs - 1) * num_between_nodes),
    twist_noise_(twist_noise),
    stress_noise_(stress_noise),
    base_pose_mean_(base_pose_mean),
    base_pose_noise_(base_pose_noise)
{
    rod_ = std::make_unique<CosseratRodModel>(
        num_nodes_, K_inv, twist_noise, stress_noise);

    init_tendon_disc_config(tendon_input);
}


void TendonRobotModel::init_tendon_disc_config(TendonInput routing) {
    tendon_config_.num_discs = num_discs_;
    tendon_config_.disc_pose_idx.reserve(num_discs_);
    tendon_config_.routing_radius = routing.routing_radius;
    tendon_config_.hole_locations.reserve(num_discs_);

    // Compute normalized arc-length positions for poses and discs
    std::vector<double> pose_s(num_nodes_);
    std::vector<double> disc_s(num_discs_);

    for (int i = 0; i < num_nodes_; ++i)
        pose_s[i] = static_cast<double>(i) / (num_nodes_ - 1);

    for (int i = 0; i < num_discs_; ++i)
        disc_s[i] = static_cast<double>(i) / (num_discs_ - 1);

    // For each disc, find the closest pose index
    for (int disc_idx = 0; disc_idx < num_discs_; ++disc_idx) {
        double s = disc_s[disc_idx];

        // Find closest pose index to this disc
        int closest_pose_idx = 0;
        double min_dist = std::abs(s - pose_s[0]);

        for (int i = 1; i < num_nodes_; ++i) {
            double dist = std::abs(s - pose_s[i]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_pose_idx = i;
            }
        }

        tendon_config_.disc_pose_idx.push_back(closest_pose_idx);
        std::array<Vector3, NUM_TENDONS> holes;

        for (int tendon_idx = 0; tendon_idx < NUM_TENDONS; ++tendon_idx) {
            double theta;

            if (routing.functions[tendon_idx] == RoutingAngleFunction::CONSTANT) {
                theta = routing.params[tendon_idx].angle_offset;
            } else if (routing.functions[tendon_idx] == RoutingAngleFunction::LINEAR) {
                theta = routing.params[tendon_idx].angle_offset + s * routing.params[tendon_idx].total_angle;
            } else {
                theta = 0.0;
            }

            double x = routing.routing_radius * std::cos(theta);
            double y = routing.routing_radius * std::sin(theta);
            double z = 0.0;

            holes[tendon_idx] = Vector3(x, y, z);
        }

        tendon_config_.hole_locations.push_back(holes);
    }

    std::unordered_set<int> disc_pose_set(
        tendon_config_.disc_pose_idx.begin(), 
        tendon_config_.disc_pose_idx.end());

    tendon_config_.no_disc_pose_idx.reserve(num_nodes_ - num_discs_);

    for (int i = 0; i < num_nodes_; ++i) {
        if (disc_pose_set.find(i) == disc_pose_set.end()) {
            tendon_config_.no_disc_pose_idx.push_back(i);
        }
    }
}


Key TendonRobotModel::get_tensions_key() const { 
    return Symbol('Q', 424242); 
}


Key TendonRobotModel::get_disc_wrench_key(int disc_idx) const { 
    return Symbol('D', disc_idx); 
}


Key TendonRobotModel::get_external_wrench_key(int node_idx) const {
    // If we are at a disc, use disc wrench key
    for (size_t disc_idx = 0; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        if (tendon_config_.disc_pose_idx[disc_idx] == node_idx) {
            return get_disc_wrench_key(disc_idx);
        }
    }

    // Else use wrench key from rod model
    return rod_->get_wrench_key(node_idx);
}


Values TendonRobotModel::get_initial_values() const {
    Values values;

    values.insert(rod_->get_initial_values(rod_length_));
    
    Eigen::Vector<double, NUM_TENDONS> zero = Eigen::Vector<double, NUM_TENDONS>::Zero();
    values.insert(get_tensions_key(), zero);

    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        values.insert(get_disc_wrench_key(disc_idx), Vector6(Vector6::Zero()));
    }

    return values;
}


NonlinearFactorGraph TendonRobotModel::build_graph(const Vector4Gaussian& tensions_) const 
{
    // To fully constrain a Cosserat rod graph, all we need to do is add:
    //   1. Base pose prior constraint
    //   2. All wrenches except base wrench need to be constrained somehow
    NonlinearFactorGraph graph = rod_->build_graph(rod_length_);
    
    // Base frame prior constraint
    graph.add(PriorFactor<Pose3>(rod_->get_pose_key(0), base_pose_mean_, base_pose_noise_));

    // Priors for discs (using disc indices), start at 1, no force at base disc
    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
        int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];
        std::array<Vector3, NUM_TENDONS> holes_prev = tendon_config_.hole_locations[disc_idx - 1];
        std::array<Vector3, NUM_TENDONS> holes = tendon_config_.hole_locations[disc_idx];

        // Some inputs change based on whether we are at the final disc
        bool is_tip;
        int pose_idx_next; 
        std::array<Vector3, NUM_TENDONS> holes_next;

        // TODO change to: bool is_tip = (disc_idx == ...)
        if (disc_idx == (tendon_config_.disc_pose_idx.size() - 1)) {
            is_tip = true;
            pose_idx_next = rod_->get_pose_key(0); // Dummy pose for tip factor, not used for tip disc
            holes_next = tendon_config_.hole_locations[0]; // Dummy holes, not used in factor
        } else {
            is_tip = false;
            pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];
            holes_next = tendon_config_.hole_locations[disc_idx + 1];
        }

        graph.add(TendonDiscWrenchFactor(
            rod_->get_pose_key(pose_idx_prev), 
            rod_->get_pose_key(pose_idx), 
            rod_->get_pose_key(pose_idx_next), 
            rod_->get_wrench_key(pose_idx),
            get_tensions_key(), 
            get_disc_wrench_key(disc_idx),
            is_tip, 
            holes_prev, 
            holes, 
            holes_next, 
            stress_noise_));  // TODO: this could actually be a friction error noise instead of being "small"
    }

    // Measurement prior on tensions
    graph.add(PriorFactor<Vector4>(
        get_tensions_key(), 
        tensions_.mean, 
        noiseModel::Gaussian::Covariance(tensions_.cov)));

    // Now we need to constrain all disc wrenches AND all internal wrenches that are not at discs.
    // These are collected into a set of external wrenches.
    for (int i = 1; i < num_nodes_; ++i) {
        Key key = get_external_wrench_key(i);

        // For now, set to zero TODO
        graph.add(PriorFactor<Vector6>(
            key, 
            Vector6::Zero(), 
            stress_noise_));
    }

    return graph;
}


void TendonRobotModel::get_J_pose_tensions(const Marginals& marginals, TendonRobotMarginals& out) const{
    // Get joint marginal between tip pose and tensions
    Key Q = get_tensions_key();
    Key T = rod_->get_pose_key(num_nodes_ - 1);
    KeyVector keys;
    keys.push_back(Q);
    keys.push_back(T);
    JointMarginal joint = marginals.jointMarginalCovariance(keys);

    // Get individual blocks
    Matrix4 sigma_QQ = joint(Q, Q);
    Matrix64 sigma_TQ = joint(T, Q);

    // Compute J_pose_tensions = sigma_TQ * inv(sigma_QQ)
    Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma_QQ);
    out.J_pose_tensions = sigma_TQ * ldlt.solve(Matrix4::Identity());
}


TendonRobotMarginals TendonRobotModel::get_marginals(
    const Values& values, 
    const Marginals& marginals) const 
{
    TendonRobotMarginals m;

    m.rod = rod_->get_marginals(values, marginals);
    // TODO samples, external wrenches
    m.tendon_config = tendon_config_;
    
    m.tensions.mean = values.at<Vector4>(get_tensions_key());
    m.tensions.cov = marginals.marginalCovariance(get_tensions_key());

    get_J_pose_tensions(marginals, m);

    return m;
}

    //     for (int i = 0; i < num_backbone_poses_; ++i) {
    //         solution.backbone_pose_mean[i] = values_.at<Pose3>(T(i)).matrix();
    //         solution.backbone_pose_cov[i] = marginals_.marginalCovariance(T(i));

    //         // No applied force at the base pose
    //         if (i > 0) {
    //             solution.applied_wrench_mean[i - 1] = values_.at<Vector6>(F(i));
    //             solution.applied_wrench_cov[i - 1] = marginals_.marginalCovariance(F(i));
    //         }
    //     }