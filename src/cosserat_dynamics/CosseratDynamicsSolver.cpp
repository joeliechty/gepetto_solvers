#include "CosseratDynamicsSolver.h"

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include "cosserat_dynamics/CosseratAccelerationFactor.h"
#include "cosserat_dynamics/CosseratVelocityFactor.h"
#include "cosserat_rod/CosseratRodSolver.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "utils/MiscInline.h"

using namespace gtsam;


CosseratDynamicsSolver::CosseratDynamicsSolver(const CosseratDynamicsConfig& config) 
:   
    num_nodes_(config.rod.num_nodes),
    dt_(config.dt),
    rod_length_(config.rod.rod_length),
    linear_damping_(config.linear_damping),
    rotational_damping_(config.rotational_damping),
    linear_inertia_(config.linear_inertia),
    rotational_inertia_(config.rotational_inertia)
{
    auto static_solver = CosseratRodSolver(config.rod);

    std::optional<Vector6Gaussian> tip_wrench = Vector6Gaussian{
        config.initial_tip_wrench,
        1e-6 * Matrix6::Identity()
    };

    static_solution_ = static_solver.solve(tip_wrench, std::nullopt, std::nullopt);

    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.rod.sigma_twist_rot, config.rod.sigma_twist_pos); 
    
    SharedDiagonal small_wrench_noise = get_noise_model_rot_pos(
        config.rod.sigma_small_moment, config.rod.sigma_small_force); 
    
    base_pose_noise_ = get_noise_model_rot_pos(
        config.rod.sigma_base_pose_rot, config.rod.sigma_base_pose_pos);
    
    acceleration_noise_ = noiseModel::Isotropic::Sigma(6, config.acceleration_noise_sigma);
    
    rod_ = std::make_unique<CosseratRodModel>(
        config.rod.num_nodes, config.rod.K_inv, twist_noise, small_wrench_noise);

    get_initial_values();
    init_prev_marginals();
}


Key get_v_prev_key(int node_idx) { return Symbol('v', node_idx); }


Key get_v_key(int node_idx) { return Symbol('u', node_idx); }


Key get_pose_prev_key(int node_idx) { return Symbol('g', node_idx); }


void CosseratDynamicsSolver::get_initial_values() {
    values_.clear();

    values_.insert(rod_->get_initial_values());

    for (int i = 0; i < num_nodes_; i++) {
        auto& state = static_solution_.marginals.states[i];
        values_.update(rod_->get_pose_key(i), Pose3(state.pose.mean));
        values_.update(rod_->get_stress_key(i), state.stress.mean);
        values_.update(rod_->get_wrench_key(i), state.wrench.mean);
        
        values_.insert(get_pose_prev_key(i), Pose3(state.pose.mean));
        values_.insert(get_v_prev_key(i), Vector6(Vector6::Zero()));
        values_.insert(get_v_key(i), Vector6(Vector6::Zero()));
    }
}


void CosseratDynamicsSolver::init_prev_marginals() {
    rod_marginals_.rod = static_solution_.marginals;

    // Start at 1 since base pose is essentially fixed
    for (int i = 1; i < num_nodes_; i++) {
        Vector6Gaussian v_gaussian;
        v_gaussian.mean = Vector6::Zero();
        v_gaussian.cov = 1e-6 * Matrix6::Identity();
        rod_marginals_.velocities.push_back(v_gaussian);
    }
}


void CosseratDynamicsSolver::build_graph() {
    // First add all twist/stress factors for the current time step.
    graph_ = rod_->build_graph(rod_length_);

    // Constrain base pose to identity 
    graph_.add(PriorFactor<Pose3>(rod_->get_pose_key(0), Pose3::Identity(), base_pose_noise_));

    // Dynamics factors for each node constrain current pose/velocity/accel to previous pose/velocity
    // Skip the base node, since we need a reaction force at the rod base 
    for (int i = 1; i < num_nodes_; i++) {
        graph_.add(CosseratAccelerationFactor(
            get_v_prev_key(i),
            rod_->get_pose_key(i),
            get_v_key(i),
            rod_->get_wrench_key(i),
            acceleration_noise_,
            dt_,
            linear_damping_,
            rotational_damping_,
            linear_inertia_,
            rotational_inertia_));
        
        graph_.add(CosseratVelocityFactor(
            get_pose_prev_key(i),
            get_v_prev_key(i),
            rod_->get_pose_key(i),
            get_v_key(i),
            rod_->get_wrench_key(i),
            base_pose_noise_,
            dt_));
    }

    // Prior factors for previous poses and velocities from last time step
    // Skip the base node, since its pose is essentially fixed
    for (int i = 1; i < num_nodes_; i++){
        graph_.add(PriorFactor<Pose3>(
            get_pose_prev_key(i),
            Pose3(rod_marginals_.rod.states[i].pose.mean),
            rod_marginals_.rod.states[i].pose.cov));
        
        // Velocity uses i - 1 since velocities start from node 1
        graph_.add(PriorFactor<Vector6>(
            get_v_prev_key(i),
            rod_marginals_.velocities[i - 1].mean,
            rod_marginals_.velocities[i - 1].cov));
    }
}


void CosseratDynamicsSolver::extract_solution() {
    rod_marginals_.rod = rod_->get_marginals(values_, marginals_);

    // Again, skip base node
    for (int i = 1; i < num_nodes_; i++) {
        // i - 1 because velocities start from node 1. TRICKY
        rod_marginals_.velocities[i - 1].mean = values_.at<Vector6>(get_v_key(i));
        rod_marginals_.velocities[i - 1].cov = marginals_.marginalCovariance(get_v_key(i));
    }
}


Solution<CosseratDynamicsMarginals> CosseratDynamicsSolver::solve() {
    Solution<CosseratDynamicsMarginals> solution;
    solution.meta = optimize();
    solution.marginals = rod_marginals_;

    return solution;
}