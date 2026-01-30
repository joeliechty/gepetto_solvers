#include "CosseratDynamicsSolver.h"

#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <ios>
#include <memory>

#include "cosserat_dynamics/CosseratDynamicsFactor.h"
#include "cosserat_rod/CosseratRodModel.h"

using namespace gtsam;


CosseratDynamicsSolver::CosseratDynamicsSolver(const CosseratDynamicsConfig& config) 
:   
    num_time_steps_(config.num_time_steps),
    num_nodes_(config.num_nodes),
    dt_(config.dt),
    rod_length_(config.rod_length),
    linear_damping_(config.linear_damping),
    rotational_damping_(config.rotational_damping),
    linear_inertia_(config.linear_inertia),
    rotational_inertia_(config.rotational_inertia),
    init_wrench_mean_(config.init_wrench_mean)
{
    wrench_noise_ = noiseModel::Isotropic::Sigma(6, config.wrench_noise_sigma);
    twist_noise_ = noiseModel::Isotropic::Sigma(6, config.twist_noise_sigma);
    dynamics_noise_ = noiseModel::Isotropic::Sigma(6, config.dynamics_noise_sigma);
    init_wrench_noise_ = noiseModel::Isotropic::Sigma(6, config.init_wrench_sigma);

    // Init rod model graph for each time step
    for (int t = 0; t < num_time_steps_; t++) {
        rods_t_.push_back(std::make_unique<CosseratRodModel>(
            num_nodes_, config.K_inv, twist_noise_, wrench_noise_));
    }

    get_initial_values();
}


void CosseratDynamicsSolver::get_initial_values() {
    values_.clear();

    for (auto& rod_t : rods_t_) {
        values_.insert(rod_t->get_initial_values(rod_length_));
    }
}


void CosseratDynamicsSolver::build_graph() {
    graph_.resize(0);

    for (int t = 0; t < num_time_steps_; t++) {
        // First add all twist/stress factors to all the rods
        graph_.add(rods_t_[t]->build_graph(rod_length_));

        // Base pose prior
        graph_.add(PriorFactor<Pose3>(rods_t_[t]->get_pose_key(0), Pose3::Identity(), twist_noise_));
    }

    // Enforce static conditions at t = 0
    for (size_t i = 1; i < num_nodes_; ++i) {
        graph_.add(PriorFactor<Vector6>(
            rods_t_[0]->get_wrench_key(i),
            i == num_nodes_ - 1 ? init_wrench_mean_ : Vector6::Zero(),
            i == num_nodes_ - 1 ? init_wrench_noise_ : wrench_noise_));
    }

    // Set velocity between factors for t = 1
    for (size_t i = 1; i < num_nodes_; ++i) {
        graph_.add(BetweenFactor<Pose3>(
            rods_t_[0]->get_pose_key(i),
            rods_t_[1]->get_pose_key(i),
            Pose3::Identity(),
            twist_noise_));
    }

    // Set wrench priors for t = 1
    for (size_t i = 1; i + 1 < num_nodes_; ++i) {
        graph_.add(PriorFactor<Vector6>(
            rods_t_[1]->get_wrench_key(i),
            Vector6::Zero(),
            wrench_noise_));
    }


    // Now add all the finite difference dynamics factors at each time step
    // Skip 0 since that is the static config, already handled above
    for (int t = 1; t < num_time_steps_; t++) {
        int p0_idx = t - 1;
        int p1_idx = t;
        int p2_idx = t + 1;
        int w_idx = t;

        if (t == num_time_steps_ - 1) {
            p0_idx = t - 2;
            p1_idx = t - 1;
            p2_idx = t;
        }

        if (t == 1) {
            w_idx = 2;
        }

        for (int i = 1; i < num_nodes_; i++) {
            graph_.add(CosseratDynamicsFactor(
                rods_t_[p0_idx]->get_pose_key(i),
                rods_t_[p1_idx]->get_pose_key(i),
                rods_t_[p2_idx]->get_pose_key(i),
                rods_t_[w_idx]->get_wrench_key(i),
                dynamics_noise_,
                dt_,
                linear_damping_,
                rotational_damping_,
                linear_inertia_,
                rotational_inertia_));
        }
    }
}


void CosseratDynamicsSolver::extract_solution() {
    for (auto& rod_t : rods_t_) {
        Solution<CosseratRodMarginals> sol_i;
        sol_i.marginals = rod_t->get_marginals(values_, marginals_);
        extracted_.rods_t.push_back(sol_i);
    }
}


Solution<CosseratDynamicsMarginals> CosseratDynamicsSolver::solve() {
    Solution<CosseratDynamicsMarginals> solution;
    // delta_initial_ = 1e-2;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}