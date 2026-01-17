#include "CosseratDynamicsSolver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include "cosserat_rod/CosseratRodSolver.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "CosseratDynamicsFactor.h"
#include "utils/MiscInline.h"

using namespace gtsam;


CosseratDynamicsSolver::CosseratDynamicsSolver(const CosseratDynamicsConfig& config) 
:   
    num_time_steps_(config.num_time_steps),
    num_nodes_(config.rod_config.num_nodes),
    dt_(config.dt),
    rod_length_(config.rod_config.rod_length),
    linear_damping_(config.linear_damping),
    rotational_damping_(config.rotational_damping),
    linear_inertia_(config.linear_inertia),
    rotational_inertia_(config.rotational_inertia)
{
    auto static_solver = CosseratRodSolver(config.rod_config);

    std::optional<Vector6Gaussian> tip_wrench = Vector6Gaussian{
        config.initial_tip_wrench,
        1e-6 * Matrix6::Identity()
    };

    static_solution_ = static_solver.solve(
        tip_wrench,
        std::nullopt,
        std::nullopt);

    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.rod_config.sigma_twist_rot, config.rod_config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.rod_config.sigma_small_moment, config.rod_config.sigma_small_force); 
    
    base_pose_noise_ = get_noise_model_rot_pos(
        config.rod_config.sigma_base_pose_rot, config.rod_config.sigma_base_pose_pos);

    rods_t_.resize(num_time_steps_);

    for (auto& rod_t : rods_t_) {
        rod_t = std::make_unique<CosseratRodModel>(
            config.rod_config.num_nodes, config.rod_config.K_inv, twist_noise, small_wrench_noise_);
    }

    get_initial_values();
}


void CosseratDynamicsSolver::get_initial_values() {
    values_.clear();

    for (auto& rod_t : rods_t_) {
        values_.insert(rod_t->get_initial_values());

        for (int i = 0; i < num_nodes_; i++) {
            auto& state = static_solution_.marginals.states[i];
            values_.update(rod_t->get_pose_key(i), Pose3(state.pose.mean));
            values_.update(rod_t->get_stress_key(i), state.stress.mean);
            values_.update(rod_t->get_wrench_key(i), state.wrench.mean);
        }
    }
}


void CosseratDynamicsSolver::build_graph() {
    graph_.resize(0);

    // First add all twist/stress factors to all the rods
    for (int t = 0; t < num_time_steps_; t++) {
        auto rod_graph = rods_t_[t]->build_graph(rod_length_);
        graph_.push_back(rod_graph.begin(), rod_graph.end());
    }

    // Base pose prior
    for (int t = 0; t < num_time_steps_; t++) { 
        graph_.add(PriorFactor<Pose3>(rods_t_[t]->get_pose_key(0), Pose3::Identity(), base_pose_noise_));
    }

    // Initial conditions: first two rods in time known from static config
    for (int t = 0; t < 2; t++) {
        // Skip base pose, already taken care of above
        for (int i = 1; i < num_nodes_; i++){
            graph_.add(PriorFactor<Pose3>(
                rods_t_[t]->get_pose_key(i),
                Pose3(static_solution_.marginals.states[i].pose.mean), 
                base_pose_noise_));
        }
    }

    // Need to constrain last wrenches in time: approx equal to second to last
    for (int i = 0; i < num_nodes_; i++) {
        graph_.add(BetweenFactor<Vector6>(
            rods_t_[num_time_steps_ - 1]->get_wrench_key(i),
            rods_t_[num_time_steps_ - 2]->get_wrench_key(i),
            Vector6::Zero(),
            small_wrench_noise_));
    }

    // Now add all the finite difference dynamics factors at each time step
    for (int t = 2; t + 1 < num_time_steps_; t++) {
        for (int i = 1; i < num_nodes_; i++) {
            graph_.add(CosseratDynamicsFactor(
                rods_t_[t - 1]->get_pose_key(i),
                rods_t_[t + 0]->get_pose_key(i),
                rods_t_[t + 1]->get_pose_key(i),
                rods_t_[t + 0]->get_wrench_key(i),
                small_wrench_noise_,
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
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}