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
    init_velocity_mean_(config.init_velocity_mean),
    init_velocity_idx_(config.init_velocity_idx)
{
    wrench_noise_ = noiseModel::Isotropic::Sigma(6, config.wrench_noise_sigma);
    twist_noise_ = noiseModel::Isotropic::Sigma(6, config.twist_noise_sigma);
    dynamics_noise_ = noiseModel::Isotropic::Sigma(6, config.dynamics_noise_sigma);

    init_velocity_noise_ = noiseModel::Diagonal::Sigmas((Vector6() << 
        config.twist_noise_sigma, config.twist_noise_sigma, config.twist_noise_sigma, 
        config.init_velocity_sigma, config.twist_noise_sigma, config.twist_noise_sigma).finished() );

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


void CosseratDynamicsSolver::add_initial_conditions_factors() {
    // Enforce static conditions at t = 0, first make internal wrenches zero 
    for (size_t i = 1; i < num_nodes_; ++i) {
        graph_.add(PriorFactor<Vector6>(
            rods_t_[0]->get_wrench_key(i),
            Vector6::Zero(),
            wrench_noise_));
    }

    // Works but hacky
    for (size_t i = 1; i < num_nodes_; ++i) {
        Vector6 wrench_mean = Vector6::Zero();
        double w_x = -10.0;

        if (i == init_velocity_idx_ - 1 || i == init_velocity_idx_ + 1) {
            wrench_mean = (Vector6() << 0, 0, 0, -w_x / 2, 0, 0).finished();
        } 
        
        if (i == init_velocity_idx_) {
            wrench_mean = (Vector6() << 0, 0, 0, w_x, 0, 0).finished();
        }

        graph_.add(PriorFactor<Vector6>(
            rods_t_[1]->get_wrench_key(i),
            wrench_mean,
            wrench_noise_));
    }





    // // t = 1: all wrenches = 0 except the three near init velocity
    // for (int i = 0; i < num_nodes_; i++) {
    //     // If we are at the before or after nodes, set pose equal to t = 0
    //     // if (i == init_velocity_idx_ - 1 || i == init_velocity_idx_ + 1) {
    //     //     graph_.add(BetweenFactor<Pose3>(
    //     //         rods_t_[0]->get_pose_key(i),
    //     //         rods_t_[1]->get_pose_key(i),
    //     //         Pose3::Identity(),
    //     //         twist_noise_));
    //     //     continue;
    //     // }

    //     // // If we are at the velocity node, set it to the dp
    //     // if (i == init_velocity_idx_) {
    //     //     graph_.add(BetweenFactor<Pose3>(
    //     //         rods_t_[0]->get_pose_key(i),
    //     //         rods_t_[1]->get_pose_key(i),
    //     //         Pose3(Rot3::Identity(), Point3(init_velocity_mean_ * dt_, 0, 0)),
    //     //         init_velocity_noise_));
    //     //     continue;
    //     // } 
        
    //     // If we are at a normal node, set wrench to 0
    //     graph_.add(PriorFactor<Vector6>(
    //         rods_t_[1]->get_wrench_key(i),
    //         Vector6::Zero(),
    //         wrench_noise_));
    // }

    // // graph_.add(BetweenFactor<Vector6>(
    // //             rods_t_[1]->get_wrench_key(init_velocity_idx_ + 1),
    // //             rods_t_[1]->get_wrench_key(init_velocity_idx_ - 1),
    // //             Vector6::Zero(),
    // //             wrench_noise_));
    // // Enforce initial velocity conditions: single node has nonzero init velocity




    // for (int i = 1; i < num_nodes_; i++) {
    //     Point3 dp = i == init_velocity_idx_ ? 
    //         Point3(init_velocity_mean_ * dt_, 0, 0) : 
    //         Point3::Zero();
        
    //     graph_.add(BetweenFactor<Pose3>(
    //         rods_t_[0]->get_pose_key(i),
    //         rods_t_[1]->get_pose_key(i),
    //         Pose3(Rot3::Identity(), dp),
    //         init_velocity_noise_));  // TODO should only be big for x component
    // }

    // for (size_t i = 1; i < num_nodes_; ++i) {
    //     graph_.add(PriorFactor<Vector6>(
    //         rods_t_[1]->get_wrench_key(i),
    //         Vector6::Zero(),
    //         wrench_noise_));
    // }
}


void CosseratDynamicsSolver::build_graph() {
    graph_.resize(0);

    for (int t = 0; t < num_time_steps_; t++) {
        // First add all twist/stress factors to all the rods
        graph_.add(rods_t_[t]->build_graph(rod_length_));

        // Base pose prior
        graph_.add(PriorFactor<Pose3>(rods_t_[t]->get_pose_key(0), Pose3::Identity(), twist_noise_));
    }

    add_initial_conditions_factors();

    // Now add all the center finite difference dynamics factors at each time step
    // Skip 0 since that is the static config, already handled above
    for (int t = 2; t < num_time_steps_; t++) {
        // Skip 0 since we need a base reaction force
        for (int i = 1; i < num_nodes_; i++) {
            graph_.add(CosseratDynamicsFactor(
                rods_t_[t - 2]->get_pose_key(i),
                rods_t_[t - 1]->get_pose_key(i),
                rods_t_[t]->get_pose_key(i),
                rods_t_[t]->get_wrench_key(i),
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