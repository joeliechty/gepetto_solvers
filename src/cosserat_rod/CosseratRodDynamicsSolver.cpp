#include "CosseratRodDynamicsSolver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <optional>
#include <gtsam/linear/NoiseModel.h>

#include "cosserat_rod/CosseratRodSolver.h"
#include "cosserat_rod/CosseratRodModel.h"
#include "CosseratDynamicsFactor.h"

using namespace gtsam;


CosseratRodDynamicsSolver::CosseratRodDynamicsSolver(const CosseratRodDynamicsConfig& config) 
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

    static_solution_ = static_solver.solve(
        config.initial_tip_wrench,
        1e-6 * Matrix6::Identity(),
        std::nullopt,
        std::nullopt,
        std::nullopt);

    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.rod_config.sigma_twist_rot, config.rod_config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.rod_config.sigma_small_moment, config.rod_config.sigma_small_force); 
    
    base_pose_noise_ = get_noise_model_rot_pos(
        config.rod_config.sigma_base_pose_rot, config.rod_config.sigma_base_pose_pos);
    
    dynamics_noise_ = noiseModel::Isotropic::Sigma(6, config.sigma_dynamics_noise);

    rods_t_.resize(num_time_steps_);

    for (auto& rod_t : rods_t_) {
        rod_t = std::make_unique<CosseratRodModel>(
            config.rod_config.num_nodes, config.rod_config.K_inv, twist_cov, small_wrench_noise_);
    }

    init_values();
}


void CosseratRodDynamicsSolver::init_values() {
    values_.clear();

    for (auto& rod_t : rods_t_) {
        values_.insert(rod_t->get_initial_values());

        for (int i = 0; i < num_nodes_; i++) {
            values_.update(rod_t->get_pose_key(i), Pose3(static_solution_.marginals.pose_mean[i]));
            values_.update(rod_t->get_stress_key(i), static_solution_.marginals.stress_mean[i]);
            values_.update(rod_t->get_wrench_key(i), static_solution_.marginals.wrench_mean[i]);
        }
    }
}


void CosseratRodDynamicsSolver::build_graph() {
    graph_.resize(0);

    // First add all twist/stress factors to all the rods
    for (auto& rod_t : rods_t_) {
        auto rod_graph = rod_t->build_graph(rod_length_);
        graph_.push_back(rod_graph.begin(), rod_graph.end());

        // Constrain interior wrenches to zero (skip base and tip)
        std::vector<Key> wrench_keys = rod_t->get_wrench_keys();
        for (size_t j = 1; j + 1 < wrench_keys.size(); ++j) {
            graph_.add(PriorFactor<Vector6>(wrench_keys[j], Vector6::Zero(), small_wrench_noise_));
        }

        // Base pose prior
        graph_.add(PriorFactor<Pose3>(rod_t->get_pose_key(0), Pose3::Identity(), base_pose_noise_));
    }

    // Now add all the finite difference dynamics factors at each time step
    for (int i = 1; i + 1 < num_time_steps_; i++) {
        graph_.add(CosseratDynamicsFactor(
            rods_t_[i - 1]->get_pose_key(-1),
            rods_t_[i + 0]->get_pose_key(-1),
            rods_t_[i + 1]->get_pose_key(-1),
            rods_t_[i + 0]->get_wrench_key(-1),
            dynamics_noise_,
            dt_,
            linear_damping_,
            rotational_damping_,
            linear_inertia_,
            rotational_inertia_
        ));
    }

    // Need to constrain first and last wrenches
    graph_.add(PriorFactor<Vector6>(rods_t_.front()->get_wrench_key(-1), Vector6::Zero(), small_wrench_noise_));
    graph_.add(PriorFactor<Vector6>(rods_t_.back()->get_wrench_key(-1), Vector6::Zero(), small_wrench_noise_));

    // Add initial condition factors: first two poses in time are set to known values from static config
    std::vector<Key> pose_keys_0 = rods_t_[0]->get_pose_keys();
    std::vector<Key> pose_keys_1 = rods_t_[1]->get_pose_keys();

    for (size_t i = 0; i < pose_keys_0.size(); ++i) {
        graph_.add(PriorFactor<Pose3>(
            pose_keys_0[i], 
            Pose3(static_solution_.marginals.pose_mean[i]), 
            base_pose_noise_));

        graph_.add(PriorFactor<Pose3>(
            pose_keys_1[i], 
            Pose3(static_solution_.marginals.pose_mean[i]), 
            base_pose_noise_));
    }
}


CosseratRodDynamicsSolution CosseratRodDynamicsSolver::solve() {
    auto start = std::chrono::high_resolution_clock::now();
    auto build_start = start;

    build_graph();

    auto build_stop = std::chrono::high_resolution_clock::now();
    auto optimize_start = build_stop;

    LevenbergMarquardtParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    params.setlambdaInitial(10.0);
    LevenbergMarquardtOptimizer optimizer(graph_, values_, params);

    values_ = optimizer.optimize();

    auto optimize_stop = std::chrono::high_resolution_clock::now();
    auto marginalize_start = optimize_stop;

    marginals_ = Marginals(graph_, values_);

    auto marginalize_stop = std::chrono::high_resolution_clock::now();
    auto extract_start = marginalize_stop;

    CosseratRodDynamicsSolution solution;
    
    for (auto& rod_t : rods_t_) {
        CosseratRodSolution sol_i;
        sol_i.marginals = rod_t->get_marginals(values_, marginals_);
        solution.marginals.push_back(sol_i);
    }

    auto extract_stop = std::chrono::high_resolution_clock::now();
    auto stop = extract_stop;

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.build_time_ms = std::chrono::duration<double, std::milli>(build_stop - build_start).count();
    solution.meta.optimize_time_ms = std::chrono::duration<double, std::milli>(optimize_stop - optimize_start).count();
    solution.meta.marginalize_time_ms = std::chrono::duration<double, std::milli>(marginalize_stop - marginalize_start).count();
    solution.meta.extract_time_ms = std::chrono::duration<double, std::milli>(extract_stop - extract_start).count();
    
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}