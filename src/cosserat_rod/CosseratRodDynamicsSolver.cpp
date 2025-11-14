#include "CosseratRodDynamicsSolver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include "cosserat_rod/CosseratRodModel.h"
#include "linear/NoiseModel.h"

using namespace gtsam;


CosseratRodDynamicsSolver::CosseratRodDynamicsSolver(const CosseratRodSolverConfig& config) {
    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_small_moment, config.sigma_small_force); 
    
    base_pose_noise_ = get_noise_model_rot_pos(
        config.sigma_base_pose_rot, config.sigma_base_pose_pos);
    
    for (auto& rod_t : rods_t_) {
        rod_t = std::make_unique<CosseratRodModel>(
            config.num_nodes, config.K_inv, twist_cov, small_wrench_noise_);
    }
}


void CosseratRodDynamicsSolver::build_graph() {
    graph_.resize(0);

    // First add all twist/stress factors to all the rods
    for (auto& rod_t : rods_t_) {
        // Build base cosserat rod graph
        double rod_length = 1.0;
        auto rod_graph = rod_t->build_graph(rod_length);
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
    SharedDiagonal dynamics_noise = noiseModel::Isotropic::Sigma(3, 1e-5);

    for (int i = 1; i +1 < NUM_STEPS; i++) {
        graph_.add(CosseratDynamicsFactor(
            rods_t_[i - 1]->get_pose_key(-1),
            rods_t_[i + 0]->get_pose_key(-1),
            rods_t_[i + 1]->get_pose_key(-1),
            rods_t_[i]->get_wrench_key(-1),
            dynamics_noise
        ));
    }
}


CosseratRodDynamicsSolution CosseratRodDynamicsSolver::step() {
    auto start = std::chrono::high_resolution_clock::now();
    auto build_start = start;

    build_graph();

    auto build_stop = std::chrono::high_resolution_clock::now();
    auto optimize_start = build_stop;

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    values_ = optimizer.optimize();

    auto optimize_stop = std::chrono::high_resolution_clock::now();
    auto marginalize_start = optimize_stop;

    marginals_ = Marginals(graph_, values_);

    auto marginalize_stop = std::chrono::high_resolution_clock::now();
    auto extract_start = marginalize_stop;

    CosseratRodDynamicsSolution solution;
    
    for (auto& rod_t : rods_t_) {
        solution.marginals.push_back(rod_t->get_marginals(values_, marginals_));
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