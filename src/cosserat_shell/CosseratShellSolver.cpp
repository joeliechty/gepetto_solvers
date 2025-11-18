#include "CosseratShellSolver.h"

#include <gtsam/nonlinear/DoglegOptimizer.h>

using namespace gtsam;


CosseratShellSolver::CosseratShellSolver(const CosseratShellSolverConfig& config) {
    SharedDiagonal twist_cov = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    SharedDiagonal stress_cov = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force); 
    
    shell_= std::make_unique<CosseratShellModel>(
        config.num_nodes_x,
        config.num_nodes_y, 
        config.element_size,
        config.K_inv, 
        twist_cov, 
        stress_cov);

    values_ = shell_->get_initial_values();
}


CosseratShellSolution CosseratShellSolver::solve() {
    auto start = std::chrono::high_resolution_clock::now();

    graph_ = shell_->build_graph();

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    CosseratShellSolution solution;

    auto start_solve = std::chrono::high_resolution_clock::now();

    values_ = optimizer.optimize();

    auto stop_solve = std::chrono::high_resolution_clock::now();

    marginals_ = Marginals(graph_, values_);
    solution.marginals = shell_->get_marginals(values_, marginals_);

    auto stop = std::chrono::high_resolution_clock::now();

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.optimize_time_ms = std::chrono::duration<double, std::milli>(stop_solve - start_solve).count();
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}
