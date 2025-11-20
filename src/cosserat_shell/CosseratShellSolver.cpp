#include "CosseratShellSolver.h"

#include <chrono>
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


CosseratShellSolution CosseratShellSolver::solve(
    const Matrix4& displacement_mean,
    const Matrix6& displacement_cov) 
{
    auto start = std::chrono::high_resolution_clock::now();

    graph_ = shell_->build_graph(displacement_mean, displacement_cov);

    auto stop_build = std::chrono::high_resolution_clock::now();

    DoglegParams params;
    params.setLinearSolverType("MULTIFRONTAL_QR");
    DoglegOptimizer optimizer(graph_, values_, params);

    CosseratShellSolution solution;

    values_ = optimizer.optimize();

    auto stop_solve = std::chrono::high_resolution_clock::now();

    marginals_ = Marginals(graph_, values_);

    auto stop_marginalize = std::chrono::high_resolution_clock::now();

    solution.marginals = shell_->get_marginals(values_, marginals_);

    auto stop = std::chrono::high_resolution_clock::now();

    solution.meta.total_time_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    solution.meta.optimize_time_ms = std::chrono::duration<double, std::milli>(stop_solve - stop_build).count();
    solution.meta.build_time_ms = std::chrono::duration<double, std::milli>(stop_build - start).count();
    solution.meta.marginalize_time_ms = std::chrono::duration<double, std::milli>(stop_marginalize - stop_solve).count();
    solution.meta.extract_time_ms = std::chrono::duration<double, std::milli>(stop - stop_marginalize).count();
    solution.meta.error = optimizer.error();
    solution.meta.iterations = optimizer.iterations();

    return solution;
}
