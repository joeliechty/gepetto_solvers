#include "CosseratShellSolver.h"

#include "cosserat_shell/CosseratShellModel.h"
#include "utils/Gaussians.h"
#include "utils/MiscInline.h"

using namespace gtsam;


CosseratShellSolver::CosseratShellSolver(const CosseratShellSolverConfig& config) {
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    SharedDiagonal stress_noise = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force); 
    
    shell_= std::make_unique<CosseratShellModel>(
        config.num_nodes_x,
        config.num_nodes_y, 
        config.element_size,
        config.K_inv, 
        twist_noise, 
        stress_noise);

    get_initial_values();
}


void CosseratShellSolver::get_initial_values() {
    values_ = shell_->get_initial_values();
}


void CosseratShellSolver::extract_solution() {
    extracted_ = shell_->get_marginals(values_, marginals_);
}


void CosseratShellSolver::build_graph() {
    graph_ = shell_->build_graph(displacement_);
}


Solution<CosseratShellMarginals> CosseratShellSolver::solve(
    const Pose3Gaussian& displacement) 
{
    displacement_ = displacement;

    Solution<CosseratShellMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}