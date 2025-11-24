#include "TendonRobotSolver.h"

#include "utils/MiscInline.h"

using namespace gtsam;


TendonRobotSolver::TendonRobotSolver(const TendonRobotSolverConfig& config) {
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force); 
    
    robot_ = std::make_unique<TendonRobotModel>(
        config.rod_length,
        config.num_discs,
        config.num_between_nodes,
        config.tendon_input,
        config.K_inv, 
        twist_noise,
        small_wrench_noise_);

    get_initial_values();
}


Solution<TendonRobotMarginals> TendonRobotSolver::solve(
    const gtsam::Vector4& tensions_mean,
    const gtsam::Matrix4& tensions_cov) 
{
    tensions_mean_ = tensions_mean;
    tensions_cov_ = tensions_cov;
    
    Solution<TendonRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}


// void TendonRobotSolver::build_graph() {

// }


// void TendonRobotSolver::extract_solution() {

// }


// void TendonRobotSolver::get_initial_values() {

// }


