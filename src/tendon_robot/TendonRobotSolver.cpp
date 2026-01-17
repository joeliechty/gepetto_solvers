#include "TendonRobotSolver.h"

#include "utils/Gaussians.h"
#include "utils/MiscInline.h"

using namespace gtsam;


TendonRobotSolver::TendonRobotSolver(const TendonRobotSolverConfig& config) {
    SharedDiagonal twist_noise = get_noise_model_rot_pos(
        config.sigma_twist_rot, config.sigma_twist_pos); 
    
    small_wrench_noise_ = get_noise_model_rot_pos(
        config.sigma_stress_moment, config.sigma_stress_force); 
    
    Rot3 base_rot = Rot3::Rx(-M_PI / 2).compose(Rot3::Rz(M_PI));
    Pose3 base_pose_mean = Pose3(base_rot, Point3::Zero());
    SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
        config.sigma_base_rot, config.sigma_base_pos);
    
    robot_ = std::make_unique<TendonRobotModel>(
        config.rod_length,
        config.num_discs,
        config.num_between_nodes,
        config.tendon_input,
        config.K_inv, 
        twist_noise,
        small_wrench_noise_,
        base_pose_mean,
        base_pose_noise);

    get_initial_values();
}


Solution<TendonRobotMarginals> TendonRobotSolver::solve(
    const Vector4Gaussian& tensions,
    const std::optional<Vector3Gaussian>& tip_force)
{
    tensions_ = tensions;
    tip_force_ = tip_force;

    Solution<TendonRobotMarginals> solution;
    solution.meta = optimize();
    solution.marginals = extracted_;

    return solution;
}


void TendonRobotSolver::build_graph() {
    // TODO add tip force constraint if provided
    graph_ = robot_->build_graph(tensions_);
}


void TendonRobotSolver::extract_solution() {
    extracted_ = robot_->get_marginals(values_, marginals_);
}

void TendonRobotSolver::get_initial_values() {
    values_ = robot_->get_initial_values();
}
