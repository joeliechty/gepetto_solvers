namespace gtsam {

enum class RoutingAngleFunction {
    CONSTANT = 0,
    LINEAR = 1
};

struct RoutingFunctionParams {
    double angle_offset = 0.0;  // Starting angle (radians)
    double total_angle = 0.0;   // For LINEAR: total angle change across the rod
};

struct TendonDiscConfig {
    int num_tendons;
    int num_discs;
    double routing_radius;
    std::vector<int> disc_pose_idx;
    std::vector<int> no_disc_pose_idx;
    std::vector<std::vector<Vector3>> local_holes;  // (disc, tendon)
};

struct TendonRobotSolution {
    std::vector<Matrix4> backbone_pose_mean;
    std::vector<Matrix6> backbone_pose_cov;
    std::vector<Matrix4> tip_pose_samples;
    std::vector<std::vector<Vector3>> fbg_array_samples;
    
    std::vector<Vector6> applied_wrench_mean;
    std::vector<Matrix6> applied_wrench_cov;

    Vector4 tensions_mean;
    Matrix4 tensions_cov;

    Matrix64 J_pose_tensions;
    
    double solve_time_ms = 0;
    double extract_time_ms = 0;
    double total_time_ms = 0;

    TendonDiscConfig tendon_disc_config;

    TendonRobotSolution() = default;

    TendonRobotSolution(size_t num_backbone_poses, size_t num_samples = 0) {
        backbone_pose_mean.resize(num_backbone_poses);
        backbone_pose_cov.resize(num_backbone_poses);
        applied_wrench_mean.resize(num_backbone_poses - 1);
        applied_wrench_cov.resize(num_backbone_poses - 1);

        tip_pose_samples.resize(num_samples);
        fbg_array_samples.resize(num_samples); // each can be filled with pose-count-long vectors
    }
};

struct TendonRobotConfig{
    // Phsysical parameters 
    int num_discs = 9;
    int poses_between_discs = 2;
    double rod_length = 0.2; 
    double rod_diameter = 1.0e-3;
    double youngs_modulus = 35.0e9;  // Nitinol
    double shear_modulus = 12.0e9;  // Nitinol
    double routing_radius = 0.01;
    bool use_midpoint = true;
    
    // General noise parameters
    double cosserat_twist_r_std = 1e-1;
    double small_force_std = 1e-5;
    double small_moment_std = 1e-5;
    double small_r_std = 1e-3;
    double small_p_std = 1e-5;

    // External load parameters
    double tip_force_prior_std = 1e-4;
    double dist_load_prior_std = 1e-1;
    double dist_load_smoothness_std = 1e-3;

    // Measurement noise parameters
    double tension_meas_std = 5e-2;
    double tip_position_meas_std = 1e-3;
    double fbg_strain_meas_std = 5e-6;

    // Params specifying how much these are allowed to change wrt time
    double tension_drift_std = 1e-1;
    double tip_force_drift_std = 1e-1;
    double dist_load_drift_std = 1e-2;
    
    // Routing configuration
    std::vector<RoutingAngleFunction> angle_functions;
    std::vector<RoutingFunctionParams> angle_params;
};
}