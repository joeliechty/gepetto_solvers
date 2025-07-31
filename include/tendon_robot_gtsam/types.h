namespace gtsam {

enum class RoutingAngleFunction {
    CONSTANT = 0,
    LINEAR = 1
};

struct RoutingParams {
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

    Vector6 tip_wrench_mean;
    Matrix6 tip_wrench_cov;

    Vector4 tensions_mean;
    Matrix4 tensions_cov;

    double solve_time_ms;
    double extract_time_ms;
    double total_time_ms;

    TendonDiscConfig tendon_disc_config;
};

struct TendonRobotGtsamConfig{
    // Phsysical parameters 
    int num_discs = 9;
    int poses_between_discs = 2;
    double rod_length = 0.2; 
    double rod_diameter = 1.0e-3;
    double youngs_modulus = 35.0e9;  // Nitinol
    double shear_modulus = 12.0e9;  // Nitinol
    double routing_radius = 0.01;

    // Model parameters
    double tension_std = 5e-2;
    double small_force_std = 1e-5;
    double small_moment_std = 1e-5;
    double cosserat_twist_r_std = 1e-1;
    double small_r_std = 1e-3;
    double small_p_std = 1e-5;
    double tip_force_std = 1e-4;

    // Temporal drift parameters
    double tip_accel_std = 1e-3;
    double tension_drift_std = 1e-1;
    double wrench_drift_std = 1e-1;

    // Measurement noise parameters
    double tip_pose_r_meas_std = 3e0;
    double tip_pose_p_meas_std = 1e-3;

    // Routing configuration
    std::vector<RoutingAngleFunction> angle_functions;
    std::vector<RoutingParams> angle_params;
};
}