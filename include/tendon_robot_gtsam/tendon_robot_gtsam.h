#include <chrono>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>

#include "factors.h"


namespace gtsam{

enum class RoutingAngleFunction {
    CONSTANT = 0,
    LINEAR = 1
};

struct RoutingParams {
    double offset = 0.0;  // Starting angle (radians)
    double angle = 0.0;   // For LINEAR: total angle change across the rod
};

struct TendonDiscConfig {
    int num_tendons;
    int num_discs;
    double routing_radius;
    std::vector<int> disc_pose_idx;
    std::vector<std::vector<Vector3>> local_holes;  // (disc, tendon)
};

TendonDiscConfig generate_tendon_disc_config(
    int num_discs,
    int num_poses,
    double routing_radius,
    const std::vector<RoutingAngleFunction>& angle_functions, 
    const std::vector<RoutingParams>& angle_params)
{
    int num_tendons = angle_functions.size();

    TendonDiscConfig config;
    
    config.num_tendons = num_tendons;
    config.num_discs = num_discs;
    config.disc_pose_idx.reserve(num_discs);
    config.routing_radius = routing_radius;
    config.local_holes.reserve(num_discs);

    // Compute normalized arc-length positions for poses and discs
    std::vector<double> pose_s(num_poses);
    std::vector<double> disc_s(num_discs);

    for (int i = 0; i < num_poses; ++i)
        pose_s[i] = static_cast<double>(i) / (num_poses - 1);

    for (int i = 0; i < num_discs; ++i)
        disc_s[i] = static_cast<double>(i) / (num_discs - 1);

    // For each disc, find the closest pose index
    for (int disc_idx = 0; disc_idx < num_discs; ++disc_idx) {
        double s = disc_s[disc_idx];

        // Find closest pose index to this disc
        int closest_pose_idx = 0;
        double min_dist = std::abs(s - pose_s[0]);

        for (int i = 1; i < num_poses; ++i) {
            double dist = std::abs(s - pose_s[i]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_pose_idx = i;
            }
        }

        config.disc_pose_idx.push_back(closest_pose_idx);
        std::vector<Vector3> holes;
        holes.reserve(num_tendons);

        for (int tendon_idx = 0; tendon_idx < num_tendons; ++tendon_idx) {
            double theta;
            if (angle_functions[tendon_idx] == RoutingAngleFunction::CONSTANT) {
                theta = angle_params[tendon_idx].offset;
            } else if (angle_functions[tendon_idx] == RoutingAngleFunction::LINEAR) {
                theta = angle_params[tendon_idx].offset + s * angle_params[tendon_idx].angle;
            } else {
                theta = 0.0;
            }

            double x = routing_radius * std::cos(theta);
            double y = routing_radius * std::sin(theta);
            double z = 0.0;

            holes.emplace_back(x, y, z);
        }

        config.local_holes.push_back(holes);
    }

    return config;
}


struct TendonRobotSolution {
    std::vector<Pose3> backbone_pose_mean;
    std::vector<Matrix6> backbone_pose_cov;

    Vector6 tip_wrench_mean;
    Matrix6 tip_wrench_cov;

    Vector4 tensions_mean;
    Matrix4 tensions_cov;

    double solve_time_ms;

    TendonDiscConfig tendon_disc_config;
};

struct TendonRobotGtsamConfig{
    // Backbone parameters 
    int num_discs = 12;
    int poses_between_each = 3;
    double rod_length = 0.2; 
    double rod_diameter = 1.0e-3;
    double youngs_modulus = 35.0e9;  // Nitinol
    double shear_modulus = 12.0e9;  // Nitinol
    
    // Noise parameters
    double tension_std = 5e-2;
    double small_force_std = 1e-5;
    double small_moment_std = 1e-5;
    double small_stress_std = 1e-5;
    double cosserat_twist_r_std = 1e-1; // This just looks right
    double small_r_std = 1e-3;
    double small_p_std = 1e-5;

    // Routing configuration
    double routing_radius;
    std::vector<RoutingAngleFunction> angle_functions;
    std::vector<RoutingParams> angle_params;

    //double tip_force_std = 1e-3;
    //double tip_moment_std = 1e-3;
};

TendonRobotGtsamConfig get_default_config(){
    gtsam::TendonRobotGtsamConfig config;

    config.routing_radius = 0.008;

    config.angle_functions = {
        gtsam::RoutingAngleFunction::LINEAR,
        gtsam::RoutingAngleFunction::CONSTANT,
        gtsam::RoutingAngleFunction::CONSTANT,
        gtsam::RoutingAngleFunction::CONSTANT
    };

    config.angle_params = {
        {0.0, 2 * M_PI},
        {M_PI, 0.0},
        {3 * M_PI / 2, 0.0},
        {0.0, 0.0}
    };

    return config;
}

using symbol_shorthand::T;
using symbol_shorthand::F;
using symbol_shorthand::S;
using symbol_shorthand::Q;
// using symbol_shorthand::D;

class TendonRobotGtsam {
public:
    TendonRobotGtsam(TendonRobotGtsamConfig config) {
        int num_backbone_poses = config.num_discs + (config.num_discs - 1) * config.poses_between_each;
        
        backbone_idx_start_ = 0;
        backbone_idx_end_ = backbone_idx_start_ + num_backbone_poses - 1;

        ds_ = config.rod_length / (num_backbone_poses - 1);

        // Build stiffness matrix
        double cross_section_area = M_PI * std::pow(config.rod_diameter, 2) / 4.0;
        double cross_section_moment = M_PI * std::pow(config.rod_diameter, 4) / 64.0;

        double k_bending = config.youngs_modulus * cross_section_moment;
        double k_torsion = 2.0 * config.shear_modulus * cross_section_moment;
        double k_shear = config.shear_modulus * cross_section_area;
        double k_extension = config.youngs_modulus * cross_section_area;

        K_inv_ = Matrix66::Zero();
        K_inv_(0, 0) = 1 / k_bending;
        K_inv_(1, 1) = 1 / k_bending;
        K_inv_(2, 2) = 1 / k_torsion;
        K_inv_(3, 3) = 1 / k_shear;
        K_inv_(4, 4) = 1 / k_shear;
        K_inv_(5, 5) = 1 / k_extension;

        // Tendon/Disc config
        tendon_config_ = generate_tendon_disc_config(
            config.num_discs, num_backbone_poses, config.routing_radius, config.angle_functions, config.angle_params);

        // Noise models
        tensions_cov_ = noiseModel::Isotropic::Sigma(4, config.tension_std);

        small_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_moment_std, config.small_moment_std, config.small_moment_std, 
            config.small_force_std, config.small_force_std, config.small_force_std).finished());
        
        base_frame_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_r_std, config.small_r_std, config.small_r_std, 
            config.small_p_std, config.small_p_std, config.small_p_std).finished());
        
        double r_std = config.cosserat_twist_r_std * ds_;
        
        cosserat_cov_ = noiseModel::Diagonal::Sigmas((Vector(12) << 
            r_std, r_std, r_std, 
            config.small_p_std, config.small_p_std, config.small_p_std,
            config.small_stress_std, config.small_stress_std, config.small_stress_std, 
            config.small_stress_std, config.small_stress_std, config.small_stress_std).finished());

        tip_stress_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_stress_std, config.small_stress_std, config.small_stress_std, 
            config.small_stress_std, config.small_stress_std, config.small_stress_std).finished());
    }

private:
    int backbone_idx_start_;
    int backbone_idx_end_;

    TendonDiscConfig tendon_config_;

    double ds_;
    Matrix66 K_inv_;

    noiseModel::Diagonal::shared_ptr tensions_cov_;
    noiseModel::Diagonal::shared_ptr small_wrench_cov_;
    noiseModel::Diagonal::shared_ptr base_frame_cov_;
    noiseModel::Diagonal::shared_ptr cosserat_cov_;
    noiseModel::Diagonal::shared_ptr tip_stress_cov_;

    NonlinearFactorGraph graph_;
    Values initial_values_;
    Values last_values_;

    void initialize_values(const Vector4 tensions, const Vector6 tip_wrench){
        initial_values_.clear();
        // TODO init tip_wrench
        initial_values_.insert(Q(0), tensions);

        for (size_t i = backbone_idx_start_; i <= backbone_idx_end_; ++i) {
            // Initialize pose
            if (last_values_.exists(T(i))) {
                initial_values_.insert(T(i), last_values_.at<Pose3>(T(i)));
            } else {
                // Initialize pose to be pure z translation, TODO change if base frame is not identity. This is actually wrong
                initial_values_.insert(T(i), Pose3(Rot3::Roll(0.01 * i), Point3(0.0, 0.01 * i, i * ds_)));
            }

            // Initialize stress
            if (last_values_.exists(S(i))) {
                initial_values_.insert(S(i), last_values_.at<Vector6>(S(i)));
            } else {
                initial_values_.insert(S(i), Vector6(Vector6::Zero()));
            }

            // Initialize wrench, (no wrench at i = 0)
            if (i > backbone_idx_start_) {
                if (last_values_.exists(F(i))) {
                    initial_values_.insert(F(i), last_values_.at<Vector6>(F(i)));
                } else {
                    initial_values_.insert(F(i), Vector6(Vector6::Zero()));
                }
            }
        }
    }

public:

    TendonRobotSolution solve(const Vector6& tip_wrench, const Vector4& tensions) {
        graph_.resize(0);
        initialize_values(tensions, tip_wrench);

        // Tendon tensions prior
        graph_.add(PriorFactor<Vector4>(Q(0), tensions, tensions_cov_));
        
        // Tip force prior
        // auto tip_wrench_cov = noiseModel::Diagonal::Sigmas((Vector(6) << tip_moment_std_, tip_moment_std_, tip_moment_std_, tip_force_std_, tip_force_std_, tip_force_std_).finished());
        // new_graph.add(PriorFactor<Vector6>(F(t_), tip_wrench_mean, tip_wrench_cov));
        // new_values.insert(tip_wrench_key_, tip_wrench_mean);    

        // Priors for wrenches along backbone
        for (int i = 1; i < backbone_idx_end_; i++) {
            bool is_tip = false;
            auto it = std::find(tendon_config_.disc_pose_idx.begin(),
                                tendon_config_.disc_pose_idx.end(), i);
            
            // If we are at a disc, add a disc wrench factor
            if (it != tendon_config_.disc_pose_idx.end()) {
                int disc_idx = std::distance(tendon_config_.disc_pose_idx.begin(), it);
                int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];
                int pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];

                auto factor = TendonDiscWrenchFactor(
                    T(pose_idx_prev), T(i), T(pose_idx_next), F(i), Q(0),
                    is_tip,
                    tendon_config_.local_holes[disc_idx - 1],
                    tendon_config_.local_holes[disc_idx],
                    tendon_config_.local_holes[disc_idx + 1],
                    small_wrench_cov_);
                
                graph_.add(factor);

            } else {
                Vector6 tip_wrench_mean = (i == backbone_idx_end_ - 1) ? tip_wrench : Vector6::Zero();
                graph_.add(PriorFactor<Vector6>(F(i), tip_wrench_mean, small_wrench_cov_));
            }
        }

        // For the last force, use a tip tendon wrench factor instead
        bool is_tip = true;
        graph_.add(TendonDiscWrenchFactor(T(tendon_config_.disc_pose_idx[tendon_config_.num_discs - 2]),
                                          T(tendon_config_.disc_pose_idx[tendon_config_.num_discs - 1]),
                                          T(0), // Dummy pose for tip factor, not used
                                          F(backbone_idx_end_),
                                          Q(0),
                                          is_tip,
                                          tendon_config_.local_holes[tendon_config_.num_discs - 2],
                                          tendon_config_.local_holes[tendon_config_.num_discs - 1],
                                          tendon_config_.local_holes[0], // Dummy, not used 
                                          small_wrench_cov_));

        // Base frame soft constraint
        graph_.add(PriorFactor<Pose3>(T(backbone_idx_start_), Pose3::Identity(), base_frame_cov_));

        // Cosserat twist factors
        for (size_t i = backbone_idx_start_; i < backbone_idx_end_; ++i) {
            auto factor = std::make_shared<CosseratRodFactor>(
                T(i), T(i + 1), S(i), S(i + 1), F(i + 1), ds_, K_inv_, cosserat_cov_);
            graph_.add(factor);
        }

        // Near-zero constraint for tip stress
        graph_.add(PriorFactor<Vector6>(
            S(backbone_idx_end_), Vector6::Zero(), tip_stress_cov_));

        graph_.saveGraph("factor_graph.dot", initial_values_);

        TendonRobotSolution output;

        auto start_solve = std::chrono::high_resolution_clock::now();

        LevenbergMarquardtParams params;
        params.setMaxIterations(15);
        params.setVerbosityLM("SUMMARY");
        params.setLinearSolverType("MULTIFRONTAL_QR");
        LevenbergMarquardtOptimizer optimizer(graph_, initial_values_, params);
        
        // GaussNewtonParams params;
        // params.setLinearSolverType("MULTIFRONTAL_QR");
        // params.setVerbosity("TERMINATION");
        // GaussNewtonOptimizer optimizer(graph_, initial_values_, params);
        
        Values result = optimizer.optimize();
        Marginals marginals(graph_, result);
        
        // int num_samples = 100;
        // std::vector<gtsam::Values> samples = sample_joint_distribution(graph_, result, num_samples);

        for (int i = backbone_idx_start_; i <= backbone_idx_end_; ++i) {
            output.backbone_pose_mean.push_back(result.at<Pose3>(T(i)));
            output.backbone_pose_cov.push_back(marginals.marginalCovariance(T(i)));
        }

        // for (int i = 0; i < num_samples; i++){
        //     const Values joint_sample = samples[i];

        //     std::vector<Pose3> backbone_pose_sample;
        //     backbone_pose_sample.reserve(backbone_idx_end_ - backbone_idx_start_ + 1);

        //     for (int j = backbone_idx_start_; j <= backbone_idx_end_; j++){
        //         backbone_pose_sample.push_back(joint_sample.at<Pose3>(T(j)));
        //     }

        //     output.backbone_pose_samples.push_back(backbone_pose_sample);
        // }

        output.tip_wrench_mean = result.at<Vector6>(F(backbone_idx_end_ - 1));
        output.tip_wrench_cov = marginals.marginalCovariance(F(backbone_idx_end_ - 1));

        output.tensions_mean = result.at<Vector4>(Q(0));
        output.tensions_cov = marginals.marginalCovariance(Q(0));

        auto end_solve = std::chrono::high_resolution_clock::now();
        output.solve_time_ms = std::chrono::duration<double, std::milli>(end_solve - start_solve).count();

        output.tendon_disc_config = tendon_config_;

        last_values_ = result;

        return output;
    }
};
}