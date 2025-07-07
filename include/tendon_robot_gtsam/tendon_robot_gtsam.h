#include <chrono>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
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
    std::vector<std::vector<Vector3>> local_hole_locations;  // (disc, tendon)
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
    config.local_hole_locations.reserve(num_discs);

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

        config.local_hole_locations.push_back(holes);
    }

    return config;
}

std::vector<gtsam::Values> sample_joint_distribution(
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::Values& map_values,
    int num_samples)
{
    auto linear_graph = graph.linearize(map_values);
    auto bayesNet = linear_graph->eliminateSequential();

    std::vector<Values> samples;
    for (int i = 0; i < num_samples; ++i) {
        VectorValues del_values = bayesNet->sample();
        Values sample = map_values.retract(del_values);
        samples.push_back(sample);
    }
    
    return samples;
}

struct TendonRobotSolution {
    std::vector<Pose3> backbone_pose_mean;
    std::vector<Matrix6> backbone_pose_cov;

    std::vector<std::vector<Pose3>> backbone_pose_samples;

    Vector6 tip_wrench_mean;
    Matrix6 tip_wrench_cov;

    Vector4 tensions_mean;
    Matrix4 tensions_cov;

    double solve_time_ms;

    TendonDiscConfig tendon_disc_config;
};

using symbol_shorthand::T;
using symbol_shorthand::F;
using symbol_shorthand::S;
using symbol_shorthand::Q;
// using symbol_shorthand::D;

class TendonRobotGtsam {
public:
    TendonRobotGtsam(int num_discs = 12, int poses_between_each = 5) {
        int num_backbone_poses = num_discs + (num_discs - 1) * poses_between_each;
        backbone_idx_start_ = 0;
        backbone_idx_end_ = backbone_idx_start_ + num_backbone_poses - 1;

        tensions_key_ = Q(0);

        K_ = Matrix66::Zero();
        K_(0, 0) = k_bending_;
        K_(1, 1) = k_bending_;
        K_(2, 2) = k_torsion_;
        K_(3, 3) = k_shear_;
        K_(4, 4) = k_shear_;
        K_(5, 5) = k_extension_;

        K_inv_ = K_.inverse();

        ds_ = rod_length_ / (num_backbone_poses - 1);

        std::vector<RoutingAngleFunction> angle_functions;
        angle_functions.push_back(RoutingAngleFunction::LINEAR);
        angle_functions.push_back(RoutingAngleFunction::CONSTANT);
        angle_functions.push_back(RoutingAngleFunction::CONSTANT);
        angle_functions.push_back(RoutingAngleFunction::CONSTANT);

        std::vector<RoutingParams> angle_params;
        angle_params.push_back({0.0,     2 * M_PI});
        angle_params.push_back({M_PI,         0.0});
        angle_params.push_back({3 * M_PI / 2, 0.0});
        angle_params.push_back({0.0,          0.0});

        double routing_radius = 0.008;
        tendon_config_ = generate_tendon_disc_config(
            num_discs, num_backbone_poses, routing_radius, angle_functions, angle_params);
    }

private:
    int backbone_idx_start_;
    int backbone_idx_end_;

    TendonDiscConfig tendon_config_;

    double ds_;

    Key tensions_key_;

    Values last_result_;

    // Parameters
    static constexpr double tension_std = 1e-2;

    static constexpr double cosserat_twist_r_std_ = 1e-1; // This just looks right
    static constexpr double small_r_std_ = 1e-3;
    static constexpr double small_p_std_ = 1e-5;

    // static constexpr double tip_force_std_ = 1e-3;
    // static constexpr double tip_moment_std_ = 1e-3;

    static constexpr double rod_length_ = 0.2; 
    static constexpr double rod_diameter_ = 1.0e-3;
    static constexpr double youngs_modulus_ = 35.0e9;  // Nitinol
    static constexpr double shear_modulus_ = 12.0e9;  // Nitinol
    
    static constexpr double cross_section_area_ = M_PI * std::pow(rod_diameter_, 2) / 4.0;
    static constexpr double cross_section_moment_ = M_PI * std::pow(rod_diameter_, 4) / 64.0;

    static constexpr double k_bending_ = youngs_modulus_ * cross_section_moment_;
    static constexpr double k_torsion_ = 2.0 * shear_modulus_ * cross_section_moment_;
    static constexpr double k_shear_ = shear_modulus_ * cross_section_area_;
    static constexpr double k_extension_ = youngs_modulus_ * cross_section_area_;

    Matrix66 K_, K_inv_;

public:
    TendonRobotSolution update(const Vector6& tip_wrench_mean, const Vector4& tensions_mean) {
        NonlinearFactorGraph graph;
        Values initial_values;

        // Tendon tensions prior
        auto tensions_cov = noiseModel::Isotropic::Sigma(4, tension_std);
        graph.add(PriorFactor<Vector4>(tensions_key_, tensions_mean, tensions_cov));
        initial_values.insert(tensions_key_, tensions_mean);

        // Tip force prior
        // auto tip_wrench_cov = noiseModel::Diagonal::Sigmas((Vector(6) << tip_moment_std_, tip_moment_std_, tip_moment_std_, tip_force_std_, tip_force_std_, tip_force_std_).finished());
        // new_graph.add(PriorFactor<Vector6>(F(t_), tip_wrench_mean, tip_wrench_cov));
        // new_values.insert(tip_wrench_key_, tip_wrench_mean);    

        // Priors for forces
        double small_force_std = 1e-5;
        double small_moment_std = 1e-5;
        auto small_wrench_cov = noiseModel::Diagonal::Sigmas(
            (Vector(6) << small_moment_std, small_moment_std, small_moment_std, 
                          small_force_std, small_force_std, small_force_std).finished());
        
        // double disc_force_std = 1e-2;
        // double disc_moment_std = 1e-3;
        // auto disc_wrench_cov = noiseModel::Diagonal::Sigmas(
        //     (Vector(6) << disc_moment_std, disc_moment_std, disc_moment_std, 
        //                   disc_force_std, disc_force_std, disc_force_std).finished());
        
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
                    T(pose_idx_prev), T(i), T(pose_idx_next), F(i), tensions_key_,
                    is_tip,
                    tendon_config_.local_hole_locations[disc_idx - 1],
                    tendon_config_.local_hole_locations[disc_idx],
                    tendon_config_.local_hole_locations[disc_idx + 1],
                    small_wrench_cov);
                
                graph.add(factor);

            } else {
                graph.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), small_wrench_cov));
            }
        }

        // For the last force, use a tip tendon wrench factor instead
        bool is_tip = true;
        graph.add(TendonDiscWrenchFactor(T(tendon_config_.disc_pose_idx[tendon_config_.num_discs - 2]),
                                         T(tendon_config_.disc_pose_idx[tendon_config_.num_discs - 1]),
                                         T(0), // Dummy pose for tip factor, not used
                                         F(backbone_idx_end_),
                                         tensions_key_,
                                         is_tip,
                                         tendon_config_.local_hole_locations[tendon_config_.num_discs - 2],
                                         tendon_config_.local_hole_locations[tendon_config_.num_discs - 1],
                                         tendon_config_.local_hole_locations[0], // Dummy, not used 
                                         small_wrench_cov));

        // Base frame constraint
        auto base_frame_cov = noiseModel::Diagonal::Sigmas((Vector(6) << small_r_std_, small_r_std_, small_r_std_, small_p_std_, small_p_std_, small_p_std_).finished());
        graph.add(PriorFactor<Pose3>(T(backbone_idx_start_), Pose3::Identity(), base_frame_cov));

        // Cosserat twist factors
        double r_std = cosserat_twist_r_std_ * ds_;
        double stress_std = 1e-5;

        auto cosserat_cov = noiseModel::Diagonal::Sigmas((Vector(12) << 
            r_std, r_std, r_std, small_p_std_, small_p_std_, small_p_std_,
            stress_std, stress_std, stress_std, stress_std, stress_std, stress_std).finished());
        // auto cosserat_cov = noiseModel::Constrained::MixedSigmas((Vector(12) << 
        //     r_std, r_std, r_std, small_p_std_, small_p_std_, small_p_std_,
        //     0, 0, 0, 0, 0, 0).finished());

        for (size_t i = backbone_idx_start_; i < backbone_idx_end_; ++i) {
            auto factor = std::make_shared<CosseratRodFactor>(
                T(i), T(i + 1), S(i), S(i + 1), F(i + 1), ds_, K_inv_, cosserat_cov);
            graph.add(factor);
        }

        // Near-zero constraint for tip stress
        auto tip_stress_cov = noiseModel::Diagonal::Sigmas((Vector(6) << 
            stress_std, stress_std, stress_std, stress_std, stress_std, stress_std).finished());
        // auto tip_stress_cov = noiseModel::Constrained::All(6);
        graph.add(PriorFactor<Vector6>(
            S(backbone_idx_end_), Vector6::Zero(), tip_stress_cov));

        // Initialize values
        for (size_t i = backbone_idx_start_; i <= backbone_idx_end_; ++i) {
            // Initialize pose
            if (last_result_.exists(T(i))) {
                initial_values.insert(T(i), last_result_.at<Pose3>(T(i)));
            } else {
                initial_values.insert(T(i), Pose3::Identity());
            }

            // Initialize stress
            if (last_result_.exists(S(i))) {
                initial_values.insert(S(i), last_result_.at<Vector6>(S(i)));
            } else {
                initial_values.insert(S(i), Vector6(Vector6::Zero()));
            }

            // Initialize wrench, (no wrench at i = 0)
            if (i > backbone_idx_start_) {
                if (last_result_.exists(F(i))) {
                    initial_values.insert(F(i), last_result_.at<Vector6>(F(i)));
                } else {
                    initial_values.insert(F(i), Vector6(Vector6::Zero()));
                }
            }
        }
        
        graph.saveGraph("factor_graph.dot", initial_values);

        TendonRobotSolution output;

        auto start_solve = std::chrono::high_resolution_clock::now();

        LevenbergMarquardtParams params;
        params.setVerbosityLM("SUMMARY");
        params.setlambdaInitial(0.1);
        LevenbergMarquardtOptimizer optimizer(graph, initial_values, params);

        Values result = optimizer.optimize();
        Marginals marginals(graph, result);
        
        int num_samples = 100;
        std::vector<gtsam::Values> samples = sample_joint_distribution(graph, result, num_samples);

        for (int i = backbone_idx_start_; i <= backbone_idx_end_; ++i) {
            output.backbone_pose_mean.push_back(result.at<Pose3>(T(i)));
            output.backbone_pose_cov.push_back(marginals.marginalCovariance(T(i)));
        }

        for (int i = 0; i < num_samples; i++){
            const Values joint_sample = samples[i];

            std::vector<Pose3> backbone_pose_sample;
            backbone_pose_sample.reserve(backbone_idx_end_ - backbone_idx_start_ + 1);

            for (int j = backbone_idx_start_; j <= backbone_idx_end_; j++){
                backbone_pose_sample.push_back(joint_sample.at<Pose3>(T(j)));
            }

            output.backbone_pose_samples.push_back(backbone_pose_sample);
        }

        // output.tip_wrench_mean = result.at<Vector6>(F(t_));
        // output.tip_wrench_cov = marginals.marginalCovariance(F(t_));

        // output.tensions_mean = result.at<Vector4>(tensions_key_);
        // output.tensions_cov = marginals.marginalCovariance(tensions_key_);

        auto end_solve = std::chrono::high_resolution_clock::now();
        output.solve_time_ms = std::chrono::duration<double, std::milli>(end_solve - start_solve).count();

        output.tendon_disc_config = tendon_config_;

        last_result_ = result;

        return output;
    }
};
}