#include <chrono>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>

#include "factors.h"
#include "types.h"

namespace gtsam{

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
                theta = angle_params[tendon_idx].angle_offset;
            } else if (angle_functions[tendon_idx] == RoutingAngleFunction::LINEAR) {
                theta = angle_params[tendon_idx].angle_offset + s * angle_params[tendon_idx].total_angle;
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
        num_backbone_poses_ = config.num_discs + (config.num_discs - 1) * config.poses_between_discs;
        ds_ = config.rod_length / (num_backbone_poses_ - 1);

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
            config.num_discs, num_backbone_poses_, config.routing_radius, config.angle_functions, config.angle_params);

        // Noise models
        tensions_cov_ = noiseModel::Isotropic::Sigma(4, config.tension_std);

        small_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_moment_std, config.small_moment_std, config.small_moment_std, 
            config.small_force_std, config.small_force_std, config.small_force_std).finished());

        tip_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_moment_std, config.small_moment_std, config.small_moment_std, 
            config.tip_force_std, config.tip_force_std, config.tip_force_std).finished());
        
        base_frame_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_r_std, config.small_r_std, config.small_r_std, 
            config.small_p_std, config.small_p_std, config.small_p_std).finished());
        
        cosserat_twist_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.cosserat_twist_r_std, config.cosserat_twist_r_std, config.cosserat_twist_r_std, 
            config.small_p_std, config.small_p_std, config.small_p_std).finished());

        prior_pose_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            3 * M_PI, 3 * M_PI, 3 * M_PI, 3 * config.rod_length, 3 * config.rod_length, 3 * config.rod_length).finished());
    }

private:
    int num_backbone_poses_;
    double ds_;
    Matrix66 K_inv_;

    TendonDiscConfig tendon_config_;

    noiseModel::Diagonal::shared_ptr tensions_cov_;
    noiseModel::Diagonal::shared_ptr small_wrench_cov_;
    noiseModel::Diagonal::shared_ptr tip_wrench_cov_;
    noiseModel::Diagonal::shared_ptr base_frame_cov_;
    noiseModel::Diagonal::shared_ptr cosserat_twist_cov_;
    noiseModel::Diagonal::shared_ptr prior_pose_cov_;

    Values values_;

    void initialize_values(const Vector4& tensions, const Vector6& tip_wrench){
        if (values_.exists(Q(0))) {
            values_.update(Q(0), tensions);
        } else {
            values_.insert(Q(0), tensions);
        }
        
        for (int i = 0; i < num_backbone_poses_; ++i) {
            // Initialize pose
            if (!values_.exists(T(i))) {
                values_.insert(T(i), Pose3(Rot3::Roll(0.01 * i), Point3(0.0, 0.01 * i, i * ds_)));
            }

            // Initialize stress
            if (!values_.exists(S(i))) {
                values_.insert(S(i), Vector6(Vector6::Zero()));
            }

            // Initialize wrench, (no wrench at i = 0) TODO add tip wrench
            if (i > 0 && !values_.exists(F(i))) {
                values_.insert(F(i), Vector6(Vector6::Zero()));
            }
        }
    }

    NonlinearFactorGraph build_graph(const Vector4& tensions_mean, const Vector6& tip_wrench_mean) {
        NonlinearFactorGraph graph;

        // Priors for wrenches along backbone
        for (int i = 1; i + 2 < num_backbone_poses_; i++) {
            bool is_tip = false;
            auto it = std::find(tendon_config_.disc_pose_idx.begin(),
                                tendon_config_.disc_pose_idx.end(), i);
            
            // If we are at a disc, add a disc wrench factor
            if (it != tendon_config_.disc_pose_idx.end()) {
                int disc_idx = std::distance(tendon_config_.disc_pose_idx.begin(), it);
                int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];
                int pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];

                graph.add(TendonDiscWrenchFactor(
                    T(pose_idx_prev), T(i), T(pose_idx_next), F(i), Q(0),
                    is_tip,
                    tendon_config_.local_holes[disc_idx - 1],
                    tendon_config_.local_holes[disc_idx],
                    tendon_config_.local_holes[disc_idx + 1],
                    small_wrench_cov_));
            } else {
                graph.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), small_wrench_cov_));
            }
        }

        // For the last force, use a tip tendon wrench factor instead
        bool is_tip = true;
        graph.add(TendonDiscWrenchFactor(T(tendon_config_.disc_pose_idx[tendon_config_.num_discs - 2]),
                                          T(tendon_config_.disc_pose_idx[tendon_config_.num_discs - 1]),
                                          T(0), // Dummy pose for tip factor, not used
                                          F(num_backbone_poses_ - 1),
                                          Q(0),
                                          is_tip,
                                          tendon_config_.local_holes[tendon_config_.num_discs - 2],
                                          tendon_config_.local_holes[tendon_config_.num_discs - 1],
                                          tendon_config_.local_holes[0], // Dummy, not used 
                                          small_wrench_cov_));

        // Base frame soft constraint?
        graph.add(PriorFactor<Pose3>(T(0), Pose3::Identity(), base_frame_cov_));

        // Soft pose prior for stability
        for (int i = 0; i < num_backbone_poses_; ++i) {
            graph.add(PriorFactor<Pose3>(T(i), Pose3::Identity(), prior_pose_cov_));
        }

        // Cosserat factors
        for (int i = 0; i + 1 < num_backbone_poses_; ++i) {
            graph.add(CosseratRodTwistFactor(
                T(i), T(i + 1), S(i), S(i + 1), ds_, K_inv_, cosserat_twist_cov_));
            graph.add(CosseratRodStressFactor(
                T(i), T(i + 1), S(i), S(i + 1), F(i + 1), small_wrench_cov_));
        }

        // Near-zero constraint for tip stress
        graph.add(PriorFactor<Vector6>(
            S(num_backbone_poses_ - 1), Vector6::Zero(), small_wrench_cov_));

        // Tendon tensions prior
        graph.add(PriorFactor<Vector4>(Q(0), tensions_mean, tensions_cov_));

        // Tip wrench actually applied at tip - 1, since tip already has a disc wrench
        graph.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 2), tip_wrench_mean, tip_wrench_cov_));

        return graph;
    }

    void solve_graph(const NonlinearFactorGraph& graph)
    {
        // LevenbergMarquardtParams params;
        // params.setVerbosityLM("SUMMARY");
        // params.setLinearSolverType("MULTIFRONTAL_QR");
        // LevenbergMarquardtOptimizer optimizer(graph, values_, params);

        DoglegParams params;
        // params.setDeltaInitial(0.01);
        params.setVerbosity("ERROR");
        // params.setLinearSolverType("MULTIFRONTAL_QR");
        // params.setLinearSolverType("SEQUENTIAL_QR");
        // params.setOrderingType("METIS");
        DoglegOptimizer optimizer(graph, values_, params);

        // GaussNewtonParams params;
        // params.setVerbosity("ERROR");
        // params.setLinearSolverType("MULTIFRONTAL_QR");
        // // params.setOrderingType("METIS");
        // GaussNewtonOptimizer optimizer(graph, values_, params);

        values_ = optimizer.optimize();
    }

    TendonRobotSolution extract_solution(const NonlinearFactorGraph& graph, const Marginals& marginals){
        TendonRobotSolution solution;

        for (int i = 0; i < num_backbone_poses_; ++i) {
            solution.backbone_pose_mean.push_back(values_.at<Pose3>(T(i)).matrix());
            solution.backbone_pose_cov.push_back(marginals.marginalCovariance(T(i)));
        }

        solution.tip_wrench_mean = values_.at<Vector6>(F(num_backbone_poses_ - 1 - 1));
        solution.tip_wrench_cov = marginals.marginalCovariance(F(num_backbone_poses_ - 1 - 1));

        solution.tensions_mean = values_.at<Vector4>(Q(0));
        solution.tensions_cov = marginals.marginalCovariance(Q(0));

        solution.tendon_disc_config = tendon_config_;

        return solution;
    }

    std::vector<Vector> sample_joint_cov(const Matrix& cov, int num_samples) {
        const int dim = cov.rows();
        Eigen::LLT<Matrix> llt(cov);
        Matrix L = llt.matrixL();

        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<> normal(0.0, 1.0);  // N(0,1)

        std::vector<Vector> samples;
        samples.reserve(num_samples);

        for (int n = 0; n < num_samples; ++n) {
            Vector z(dim);
            for (int i = 0; i < dim; ++i)
                z(i) = normal(gen);

            Vector delta = L * z;
            samples.push_back(delta);
        }

        return samples;
    }

    void sample_backbone_poses(const Marginals& marginals, TendonRobotSolution& solution, const int num_samples) {
        KeyVector pose_keys;
        for (int i = 0; i < num_backbone_poses_; ++i) {
            pose_keys.push_back(T(i));
        }

        Matrix pose_joint_cov = marginals.jointMarginalCovariance(pose_keys).fullMatrix();
        std::vector<Vector> deltas = sample_joint_cov(pose_joint_cov, num_samples);

        for(int i = 0; i < num_samples; i++) {
            VectorValues vector_values;
            int idx = 0;
            for (const auto& key : pose_keys) {
                Vector6 increment = deltas[i].segment(idx, 6);  // Pose3 tangent dim is 6
                vector_values.insert(key, increment);
                idx += 6;
            }

            Values values_sample = values_.retract(vector_values);
            std::vector<Matrix4> backbone_sample;
            for (int j = 0; j < num_backbone_poses_; j++) {
                backbone_sample.push_back(values_sample.at<Pose3>(T(j)).matrix());
            }
            solution.backbone_pose_samples.push_back(backbone_sample);
        }
    }

    void sample_solution(
        const NonlinearFactorGraph& graph, 
        const Marginals& marginals, 
        TendonRobotSolution& solution,
        int num_samples) 
    {
        sample_backbone_poses(marginals, solution, num_samples);
    }
        
public:

    TendonRobotSolution solve(const Vector4& tensions, const Vector6& tip_wrench, int num_samples) {
        
        auto start_initialize = std::chrono::high_resolution_clock::now();

        initialize_values(tensions, tip_wrench);
        NonlinearFactorGraph graph = build_graph(tensions, tip_wrench);

        auto end_initialize = std::chrono::high_resolution_clock::now();

        // if(save_graph)
        //     graph.saveGraph("tendon_robot_graph.dot", initial_values);

        auto start_solve = std::chrono::high_resolution_clock::now();

        solve_graph(graph);

        auto end_solve = std::chrono::high_resolution_clock::now();

        auto start_extract = std::chrono::high_resolution_clock::now();

        Marginals marginals(graph, values_);
        TendonRobotSolution solution = extract_solution(graph, marginals);
        sample_solution(graph, marginals, solution, num_samples);

        auto end_extract = std::chrono::high_resolution_clock::now();

        solution.build_time_ms = std::chrono::duration<double, std::milli>(end_initialize - start_initialize).count();
        solution.solve_time_ms = std::chrono::duration<double, std::milli>(end_solve - start_solve).count();
        solution.extract_time_ms = std::chrono::duration<double, std::milli>(end_extract - start_extract).count();
        solution.total_time_ms = std::chrono::duration<double, std::milli>(end_extract - start_initialize).count();

        return solution;
    }
};
}