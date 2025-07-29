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

    std::unordered_set<int> disc_pose_set(config.disc_pose_idx.begin(), config.disc_pose_idx.end());
    config.no_disc_pose_idx.reserve(num_poses - num_discs);

    for (int i = 0; i < num_poses; ++i) {
        if (disc_pose_set.find(i) == disc_pose_set.end()) {
            config.no_disc_pose_idx.push_back(i);
        }
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
        
        tip_pose_meas_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.tip_pose_r_meas_std, config.tip_pose_r_meas_std, config.tip_pose_r_meas_std,
            config.tip_pose_p_meas_std, config.tip_pose_p_meas_std, config.tip_pose_p_meas_std).finished());

        last_tip_pose_mean_ = Pose3(Rot3::Identity(), Point3(0, 0, config.rod_length));
        last_tip_pose_cov_ = 0.001 * Matrix6::Identity();
        tip_pose_drift_cov_ = (Vector(6) <<
            config.pose_drift_r_std, config.pose_drift_r_std, config.pose_drift_r_std, 
            config.pose_drift_p_std, config.pose_drift_p_std, config.pose_drift_p_std).finished().array().square().matrix().asDiagonal();

        last_tensions_mean_ = Vector4::Zero();
        last_tensions_cov_ = 0.001 * Matrix4::Identity();
        tensions_drift_cov_ = (config.tension_drift_std * Vector4::Ones()).array().square().matrix().asDiagonal();

        last_tip_wrench_mean_ = Vector6::Zero();
        last_tip_wrench_cov_ = 0.001 * Matrix6::Identity();
        tip_wrench_drift_cov_ = (config.wrench_drift_std * Vector6::Ones()).array().square().matrix().asDiagonal();

        initialize_values();
    }

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
    noiseModel::Diagonal::shared_ptr tip_pose_meas_cov_;

    Matrix6 tip_velocity_drift_cov_;
    Matrix4 tensions_drift_cov_;
    Matrix6 tip_wrench_drift_cov_;

    Pose3 last_tip_pose_mean_;
    Vector6 last_tip_velocity_mean_;
    
    Matrix6 last_tip_pose_cov_;
    Vector4 last_tensions_mean_;
    Matrix4 last_tensions_cov_;
    Vector6 last_tip_wrench_mean_;
    Matrix6 last_tip_wrench_cov_;

    Ordering ordering_;
    bool is_first_solve_ = true;
    Values values_;

    void initialize_values(){
        if (!values_.exists(Q(0))) {
            values_.insert(Q(0), Vector4(Vector4::Zero()));
        }
        
        for (int i = 0; i < num_backbone_poses_; ++i) {
            // Initialize pose
            if (!values_.exists(T(i))) {
                values_.insert(T(i), Pose3(Rot3::Identity(), Point3(0.0, 0.0, i * ds_)));
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

    NonlinearFactorGraph build_graph_base() {
        NonlinearFactorGraph graph;
        
        // Priors for discs (using disc indices), start at 1, no force at base disc
        for (size_t disc_idx = 1; disc_idx + 1 < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
            int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
            int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];
            int pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];
            bool is_tip = false;

            graph.add(TendonDiscWrenchFactor(
                T(pose_idx_prev), T(pose_idx), T(pose_idx_next), F(pose_idx), Q(0),
                is_tip,
                tendon_config_.local_holes[disc_idx - 1],
                tendon_config_.local_holes[disc_idx],
                tendon_config_.local_holes[disc_idx + 1],
                small_wrench_cov_));
        }

        // For the last disc, use a tip tendon wrench factor instead
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

        // Base frame soft constraint
        graph.add(PriorFactor<Pose3>(T(0), Pose3::Identity(), base_frame_cov_));

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
        
        // Priors from last time step to smooth in time, like Kalman filter
        add_last_state_prior(graph);

        return graph;
    }

    void add_last_state_prior(NonlinearFactorGraph& graph) {
        graph.add(PriorFactor<Pose3>(T(num_backbone_poses_ - 1), last_tip_pose_mean_, 
            noiseModel::Gaussian::Covariance(last_tip_pose_cov_ + tip_pose_drift_cov_)));

        graph.add(PriorFactor<Vector4>(Q(0), last_tensions_mean_,
            noiseModel::Gaussian::Covariance(last_tensions_cov_ + tensions_drift_cov_)));

        graph.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 2), last_tip_wrench_mean_, 
            noiseModel::Gaussian::Covariance(last_tip_wrench_cov_ + tip_wrench_drift_cov_)));
    }

    void solve_graph(const NonlinearFactorGraph& graph)
    {
        // Reusing the variable ordering can save a few ms.
        // Note that graph structure must be identical every call.
        if (is_first_solve_) {
            ordering_ = Ordering::Colamd(graph);
            is_first_solve_ = false;
        }
        
        // GaussNewtonParams params;
        // // params.setOrdering(ordering_);
        // params.setVerbosity("TERMINATION");
        // // params.setLinearSolverType("MULTIFRONTAL_QR");
        // GaussNewtonOptimizer optimizer(graph, values_, params);

        DoglegParams params;
        params.setVerbosity("TERMINATION");
        params.setOrdering(ordering_);
        // params.setLinearSolverType("MULTIFRONTAL_QR");
        DoglegOptimizer optimizer(graph, values_, params);

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

    void sample_solution(
        const NonlinearFactorGraph& graph, 
        const Marginals& marginals, 
        TendonRobotSolution& solution,
        int num_samples) 
    {
        Pose3 tip_pose_mean = Pose3(solution.backbone_pose_mean.back());
        Matrix6 tip_pose_cov = solution.backbone_pose_cov.back();

        std::vector<Vector> d_tip_pose = sample_joint_cov(tip_pose_cov, num_samples);

        for (int i = 0; i < num_samples; i++) {
            solution.tip_pose_samples.push_back(
                tip_pose_mean.retract(d_tip_pose[i]).matrix());
        }
    }
        
    TendonRobotSolution update(const NonlinearFactorGraph& graph, int num_samples) {
        auto start_solve = std::chrono::high_resolution_clock::now();

        solve_graph(graph);

        auto end_solve = std::chrono::high_resolution_clock::now();

        auto start_extract = std::chrono::high_resolution_clock::now();

        Marginals marginals(graph, values_);
        TendonRobotSolution solution = extract_solution(graph, marginals);
        sample_solution(graph, marginals, solution, num_samples);

        auto end_extract = std::chrono::high_resolution_clock::now();

        solution.solve_time_ms = std::chrono::duration<double, std::milli>(end_solve - start_solve).count();
        solution.extract_time_ms = std::chrono::duration<double, std::milli>(end_extract - start_extract).count();
        solution.total_time_ms = std::chrono::duration<double, std::milli>(end_extract - start_solve).count();

        last_tip_pose_mean_ = values_.at<Pose3>(T(num_backbone_poses_ - 1));
        last_tip_pose_cov_ = marginals.marginalCovariance(T(num_backbone_poses_ - 1));
        last_tensions_mean_ = values_.at<Vector4>(Q(0));
        last_tensions_cov_ = marginals.marginalCovariance(Q(0));
        last_tip_wrench_mean_ = values_.at<Vector6>(F(num_backbone_poses_ - 2));
        last_tip_wrench_cov_ = marginals.marginalCovariance(F(num_backbone_poses_ - 2));

        return solution;
    }
};

class TipForceSim : public TendonRobotGtsam {
public:
    using TendonRobotGtsam::TendonRobotGtsam;

private:
    void add_loading_factors(const Vector4& tensions, const Vector3& tip_force, NonlinearFactorGraph& graph) {
        // Wrench priors at non-disc poses
        for (int i : tendon_config_.no_disc_pose_idx) {
            if (i == num_backbone_poses_ - 2)
                continue;  // tip wrench is handled below
            graph.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), small_wrench_cov_));
        }

        // Tip wrench actually applied at tip - 1, since tip already has a disc wrench
        Vector6 tip_wrench_mean;
        tip_wrench_mean.head<3>() = Vector3::Zero();
        tip_wrench_mean.tail<3>() = tip_force;
        graph.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 2), tip_wrench_mean, tip_wrench_cov_));

        // Tendon tensions measurement prior
        graph.add(PriorFactor<Vector4>(Q(0), tensions, tensions_cov_));
    }

public:
    TendonRobotSolution step(const Vector4& tensions, const Vector3& tip_force, int num_samples) {
        NonlinearFactorGraph graph = build_graph_base();
        add_loading_factors(tensions, tip_force, graph);

        return update(graph, num_samples);
    }
};

class TipForceSolver : public TendonRobotGtsam {
public:
    using TendonRobotGtsam::TendonRobotGtsam;

private:
    void add_measurement_factors(const Vector4& tensions_meas, const Vector3& tip_position_meas, NonlinearFactorGraph& graph) {
        // Wrench priors at non-disc poses
        for (int i : tendon_config_.no_disc_pose_idx) {
            if (i == num_backbone_poses_ - 2)
                continue;  // tip wrench is handled below
            graph.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), small_wrench_cov_));
        }

        // Tip wrench actually applied at tip - 1, since tip already has a disc wrench
        graph.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 2), Vector6::Zero(), tip_wrench_cov_));

        Pose3 tip_pose_mean = Pose3(Rot3::Identity(), tip_position_meas);
        graph.add(PriorFactor<Pose3>(T(num_backbone_poses_ - 1), tip_pose_mean, tip_pose_meas_cov_));

        // Tendon tensions measurement prior
        graph.add(PriorFactor<Vector4>(Q(0), tensions_meas, tensions_cov_));
    }

public:
    TendonRobotSolution step(const Vector4& tensions_meas, const Vector3& tip_position_meas, int num_samples) {
        NonlinearFactorGraph graph = build_graph_base();
        add_measurement_factors(tensions_meas, tip_position_meas, graph);

        return update(graph, num_samples);
    }
};
}



