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
    const std::vector<RoutingFunctionParams>& angle_params)
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

TendonRobotConfig get_default_config(){
    gtsam::TendonRobotConfig config;

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

using symbol_shorthand::T; // poses
using symbol_shorthand::F; // applied wrenches
using symbol_shorthand::D; // disc wrenches
using symbol_shorthand::S; // internal stresses
using symbol_shorthand::Q; // tendon tensions

class TendonRobotGtsam {
public:
    TendonRobotGtsam(const TendonRobotConfig& config) {
        num_backbone_poses_ = config.num_discs + (config.num_discs - 1) * config.poses_between_discs;
        ds_ = config.rod_length / (num_backbone_poses_ - 1);
        rod_diameter_ = config.rod_diameter;
        use_midpoint_ = config.use_midpoint;

        // Build stiffness matrix
        double cross_section_area = M_PI * std::pow(config.rod_diameter, 2) / 4.0;
        double cross_section_moment = M_PI * std::pow(config.rod_diameter, 4) / 64.0;

        double k_bending = config.youngs_modulus * cross_section_moment;
        double k_torsion = 2.0 * config.shear_modulus * cross_section_moment;
        double k_shear = config.shear_modulus * cross_section_area;
        double k_extension = config.youngs_modulus * cross_section_area;

        K_inv_ = Matrix6::Zero();
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
        tensions_cov_ = noiseModel::Isotropic::Sigma(4, config.tension_meas_std);

        small_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_moment_std, config.small_moment_std, config.small_moment_std, 
            config.small_force_std, config.small_force_std, config.small_force_std).finished());
        
        base_frame_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_r_std, config.small_r_std, config.small_r_std, 
            config.small_p_std, config.small_p_std, config.small_p_std).finished());
        
        cosserat_twist_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.cosserat_twist_r_std, config.cosserat_twist_r_std, config.cosserat_twist_r_std, 
            config.small_p_std, config.small_p_std, config.small_p_std).finished());

        initialize_values();
    }

    int num_backbone_poses_;
    bool use_midpoint_;
    double ds_;
    double rod_diameter_;
    Matrix66 K_inv_;

    TendonDiscConfig tendon_config_;

    noiseModel::Diagonal::shared_ptr tensions_cov_;
    noiseModel::Diagonal::shared_ptr small_wrench_cov_;
    noiseModel::Diagonal::shared_ptr base_frame_cov_;
    noiseModel::Diagonal::shared_ptr cosserat_twist_cov_;

    Ordering ordering_;
    bool is_first_solve_ = true;
    Values values_;
    NonlinearFactorGraph graph_;
    Marginals marginals_;
    TendonRobotSolution solution_;

    void initialize_values(){
        values_.clear();

        values_.insert(Q(0), Vector4(Vector4::Zero()));

        for (int i = 0; i < num_backbone_poses_; ++i) {
            values_.insert(T(i), Pose3(Rot3::Identity(), Point3(0.0, 0.0, i * ds_)));
            values_.insert(S(i), Vector6(Vector6::Zero()));
            if (i > 0) { values_.insert(F(i), Vector6(Vector6::Zero())); }
        }

        for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
            values_.insert(D(tendon_config_.disc_pose_idx[disc_idx]), Vector6(Vector6::Zero()));
        }
    }

    void build_graph_base(const Vector4& tensions) {
        graph_.resize(0);
        
        // Prior on tensions measurements
        graph_.add(PriorFactor<Vector4>(Q(0), tensions, tensions_cov_));

        // Priors for discs (using disc indices), start at 1, no force at base disc
        for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
            int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
            int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];
            std::vector<Vector3> holes_prev = tendon_config_.local_holes[disc_idx - 1];
            std::vector<Vector3> holes = tendon_config_.local_holes[disc_idx];

            // Some inputs change based on whether we are at the final disc
            bool is_tip;
            int pose_idx_next; 
            std::vector<Vector3> holes_next;

            if (disc_idx == (tendon_config_.disc_pose_idx.size() - 1)) {
                is_tip = true;
                pose_idx_next = T(0); // Dummy pose for tip factor, not used for tip disc
                holes_next = tendon_config_.local_holes[0]; // Dummy holes, not used in factor
            } else {
                is_tip = false;
                pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];
                holes_next = tendon_config_.local_holes[disc_idx + 1];
            }

            graph_.add(TendonDiscWrenchFactor(
                       T(pose_idx_prev), T(pose_idx), T(pose_idx_next), D(pose_idx), Q(0), F(pose_idx),
                       is_tip, holes_prev, holes, holes_next, small_wrench_cov_));
        }

        // Base frame soft constraint
        graph_.add(PriorFactor<Pose3>(T(0), Pose3::Identity(), base_frame_cov_));

        // Cosserat twist factors
        for (int i = 0; i + 1 < num_backbone_poses_; ++i) {
            graph_.add(CosseratRodTwistFactor(
                T(i), T(i + 1), S(i), S(i + 1), ds_, K_inv_, use_midpoint_, cosserat_twist_cov_));
        }

        // Cosserat stress factors
        for (int i = 0; i + 1 < num_backbone_poses_; ++i) {
            // If a disc is next, then that changes whether we use an applied wrench F or a disc wrench D
            bool is_disc_next = std::find(tendon_config_.disc_pose_idx.begin(), tendon_config_.disc_pose_idx.end(), i + 1) 
                != tendon_config_.disc_pose_idx.end();
            
            graph_.add(CosseratRodStressFactor(
                T(i), T(i + 1), 
                S(i), S(i + 1),
                is_disc_next ? D(i + 1) : F(i + 1), 
                is_disc_next ? false : true,
                small_wrench_cov_));
        }

        // Near-zero prior constraint for tip stress
        graph_.add(PriorFactor<Vector6>(S(num_backbone_poses_ - 1), Vector6::Zero(), small_wrench_cov_));
    }

    void solve_graph()
    {
        // Reusing the variable ordering can save a few ms.
        if (is_first_solve_) {
            ordering_ = Ordering::Colamd(graph_);
            is_first_solve_ = false;
        }
        
        // LevenbergMarquardtParams params;
        // params.setVerbosityLM("SUMMARY");
        // GaussNewtonParams params;
        DoglegParams params;
        params.setVerbosity("TERMINATION");
        params.setOrdering(ordering_);
        params.setLinearSolverType("MULTIFRONTAL_QR");
        // GaussNewtonOptimizer optimizer(graph_, values_, params);
        DoglegOptimizer optimizer(graph_, values_, params);
        // LevenbergMarquardtOptimizer optimizer(graph_, values_, params);

        values_ = optimizer.optimize();
    }

    void extract_solution(){
        marginals_ = Marginals(graph_, values_);

        for (int i = 0; i < num_backbone_poses_; ++i) {
            solution_.backbone_pose_mean[i] = values_.at<Pose3>(T(i)).matrix();
            solution_.backbone_pose_cov[i] = marginals_.marginalCovariance(T(i));

            // No applied force at the base pose
            if (i > 0) {
                solution_.applied_wrench_mean[i - 1] = values_.at<Vector6>(F(i));
                solution_.applied_wrench_cov[i - 1] = marginals_.marginalCovariance(F(i));
            }
        }

        solution_.tensions_mean = values_.at<Vector4>(Q(0));
        solution_.tensions_cov = marginals_.marginalCovariance(Q(0));

        solution_.tendon_disc_config = tendon_config_;

        KeyVector keys;
        keys.push_back(Q(0));
        keys.push_back(T(num_backbone_poses_ - 1));
        JointMarginal tensions_pose_joint = marginals_.jointMarginalCovariance(keys);

        Matrix4 sigma_tensions_tensions = tensions_pose_joint(Q(0), Q(0));
        Matrix64 sigma_pose_tensions = tensions_pose_joint(T(num_backbone_poses_ - 1), Q(0));

        Eigen::LDLT<Eigen::MatrixXd> ldlt(sigma_tensions_tensions);
        solution_.J_pose_tensions = sigma_pose_tensions * ldlt.solve(Matrix4::Identity());
    }

    std::vector<Vector> sample_cov(const Matrix& cov, int num_samples) {
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

    void sample_tip_pose(int num_samples) {
        Pose3 tip_pose_mean = Pose3(solution_.backbone_pose_mean.back());
        Matrix6 tip_pose_cov = solution_.backbone_pose_cov.back();

        std::vector<Vector> d_tip_pose = sample_cov(tip_pose_cov, num_samples);
        d_tip_pose.reserve(num_samples);

        for (int i = 0; i < num_samples; i++) {
            solution_.tip_pose_samples[i] = tip_pose_mean.retract(d_tip_pose[i]).matrix();
        }
    }

    void sample_fbg_array(int num_samples) {   
        KeyVector stress_keys;
        stress_keys.reserve(num_backbone_poses_);

        for (int i = 0; i < num_backbone_poses_; ++i) {
            stress_keys.push_back(S(i));
        }

        Matrix joint_stress_cov = marginals_.jointMarginalCovariance(stress_keys).fullMatrix();
        std::vector<Vector> joint_d_stresses = sample_cov(joint_stress_cov, num_samples);
        
        for (int i = 0; i < num_samples; ++i) {
            std::vector<Vector3> fbg_array_sample;
            fbg_array_sample.reserve(num_backbone_poses_);

            for (int j = 0; j < num_backbone_poses_; ++j) {
                Vector6 stress_mean = values_.at<Vector6>(S(j));

                Vector6 d_stress = joint_d_stresses[i].segment<6>(6 * j);
                Vector6 stress = stress_mean + d_stress;

                fbg_array_sample.push_back(stress_to_fbg_signal(stress, K_inv_, rod_diameter_));
            }
            solution_.fbg_array_samples[i] = fbg_array_sample;
        }
    }

    void sample_solution(int num_samples) 
    {
        sample_tip_pose(num_samples);
        sample_fbg_array(num_samples);
    }
        
    void update(int num_samples) {
        // graph.saveGraph("graph.dot", values_);
        solution_ = TendonRobotSolution(num_backbone_poses_, num_samples);

        auto start_solve = std::chrono::high_resolution_clock::now();

        solve_graph();

        auto start_extract = std::chrono::high_resolution_clock::now();

        extract_solution();
        sample_solution(num_samples);
        
        auto end_extract = std::chrono::high_resolution_clock::now();

        solution_.solve_time_ms = std::chrono::duration<double, std::milli>(start_extract - start_solve).count();
        solution_.extract_time_ms = std::chrono::duration<double, std::milli>(end_extract - start_extract).count();
        solution_.total_time_ms = std::chrono::duration<double, std::milli>(end_extract - start_solve).count();
    }
};


class TipForceSolver : public TendonRobotGtsam {
public:
    TipForceSolver(const TendonRobotConfig& config) 
        : TendonRobotGtsam(config)
    {
        tip_wrench_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_moment_std, config.small_moment_std, config.small_moment_std, 
            config.tip_force_prior_std, config.tip_force_prior_std, config.tip_force_prior_std).finished());
            
        tip_position_meas_cov_ = noiseModel::Isotropic::Sigma(3, config.tip_position_meas_std);
    }

private:
    noiseModel::Diagonal::shared_ptr tip_position_meas_cov_;
    noiseModel::Diagonal::shared_ptr tip_wrench_cov_;

    void add_common_factors() {
        // Applied wrenches are all zero, exect at the tip
        for (int i = 1; i + 1 < num_backbone_poses_; i++) {
            graph_.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), small_wrench_cov_));
        }
    }

public:
    TendonRobotSolution step(const Vector4& tensions_meas, const Vector3& tip_position_meas, int num_samples) {
        build_graph_base(tensions_meas);
        add_common_factors();

        // Tip force prior is zero with big uncertainty
        graph_.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 1), Vector6::Zero(), tip_wrench_cov_));

        // Tip pose measurement prior
        graph_.add(PositionMeasurementFactor(T(num_backbone_poses_ - 1), tip_position_meas, tip_position_meas_cov_));

        // Run the optimizer, etc
        update(num_samples);

        return solution_;
    }

    TendonRobotSolution simulation_step(const Vector4& tensions, const Vector3& tip_force) {
        build_graph_base(tensions);
        add_common_factors();
        
        // Known tip force factor
        Vector6 tip_wrench_mean;
        tip_wrench_mean.head<3>() = Vector3::Zero();
        tip_wrench_mean.tail<3>() = tip_force;
        graph_.add(PriorFactor<Vector6>(F(num_backbone_poses_ - 1), tip_wrench_mean, small_wrench_cov_));
        
        update(1);

        return solution_;
    }
};


class DistLoadSolver : public TendonRobotGtsam {
public:
    DistLoadSolver(const TendonRobotConfig& config) 
        : TendonRobotGtsam(config)
    {
        dist_load_prior_cov_ = noiseModel::Diagonal::Sigmas((Vector(6) << 
            config.small_moment_std, config.small_moment_std, config.small_moment_std, 
            config.dist_load_prior_std, config.dist_load_prior_std, config.dist_load_prior_std).finished());
        
        dist_load_smoothing_cov_ = noiseModel::Isotropic::Sigma(3, config.dist_load_smoothness_std);

        fbg_strain_meas_cov_ = noiseModel::Isotropic::Sigma(3, config.fbg_strain_meas_std);
    }

private:
    noiseModel::Diagonal::shared_ptr dist_load_prior_cov_;
    noiseModel::Isotropic::shared_ptr dist_load_smoothing_cov_;
    noiseModel::Isotropic::shared_ptr fbg_strain_meas_cov_;

public:
    TendonRobotSolution step(const Vector4& tensions_meas, const std::vector<Vector3>& fbg_signals_meas, int num_samples) {
        build_graph_base(tensions_meas);

        // Magnitude prior factors for distributed load
        graph_.add(PriorFactor<Vector6>(F(1), Vector6::Zero(), small_wrench_cov_));
        for (int i = 2; i < num_backbone_poses_; i++) {
            graph_.add(PriorFactor<Vector6>(F(i), Vector6::Zero(), dist_load_prior_cov_));
        }

        // Smoothing prior factors for distributed load
        for (int i = 1; i + 3 < num_backbone_poses_; ++i) {
            graph_.add(DistLoadSmoothingFactor(F(i), F(i + 1), F(i + 2), F(i + 3), dist_load_smoothing_cov_));
        }

        // FBG strain measurement factors
        for (int i = 0; i < num_backbone_poses_; ++i) {
            graph_.add(FbgMeasurementFactor(S(i), fbg_signals_meas[i], K_inv_, rod_diameter_, fbg_strain_meas_cov_));
        }

        update(num_samples);

        return solution_;
    }

    TendonRobotSolution step_simulation(const Vector4& tensions, const std::vector<Vector3>& forces) {
        build_graph_base(tensions);

        // Add applied loads with small uncertainty
        for (int i = 1; i < num_backbone_poses_; i++) {
            Vector6 applied_wrench;
            applied_wrench.head<3>() = Vector3::Zero();
            applied_wrench.tail<3>() = forces[i - 1];
            graph_.add(PriorFactor<Vector6>(F(i), applied_wrench, small_wrench_cov_));
        }

        update(1);

        return solution_;
    }
};
}

