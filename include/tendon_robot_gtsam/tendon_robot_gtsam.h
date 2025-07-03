#include <chrono>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>

#include "factors.h"


namespace gtsam{

struct TendonRobotSolution {
    std::vector<Pose3> backbone_pose_mean;
    std::vector<Matrix6> backbone_pose_cov;

    Vector6 tip_wrench_mean;
    Matrix6 tip_wrench_cov;

    Vector4 tensions_mean;
    Matrix4 tensions_cov;

    double solve_time_ms;
};

using symbol_shorthand::T;
using symbol_shorthand::F;
using symbol_shorthand::S;
// using symbol_shorthand::Q;
// using symbol_shorthand::D;

class TendonRobotGtsam {
public:
    TendonRobotGtsam(size_t num_backbone_poses = 20) {
        backbone_idx_start_ = 0;
        backbone_idx_end_ = backbone_idx_start_ + num_backbone_poses - 1;

        // tip_wrench_key_ = F(0);
        // tensions_key_ = Q(0);

        K_ = Matrix66::Zero();
        K_(0, 0) = k_bending_;
        K_(1, 1) = k_bending_;
        K_(2, 2) = k_torsion_;
        K_(3, 3) = k_shear_;
        K_(4, 4) = k_shear_;
        K_(5, 5) = k_extension_;

        K_inv_ = K_.inverse();

        ds_ = rod_length_ / (num_backbone_poses - 1);
    }

private:
    int backbone_idx_start_;
    int backbone_idx_end_;

    double ds_;

    // Key tip_wrench_key_;
    // Key tensions_key_;

    Values last_result_;

    // Parameters
    static constexpr double tension_std = 1e-3;

    static constexpr double cosserat_twist_r_std_ = 1e0; // This just looks right
    static constexpr double small_r_std_ = 1e-3;
    static constexpr double small_p_std_ = 1e-5;

    static constexpr double tip_force_std_ = 1e-3;
    static constexpr double tip_moment_std_ = 1e-3;

    static constexpr double rod_length_ = 0.05; 
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
        // auto tensions_cov = noiseModel::Isotropic::Sigma(4, tension_std);
        // graph.add(PriorFactor<Vector4>(tensions_key_, tensions_mean, tensions_cov));
        // initial_values.insert(tensions_key_, tensions_mean);

        // Tip force prior
        // auto tip_wrench_cov = noiseModel::Diagonal::Sigmas((Vector(6) << tip_moment_std_, tip_moment_std_, tip_moment_std_, tip_force_std_, tip_force_std_, tip_force_std_).finished());
        // new_graph.add(PriorFactor<Vector6>(F(t_), tip_wrench_mean, tip_wrench_cov));
        // new_values.insert(tip_wrench_key_, tip_wrench_mean);

        // Tip disc wrench prior
        // auto disc_wrench_cov = noiseModel::Isotropic::Sigma(6, 1e-2);

        // std::vector<Point2> xy_hole_loc_base;
        // for(int i = 0; i < 4; i++)
        //     xy_hole_loc_base.push_back(Point2(0.1, 0));

        // std::vector<Point2> xy_hole_loc_tip;
        // for(int i = 0; i < 4; i++)
        //     xy_hole_loc_tip.push_back(Point2(0.1, 0));
        
        // graph.add(TipTendonDiscWrenchFactor(
        //     T(backbone_pose_idx_start_), T(tip_pose_idx_), 
        //     D(0), tensions_key_, 
        //     xy_hole_loc_base, xy_hole_loc_tip, disc_wrench_cov));
        
        
        // if (last_result_.exists(D(0))) {
        //     initial_values.insert(D(0), last_result_.at<Vector6>(D(0)));
        // } else {
        //     initial_values.insert(D(0), Vector6(Vector6::Zero()));
        // }

        // Priors for forces
        double small_force_std = 1e-5;
        double small_moment_std = 1e-5;
        auto small_wrench_cov = noiseModel::Diagonal::Sigmas((Vector(6) << small_moment_std, small_moment_std, small_moment_std, small_force_std, small_force_std, small_force_std).finished());
        
        int mid_idx_ = 10;
        for(int i = 1; i <= backbone_idx_end_; i++){
            if(i == mid_idx_) continue;
            graph.add(PriorFactor<Vector6>(F(i), Vector6(Vector6::Zero()), small_wrench_cov));
        }
        
        Vector6 mid_wrench = ((Vector(6) << 0, tensions_mean[0], tensions_mean[1], tensions_mean[2], 0, tensions_mean[3]).finished());
        graph.add(PriorFactor<Vector6>(F(mid_idx_), mid_wrench, small_wrench_cov));

        // Base frame constraint
        auto base_frame_cov = noiseModel::Diagonal::Sigmas((Vector(6) << small_r_std_, small_r_std_, small_r_std_, small_p_std_, small_p_std_, small_p_std_).finished());
        graph.add(PriorFactor<Pose3>(T(backbone_idx_start_), Pose3::Identity(), base_frame_cov));

        // Cosserat twist factors
        double r_std = cosserat_twist_r_std_ * ds_;
        double stress_std = 1e-5;

        auto cosserat_cov = noiseModel::Diagonal::Sigmas((Vector(12) << 
            r_std, r_std, r_std, small_p_std_, small_p_std_, small_p_std_,
            stress_std, stress_std, stress_std, stress_std, stress_std, stress_std).finished());

        for (size_t i = backbone_idx_start_; i < backbone_idx_end_; ++i) {
            auto factor = std::make_shared<CosseratRodFactor>(
                T(i), T(i + 1), S(i), S(i + 1), F(i + 1), ds_, K_inv_, cosserat_cov);
            graph.add(factor);
        }

        // Near-zero constraint for tip stress
        auto tip_stress_cov = noiseModel::Diagonal::Sigmas((Vector(6) << 
            stress_std, stress_std, stress_std, stress_std, stress_std, stress_std).finished());
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
        

        TendonRobotSolution output;

        auto start_solve = std::chrono::high_resolution_clock::now();

        LevenbergMarquardtParams params;
        params.setVerbosityLM("SUMMARY");
        params.setlambdaInitial(0.1);
        LevenbergMarquardtOptimizer optimizer(graph, initial_values, params);

        Values result = optimizer.optimize();
        Marginals marginals(graph, result);

        for (size_t i = backbone_idx_start_; i <= backbone_idx_end_; ++i) {
            output.backbone_pose_mean.push_back(result.at<Pose3>(T(i)));
            output.backbone_pose_cov.push_back(marginals.marginalCovariance(T(i)));
        }

        // output.tip_wrench_mean = result.at<Vector6>(F(t_));
        // output.tip_wrench_cov = marginals.marginalCovariance(F(t_));

        // output.tensions_mean = result.at<Vector4>(tensions_key_);
        // output.tensions_cov = marginals.marginalCovariance(tensions_key_);

        auto end_solve = std::chrono::high_resolution_clock::now();
        output.solve_time_ms = std::chrono::duration<double, std::milli>(end_solve - start_solve).count();


        last_result_ = result;

        return output;
    }
};
}