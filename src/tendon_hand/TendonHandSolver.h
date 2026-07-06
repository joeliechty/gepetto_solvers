#pragma once

#include "tendon_hand/TendonHandModel.h"
#include "tendon_finger/TendonFingerSolver.h"   // TendonFingerSolverConfig
#include "utils/Gaussians.h"
#include "utils/SolverBase.h"

#include <gtsam/base/Matrix.h>

#include <memory>
#include <string>
#include <vector>


struct TendonHandSolverConfig {
    SolverBaseConfig base;

    // Shared floating wrist base pose (world frame). Identity by default; each
    // finger is placed by its own hand_base_offset relative to this.
    gtsam::Matrix4 wrist_pose = gtsam::Matrix4::Identity();

    // Prior tightness on the shared wrist variable. Tight (default) => rigidly
    // anchored gauge; loosen for a more free-floating wrist.
    double sigma_wrist_pos = 1e-4;
    double sigma_wrist_rot = 1e-3;

    // (Interior/tip external-wrench prior noise is taken per finger from each
    // finger's sigma_stress_moment/force, matching TendonFingerSolver.)
};


// Static/kinematic solver for a multi-finger tendon hand whose fingers share one
// floating wrist base. Thin SolverBase wrapper around TendonHandModel; when any
// finger carries a contact constraint the solve runs on the Augmented Lagrangian
// path (like the single-finger TendonFingerSolver).
class TendonHandSolver : SolverBase {
public:
    TendonHandSolver(
        const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
        const TendonHandSolverConfig& config);

    Solution<TendonHandMarginals> solve(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches);

    int num_fingers() const { return hand_->num_fingers(); }

    // Re-expose SolverBase diagnostics (privately inherited).
    std::vector<std::tuple<std::string, int, double>>
        get_factor_error_summary() const { return SolverBase::get_factor_error_summary(); }

private:
    void build_graph() override;
    void extract_solution() override;
    void get_initial_values() override;

    std::unique_ptr<TendonHandModel> hand_;

    std::vector<VectorXGaussian> tensions_;
    std::vector<Vector6Gaussian> tip_wrenches_;

    TendonHandMarginals extracted_;
};
