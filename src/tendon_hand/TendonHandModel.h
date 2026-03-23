#pragma once

#include "tendon_robot/TendonRobotModel.h"
#include "tendon_robot/TendonRobotSolver.h"
#include "utils/Gaussians.h"
#include <gtsam/linear/NoiseModel.h>
#include <memory>
#include <vector>
#include <string>
#include <openvdb/openvdb.h>

struct TendonHandMarginals {
    std::vector<TendonRobotMarginals> fingers;
    std::vector<std::string> finger_names;
};


class TendonHandModel {
public:
    TendonHandModel(
        const std::vector<std::pair<std::string, TendonRobotSolverConfig>>& finger_configs,
        gtsam::SharedDiagonal small_wrench_noise);

    gtsam::NonlinearFactorGraph build_graph(
        const std::vector<VectorXGaussian>& tensions,
        const std::vector<Vector6Gaussian>& tip_wrenches,
        const gtsam::Values& current_values
    );

    gtsam::Values get_initial_values() const;

    TendonHandMarginals get_marginals(
        const gtsam::Values& values,
        const gtsam::Marginals& marginals) const;

    int num_fingers() const { return fingers_.size(); }

    void set_object(const std::string& vdb_path,
                    const gtsam::Pose3& initial_pose,
                    const Eigen::VectorXd& prior_sigmas);

private:
    std::vector<std::string> finger_names_;
    std::vector<int> num_tendons_per_finger_;
    gtsam::SharedDiagonal small_wrench_noise_;
    bool has_object_ = false;
    openvdb::FloatGrid::Ptr sdf_grid_;
    gtsam::Key object_key_ = gtsam::Symbol('O', 0);
    gtsam::Pose3 object_prior_mean_;
    gtsam::SharedDiagonal object_prior_noise_;

    // We use variant to handle different numbers of tendons per finger
    using FingerVariant = std::variant<
        std::unique_ptr<TendonRobotModel<1>>,
        std::unique_ptr<TendonRobotModel<2>>,
        std::unique_ptr<TendonRobotModel<3>>,
        std::unique_ptr<TendonRobotModel<4>>,
        std::unique_ptr<TendonRobotModel<5>>,
        std::unique_ptr<TendonRobotModel<6>>,
        std::unique_ptr<TendonRobotModel<7>>,
        std::unique_ptr<TendonRobotModel<8>>,
        std::unique_ptr<TendonRobotModel<9>>,
        std::unique_ptr<TendonRobotModel<10>>
    >;

    std::vector<FingerVariant> fingers_;
};
