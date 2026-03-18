#include "TendonHandModel.h"
#include "tendon_robot/TendonRobotSolver.h"
#include "utils/MiscInline.h"

#include <gtsam/slam/PriorFactor.h>

using namespace gtsam;


TendonHandModel::TendonHandModel(
    const std::vector<std::pair<std::string, TendonRobotSolverConfig>>& finger_configs,
    SharedDiagonal small_wrench_noise)
:
    small_wrench_noise_(small_wrench_noise)
{
    finger_names_.reserve(finger_configs.size());
    num_tendons_per_finger_.reserve(finger_configs.size());
    fingers_.reserve(finger_configs.size());

    for (const auto& [name, config] : finger_configs) {
        finger_names_.push_back(name);
        num_tendons_per_finger_.push_back(config.num_tendons);

        // Create noise models
        SharedDiagonal twist_noise = get_noise_model_rot_pos(
            config.sigma_twist_rot, config.sigma_twist_pos);
        SharedDiagonal stress_noise = get_noise_model_rot_pos(
            config.sigma_stress_moment, config.sigma_stress_force);

        Pose3 base_pose_mean;
        if (config.base_pose.isZero()) {
            Rot3 base_rot = Rot3::Rx(-M_PI / 2).compose(Rot3::Rz(M_PI));
            base_pose_mean = Pose3(base_rot, Point3::Zero());
        } else {
            base_pose_mean = Pose3(config.base_pose);
        }
        SharedDiagonal base_pose_noise = get_noise_model_rot_pos(
            config.sigma_base_rot, config.sigma_base_pos);

        // Create TendonRobotModel based on number of tendons
        int N = config.num_tendons;

        if (config.per_disc_tendon_input.is_populated()) {
            // Per-disc routing path
            if (config.K_inv_per_segment.empty()) {
                switch (N) {
                    case 1:  fingers_.push_back(std::make_unique<TendonRobotModel<1>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 2:  fingers_.push_back(std::make_unique<TendonRobotModel<2>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 3:  fingers_.push_back(std::make_unique<TendonRobotModel<3>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 4:  fingers_.push_back(std::make_unique<TendonRobotModel<4>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 5:  fingers_.push_back(std::make_unique<TendonRobotModel<5>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 6:  fingers_.push_back(std::make_unique<TendonRobotModel<6>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 7:  fingers_.push_back(std::make_unique<TendonRobotModel<7>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 8:  fingers_.push_back(std::make_unique<TendonRobotModel<8>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 9:  fingers_.push_back(std::make_unique<TendonRobotModel<9>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 10: fingers_.push_back(std::make_unique<TendonRobotModel<10>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    default: throw std::invalid_argument(
                        "num_tendons must be between 1 and 10, got " + std::to_string(N));
                }
            } else {
                switch (N) {
                    case 1:  fingers_.push_back(std::make_unique<TendonRobotModel<1>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 2:  fingers_.push_back(std::make_unique<TendonRobotModel<2>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 3:  fingers_.push_back(std::make_unique<TendonRobotModel<3>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 4:  fingers_.push_back(std::make_unique<TendonRobotModel<4>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 5:  fingers_.push_back(std::make_unique<TendonRobotModel<5>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 6:  fingers_.push_back(std::make_unique<TendonRobotModel<6>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 7:  fingers_.push_back(std::make_unique<TendonRobotModel<7>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 8:  fingers_.push_back(std::make_unique<TendonRobotModel<8>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 9:  fingers_.push_back(std::make_unique<TendonRobotModel<9>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 10: fingers_.push_back(std::make_unique<TendonRobotModel<10>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.per_disc_tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    default: throw std::invalid_argument(
                        "num_tendons must be between 1 and 10, got " + std::to_string(N));
                }
            }
        } else {
            // Simple TendonInput path
            if (config.K_inv_per_segment.empty()) {
                switch (N) {
                    case 1:  fingers_.push_back(std::make_unique<TendonRobotModel<1>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 2:  fingers_.push_back(std::make_unique<TendonRobotModel<2>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 3:  fingers_.push_back(std::make_unique<TendonRobotModel<3>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 4:  fingers_.push_back(std::make_unique<TendonRobotModel<4>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 5:  fingers_.push_back(std::make_unique<TendonRobotModel<5>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 6:  fingers_.push_back(std::make_unique<TendonRobotModel<6>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 7:  fingers_.push_back(std::make_unique<TendonRobotModel<7>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 8:  fingers_.push_back(std::make_unique<TendonRobotModel<8>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 9:  fingers_.push_back(std::make_unique<TendonRobotModel<9>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 10: fingers_.push_back(std::make_unique<TendonRobotModel<10>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    default: throw std::invalid_argument(
                        "num_tendons must be between 1 and 10, got " + std::to_string(N));
                }
            } else {
                switch (N) {
                    case 1:  fingers_.push_back(std::make_unique<TendonRobotModel<1>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 2:  fingers_.push_back(std::make_unique<TendonRobotModel<2>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 3:  fingers_.push_back(std::make_unique<TendonRobotModel<3>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 4:  fingers_.push_back(std::make_unique<TendonRobotModel<4>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 5:  fingers_.push_back(std::make_unique<TendonRobotModel<5>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 6:  fingers_.push_back(std::make_unique<TendonRobotModel<6>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 7:  fingers_.push_back(std::make_unique<TendonRobotModel<7>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 8:  fingers_.push_back(std::make_unique<TendonRobotModel<8>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 9:  fingers_.push_back(std::make_unique<TendonRobotModel<9>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    case 10: fingers_.push_back(std::make_unique<TendonRobotModel<10>>(
                        config.rod_length, config.num_discs, config.num_between_nodes,
                        config.tendon_input, config.K_inv_per_segment, twist_noise, stress_noise,
                        base_pose_mean, base_pose_noise, config.disc_positions_normalized)); break;
                    default: throw std::invalid_argument(
                        "num_tendons must be between 1 and 10, got " + std::to_string(N));
                }
            }
        }
    }
}


NonlinearFactorGraph TendonHandModel::build_graph(
    const std::vector<VectorXGaussian>& tensions,
    const std::vector<Vector6Gaussian>& tip_wrenches)
{
    if (tensions.size() != fingers_.size())
        throw std::invalid_argument(
            "tensions size (" + std::to_string(tensions.size()) +
            ") must match number of fingers (" + std::to_string(fingers_.size()) + ")");

    if (tip_wrenches.size() != fingers_.size())
        throw std::invalid_argument(
            "tip_wrenches size (" + std::to_string(tip_wrenches.size()) +
            ") must match number of fingers (" + std::to_string(fingers_.size()) + ")");

    NonlinearFactorGraph graph;

    // Add each finger's graph to the combined graph
    for (size_t i = 0; i < fingers_.size(); ++i) {
        std::visit([&](auto& finger_ptr) {
            using FingerType = typename std::remove_reference_t<decltype(*finger_ptr)>;
            constexpr int N = FingerType::NumTendons;

            // Convert VectorXGaussian to VectorNGaussian<N>
            if (tensions[i].mean.size() != N)
                throw std::invalid_argument(
                    "Finger " + std::to_string(i) + " expects " + std::to_string(N) +
                    " tendons, got " + std::to_string(tensions[i].mean.size()));

            VectorNGaussian<N> t_fixed;
            t_fixed.mean = tensions[i].mean;
            t_fixed.cov = tensions[i].cov;

            // Build the finger's graph and add it to the combined graph
            graph.add(finger_ptr->build_graph(t_fixed));

            // Constrain all external wrenches except base and tip
            int num_nodes = finger_ptr->get_num_nodes();
            for (int j = 1; j + 1 < num_nodes; ++j) {
                graph.add(PriorFactor<Vector6>(
                    finger_ptr->get_external_wrench_key(j),
                    Vector6::Zero(),
                    small_wrench_noise_));
            }

            // Add tip wrench constraint
            graph.add(PriorFactor<Vector6>(
                finger_ptr->get_external_wrench_key(num_nodes - 1),
                tip_wrenches[i].mean,
                noiseModel::Gaussian::Covariance(tip_wrenches[i].cov)));

        }, fingers_[i]);
    }

    // TODO: Add coupling constraints between fingers here in the future
    // e.g., contact constraints, palm constraints, etc.

    return graph;
}


Values TendonHandModel::get_initial_values() const {
    Values values;

    for (const auto& finger : fingers_) {
        std::visit([&](const auto& finger_ptr) {
            values.insert(finger_ptr->get_initial_values());
        }, finger);
    }

    return values;
}


TendonHandMarginals TendonHandModel::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    TendonHandMarginals solution;
    solution.fingers.reserve(fingers_.size());
    solution.finger_names = finger_names_;

    for (const auto& finger : fingers_) {
        std::visit([&](const auto& finger_ptr) {
            solution.fingers.push_back(finger_ptr->get_marginals(values, marginals));
        }, finger);
    }

    return solution;
}
