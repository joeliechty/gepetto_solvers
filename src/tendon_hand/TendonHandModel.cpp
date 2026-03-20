#include "TendonHandModel.h"
#include "tendon_robot/TendonRobotSolver.h"
#include "utils/MiscInline.h"
#include "SphereContactFactor.h"

#include <gtsam/slam/PriorFactor.h>

#ifdef CREST_USE_OPENMP
#include <omp.h>
#endif

#include <vector>
#include <tuple>

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
    const std::vector<Vector6Gaussian>& tip_wrenches,
    const Values& current_values)
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

    // ----- Collision avoidance factors -----
    // want the contact to act like a very stiff spring so will use sigma of 1e-4
    auto contact_noise = noiseModel::Isotropic::Sigma(1, 1e-4);
    double radius = 0.005;      // TODO: 5mm radius for collision spheres, make this something that is passed in
    double broad_phase_dist = 0.02;    // only add factor if we are within 2cm of the contact point

    // helper lambdas to cleanly extract data from the std::variant finger array
    auto get_num_nodes = [&](int f_idx) -> int {
        int n = 0;
        std::visit([&](const auto& ptr){n = ptr->get_num_nodes();}, fingers_[f_idx]);
        return n;
    };
    auto get_pose_key = [&](int f_idx, int n_idx) -> Key {
        Key k;
        std::visit([&](const auto& ptr){k = ptr->rod_->get_pose_key(n_idx);}, fingers_[f_idx]);
        return k;
    };
    auto get_num_between_nodes = [&](int f_idx) -> int {
        int n = 0;
        std::visit([&](const auto& ptr){n = ptr->get_num_between_nodes();}, fingers_[f_idx]);
        return n;
    };

    // Check if a node is in a joint segment. Joints are rolling articulation points
    // between bones with no physical material to collide. In the standard finger
    // pattern (bone-joint-bone-joint-bone-joint-bone), joint segments are at odd
    // disc gaps (1, 3, 5, ...).
    auto is_in_joint_segment = [](int node_idx, int num_between_nodes) -> bool {
        int nodes_per_segment = num_between_nodes + 1;
        int segment_idx = node_idx / nodes_per_segment;
        // Odd segments (1, 3, 5, ...) are joints
        return (segment_idx % 2) == 1;
    };

    // Build list of finger pairs to check (f1 < f2)
    std::vector<std::pair<int, int>> finger_pairs;
    for (size_t f1 = 0; f1 < fingers_.size(); ++f1) {
        for (size_t f2 = f1 + 1; f2 < fingers_.size(); ++f2) {
            finger_pairs.emplace_back(f1, f2);
        }
    }

    // Collect collision pairs in parallel, then add to graph
    // Each element: (key1, key2)
    std::vector<std::pair<Key, Key>> collision_pairs;

#ifdef CREST_USE_OPENMP
    // Thread-local storage for collision pairs
    std::vector<std::vector<std::pair<Key, Key>>> thread_local_pairs;

    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();

        #pragma omp single
        {
            thread_local_pairs.resize(num_threads);
        }

        #pragma omp for schedule(dynamic)
        for (size_t pair_idx = 0; pair_idx < finger_pairs.size(); ++pair_idx) {
            int f1 = finger_pairs[pair_idx].first;
            int f2 = finger_pairs[pair_idx].second;

            int nodes_f1 = get_num_nodes(f1);
            int nodes_f2 = get_num_nodes(f2);
            int between_nodes_f1 = get_num_between_nodes(f1);
            int between_nodes_f2 = get_num_between_nodes(f2);
            int metacarpal_nodes_f1 = between_nodes_f1 + 1;
            int metacarpal_nodes_f2 = between_nodes_f2 + 1;

            for (int n1 = 0; n1 < nodes_f1; ++n1) {
                if (is_in_joint_segment(n1, between_nodes_f1)) continue;

                for (int n2 = 0; n2 < nodes_f2; ++n2) {
                    if (is_in_joint_segment(n2, between_nodes_f2)) continue;
                    if (n1 < metacarpal_nodes_f1 && n2 < metacarpal_nodes_f2) continue;

                    Key k1 = get_pose_key(f1, n1);
                    Key k2 = get_pose_key(f2, n2);

                    if (current_values.exists(k1) && current_values.exists(k2)) {
                        Pose3 p1 = current_values.at<Pose3>(k1);
                        Pose3 p2 = current_values.at<Pose3>(k2);

                        if (gtsam::distance3(p1.translation(), p2.translation()) < broad_phase_dist) {
                            thread_local_pairs[thread_id].emplace_back(k1, k2);
                        }
                    }
                }
            }
        }
    }

    // Merge thread-local results
    for (const auto& local_pairs : thread_local_pairs) {
        collision_pairs.insert(collision_pairs.end(), local_pairs.begin(), local_pairs.end());
    }

#else
    // Sequential fallback
    for (const auto& [f1, f2] : finger_pairs) {
        int nodes_f1 = get_num_nodes(f1);
        int nodes_f2 = get_num_nodes(f2);
        int between_nodes_f1 = get_num_between_nodes(f1);
        int between_nodes_f2 = get_num_between_nodes(f2);
        int metacarpal_nodes_f1 = between_nodes_f1 + 1;
        int metacarpal_nodes_f2 = between_nodes_f2 + 1;

        for (int n1 = 0; n1 < nodes_f1; ++n1) {
            if (is_in_joint_segment(n1, between_nodes_f1)) continue;

            for (int n2 = 0; n2 < nodes_f2; ++n2) {
                if (is_in_joint_segment(n2, between_nodes_f2)) continue;
                if (n1 < metacarpal_nodes_f1 && n2 < metacarpal_nodes_f2) continue;

                Key k1 = get_pose_key(f1, n1);
                Key k2 = get_pose_key(f2, n2);

                if (current_values.exists(k1) && current_values.exists(k2)) {
                    Pose3 p1 = current_values.at<Pose3>(k1);
                    Pose3 p2 = current_values.at<Pose3>(k2);

                    if (gtsam::distance3(p1.translation(), p2.translation()) < broad_phase_dist) {
                        collision_pairs.emplace_back(k1, k2);
                    }
                }
            }
        }
    }
#endif

    // Add all collision factors to the graph (must be sequential)
    for (const auto& [k1, k2] : collision_pairs) {
        graph.add(crest_sparse::SphereContactFactor(k1, k2, radius, radius, contact_noise));
    }

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
