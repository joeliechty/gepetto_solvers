#include "gepetto_solvers/digits/tendon/TendonFingerModel.h"
#include "gepetto_solvers/cosserat_rod/CosseratRodModel.h"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <memory>

#include "gepetto_solvers/digits/tendon/TendonDiscWrenchFactor.h"
#include "gepetto_solvers/digits/tendon/TendonLengthFactor.h"
#include "gepetto_solvers/utils/Gaussians.h"

using namespace gtsam;


// --- TendonInput constructors ---

template<int N>
TendonFingerModel<N>::TendonFingerModel(
    double rod_length,
    int num_discs,
    int num_between_nodes,
    TendonInput tendon_input,
    const Matrix6& K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise,
    const std::vector<double>& disc_positions_normalized)
:
    id_(next_id_++),
    rod_length_(rod_length),
    num_discs_(num_discs),
    num_nodes_(num_discs + (num_discs - 1) * num_between_nodes),
    num_between_nodes_(num_between_nodes),
    twist_noise_(twist_noise),
    stress_noise_(stress_noise),
    base_pose_mean_(base_pose_mean),
    base_pose_noise_(base_pose_noise)
{
    rod_ = std::make_unique<CosseratRodModel>(
        num_nodes_, K_inv, twist_noise, stress_noise);

    init_tendon_disc_config(tendon_input, disc_positions_normalized);
}


template<int N>
TendonFingerModel<N>::TendonFingerModel(
    double rod_length,
    int num_discs,
    int num_between_nodes,
    TendonInput tendon_input,
    const std::vector<Matrix6>& K_inv_per_segment,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise,
    const std::vector<double>& disc_positions_normalized)
:
    id_(next_id_++),
    rod_length_(rod_length),
    num_discs_(num_discs),
    num_nodes_(num_discs + (num_discs - 1) * num_between_nodes),
    num_between_nodes_(num_between_nodes),
    twist_noise_(twist_noise),
    stress_noise_(stress_noise),
    base_pose_mean_(base_pose_mean),
    base_pose_noise_(base_pose_noise)
{
    rod_ = std::make_unique<CosseratRodModel>(
        num_nodes_, K_inv_per_segment, twist_noise, stress_noise);

    init_tendon_disc_config(tendon_input, disc_positions_normalized);
}


// --- PerDiscTendonInput constructors ---

template<int N>
TendonFingerModel<N>::TendonFingerModel(
    double rod_length,
    int num_discs,
    int num_between_nodes,
    PerDiscTendonInput per_disc_input,
    const Matrix6& K_inv,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise,
    const std::vector<double>& disc_positions_normalized)
:
    id_(next_id_++),
    rod_length_(rod_length),
    num_discs_(num_discs),
    num_nodes_(num_discs + (num_discs - 1) * num_between_nodes),
    num_between_nodes_(num_between_nodes),
    twist_noise_(twist_noise),
    stress_noise_(stress_noise),
    base_pose_mean_(base_pose_mean),
    base_pose_noise_(base_pose_noise)
{
    rod_ = std::make_unique<CosseratRodModel>(
        num_nodes_, K_inv, twist_noise, stress_noise);

    init_tendon_disc_config_per_disc(per_disc_input, disc_positions_normalized);
}


template<int N>
TendonFingerModel<N>::TendonFingerModel(
    double rod_length,
    int num_discs,
    int num_between_nodes,
    PerDiscTendonInput per_disc_input,
    const std::vector<Matrix6>& K_inv_per_segment,
    SharedDiagonal twist_noise,
    SharedDiagonal stress_noise,
    Pose3 base_pose_mean,
    SharedDiagonal base_pose_noise,
    const std::vector<double>& disc_positions_normalized)
:
    id_(next_id_++),
    rod_length_(rod_length),
    num_discs_(num_discs),
    num_nodes_(num_discs + (num_discs - 1) * num_between_nodes),
    num_between_nodes_(num_between_nodes),
    twist_noise_(twist_noise),
    stress_noise_(stress_noise),
    base_pose_mean_(base_pose_mean),
    base_pose_noise_(base_pose_noise)
{
    rod_ = std::make_unique<CosseratRodModel>(
        num_nodes_, K_inv_per_segment, twist_noise, stress_noise);

    init_tendon_disc_config_per_disc(per_disc_input, disc_positions_normalized);
}


// --- Shared helper: compute disc positions and segment lengths ---

template<int N>
void TendonFingerModel<N>::compute_disc_positions_and_segments(
    const std::vector<double>& disc_positions_normalized)
{
    std::vector<double> disc_s(num_discs_);

    // Use custom disc positions if provided, otherwise uniform spacing
    if (!disc_positions_normalized.empty()) {
        if (static_cast<int>(disc_positions_normalized.size()) != num_discs_)
            throw std::invalid_argument(
                "TendonFingerModel: disc_positions_normalized must have exactly num_discs entries");
        disc_s = disc_positions_normalized;
    } else {
        for (int i = 0; i < num_discs_; ++i)
            disc_s[i] = static_cast<double>(i) / (num_discs_ - 1);
    }

    // Compute per-segment arc lengths from disc positions
    int num_between_nodes = (num_discs_ > 1) ? (num_nodes_ - num_discs_) / (num_discs_ - 1) : 0;
    segment_lengths_.clear();
    segment_lengths_.reserve(num_nodes_ - 1);
    for (int i = 0; i < num_discs_ - 1; ++i) {
        double inter_disc_len = (disc_s[i + 1] - disc_s[i]) * rod_length_;
        double ds_sub = inter_disc_len / (num_between_nodes + 1);
        for (int k = 0; k < num_between_nodes + 1; ++k)
            segment_lengths_.push_back(ds_sub);
    }

    // Compute disc pose indices and non-disc indices
    tendon_config_.disc_pose_idx.clear();
    tendon_config_.disc_pose_idx.reserve(num_discs_);
    tendon_config_.no_disc_pose_idx.clear();

    for (int disc_idx = 0; disc_idx < num_discs_; ++disc_idx) {
        int closest_pose_idx = disc_idx * (num_between_nodes + 1);
        tendon_config_.disc_pose_idx.push_back(closest_pose_idx);
    }

    std::vector<bool> is_disc(num_nodes_, false);
    for (int idx : tendon_config_.disc_pose_idx)
        is_disc[idx] = true;
    for (int i = 0; i < num_nodes_; ++i)
        if (!is_disc[i])
            tendon_config_.no_disc_pose_idx.push_back(i);
}


// --- init from TendonInput (simple global routing) ---

template<int N>
void TendonFingerModel<N>::init_tendon_disc_config(TendonInput routing, const std::vector<double>& disc_positions_normalized) {
    if (static_cast<int>(routing.functions.size()) != N)
        throw std::invalid_argument(
            "TendonInput functions size (" + std::to_string(routing.functions.size()) +
            ") must match template parameter N (" + std::to_string(N) + ")");

    tendon_config_.num_discs = num_discs_;
    tendon_config_.num_tendons = N;
    tendon_config_.routing_radius = routing.routing_radius;
    tendon_config_.hole_locations.reserve(num_discs_);

    // Compute disc positions
    std::vector<double> disc_s(num_discs_);
    if (!disc_positions_normalized.empty()) {
        if (static_cast<int>(disc_positions_normalized.size()) != num_discs_)
            throw std::invalid_argument(
                "TendonFingerModel: disc_positions_normalized must have exactly num_discs entries");
        disc_s = disc_positions_normalized;
    } else {
        for (int i = 0; i < num_discs_; ++i)
            disc_s[i] = static_cast<double>(i) / (num_discs_ - 1);
    }

    compute_disc_positions_and_segments(disc_positions_normalized);

    // For each disc, compute hole locations for all tendons
    for (int disc_idx = 0; disc_idx < num_discs_; ++disc_idx) {
        double s = disc_s[disc_idx];

        std::vector<std::optional<Vector3>> holes(N);

        for (int tendon_idx = 0; tendon_idx < N; ++tendon_idx) {
            double theta;

            if (routing.functions[tendon_idx] == RoutingAngleFunction::CONSTANT) {
                theta = routing.params[tendon_idx].angle_offset;
            } else if (routing.functions[tendon_idx] == RoutingAngleFunction::LINEAR) {
                theta = routing.params[tendon_idx].angle_offset + s * routing.params[tendon_idx].total_angle;
            } else {
                theta = 0.0;
            }

            double x = routing.routing_radius * std::cos(theta);
            double y = routing.routing_radius * std::sin(theta);
            double z = 0.0;

            holes[tendon_idx] = Vector3(x, y, z);
        }

        tendon_config_.hole_locations.push_back(holes);
    }
}


// --- init from PerDiscTendonInput (per-disc routing with early termination) ---

template<int N>
void TendonFingerModel<N>::init_tendon_disc_config_per_disc(PerDiscTendonInput input, const std::vector<double>& disc_positions_normalized) {
    if (input.num_tendons != N)
        throw std::invalid_argument(
            "PerDiscTendonInput num_tendons (" + std::to_string(input.num_tendons) +
            ") must match template parameter N (" + std::to_string(N) + ")");
    if (static_cast<int>(input.hole_angles.size()) != num_discs_)
        throw std::invalid_argument(
            "PerDiscTendonInput hole_angles outer size (" + std::to_string(input.hole_angles.size()) +
            ") must equal num_discs (" + std::to_string(num_discs_) + ")");

    bool has_per_tendon_radii = !input.hole_radii.empty();
    if (has_per_tendon_radii) {
        if (static_cast<int>(input.hole_radii.size()) != num_discs_)
            throw std::invalid_argument(
                "PerDiscTendonInput hole_radii outer size (" + std::to_string(input.hole_radii.size()) +
                ") must equal num_discs (" + std::to_string(num_discs_) + ")");
    }

    tendon_config_.num_discs = num_discs_;
    tendon_config_.num_tendons = N;
    tendon_config_.routing_radius = input.routing_radius;
    tendon_config_.hole_locations.reserve(num_discs_);

    compute_disc_positions_and_segments(disc_positions_normalized);

    for (int disc_idx = 0; disc_idx < num_discs_; ++disc_idx) {
        if (static_cast<int>(input.hole_angles[disc_idx].size()) != N)
            throw std::invalid_argument(
                "PerDiscTendonInput hole_angles[" + std::to_string(disc_idx) +
                "] size must equal N (" + std::to_string(N) + ")");

        if (has_per_tendon_radii && static_cast<int>(input.hole_radii[disc_idx].size()) != N)
            throw std::invalid_argument(
                "PerDiscTendonInput hole_radii[" + std::to_string(disc_idx) +
                "] size must equal N (" + std::to_string(N) + ")");

        std::vector<std::optional<Vector3>> holes(N);

        for (int t = 0; t < N; ++t) {
            double angle = input.hole_angles[disc_idx][t];
            if (std::isnan(angle)) {
                holes[t] = std::nullopt;  // Tendon terminated at this disc
            } else {
                double r = has_per_tendon_radii ? input.hole_radii[disc_idx][t] : input.routing_radius;
                double x = r * std::cos(angle);
                double y = r * std::sin(angle);
                holes[t] = Vector3(x, y, 0.0);
            }
        }

        tendon_config_.hole_locations.push_back(holes);
    }
}


template<int N>
Key TendonFingerModel<N>::get_tensions_key() const {
    return Symbol('Q', 1000 * id_);
}


template<int N>
Key TendonFingerModel<N>::get_lengths_key() const {
    return Symbol('L', 1000 * id_);
}


template<int N>
Key TendonFingerModel<N>::get_disc_wrench_key(int disc_idx) const {
    // We dont ever want to include disc wrenches for base disc
    if (disc_idx < 1)
        throw std::out_of_range("TendonFinger: invalid disc wrench index");

    return Symbol('D', 1000 * id_ + disc_idx);
}


template<int N>
Key TendonFingerModel<N>::get_external_wrench_key(int node_idx) const {
    // If we are at a disc, use disc wrench key
    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        if (tendon_config_.disc_pose_idx[disc_idx] == node_idx) {
            return get_disc_wrench_key(disc_idx);
        }
    }

    // Else use wrench key from rod model
    return rod_->get_wrench_key(node_idx);
}


template<int N>
Eigen::Vector<double, N> TendonFingerModel<N>::compute_tendon_lengths(const Values& values) const {
    Eigen::Vector<double, N> lengths = Eigen::Vector<double, N>::Zero();

    for (int t = 0; t < N; ++t) {
        double length = 0.0;
        bool has_prev = false;
        Point3 p_prev;

        for (size_t disc_idx = 0; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
            auto& hole_opt = tendon_config_.hole_locations[disc_idx][t];

            if (hole_opt.has_value()) {
                int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
                // Base disc (pose idx 0) is derived from the hand base when reparameterized.
                Pose3 disc_pose = (use_hand_base_ && pose_idx == 0)
                    ? values.at<Pose3>(rod_->get_root_base_key()).compose(hand_base_offset_)
                    : values.at<Pose3>(rod_->get_pose_key(pose_idx));
                Point3 p_curr = disc_pose.transformFrom(hole_opt.value());

                if (has_prev) {
                    length += (p_curr - p_prev).norm();
                }
                p_prev = p_curr;
                has_prev = true;
            } else {
                break;
            }
        }
        lengths[t] = length;
    }

    return lengths;
}


template<int N>
void TendonFingerModel<N>::set_hand_base(const Pose3& offset,
                                         std::optional<gtsam::Key> shared_key) {
    use_hand_base_ = true;
    hand_base_offset_ = offset;
    rod_->set_root_reparameterization(offset, shared_key);
}


template<int N>
Values TendonFingerModel<N>::get_initial_values() const {
    Values values;

    // Seed the rod nodes along the actual base orientation (base_pose_mean_),
    // not the world frame. The base-pose prior anchors node 0 at
    // base_pose_mean_ (e.g. Rx(-pi/2)*Rz(pi), i.e. the finger lying
    // horizontally), so seeding from Identity would start the whole rod
    // pointing straight up — a poor initial guess that makes the optimizer
    // cross a large rod-bending gap and is prone to local minima, especially
    // for contact problems. Since the base pose is precisely known (the finger
    // is rigidly mounted), starting the rod there is well justified.
    values.insert(rod_->get_initial_values(segment_lengths_, base_pose_mean_));

    Eigen::Vector<double, N> zero = Eigen::Vector<double, N>::Zero();
    values.insert(get_tensions_key(), zero);

    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        values.insert(get_disc_wrench_key(disc_idx), Vector6(Vector6::Zero()));
    }

    // Compute initial tendon lengths from the straight-rod initial poses
    Eigen::Vector<double, N> init_lengths = compute_tendon_lengths(values);
    values.insert(get_lengths_key(), init_lengths);

    return values;
}


template<int N>
NonlinearFactorGraph TendonFingerModel<N>::build_graph(const VectorNGaussian<N>& tensions_) const
{
    // To fully constrain a Cosserat rod graph, all we need to do is add:
    //   1. Base pose prior constraint
    //   2. All wrenches except base wrench need to be constrained somehow
    NonlinearFactorGraph graph = rod_->build_graph(segment_lengths_);

    // Base frame prior constraint. With the hand-base reparameterization the
    // node-0 pose is no longer a variable; anchor the hand base T_base such that
    // T_0 = T_base o offset = base_pose_mean_, i.e. T_base = base_pose_mean_ o offset^{-1}.
    // Skipped when emit_base_prior_ is false (an owner anchors the shared base).
    if (emit_base_prior_) {
        if (use_hand_base_) {
            graph.add(PriorFactor<Pose3>(
                rod_->get_root_base_key(),
                base_pose_mean_ * hand_base_offset_.inverse(),
                base_pose_noise_));
        } else {
            graph.add(PriorFactor<Pose3>(rod_->get_pose_key(0), base_pose_mean_, base_pose_noise_));
        }
    }

    // Priors for discs (using disc indices), start at 1, no force at base disc
    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
        int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];

        // Convert from vector<optional<Vector3>> to array<Point3, N> + bool masks
        std::array<Point3, N> holes_prev, holes, holes_next;
        std::array<bool, N> active, active_prev, active_next;

        for (int t = 0; t < N; ++t) {
            // Current disc
            auto& h = tendon_config_.hole_locations[disc_idx][t];
            active[t] = h.has_value();
            holes[t] = h.value_or(Point3::Zero());

            // Previous disc
            auto& hp = tendon_config_.hole_locations[disc_idx - 1][t];
            active_prev[t] = hp.has_value();
            holes_prev[t] = hp.value_or(Point3::Zero());

            // Next disc
            if (disc_idx + 1 < tendon_config_.disc_pose_idx.size()) {
                auto& hn = tendon_config_.hole_locations[disc_idx + 1][t];
                active_next[t] = hn.has_value();
                holes_next[t] = hn.value_or(Point3::Zero());
            } else {
                active_next[t] = false;
                holes_next[t] = Point3::Zero();
            }
        }

        // Next disc variables
        int pose_idx_next;

        // They change whether or not we are at the tip (no next disc exists)
        bool is_tip = false;
        if (disc_idx == (tendon_config_.disc_pose_idx.size() - 1)) {
            is_tip = true;
            pose_idx_next = 0; // Dummy pose for tip factor, not used for tip disc
        } else {
            pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];
        }

        // When reparameterized, the base disc (node 0) is the hand base composed
        // with the fixed offset; pass the hand-base key + offset for that prev pose.
        bool prev_is_root = use_hand_base_ && (pose_idx_prev == 0);
        Key prev_pose_key = prev_is_root
            ? rod_->get_root_base_key()
            : rod_->get_pose_key(pose_idx_prev);
        std::optional<Pose3> prev_offset =
            prev_is_root ? std::optional<Pose3>(hand_base_offset_) : std::nullopt;

        // The tip disc passes node 0 (pose_idx_next == 0) as an unused dummy
        // "next" pose (the is_tip branch in the factor never reads it). When
        // reparameterized, node 0 is not a Values variable, so route this dummy
        // to the hand-base key — which exists — to keep the Values resolvable.
        // Its value/Jacobian remain unused since is_tip skips the next-disc term.
        Key next_pose_key = (use_hand_base_ && pose_idx_next == 0)
            ? rod_->get_root_base_key()
            : rod_->get_pose_key(pose_idx_next);

        // Add the factor that relates poses, tensions, wrenches together for the disc
        graph.add(TendonDiscWrenchFactor<N>(
            prev_pose_key,
            rod_->get_pose_key(pose_idx),
            next_pose_key,
            rod_->get_wrench_key(pose_idx), // Spatial
            get_tensions_key(),
            get_disc_wrench_key(disc_idx), // Spatial
            is_tip,
            holes_prev,
            holes,
            holes_next,
            active,
            active_prev,
            active_next,
            stress_noise_,  // This could be a separate friction noise
            prev_offset));
    }

    // Tendon length inextensibility constraint
    {
        std::vector<Key> disc_pose_keys;
        disc_pose_keys.reserve(num_discs_);
        for (int d = 0; d < num_discs_; ++d) {
            int pidx = tendon_config_.disc_pose_idx[d];
            // The base disc (pose idx 0) is the hand base when reparameterized.
            if (use_hand_base_ && pidx == 0)
                disc_pose_keys.push_back(rod_->get_root_base_key());
            else
                disc_pose_keys.push_back(rod_->get_pose_key(pidx));
        }

        std::optional<Pose3> first_disc_offset =
            use_hand_base_ ? std::optional<Pose3>(hand_base_offset_) : std::nullopt;

        auto length_noise = noiseModel::Isotropic::Sigma(N, sigma_length_);
        graph.add(TendonLengthFactor<N>(
            get_lengths_key(), disc_pose_keys,
            tendon_config_.hole_locations, length_noise, first_disc_offset));
    }

    // Measurement prior on tensions
    graph.add(PriorFactor<Eigen::Vector<double, N>>(
        get_tensions_key(),
        tensions_.mean,
        noiseModel::Gaussian::Covariance(tensions_.cov)));

    return graph;
}


template<int N>
NonlinearFactorGraph TendonFingerModel<N>::build_graph_kinematic() const
{
    NonlinearFactorGraph graph = rod_->build_graph(segment_lengths_);

    // Base frame prior constraint. With the hand-base reparameterization the
    // node-0 pose is no longer a variable; anchor the hand base T_base such that
    // T_0 = T_base o offset = base_pose_mean_, i.e. T_base = base_pose_mean_ o offset^{-1}.
    // Skipped when emit_base_prior_ is false (an owner anchors the shared base).
    if (emit_base_prior_) {
        if (use_hand_base_) {
            graph.add(PriorFactor<Pose3>(
                rod_->get_root_base_key(),
                base_pose_mean_ * hand_base_offset_.inverse(),
                base_pose_noise_));
        } else {
            graph.add(PriorFactor<Pose3>(rod_->get_pose_key(0), base_pose_mean_, base_pose_noise_));
        }
    }

    // Priors for discs (using disc indices), start at 1, no force at base disc
    for (size_t disc_idx = 1; disc_idx < tendon_config_.disc_pose_idx.size(); ++disc_idx) {
        int pose_idx = tendon_config_.disc_pose_idx[disc_idx];
        int pose_idx_prev = tendon_config_.disc_pose_idx[disc_idx - 1];

        // Convert from vector<optional<Vector3>> to array<Point3, N> + bool masks
        std::array<Point3, N> holes_prev, holes, holes_next;
        std::array<bool, N> active, active_prev, active_next;

        for (int t = 0; t < N; ++t) {
            // Current disc
            auto& h = tendon_config_.hole_locations[disc_idx][t];
            active[t] = h.has_value();
            holes[t] = h.value_or(Point3::Zero());

            // Previous disc
            auto& hp = tendon_config_.hole_locations[disc_idx - 1][t];
            active_prev[t] = hp.has_value();
            holes_prev[t] = hp.value_or(Point3::Zero());

            // Next disc
            if (disc_idx + 1 < tendon_config_.disc_pose_idx.size()) {
                auto& hn = tendon_config_.hole_locations[disc_idx + 1][t];
                active_next[t] = hn.has_value();
                holes_next[t] = hn.value_or(Point3::Zero());
            } else {
                active_next[t] = false;
                holes_next[t] = Point3::Zero();
            }
        }

        // Next disc variables
        int pose_idx_next;

        // They change whether or not we are at the tip (no next disc exists)
        bool is_tip = false;
        if (disc_idx == (tendon_config_.disc_pose_idx.size() - 1)) {
            is_tip = true;
            pose_idx_next = 0; // Dummy pose for tip factor, not used for tip disc
        } else {
            pose_idx_next = tendon_config_.disc_pose_idx[disc_idx + 1];
        }

        // When reparameterized, the base disc (node 0) is the hand base composed
        // with the fixed offset; pass the hand-base key + offset for that prev pose.
        bool prev_is_root = use_hand_base_ && (pose_idx_prev == 0);
        Key prev_pose_key = prev_is_root
            ? rod_->get_root_base_key()
            : rod_->get_pose_key(pose_idx_prev);
        std::optional<Pose3> prev_offset =
            prev_is_root ? std::optional<Pose3>(hand_base_offset_) : std::nullopt;

        // The tip disc passes node 0 (pose_idx_next == 0) as an unused dummy
        // "next" pose (the is_tip branch in the factor never reads it). When
        // reparameterized, node 0 is not a Values variable, so route this dummy
        // to the hand-base key — which exists — to keep the Values resolvable.
        // Its value/Jacobian remain unused since is_tip skips the next-disc term.
        Key next_pose_key = (use_hand_base_ && pose_idx_next == 0)
            ? rod_->get_root_base_key()
            : rod_->get_pose_key(pose_idx_next);

        // Add the factor that relates poses, tensions, wrenches together for the disc
        graph.add(TendonDiscWrenchFactor<N>(
            prev_pose_key,
            rod_->get_pose_key(pose_idx),
            next_pose_key,
            rod_->get_wrench_key(pose_idx), // Spatial
            get_tensions_key(),
            get_disc_wrench_key(disc_idx), // Spatial
            is_tip,
            holes_prev,
            holes,
            holes_next,
            active,
            active_prev,
            active_next,
            stress_noise_,  // This could be a separate friction noise
            prev_offset));
    }

    // Tendon length inextensibility constraint
    {
        std::vector<Key> disc_pose_keys;
        disc_pose_keys.reserve(num_discs_);
        for (int d = 0; d < num_discs_; ++d) {
            int pidx = tendon_config_.disc_pose_idx[d];
            // The base disc (pose idx 0) is the hand base when reparameterized.
            if (use_hand_base_ && pidx == 0)
                disc_pose_keys.push_back(rod_->get_root_base_key());
            else
                disc_pose_keys.push_back(rod_->get_pose_key(pidx));
        }

        std::optional<Pose3> first_disc_offset =
            use_hand_base_ ? std::optional<Pose3>(hand_base_offset_) : std::nullopt;

        auto length_noise = noiseModel::Isotropic::Sigma(N, sigma_length_);
        graph.add(TendonLengthFactor<N>(
            get_lengths_key(), disc_pose_keys,
            tendon_config_.hole_locations, length_noise, first_disc_offset));
    }

    return graph;
}


template<int N>
void TendonFingerModel<N>::get_J_pose_tensions(const Marginals& marginals, TendonFingerMarginals& out) const{
    // Get joint marginal between tip pose and tensions
    Key Q = get_tensions_key();
    Key T = rod_->get_pose_key(-1);
    JointMarginal joint = marginals.jointMarginalCovariance({Q, T});

    Eigen::Matrix<double, 6, N> sigma_TQ = joint(T, Q);
    Eigen::Matrix<double, N, N> sigma_QQ_inv = marginals.marginalInformation(Q);

    out.J_pose_tensions = sigma_TQ * sigma_QQ_inv;
}


template<int N>
void TendonFingerModel<N>::get_J_pose_tensions(const JointFn& joint_of, TendonFingerMarginals& out) const{
    // joint_of(T, Q) returns a (6+N) x (6+N) joint covariance ordered [T, Q].
    Key Q = get_tensions_key();
    Key T = rod_->get_pose_key(-1);
    Eigen::MatrixXd joint = joint_of(T, Q);

    Eigen::Matrix<double, 6, N> sigma_TQ = joint.template block<6, N>(0, 6);
    Eigen::Matrix<double, N, N> sigma_QQ = joint.template block<N, N>(6, 6);

    out.J_pose_tensions = sigma_TQ * sigma_QQ.inverse();
}


template<int N>
TendonFingerMarginals TendonFingerModel<N>::get_marginals(
    const Values& values,
    const Marginals& marginals) const
{
    return get_marginals(
        values,
        [&](Key k) { return marginals.marginalCovariance(k); },
        [&](Key a, Key b) {
            JointMarginal jm = marginals.jointMarginalCovariance({a, b});
            // Stitch into a (dim_a + dim_b) x (dim_a + dim_b) dense block, ordered [a, b].
            Eigen::MatrixXd Saa = jm(a, a);
            Eigen::MatrixXd Sbb = jm(b, b);
            Eigen::MatrixXd Sab = jm(a, b);
            Eigen::MatrixXd out(Saa.rows() + Sbb.rows(), Saa.cols() + Sbb.cols());
            out.topLeftCorner(Saa.rows(), Saa.cols())     = Saa;
            out.topRightCorner(Sab.rows(), Sab.cols())    = Sab;
            out.bottomLeftCorner(Sab.cols(), Sab.rows())  = Sab.transpose();
            out.bottomRightCorner(Sbb.rows(), Sbb.cols()) = Sbb;
            return out;
        });
}


template<int N>
TendonFingerMarginals TendonFingerModel<N>::get_marginals(
    const Values& values,
    const CovFn& cov_of,
    const JointFn& joint_of) const
{
    TendonFingerMarginals m;

    m.rod = rod_->get_marginals(values, cov_of);
    m.tendon_config = tendon_config_;

    // Read fixed-size from values, assign to dynamic VectorXGaussian
    Eigen::Vector<double, N> t_mean = values.at<Eigen::Vector<double, N>>(get_tensions_key());
    m.tensions.mean = t_mean;
    m.tensions.cov = cov_of(get_tensions_key());

    m.external_wrenches.resize(num_nodes_);
    for (int i = 0; i < num_nodes_; i++) {
        Key key = get_external_wrench_key(i);
        Vector6Gaussian wrench;
        wrench.mean = values.at<Vector6>(key);
        wrench.cov = cov_of(key);
        m.external_wrenches[i] = wrench;
    }

    get_J_pose_tensions(joint_of, m);

    // Calculate the length of each tendon from disc poses
    Eigen::Vector<double, N> lengths = compute_tendon_lengths(values);
    m.tendon_lengths.resize(N);
    for (int t = 0; t < N; ++t)
        m.tendon_lengths[t] = lengths[t];

    return m;
}


// Explicit instantiations
template class TendonFingerModel<1>;
template class TendonFingerModel<2>;
template class TendonFingerModel<3>;
template class TendonFingerModel<4>;
template class TendonFingerModel<5>;
template class TendonFingerModel<6>;
template class TendonFingerModel<7>;
template class TendonFingerModel<8>;
template class TendonFingerModel<9>;
template class TendonFingerModel<10>;
