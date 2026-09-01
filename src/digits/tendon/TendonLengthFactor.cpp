#include "gepetto_solvers/digits/tendon/TendonLengthFactor.h"

using namespace gtsam;


template<int N>
TendonLengthFactor<N>::TendonLengthFactor(
    Key lengths_key,
    const std::vector<Key>& disc_pose_keys,
    const std::vector<std::vector<std::optional<Vector3>>>& hole_locations,
    const SharedNoiseModel& model,
    std::optional<Pose3> first_disc_offset)
:
    NoiseModelFactor(model, /* keys: */ [&]() {
        KeyVector keys;
        keys.push_back(lengths_key);
        keys.insert(keys.end(), disc_pose_keys.begin(), disc_pose_keys.end());
        return keys;
    }()),
    num_discs_(disc_pose_keys.size()),
    hole_locations_(hole_locations),
    first_disc_offset_(first_disc_offset)
{}


template<int N>
Vector TendonLengthFactor<N>::unwhitenedError(
    const Values& x,
    OptionalMatrixVecType H) const
{
    // Extract lengths variable (key 0)
    Eigen::Vector<double, N> L = x.at<Eigen::Vector<double, N>>(keys()[0]);

    // Extract disc poses (keys 1..num_discs_)
    std::vector<Pose3> poses(num_discs_);
    for (int d = 0; d < num_discs_; ++d)
        poses[d] = x.at<Pose3>(keys()[1 + d]);

    // Hand-base reparameterization: the first disc key is the hand base; the
    // base-disc pose is T_base o offset. Keep the composition Jacobian so the
    // accumulated Jacobian for that key maps back to the hand base.
    Matrix6 H_first_compose = Matrix6::Identity();
    if (first_disc_offset_ && num_discs_ > 0) {
        poses[0] = poses[0].compose(*first_disc_offset_, H_first_compose);
    }

    // Initialize Jacobians if requested
    if (H) {
        H->resize(1 + num_discs_);
        // H[0]: d(error)/d(L) = I_N
        (*H)[0] = Eigen::Matrix<double, N, N>::Identity();
        // H[1+d]: d(error)/d(Pose_d) = Nx6, initialized to zero
        for (int d = 0; d < num_discs_; ++d)
            (*H)[1 + d] = Eigen::Matrix<double, N, 6>::Zero();
    }

    // Compute geometric lengths and Jacobians for each tendon
    Eigen::Vector<double, N> geometric_lengths = Eigen::Vector<double, N>::Zero();

    for (int t = 0; t < N; ++t) {
        bool has_prev = false;
        Point3 p_prev;
        int prev_disc = -1;
        Matrix36 J_prev_pose;  // d(p_prev)/d(Pose_{prev_disc})

        for (int d = 0; d < num_discs_; ++d) {
            auto& hole_opt = hole_locations_[d][t];

            if (!hole_opt.has_value()) {
                // Tendon terminated at this disc
                break;
            }

            // Transform local hole to world frame
            Matrix36 J_p_pose;  // d(p_curr)/d(Pose_d), 3x6
            Point3 p_curr = poses[d].transformFrom(
                hole_opt.value(), H ? &J_p_pose : nullptr);

            if (has_prev) {
                // Segment from prev disc to current disc
                Vector3 diff = p_curr - p_prev;
                double dist = diff.norm();

                geometric_lengths[t] += dist;

                // Jacobians: d(dist)/d(p_curr) and d(dist)/d(p_prev)
                if (H && dist > 1e-12) {
                    // d(dist)/d(p) = diff^T / dist  (1x3 row vector)
                    Eigen::RowVector3d d_dist_d_pcurr = diff.transpose() / dist;
                    Eigen::RowVector3d d_dist_d_pprev = -diff.transpose() / dist;

                    // error = L - geometric, so d(error)/d(pose) = -d(geometric)/d(pose)
                    // d(geometric_t)/d(Pose_d) += d(dist)/d(p_curr) * d(p_curr)/d(Pose_d)
                    (*H)[1 + d].row(t) -= d_dist_d_pcurr * J_p_pose;

                    // d(geometric_t)/d(Pose_{prev_disc}) += d(dist)/d(p_prev) * d(p_prev)/d(Pose_{prev_disc})
                    (*H)[1 + prev_disc].row(t) -= d_dist_d_pprev * J_prev_pose;
                }
            }

            p_prev = p_curr;
            prev_disc = d;
            J_prev_pose = J_p_pose;
            has_prev = true;
        }
    }

    // Map the base-disc Jacobian back to the hand base (chain rule).
    if (H && first_disc_offset_ && num_discs_ > 0) {
        (*H)[1] = (*H)[1] * H_first_compose;
    }

    // Error: e = L - geometric_lengths
    return L - geometric_lengths;
}


// Explicit instantiations
template class TendonLengthFactor<1>;
template class TendonLengthFactor<2>;
template class TendonLengthFactor<3>;
template class TendonLengthFactor<4>;
template class TendonLengthFactor<5>;
template class TendonLengthFactor<6>;
template class TendonLengthFactor<7>;
template class TendonLengthFactor<8>;
template class TendonLengthFactor<9>;
template class TendonLengthFactor<10>;
