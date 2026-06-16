#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <optional>
#include <vector>


/// Factor enforcing the inextensibility constraint between tendon lengths
/// and disc poses:  e_t = L_t - sum_j || T_{j+1} * r_{t,j+1} - T_j * r_{t,j} ||
///
/// Inherits from NoiseModelFactor (not NoiseModelFactorN) because the number
/// of keys is 1 + num_discs, which varies at runtime.
///
/// Keys: [lengths_key, disc_pose_0_key, disc_pose_1_key, ..., disc_pose_{D-1}_key]
/// Error dimension: N (one per tendon)
template<int N>
class TendonLengthFactor : public gtsam::NoiseModelFactor {
public:
    using NoiseModelFactor::unwhitenedError;

    TendonLengthFactor(
        gtsam::Key lengths_key,
        const std::vector<gtsam::Key>& disc_pose_keys,
        const std::vector<std::vector<std::optional<gtsam::Vector3>>>& hole_locations,
        const gtsam::SharedNoiseModel& model,
        // Hand-base reparameterization (Section 4): when the first disc is the
        // reparameterized node 0, disc_pose_keys[0] is the hand base and this
        // fixed offset reconstructs the base-disc pose as T_base o offset, with
        // the Jacobian chain-ruled back to the hand base. Unset = legacy behavior.
        std::optional<gtsam::Pose3> first_disc_offset = std::nullopt);

    gtsam::Vector unwhitenedError(
        const gtsam::Values& x,
        gtsam::OptionalMatrixVecType H = nullptr) const override;

private:
    int num_discs_;
    // hole_locations_[disc_idx][tendon_idx] = optional<Vector3> in local disc frame
    std::vector<std::vector<std::optional<gtsam::Vector3>>> hole_locations_;
    std::optional<gtsam::Pose3> first_disc_offset_;
};
