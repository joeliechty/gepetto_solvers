#pragma once

#include "gepetto_solvers/digits/tendon/TendonFingerModel.h"
#include "gepetto_solvers/digits/tendon/TendonFingerSolver.h"
#include "gepetto_solvers/hand/HandKinematics.h"
#include "gepetto_solvers/hand/HandSpec.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>


namespace gepetto_solvers {

// The tendon hand's kinematics: a set of TendonFingerModel<N> digits that all
// share ONE floating wrist variable.
//
// Each digit attaches to the wrist through its own fixed offset T_offset
// (config.hand_base_offset), so its node-0 pose is T_0 = T_wrist o T_offset.
// Node 0 is therefore NOT a variable: set_hand_base swaps it for the shared
// wrist key and the digit emits the Root* factor variants instead. That is what
// expresses the joint prior over the wrist and the digit bases as one Gaussian
// on the wrist times a deterministic SE(3) composition per digit, rather than as
// a soft rigidity penalty with a null space in it.
//
// Registered as "tendon". Everything here was previously inline in
// the old monolithic hand model; TendonFingerModel and its factors are unchanged.
class TendonHandKinematics : public HandKinematics {
public:
    TendonHandKinematics(const TendonHandKinematicsConfig& config,
                         const std::vector<std::string>& digit_names,
                         const gtsam::Pose3& wrist_pose,
                         gtsam::Key wrist_key);

    const std::vector<std::string>& digit_names() const override { return digit_names_; }

    void add_kinematics_factors(
        gtsam::NonlinearFactorGraph& graph,
        ConstraintTagger& tags,
        const std::vector<VectorXGaussian>& actuation,
        const std::vector<Vector6Gaussian>& tip_wrenches) override;

    gtsam::Key site_pose_key(HandSite site) const override;
    bool site_is_root(HandSite site) const override;
    gtsam::Pose3 digit_base_offset(int digit) const override;
    gtsam::Key actuation_key(int digit) const override;
    std::optional<gtsam::Key> displacement_key(int digit) const override;

    void insert_initial_values(gtsam::Values& values,
                               const gtsam::Values* warm) const override;
    void insert_from_state(gtsam::Values& values,
                           const HandState& state) const override;
    HandState extract(const gtsam::Values& values,
                      const gtsam::Marginals* marginals) const override;

    void add_temporal_gp(gtsam::NonlinearFactorGraph& graph,
                         const HandKinematics& next,
                         const Eigen::MatrixXd& gp_actuation_Qc,
                         const Eigen::MatrixXd& gp_displacement_Qc,
                         double dt) const override;
    void add_actuation_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& actuation) const override;
    void add_displacement_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& displacement) const override;

private:
    // A variant, to carry different numbers of tendons per digit.
    using FingerVariant = std::variant<
        std::unique_ptr<TendonFingerModel<1>>,
        std::unique_ptr<TendonFingerModel<2>>,
        std::unique_ptr<TendonFingerModel<3>>,
        std::unique_ptr<TendonFingerModel<4>>,
        std::unique_ptr<TendonFingerModel<5>>,
        std::unique_ptr<TendonFingerModel<6>>,
        std::unique_ptr<TendonFingerModel<7>>,
        std::unique_ptr<TendonFingerModel<8>>,
        std::unique_ptr<TendonFingerModel<9>>,
        std::unique_ptr<TendonFingerModel<10>>
    >;

    std::vector<FingerVariant> fingers_;
    std::vector<std::string> digit_names_;

    // Each digit's fixed attachment to the wrist. Kept because that relation is
    // the ONLY way back from a state bundle to the wrist: node 0 has no pose
    // variable under the reparameterization, so insert_from_state has to invert
    // it.
    std::vector<gtsam::Pose3> hand_base_offsets_;

    // Per-digit interior/tip external-wrench prior noise, derived from each
    // digit's sigma_stress_moment/force (as TendonFingerSolver does). Using the
    // digit's own tight stress noise here matters for conditioning -- a loose
    // hand-wide value leaves the wrench variables weakly pinned and makes the
    // contact system indeterminate.
    std::vector<gtsam::SharedDiagonal> small_wrench_noises_;

    gtsam::Key wrist_key_;

    const FingerVariant& finger_at(int digit) const;
};

}  // namespace gepetto_solvers
