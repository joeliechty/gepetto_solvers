#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <memory>
#include <string>
#include <vector>

// Pinocchio's headers are heavy and drag Boost.MPL configuration in with them,
// so they stay out of this header: the model and its scratch data are held
// behind a forward declaration and an opaque impl. Only the .cpp includes them.
namespace pinocchio {
template <typename, int> struct JointCollectionDefaultTpl;
template <typename, int, template <typename, int> class> struct ModelTpl;
template <typename, int, template <typename, int> class> struct DataTpl;
using Model = ModelTpl<double, 0, JointCollectionDefaultTpl>;
using Data = DataTpl<double, 0, JointCollectionDefaultTpl>;
}  // namespace pinocchio


namespace gepetto_solvers {

// One rigid-body kinematics model, shared by every factor built on it.
//
// Owns the pinocchio::Model. Immutable once built, so all of a hand's factors
// share one instance; each factor keeps its own scratch Data, because GTSAM may
// linearize factors in parallel and forwardKinematics writes into Data.
class RigidChainModel {
public:
    // Build from URDF text rather than a path, so a caller can hold the robot
    // description however it likes and a test can write one inline.
    //
    // Returns a non-const handle only because pybind11 cannot hold
    // shared_ptr<const T>; the class is immutable in practice -- everything
    // after construction is a const accessor -- and factors keep it as
    // shared_ptr<const RigidChainModel>.
    static std::shared_ptr<RigidChainModel> from_urdf_xml(
        const std::string& urdf_xml);

    static std::shared_ptr<RigidChainModel> from_urdf_file(
        const std::string& path);

    ~RigidChainModel();

    const pinocchio::Model& model() const { return *model_; }

    // Frame index for a link/frame name. Throws naming the frame if absent --
    // a typo'd frame name would otherwise be a silent zero pose.
    int frame_id(const std::string& name) const;

    // Configuration and velocity indices of a 1-DOF joint, by name. Separate
    // because they differ for joints pinocchio parameterizes with more
    // coordinates than degrees of freedom (a `continuous` joint is nq=2, nv=1).
    // Throws for a joint that is not 1-DOF: everything downstream indexes one
    // scalar per joint, and silently mishandling a ball joint would be worse
    // than refusing it.
    std::pair<int, int> joint_indices(const std::string& name) const;

    int nq() const;
    int nv() const;

    // The CONSTANT placement of a joint's frame relative to the model root, as
    // a 4x4 -- where a digit hangs off the palm.
    //
    // Only constant if no other joint lies between the root and this one, so
    // that is checked rather than assumed: a digit mounted on a moving wrist
    // sub-chain would otherwise get a "fixed" mount that silently moves with q,
    // and every caller that recovers the wrist by inverting it would be wrong.
    gtsam::Matrix4 fixed_placement_of_joint(const std::string& name) const;

    // Position limits from the URDF, one entry per configuration coordinate.
    // Nothing enforces them yet -- see the note in AllegroKinematics -- but a
    // caller seeding or clamping a configuration reads them from here.
    std::vector<double> lower_position_limits() const;
    std::vector<double> upper_position_limits() const;

private:
    RigidChainModel();
    std::unique_ptr<pinocchio::Model> model_;
};


// The kinematics likelihood of the hand posterior, for one frame:
//
//     p(T_i | T_w, q) ~ exp( -1/2 || T_i (-) f_fk,i(T_w, q) ||^2_{Sigma_fk,i} )
//
// with (-) the SE(3) on-manifold difference. The residual is
//
//     e = Log( f_fk,i(T_w, q)^-1 * T_i )   in R^6,
//     f_fk,i(T_w, q) = T_w * T_fk,i(q)
//
// where T_fk,i(q) is Pinocchio's placement of frame i in the model's root frame
// and T_w is the shared wrist variable.
//
// TERNARY, and the wrist being a variable is why. The wrist carries only a soft
// prior here (SolverBase's sigma_wrist_*), so contact routinely pulls the hand
// tens of millimetres off the commanded base pose; a factor that treated the
// base as fixed would fight that instead of moving with it.
//
// Sigma_fk,i is the per-frame kinematic relaxation, supplied as this factor's
// noise model. As Sigma -> 0 the likelihood approaches a hard kinematic
// constraint; a looser one lets the solve trade FK consistency against the task
// constraints, which is what makes this a soft formulation rather than a
// substitution.
//
// JACOBIANS. Every manifold derivative is GTSAM's own -- the factor composes the
// error out of Pose3::compose and Pose3::localCoordinates and asks them for
// their Jacobians, then injects Pinocchio only where q enters:
//
//     H_wrist = Hl1 * Hc1
//     H_q     = Hl1 * Hc2 * SWAP * J_pin^LOCAL(q)
//     H_site  = Hl2
//
// SWAP exchanges the top and bottom three rows. Pinocchio stacks a spatial
// velocity as [v; w]; GTSAM's Pose3 tangent is [w; v], and compose's
// second-argument Jacobian is in the body frame -- which is exactly what
// pinocchio::LOCAL returns. Getting that swap wrong does not raise; it just
// converges badly, which is why tests/core/test_pinocchio_env.py pins the
// convention and the factor's own test checks all three blocks against
// numericalDerivative.
class PinocchioFKFactor
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector, gtsam::Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector, gtsam::Pose3>;
    using Base::evaluateError;

    // `q_index` / `v_index` map this factor's joint vector onto the model's
    // configuration and velocity coordinates, so one shared model can serve
    // several digits that each own a slice of it. Both must have the same
    // length, which is the dimension of the `q` variable this factor reads.
    PinocchioFKFactor(gtsam::Key wrist_key,
                      gtsam::Key joint_key,
                      gtsam::Key site_key,
                      std::shared_ptr<const RigidChainModel> chain,
                      int frame_id,
                      std::vector<int> q_index,
                      std::vector<int> v_index,
                      const gtsam::SharedNoiseModel& noise);

    ~PinocchioFKFactor() override;

    gtsam::Vector evaluateError(const gtsam::Pose3& wrist,
                                const gtsam::Vector& q,
                                const gtsam::Pose3& site,
                                gtsam::OptionalMatrixType H_wrist,
                                gtsam::OptionalMatrixType H_q,
                                gtsam::OptionalMatrixType H_site) const override;

    // f_fk,i(T_w, q): the frame placement this factor predicts. Exposed because
    // seeding wants exactly this -- an initial guess that puts each site at its
    // own prediction starts the solve ON the kinematics manifold instead of
    // somewhere the FK likelihood immediately has to pull it back from.
    gtsam::Pose3 predict(const gtsam::Pose3& wrist, const gtsam::Vector& q) const;

    gtsam::NonlinearFactor::shared_ptr clone() const override;

private:
    std::shared_ptr<const RigidChainModel> chain_;
    // Scratch, per factor: forwardKinematics writes into it, and GTSAM may
    // linearize factors on several threads.
    std::unique_ptr<pinocchio::Data> data_;
    int frame_id_;
    std::vector<int> q_index_;
    std::vector<int> v_index_;

    // T_fk,i(q) and, when asked, its 6xM body-frame Jacobian already swapped
    // into GTSAM's [w; v] row order and narrowed to this factor's columns.
    gtsam::Pose3 frame_placement(const gtsam::Vector& q,
                                 gtsam::Matrix* H_local) const;
};

}  // namespace gepetto_solvers
