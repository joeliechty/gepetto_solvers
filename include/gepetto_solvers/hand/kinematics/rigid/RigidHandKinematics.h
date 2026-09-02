#pragma once

#include "gepetto_solvers/digits/rigid/PinocchioFKFactor.h"
#include "gepetto_solvers/hand/HandKinematics.h"
#include "gepetto_solvers/hand/HandSpec.h"

#include <memory>
#include <string>
#include <vector>


namespace gepetto_solvers {

// One digit of a URDF-described hand.
struct RigidDigitSpec {
    std::string name;

    // This digit's 1-DOF joints, base to tip. Their count is the dimension of
    // the digit's actuation variable q.
    std::vector<std::string> joints;

    // Frame names for sites 1..N, base to tip. Site 0 is NOT listed: it is the
    // digit's fixed mount on the palm, which is not a variable (see
    // RigidHandKinematics for why).
    std::vector<std::string> site_frames;
};


struct RigidHandKinematicsConfig : HandKinematicsConfig {
    // The robot description. Give one: `urdf_xml` wins if both are set, so a
    // caller can hold the description in memory and a test can write one inline.
    std::string urdf_xml;
    std::string urdf_path;

    std::vector<RigidDigitSpec> digits;

    // Sigma_fk: the diagonal of the kinematic relaxation covariance, ordered
    // [rot(3), pos(3)] to match GTSAM's Pose3 tangent. As Sigma -> 0 the
    // likelihood approaches a hard kinematic constraint.
    //
    // The default buys near-hardness without paying for it in iterations.
    // Tightening it does not fail -- every setting below reaches machine-zero
    // residual -- it just costs steps, because a stiffer FK likelihood means a
    // larger step in the site poses for the same step in q. Measured on the
    // Allegro hand, seeded a full 0.4 rad per joint away from the prior mean:
    //
    //     sigma rot / pos      iterations to converge
    //     1e-2 / 1e-3           4
    //     1e-3 / 1e-4           5
    //     3e-4 / 3e-5           9
    //     1e-4 / 1e-5          18     <- default
    //     3e-5 / 3e-6          44
    //     1e-5 / 1e-6         103
    //
    // 1e-5 m is ten microns: two orders below any contact tolerance in this
    // repository, so the kinematics is already exact for every purpose it is
    // put to. Seeded AT the prior mean -- the normal case, since q_init and q_S
    // are the same posture -- it converges in one iteration at any of these.
    gtsam::Vector6 sigma_fk =
        (gtsam::Vector6() << 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5).finished();

    // Per-site override of sigma_fk, indexed [digit][site-1] to match
    // `site_frames`. Empty (the default) uses `sigma_fk` everywhere. Present
    // because the formulation defines Sigma_fk,i PER FRAME: a caller that wants
    // the fingertip pinned harder than the proximal links says so here.
    std::vector<std::vector<gtsam::Vector6>> site_sigma_fk;

    // Seed configuration per digit, same order as `digits`. An empty entry
    // seeds that digit at zero.
    std::vector<std::vector<double>> q_init;
};


// A rigid-body hand whose mechanism comes from a URDF, posed by the kinematics
// likelihood p(T_i | T_w, q) rather than by a continuum model.
//
// Registered as "rigid_urdf". The Allegro hand is one CONFIG for it, not a
// subclass -- anything that is a set of serial 1-DOF chains hanging off a common
// palm is the same mechanism as far as this class is concerned.
//
// VARIABLES, per the hand posterior p(Theta) ~ p(T_w) p(q) p(T | T_w, q):
//
//   * the shared wrist T_w, owned by HandModel, which also emits p(T_w);
//   * one joint vector q^d per DIGIT (Symbol('J', ...)), not one 16-vector for
//     the hand. Each digit's FK touches only its own joints, so the split is
//     exact -- and it is what lets the per-digit actuation priors and the
//     planner's per-digit GP chains work unchanged. A shared vector would put
//     four copies of every such factor on one variable, which multiplies
//     Gaussians rather than repeating them: the prior would come out four times
//     tighter than asked for, silently. Coupling between digits, if a hand ever
//     needs it, goes in as a factor spanning several q^d keys.
//   * one pose T_i per SITE (Symbol('K', ...)), tied to (T_w, q^d) by a
//     PinocchioFKFactor. They are variables rather than derived quantities
//     because that is what every task constraint keys off -- contact, collision,
//     support plane and pre-grasp all ask site_pose_key and nothing else.
//
// SITE 0 IS THE FIXED MOUNT, and aliases the wrist key rather than carrying a
// variable of its own -- exactly as the tendon hand's node 0 does under its root
// reparameterization. That keeps the interface's promise that
// T_0^digit = T_wrist o digit_base_offset(digit), which callers rely on to
// recover the wrist from a frame alone, and it costs nothing: the mount is
// rigidly placed by the wrist, so a variable there would only have to be pinned
// back to it. site_is_root reports true for it, so the collision passes skip it.
//
// JOINT LIMITS ARE NOT ENFORCED. The URDF has them and RigidChainModel exposes
// them, but nothing here builds a constraint from them, so an IK solve is free
// to hyperextend into a configuration the real hand cannot reach. They belong as
// AL inequalities through the ConstraintTagger this class is handed -- which
// also needs HandModel to route a hand with hard kinematics constraints onto the
// Augmented Lagrangian path, something it currently decides from the task
// environment alone. Until then, p(q) is the only thing keeping q sane.
class RigidHandKinematics : public HandKinematics {
public:
    RigidHandKinematics(const RigidHandKinematicsConfig& config,
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

    // Sites per digit INCLUDING the fixed mount, so this is site_frames + 1.
    int num_sites(int digit) const;

private:
    struct Digit {
        std::string name;
        std::vector<int> q_index;      // into the model's configuration
        std::vector<int> v_index;      // into the model's velocity/Jacobian
        std::vector<int> frame_id;     // one per site 1..N
        std::vector<gtsam::Vector6> sigma;  // one per site 1..N
        gtsam::Pose3 mount;            // constant palm -> digit base
        Eigen::VectorXd q_init;
    };

    // Resolve a site index (negative counts from the tip) to 0..num_sites-1.
    int resolve_site(HandSite site) const;

    std::shared_ptr<const RigidChainModel> chain_;
    std::vector<Digit> digits_;
    std::vector<std::string> digit_names_;
    gtsam::Pose3 wrist_pose_;
    gtsam::Key wrist_key_;

    // Instance id, so two models built for two timesteps of a trajectory hand
    // out distinct keys -- the same reason CosseratRodModel carries one.
    int id_;
    inline static int next_id_ = 0;

    // Stride per digit in the key space, large enough that no hand's site count
    // reaches it; checked at construction rather than assumed.
    static constexpr int kDigitStride = 64;
};

}  // namespace gepetto_solvers
