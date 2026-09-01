#pragma once

#include "gepetto_solvers/hand/ConstraintTagger.h"
#include "gepetto_solvers/hand/HandState.h"
#include "gepetto_solvers/utils/Gaussians.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <optional>
#include <string>
#include <vector>


namespace gepetto_solvers {

// A place on the hand that a task constraint may attach to.
//
// `node` keeps the addressing EnvironmentConfig already uses (target_contact_node,
// collision_node_indices, table_contact_node, half_space_node, ...): >= 0 counts
// from the digit's base, < 0 from its tip. Those integers stay meaningful for a
// hand that is not a chain of rod nodes -- a kinematics is free to map them onto
// whatever variables it owns -- so the whole environment layer is unchanged by
// this abstraction.
struct HandSite {
    int digit;
    int node;
};


// The kinematics of ONE hand: everything internal to the mechanism.
//
// This is the seam that lets HandModel::build_graph add task constraints without
// knowing what it is posing. A hand may be a set of digits that each carry their
// own chain (the tendon hand), or a single mechanism whose digits are only an
// addressing convention over shared variables -- nothing here assumes which.
//
// The contract has two halves:
//
//   1. FACTORS. add_kinematics_factors contributes every factor that IS this
//      hand's kinematics: rod, tendon and root-reparameterization factors for
//      the tendon hand; joint, linkage or loop-closure factors for another.
//      HandModel adds nothing structural of its own beyond the shared wrist
//      prior and the task constraints.
//
//   2. GEOMETRY. site_pose_key answers "which pose variable is at this place on
//      the hand", which is the only question the task constraints ask. Every
//      contact, collision, support-plane, half-space and pre-grasp factor in
//      HandGraph.cpp is built from it.
//
// A future HandDynamics would implement the same add_*_factors shape and be
// appended alongside this one; HandModel calls its contributors in a fixed
// order, so adding one does not disturb the task-constraint ordering the AL
// multiplier transfer depends on.
class HandKinematics {
public:
    virtual ~HandKinematics() = default;

    // -- identity ---------------------------------------------------------
    virtual const std::vector<std::string>& digit_names() const = 0;
    int num_digits() const { return static_cast<int>(digit_names().size()); }

    // -- the factors that ARE this hand's kinematics ----------------------
    //
    // `actuation` is one Gaussian per digit over that digit's actuation
    // variable -- tendon tensions here, joint torques or commanded angles
    // elsewhere. `tip_wrenches` is the terminal external wrench per digit; a
    // mechanism with no such variable ignores it.
    //
    // Any HARD constraint must go through `tags`, never straight onto the
    // graph, so the kinematics and task halves share one constraint numbering.
    virtual void add_kinematics_factors(
        gtsam::NonlinearFactorGraph& graph,
        ConstraintTagger& tags,
        const std::vector<VectorXGaussian>& actuation,
        const std::vector<Vector6Gaussian>& tip_wrenches) = 0;

    // -- what the TASK constraints key off --------------------------------

    // The pose variable at `site`. Throws if the site does not exist.
    virtual gtsam::Key site_pose_key(HandSite site) const = 0;

    // Whether `site` resolves to the shared wrist/root variable rather than a
    // pose of its own. Collision gap factors read a key's translation, so a
    // site that aliases the wrist would report the wrist origin instead of the
    // place it names -- those spheres are excluded from every collision pass.
    virtual bool site_is_root(HandSite site) const = 0;

    // This digit's fixed attachment to the wrist, T_offset in
    // T_0^digit = T_wrist o T_offset. Read by the in-plane contact factor,
    // which spans the digit's mounting base.
    virtual gtsam::Pose3 digit_base_offset(int digit) const = 0;

    // The digit's actuation variable (tendon tensions / joint commands), and
    // its displacement variable (tendon lengths / joint positions) where one
    // exists. The trajectory planner links both across timesteps.
    virtual gtsam::Key actuation_key(int digit) const = 0;
    virtual std::optional<gtsam::Key> displacement_key(int digit) const = 0;

    // -- values / state round trip ----------------------------------------

    // Seed every variable this kinematics owns. `warm`, when given, supplies
    // poses to prefer over the cold guess.
    virtual void insert_initial_values(gtsam::Values& values,
                                       const gtsam::Values* warm) const = 0;

    // Re-key a solved state onto THIS instance's variables, producing a partial
    // Values suitable as the `warm` argument above. Throws if the state's shape
    // disagrees with this kinematics.
    virtual void insert_from_state(gtsam::Values& values,
                                   const HandState& state) const = 0;

    // Read the solved state back out. `marginals` of nullptr means means-only
    // (skip the covariance factorization), which is what the per-iteration
    // snapshots use.
    virtual HandState extract(const gtsam::Values& values,
                              const gtsam::Marginals* marginals) const = 0;

    // -- trajectory support -----------------------------------------------

    // Link this timestep's actuation/displacement variables to `next`'s with GP
    // temporal priors. An empty Qc disables that chain.
    virtual void add_temporal_gp(gtsam::NonlinearFactorGraph& graph,
                                 const HandKinematics& next,
                                 const Eigen::MatrixXd& gp_actuation_Qc,
                                 const Eigen::MatrixXd& gp_displacement_Qc,
                                 double dt) const = 0;

    // Direct priors on the actuation / displacement variables, on top of
    // whatever add_kinematics_factors already emitted. One Gaussian per digit.
    virtual void add_actuation_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& actuation) const = 0;
    virtual void add_displacement_priors(
        gtsam::NonlinearFactorGraph& graph,
        const std::vector<VectorXGaussian>& displacement) const = 0;
};

}  // namespace gepetto_solvers
