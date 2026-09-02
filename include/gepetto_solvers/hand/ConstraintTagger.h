#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <string>
#include <vector>


namespace gepetto_solvers {

// Stable, semantic identity for every hard constraint a hand model builds, in
// graph insertion order -- which is exactly the order ConstrainedOptProblem
// enumerates constraints in, since it collects them by a filtered walk of the
// graph.
//
// The point is transferring Lagrange multipliers across a REBUILD. lambda is
// indexed by a constraint's POSITION, so adding one constraint renumbers every
// multiplier after it; a tag like "tbl.contact|f4" names the same physical
// constraint no matter what else is in the graph. Tags are emitted at the
// insertion site rather than recovered by introspection because the graph cannot
// be read back: every equality is wrapped in ZeroCostConstraint and every
// inequality in CollisionInequalityConstraint, so the factor type says nothing,
// and e.g. a plane collision and a half-space inequality on the same node carry
// identical keys.
struct ConstraintTags {
    std::vector<std::string> eq;
    std::vector<std::string> ineq;
};


// The ONLY way a hard constraint may enter a hand graph. Both halves of the
// graph -- the kinematics contributed by a HandKinematics and the task
// constraints HandModel adds around it -- append through one tagger, so the tag
// order and the constraint order cannot drift apart. Adding a constraint any
// other way silently misaligns every tag after it.
//
// Passed by reference into HandKinematics::add_kinematics_factors so a mechanism
// whose kinematics needs a hard constraint (a closed linkage loop, say) can emit
// one without owning its own numbering. The tendon hand needs none -- its
// kinematics is all soft factors -- but the ordering guarantee has to exist
// before a hand that does need one is written, not after.
class ConstraintTagger {
public:
    void add_eq(gtsam::NonlinearFactorGraph& graph,
                const gtsam::NoiseModelFactor::shared_ptr& factor,
                std::string tag);

    void add_ineq(gtsam::NonlinearFactorGraph& graph,
                  const gtsam::NoiseModelFactor::shared_ptr& gap,
                  std::string tag);

    // Drop every tag. Called at the top of build_graph: the tags describe THIS
    // graph's constraints, and a stale entry would pair a multiplier with the
    // wrong one.
    void clear() { tags_ = ConstraintTags{}; }

    const ConstraintTags& tags() const { return tags_; }

private:
    ConstraintTags tags_;
};

}  // namespace gepetto_solvers
