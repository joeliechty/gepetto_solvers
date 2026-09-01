#include "gepetto_solvers/hand/ConstraintTagger.h"

#include "gepetto_solvers/utils/EnvironmentFactors.h"   // CollisionInequalityConstraint

#include <gtsam/constrained/NonlinearEqualityConstraint.h>

namespace gepetto_solvers {

void ConstraintTagger::add_eq(gtsam::NonlinearFactorGraph& graph,
                              const gtsam::NoiseModelFactor::shared_ptr& factor,
                              std::string tag) {
    graph.add(gtsam::ZeroCostConstraint(factor));
    tags_.eq.push_back(std::move(tag));
}


void ConstraintTagger::add_ineq(gtsam::NonlinearFactorGraph& graph,
                                const gtsam::NoiseModelFactor::shared_ptr& gap,
                                std::string tag) {
    graph.add(gepetto_solvers::CollisionInequalityConstraint(gap));
    tags_.ineq.push_back(std::move(tag));
}

}  // namespace gepetto_solvers
