#pragma once

#include "gepetto_solvers/digits/tendon/TendonFingerSolver.h"  // TendonFingerSolverConfig, SpherePrimitiveContactConfig
#include "gepetto_solvers/utils/EnvironmentFactors.h"          // gepetto_solvers::EnvironmentConfig

#include <memory>
#include <optional>
#include <string>
#include <vector>


namespace gepetto_solvers {

// The kinematics half of a hand description: opaque to everything except the
// factory registered under the matching name. Derive one per kinematics.
struct HandKinematicsConfig {
    virtual ~HandKinematicsConfig() = default;
};


// Everything the tendon kinematics needs: one finger config per digit, in digit
// order. Downcast to by the "tendon" factory.
struct TendonHandKinematicsConfig : HandKinematicsConfig {
    std::vector<TendonFingerSolverConfig> fingers;
};


// One hand, described so that HandModel can build a graph for it without knowing
// what kind of mechanism it is.
//
// The split is the whole point:
//   * `env` / `sphere_contact` are the TASK side -- which surfaces this digit
//     contacts, avoids, slides on. Kinematics-agnostic, read directly by
//     HandModel::build_graph.
//   * `kinematics_config` is the MECHANISM side -- interpreted only by the
//     factory named in `kinematics`, and never inspected by HandModel.
//
// make_tendon_hand_spec() builds one from the (name, TendonFingerSolverConfig)
// list the Python layer assembles, splitting each config's sdf_contact /
// sphere_contact off into `env` / `sphere_contact`.
struct HandSpec {
    // Registry key naming the HandKinematics factory to load.
    std::string kinematics = "tendon";

    std::vector<std::string> digit_names;

    // The digit that opposes the others in the pre-grasp constraints -- the
    // thumb on an anatomical hand. -1 (default) means the hand has no opposing
    // digit, and the pre-grasp centering / axis-alignment constraints are then
    // not built.
    //
    // An INDEX, declared by whoever built the hand, rather than a name matched
    // against the literal "thumb": a hand with two opposable digits, or one
    // whose digits are numbered rather than named, has no such string to match.
    int opposing_digit = -1;

    // Per-digit task environment. Either may be empty for a digit that carries
    // no environment at all. Both are indexed in digit order.
    std::vector<std::optional<EnvironmentConfig>> env;
    std::vector<std::optional<SpherePrimitiveContactConfig>> sphere_contact;

    std::shared_ptr<HandKinematicsConfig> kinematics_config;

    int num_digits() const { return static_cast<int>(digit_names.size()); }

    // Throws with a specific message if the per-digit vectors disagree in
    // length, a kinematics is unnamed, or the payload is missing.
    void validate() const;
};


// Build a HandSpec for the tendon kinematics from the (name, config) pairs the
// Python layer assembles, splitting each config's task env off from its rod and
// tendon geometry.
HandSpec make_tendon_hand_spec(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    int opposing_digit = -1);

}  // namespace gepetto_solvers
