#pragma once

#include "gepetto_solvers/hand/HandKinematics.h"
#include "gepetto_solvers/hand/HandSpec.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>


namespace gepetto_solvers {

// Builds one HandKinematics instance. `wrist_key` is the shared base variable
// every digit attaches to, and `wrist_pose` its mean -- a kinematics seeds its
// cold start from the two together.
using HandKinematicsFactory = std::function<std::unique_ptr<HandKinematics>(
    const HandKinematicsConfig& config,
    const std::vector<std::string>& digit_names,
    const gtsam::Pose3& wrist_pose,
    gtsam::Key wrist_key)>;


// Register a kinematics under `name`. Implementations self-register at static
// init, so linking a kinematics in is all it takes to make it loadable.
// Re-registering a name replaces the previous factory.
void register_hand_kinematics(const std::string& name, HandKinematicsFactory factory);

// Build the kinematics named by `spec.kinematics` from `spec`. Throws
// std::invalid_argument naming every registered kinematics if the name is
// unknown -- a hand silently failing to load is the trap this whole layer would
// otherwise set.
std::unique_ptr<HandKinematics> load_hand_kinematics(
    const HandSpec& spec,
    const gtsam::Pose3& wrist_pose,
    gtsam::Key wrist_key);

// Every registered name, sorted. For error messages and for the Python layer to
// report what this build can load.
std::vector<std::string> registered_hand_kinematics();

}  // namespace gepetto_solvers
