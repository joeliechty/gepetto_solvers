#include "gepetto_solvers/hand/HandKinematicsRegistry.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace gepetto_solvers {

namespace {

// Function-local static, not a namespace-scope one: a kinematics registers at
// static-init time, and a namespace-scope map might not be constructed yet when
// it does (the static initialization order fiasco). This way the first
// registration constructs it.
std::map<std::string, HandKinematicsFactory>& registry() {
    static std::map<std::string, HandKinematicsFactory> r;
    return r;
}

}  // namespace


void register_hand_kinematics(const std::string& name, HandKinematicsFactory factory) {
    registry()[name] = std::move(factory);
}


std::vector<std::string> registered_hand_kinematics() {
    std::vector<std::string> names;
    names.reserve(registry().size());
    for (const auto& [name, _] : registry()) names.push_back(name);
    return names;   // std::map iterates sorted
}


std::unique_ptr<HandKinematics> load_hand_kinematics(
    const HandSpec& spec,
    const gtsam::Pose3& wrist_pose,
    gtsam::Key wrist_key)
{
    spec.validate();

    auto it = registry().find(spec.kinematics);
    if (it == registry().end()) {
        std::string known;
        for (const auto& n : registered_hand_kinematics()) {
            if (!known.empty()) known += ", ";
            known += "\"" + n + "\"";
        }
        if (known.empty()) known = "(none -- no kinematics is linked into this build)";
        throw std::invalid_argument(
            "load_hand_kinematics: no kinematics registered as \"" +
            spec.kinematics + "\". Registered: " + known + ".");
    }

    auto kin = it->second(*spec.kinematics_config, spec.digit_names,
                          wrist_pose, wrist_key);
    if (!kin)
        throw std::invalid_argument(
            "load_hand_kinematics: the factory for \"" + spec.kinematics +
            "\" returned null.");
    if (kin->num_digits() != spec.num_digits())
        throw std::invalid_argument(
            "load_hand_kinematics: \"" + spec.kinematics + "\" built " +
            std::to_string(kin->num_digits()) + " digits but the spec names " +
            std::to_string(spec.num_digits()) + ".");
    return kin;
}


void HandSpec::validate() const {
    if (kinematics.empty())
        throw std::invalid_argument("HandSpec: kinematics name is empty.");
    if (digit_names.empty())
        throw std::invalid_argument("HandSpec: no digits.");
    if (!kinematics_config)
        throw std::invalid_argument(
            "HandSpec: kinematics_config is null; \"" + kinematics +
            "\" has nothing to build from.");
    const size_t n = digit_names.size();
    if (env.size() != n)
        throw std::invalid_argument(
            "HandSpec: env has " + std::to_string(env.size()) +
            " entries but there are " + std::to_string(n) + " digits.");
    if (sphere_contact.size() != n)
        throw std::invalid_argument(
            "HandSpec: sphere_contact has " + std::to_string(sphere_contact.size()) +
            " entries but there are " + std::to_string(n) + " digits.");
    if (opposing_digit >= static_cast<int>(n))
        throw std::invalid_argument(
            "HandSpec: opposing_digit " + std::to_string(opposing_digit) +
            " is out of range for " + std::to_string(n) + " digits.");
}


HandSpec make_tendon_hand_spec(
    const std::vector<std::pair<std::string, TendonFingerSolverConfig>>& finger_configs,
    int opposing_digit)
{
    HandSpec spec;
    spec.kinematics = "tendon";
    spec.opposing_digit = opposing_digit;

    auto kin_config = std::make_shared<TendonHandKinematicsConfig>();
    kin_config->fingers.reserve(finger_configs.size());

    spec.digit_names.reserve(finger_configs.size());
    spec.env.reserve(finger_configs.size());
    spec.sphere_contact.reserve(finger_configs.size());

    for (const auto& [name, c] : finger_configs) {
        spec.digit_names.push_back(name);
        spec.env.push_back(c.sdf_contact);
        spec.sphere_contact.push_back(c.sphere_contact);
        kin_config->fingers.push_back(c);
    }

    spec.kinematics_config = std::move(kin_config);
    return spec;
}

}  // namespace gepetto_solvers
