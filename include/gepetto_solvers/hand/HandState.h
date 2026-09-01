#pragma once

#include "gepetto_solvers/digits/tendon/TendonFingerModel.h"   // TendonFingerMarginals

#include <string>
#include <vector>


// One solved hand, per digit, in a form that does not depend on the variable
// KEYS any particular model handed out.
//
// Why key-independent: CosseratRodModel hands out keys from a global counter, so
// two separately-constructed HandModels use different Symbols for the same
// physical variable and a gtsam::Values from one cannot be merged into the other
// by key. This bundle is indexed by digit, node and disc instead, so a solve's
// output re-seeds another solver directly (HandKinematics::insert_from_state).
//
// KNOWN LIMIT -- the one place a kinematics type still shows through the neutral
// layer. `digits` is a vector of TendonFingerMarginals, so a hand whose state is
// not rod-and-tendon shaped cannot fill this bundle as-is; it would need a
// variant payload here, and a matching split in the Python readers.
//
// Left concrete deliberately, and it does not compromise the thing this refactor
// is for: HandModel::build_graph -- the part that had to stop knowing what kind
// of mechanism it is posing -- never touches a HandState. It addresses the hand
// through HandKinematics::site_pose_key alone. The state bundle is only the
// TRANSPORT for a finished solve, and it is produced and consumed entirely
// behind HandKinematics::extract / insert_from_state, so a second kinematics can
// generalize it then, against a real second payload, rather than now against a
// guess. The cost of guessing wrong is high: every Python reader of
// `state.digits[i].rod.states[j]` -- HandResult, the viser plotters, witness.py
// -- would move with it.
struct HandState {
    std::vector<std::string> digit_names;
    std::vector<TendonFingerMarginals> digits;
};
