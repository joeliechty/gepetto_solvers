#pragma once

#include "gepetto_solvers/utils/Gaussians.h"

#include <gtsam/base/Matrix.h>

#include <memory>
#include <string>
#include <vector>


// One solved hand, in a form that does not depend on the variable KEYS any
// particular model handed out, and does not name any particular mechanism.
//
// Why key-independent: models hand out keys from a global counter, so two
// separately-constructed HandModels use different Symbols for the same physical
// variable and a gtsam::Values from one cannot be merged into the other by key.
// This bundle is indexed by digit and site instead, so a solve's output re-seeds
// another solver directly (HandKinematics::insert_from_state).
//
// Why mechanism-independent: it is produced and consumed entirely behind
// HandKinematics::extract / insert_from_state, and a rod-and-tendon hand, a
// URDF-described rigid-body hand and anything else all have to fit through it.
// The parts that only ONE kind of mechanism has live behind `extras`.


// A place on a digit, and what was solved there.
//
// `stress` and `wrench` are continuum-rod quantities. A mechanism with no such
// state leaves them zero rather than absent -- they are two Vector6Gaussians, so
// carrying them costs little and every reader that wants them is rod-specific
// anyway.
struct SiteState {
    Pose3Gaussian   pose;
    Vector6Gaussian stress;
    Vector6Gaussian wrench;
};


// Per-digit state that only one kind of mechanism has. Derive one per
// kinematics; readers downcast, and check for null first.
//
// Same shape as HandKinematicsConfig on the input side: the neutral layer names
// the base, and only the kinematics that owns the payload knows the derived
// type.
struct DigitExtras {
    virtual ~DigitExtras() = default;
};


struct DigitState {
    // One per site, in the digit's own site order (site 0 at the base, the last
    // one at the tip). HandKinematics::site_pose_key addresses the same places.
    std::vector<SiteState> sites;

    // What drives this digit: tendon tensions on the tendon hand, joint
    // positions on a rigid-body one. Matches the digit's actuation variable.
    VectorXGaussian actuation;

    // The digit's displacement readout where it has one distinct from its
    // actuation (tendon lengths). Empty when actuation IS position, as it is for
    // a position-controlled rigid-body hand.
    std::vector<double> displacement;

    // Which sites carry a collision sphere, as indices into `sites`. Read off
    // the STATE rather than off a config so an overlay can never mark a sphere
    // the solve did not actually carry.
    std::vector<int> collision_sites;

    // Null on a mechanism that has none. See TendonDigitExtras.
    std::shared_ptr<DigitExtras> extras;
};


struct HandState {
    std::vector<std::string> digit_names;
    std::vector<DigitState>  digits;

    // The shared wrist, in the world frame.
    //
    // Carried explicitly because recovering it is a per-mechanism question, and
    // the Python side used to answer it with a tendon-hand-specific trick: node
    // 0 is not a variable under the rod's root reparameterization, so
    // solved_wrist_pose inverted digit 0's mounting offset out of its base pose.
    // A hand that owns its wrist variable outright simply reads it. Putting the
    // answer here lets each kinematics give it correctly and keeps every caller
    // out of the question.
    gtsam::Matrix4 wrist_pose = gtsam::Matrix4::Identity();
};
