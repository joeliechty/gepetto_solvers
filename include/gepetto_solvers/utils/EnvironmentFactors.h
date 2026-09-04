#pragma once

// Umbrella header for the environment factors.
//
// Every geometric constraint in the system used to live here, in one 2218-line
// file holding 21 classes. They now sit one per header under
// `gepetto_solvers/environment/`, and this header includes all of them -- so the
// three files that include it (HandModel.h, TendonFingerSolver.h and the
// tendon_finger bindings) are unchanged, and anything wanting a single factor can
// include just that one.
//
// The pattern is uniform across all of them: a plain `NoiseModelFactor` whose
// UNWHITENED ERROR IS THE RAW GEOMETRIC VIOLATION, wrapped in a constraint class
// that hands it to the Augmented Lagrangian optimizer --
// `gtsam::ZeroCostConstraint(f)` for an equality `c(x) = 0`, and
// `CollisionInequalityConstraint(f)` for an inequality `c(x) <= 0` (whose
// inactive branch returns zero error AND zero Jacobian, preserving sparsity).
//
// Two implementation details you will trip over if you do not know them:
//
//  1. The C-frame is held fixed inside the Gauss-Newton step. `t1`/`t2` come from
//     `frisvad_tangent_basis(n_hat)` and their derivative contribution is
//     dropped -- the standard locally-constant-gradient contact convention. Same
//     for the surface normal in the `c_N` row.
//
//  2. Every ellipsoid row measures with a real signed DISTANCE (`EllipsoidDistance`
//     -- exact orthogonal by default, Taubin's first-order approximation under
//     `EnvironmentConfig::ellipsoid_taubin`), never the raw algebraic
//     `x^T M x - 1`. All three have the identical zero set, but the raw form's
//     residual and gradient scale as ~1/min(semi_axis)^2 -- ~40x the Euclidean
//     distance on a 5 cm sphere and ~1e6x along a coin's thin axis. Under a
//     shared unit noise model the raw row swamps every other row and the AL
//     inner solve stagnates.

#include "gepetto_solvers/environment/ConstraintWrappers.h"
#include "gepetto_solvers/environment/EllipsoidDistance.h"
#include "gepetto_solvers/environment/EnvironmentConfig.h"
#include "gepetto_solvers/environment/TangentBasis.h"

#include "gepetto_solvers/environment/collision/EllipsoidCollisionGapFactor.h"
#include "gepetto_solvers/environment/collision/EllipsoidSetCollisionGapFactor.h"
#include "gepetto_solvers/environment/collision/EllipsoidSetPlanarGapFactor.h"
#include "gepetto_solvers/environment/collision/HalfSpaceGapFactor.h"
#include "gepetto_solvers/environment/collision/PlaneCollisionGapFactor.h"
#include "gepetto_solvers/environment/collision/SdfCollisionGapFactor.h"
#include "gepetto_solvers/environment/collision/SphereSphereCollisionGapFactor.h"

#include "gepetto_solvers/environment/contact/EllipsoidWitnessContactFactor.h"
#include "gepetto_solvers/environment/contact/SdfWitnessContactFactor.h"
#include "gepetto_solvers/environment/contact/SphereSphereContactFactor.h"
#include "gepetto_solvers/environment/contact/SphereWitnessContactFactor.h"

#include "gepetto_solvers/environment/grasp/GraspAlignmentFactor.h"

#include "gepetto_solvers/environment/pregrasp/PreGraspAxisAlignmentFactor.h"
#include "gepetto_solvers/environment/pregrasp/PreGraspCentroidFactor.h"
#include "gepetto_solvers/environment/pregrasp/PreGraspHandCenteringFactor.h"
