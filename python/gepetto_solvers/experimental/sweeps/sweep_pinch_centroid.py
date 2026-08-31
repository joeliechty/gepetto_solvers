"""Verification harness for the pre-grasp PINCH-CENTROID centering constraint.

Checks, in order of what would invalidate what:

1. STRUCTURE -- the constraint actually reaches the factor graph. Compares
   ``get_factor_error_summary()`` with the toggle off vs on at
   ``al_iters = 1``, so the graph is built but barely solved. A pose
   differential would not prove this: these scenes routinely stall, and a hand
   that fails to move looks identical whether the constraint was added and
   ignored or never added at all.

2. CONVERGENCE -- solving with it on drives
   :func:`solvers.pregrasp_centroid_witness` toward zero, i.e. the hand really
   does move its measured pinch point onto the target.

3. TABLE CONSISTENCY -- at the converged wrist, commanding the combination's
   STORED tensions through FK closes the fingertips around the target. This is
   the check that the centroid and the tensions in
   :data:`config.HAND_PINCH_POSES` describe the same pose; each is plausible
   alone, and only together do they mean "closing this hand grasps that
   object".

Run (from the crest-sparse ROOT, so the installed build is used rather than
the stale in-tree .so):
    python scripts/experimental/sweep_pinch_centroid.py
"""

import numpy as np

from gepetto_solvers.core.hand.config import pinch_pose
from gepetto_solvers.core.solvers import (
    HandFKSolver,
    HandIKStepper,
    HandSolveParams,
    capabilities,
    pregrasp_centroid_witness,
    solved_wrist_pose,
    tip_gap_matrix,
)

# The default 3-finger pinch the GUI opens on (index, middle, thumb).
CONTACT_FINGERS = [True, True, False, False, True]
FINGER_NAMES = ["index", "middle", "ring", "pinky", "thumb"]


def _params(**kw):
    """A pre-grasp-style scene: the object is collision geometry only, so the
    only thing positioning the hand is the constraint under test."""
    p = HandSolveParams()
    p.contact_fingers = list(CONTACT_FINGERS)
    p.object_contact = False
    p.table_contact = False
    p.collision = True
    p.table = True
    p.plane_avoidance = True
    # The wrist has to be free to move -- this constraint acts on it alone, and
    # a prior stiffer than the AL penalty ceiling would simply out-muscle it.
    p.sigma_wrist_pos = 1.0
    p.sigma_wrist_rot = 1.0
    for k, v in kw.items():
        setattr(p, k, v)
    return p


#: The graph wrapper an AL EQUALITY is registered as (TendonHandModel::add_eq).
#: This scene has no other equality constraint -- object and table contact are
#: both off -- so its count is exactly the number of pinch-centroid factors.
_EQ_FACTOR = "gtsam::ZeroCostConstraint"
#: Pose3 priors: the wrist prior, plus the object anchor. Watched but NOT
#: asserted to grow -- the pinch-centroid block only anchors the object when no
#: earlier block has, and in this scene collision already did. Measured: the
#: count is 2 either way, i.e. the `object_anchored` guard is doing its job and
#: not double-anchoring.
_POSE_PRIOR = "gtsam::PriorFactor<gtsam::Pose3>"


def check_structure():
    """The constraint is in the graph, and only when asked for.

    Counts named factor GROUPS rather than diffing a total, so the result says
    which factor appeared. A bare count delta would be satisfied by any
    incidental change to the graph.
    """
    def summarize(on):
        stepper = HandIKStepper(_params(pregrasp_centroid=on, al_iters=1))
        stepper.step()
        return {name: n for name, n, _ in
                stepper._solver.get_factor_error_summary()}

    off, on = summarize(False), summarize(True)
    eq_off, eq_on = off.get(_EQ_FACTOR, 0), on.get(_EQ_FACTOR, 0)
    pp_off, pp_on = off.get(_POSE_PRIOR, 0), on.get(_POSE_PRIOR, 0)

    print(f"  {_EQ_FACTOR}: {eq_off} -> {eq_on}  (expect 0 -> 1)")
    print(f"  {_POSE_PRIOR}: {pp_off} -> {pp_on}  "
          f"(informational; no double-anchor expected here)")

    ok = (eq_off == 0 and eq_on == 1 and pp_on <= pp_off + 1)
    print(f"  [{'PASS' if ok else 'FAIL'}] exactly one equality constraint added, "
          f"object anchored at most once")
    return ok


def check_convergence(max_steps=60):
    """Solving with the constraint on drives the pinch point onto the target."""
    p = _params(pregrasp_centroid=True)
    pose = pinch_pose([n for n, c in zip(FINGER_NAMES, CONTACT_FINGERS) if c])
    print(f"  pinch pose: centroid={np.round(pose.centroid, 5)} "
          f"tensions={pose.tensions} gap={pose.gap * 1000:+.1f} mm")

    stepper = HandIKStepper(p)
    first = last = None
    result = None
    # noqa: B007 -- `i` is not used inside the body, but it leaks out of the
    # loop and the summary below reports `i + 1` steps. B007 only inspects the
    # body, so its rename suggestion is wrong here.
    for i in range(max_steps):  # noqa: B007
        result = stepper.step()
        w = pregrasp_centroid_witness(p, result, 0)
        if w is None:
            print("  [FAIL] no witness -- constraint inactive for this digit set")
            return False, None, None
        if first is None:
            first = w[2]
        last = w[2]
        if stepper.status().state != "running":
            break

    print(f"  centroid gap: {first * 1000:.1f} mm -> {last * 1000:.1f} mm "
          f"over {i + 1} steps ({stepper.status().state})")
    ok = last < 5e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] converged to under 5 mm")
    return ok, result, p


def check_table_consistency(result, p):
    """At the converged wrist, the STORED tensions close the digits on target.

    Ties the two halves of each table entry together: the constraint only used
    ``centroid``, so this is the first thing that would notice if ``centroid``
    and ``tensions`` had drifted apart (e.g. regenerated at different settings).
    """
    names = [n for n, c in zip(FINGER_NAMES, CONTACT_FINGERS) if c]
    pose = pinch_pose(names)

    fk = _params()
    fk.wrist_pose = solved_wrist_pose(HandFKSolver(fk).configs, result.frames[0])
    fk.flexor_tensions = [pose.tensions.get(n, 0.0) for n in FINGER_NAMES]
    fk_result = HandFKSolver(fk).solve()

    frame = fk_result.frames[0]
    tips = np.stack([
        np.asarray(frame[n].marginals.rod.states[-1].pose.mean, float)[:3, 3]
        for n in names])
    radii = [r for r, c in zip(fk_result.tip_radii, CONTACT_FINGERS) if c]

    centroid = tips.mean(axis=0)
    _, target, _ = pregrasp_centroid_witness(p, result, 0)
    to_target = float(np.linalg.norm(centroid - target))
    closest = float(tip_gap_matrix(tips, radii).min())

    print(f"  achieved fingertip centroid: {np.round(centroid, 5)}")
    print(f"  target:                      {np.round(target, 5)}")
    print(f"  centroid-to-target: {to_target * 1000:.1f} mm")
    print(f"  closest tip-pair surface gap: {closest * 1000:+.1f} mm "
          f"(table says {pose.gap * 1000:+.1f} mm)")

    # Two independent things: the fingers close where the table says they do
    # (gap matches), and that place is the target (centroid matches).
    gap_ok = abs(closest - pose.gap) < 2e-3
    pos_ok = to_target < 1e-2
    print(f"  [{'PASS' if gap_ok else 'FAIL'}] stored gap reproduced")
    print(f"  [{'PASS' if pos_ok else 'FAIL'}] fingers close on the target "
          f"(under 10 mm)")
    return gap_ok and pos_ok


def main():
    if not capabilities()["pregrasp_centroid"]:
        print("SKIP: this binding has no EnvironmentConfig.pregrasp_centroid_point "
              "-- rebuild with `pip install .` from the crest-sparse root.")
        return 1

    print("\n1. STRUCTURE -- is the constraint in the graph?")
    ok_struct = check_structure()

    print("\n2. CONVERGENCE -- does the hand move its pinch point onto the target?")
    ok_conv, result, p = check_convergence()

    ok_table = False
    if ok_conv:
        print("\n3. TABLE CONSISTENCY -- do the stored tensions close there?")
        ok_table = check_table_consistency(result, p)

    every = ok_struct and ok_conv and ok_table
    print(f"\n{'PASS' if every else 'FAIL'}: structure={ok_struct} "
          f"convergence={ok_conv} table={ok_table}")
    return 0 if every else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
