"""Headless parameter sweep: find phase-0 (pre-grasp) settings for the pen that
let the phase-1 (table contact) warm start actually close, instead of stalling.

Ad-hoc diagnostic tool, not a demo script -- mirrors exactly what
viz_interactive.py's GUI does when the "warm start" latch is on and the user
checks phase0 then phase1: run phase0 to convergence, adopt the solved wrist
pose and flexor tensions onto the sliders (this is what
HandVizApp._adopt_solved_wrist/_adopt_solved_tensions do), carry the AL duals,
rebuild the stepper with phase1's preset, and run that to convergence.

Run (from the crest-sparse root, so the INSTALLED build is used):
    python -m python.tests.tendon_hand._sweep_pregrasp_pen
"""

import itertools
import math
import time

import numpy as np

from .solvers import (
    HandSolveParams, HandIKStepper, apply_phase_preset, solved_wrist_pose,
    FLEXOR_IDX)
from .scene import GRASP_FLEXOR_TENSION


PRIMITIVE = "pen"
CONTACT_FINGERS = [True, True, False, False, True]  # index, middle, thumb
MAX_STEPS = 60
FLEXOR_LO, FLEXOR_HI = 0.0, 3.0


def base_params():
    p = HandSolveParams()
    p.primitive = PRIMITIVE
    p.object_center = None
    p.object_rotation = None
    return p


def run_stepper(params, max_steps=MAX_STEPS):
    stepper = HandIKStepper(params)
    status = stepper.run(max_steps=max_steps)
    res = stepper._result(
        [stepper._frames[-1]], None, params.contact_fingers,
        [stepper._history[-1]], duals=stepper.al_duals(),
        dual_transfer=stepper.dual_transfer())
    return stepper, status, res


def adopt_wrist(fk_configs, res):
    """The pose the last solve reached, as a 4x4 -- mirrors
    HandVizApp._adopt_solved_wrist without touching any GUI sliders."""
    return solved_wrist_pose(fk_configs, res.frames[0])


def adopt_tensions(res):
    """Solved flexor tensions per finger, clamped to the slider range -- mirrors
    HandVizApp._adopt_solved_tensions."""
    out, clamped = [], False
    for name in res.finger_names:
        q = float(np.asarray(
            res.frames[0][name].marginals.tensions.mean, float)[FLEXOR_IDX])
        clamped = clamped or not (FLEXOR_LO <= q <= FLEXOR_HI)
        out.append(min(max(q, FLEXOR_LO), FLEXOR_HI))
    return out, clamped


def run_phase0_then_phase1(overrides0, overrides1=None, max_steps=MAX_STEPS):
    """One full phase0 -> warm-start -> phase1 trial. Returns a dict of
    everything worth reporting."""
    p0 = base_params()
    apply_phase_preset(p0, "phase0")
    p0.contact_fingers = list(CONTACT_FINGERS)
    for k, v in overrides0.items():
        setattr(p0, k, v)

    t0 = time.time()
    stepper0, status0, res0 = run_stepper(p0, max_steps)
    t_phase0 = time.time() - t0

    # Warm start into phase 1, exactly as the GUI's latch does at build time.
    p1 = base_params()
    apply_phase_preset(p1, "phase1")
    p1.contact_fingers = list(CONTACT_FINGERS)
    p1.initial_state = res0.state(0)
    p1.initial_duals = res0.duals
    p1.wrist_pose = adopt_wrist(stepper0.configs, res0)
    solved_tensions, clamped = adopt_tensions(res0)
    p1.flexor_tensions = solved_tensions
    if overrides1:
        for k, v in overrides1.items():
            setattr(p1, k, v)

    t1 = time.time()
    stepper1, status1, res1 = run_stepper(p1, max_steps)
    t_phase1 = time.time() - t1

    table_gap = res1.worst_table_gap(p1, 0)
    return dict(
        overrides0=overrides0, overrides1=overrides1 or {},
        phase0_state=status0.state, phase0_steps=status0.steps,
        phase0_violation=status0.violation, phase0_cost=status0.cost,
        phase0_time=t_phase0,
        phase1_state=status1.state, phase1_steps=status1.steps,
        phase1_violation=status1.violation, phase1_cost=status1.cost,
        phase1_time=t_phase1,
        table_gap=table_gap, clamped=clamped,
    )


def fmt(r):
    ov0 = ", ".join(f"{k}={v}" for k, v in r["overrides0"].items()) or "baseline"
    ov1 = ", ".join(f"{k}={v}" for k, v in r["overrides1"].items())
    tag = ov0 + (f" | p1: {ov1}" if ov1 else "")
    return (f"{tag:60s} | p0 {r['phase0_state']:9s} v={r['phase0_violation']:.2e} "
            f"({r['phase0_steps']:2d} st) | p1 {r['phase1_state']:9s} "
            f"v={r['phase1_violation']:.2e} gap={r['table_gap']*1000:+7.2f} mm "
            f"({r['phase1_steps']:2d} st)")


def main():
    print(f"Sweeping phase-0 pre-grasp params for {PRIMITIVE!r} "
          f"(contact = index/middle/thumb)\n")

    print("=== baseline ===")
    baseline = run_phase0_then_phase1({})
    print(fmt(baseline))

    sweeps = {
        "h_clear": [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06],
        "sigma_wrist_pos": [0.1, 0.3, 1.0, 3.0],
        "collision_radius": [0.0015, 0.002, 0.003, 0.004],
        "collision_sigma": [1e-6, 1e-5, 1e-4, 1e-3],
        "ik_settle_steps": [0, 1, 2, 3],
        "al_iters": [40, 80, 150],
    }

    wrist_pose_sweeps = {
        "tz": [0.03, 0.05, 0.075, 0.10, 0.125],
        "pitch": [-1.4, -1.3, -1.22, -1.1, -1.0, -0.9],
    }

    results = [("baseline", baseline)]

    print("\n=== 1-D sweeps (single field, everything else at preset default) ===")
    for field, values in sweeps.items():
        print(f"\n-- {field} --")
        for v in values:
            r = run_phase0_then_phase1({field: v})
            results.append((f"{field}={v}", r))
            print(fmt(r))

    print("\n=== wrist start pose sweeps ===")
    from .solvers import DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY, wrist_pose_from_xyzrpy
    x0, y0, z0 = DEFAULT_WRIST_XYZ
    r0, p0_, yw0 = DEFAULT_WRIST_RPY
    for field, values in wrist_pose_sweeps.items():
        print(f"\n-- start {field} --")
        for v in values:
            if field == "tz":
                pose = wrist_pose_from_xyzrpy((x0, y0, v), (r0, p0_, yw0))
            else:
                pose = wrist_pose_from_xyzrpy((x0, y0, z0), (r0, v, yw0))
            r = run_phase0_then_phase1({"wrist_pose": pose})
            results.append((f"start_{field}={v}", r))
            print(fmt(r))

    print("\n=== phase-1 wrist-prior looseness (does relaxing the settle help?) ===")
    for v in [0.01, 0.03, 0.1, 0.3, 1.0]:
        r = run_phase0_then_phase1(
            {}, overrides1={"sigma_wrist_pos": v, "sigma_wrist_rot": v})
        results.append((f"phase1_sigma_wrist={v}", r))
        print(fmt(r))

    print("\n=== summary, sorted by (phase1 converged first, then table gap) ===")
    def score(item):
        _, r = item
        converged = 0 if r["phase1_state"] == "converged" else 1
        return (converged, abs(r["table_gap"]))
    for name, r in sorted(results, key=score)[:15]:
        print(f"{name:40s} -> {fmt(r)}")


if __name__ == "__main__":
    main()
