"""Section 1.8 phased controller: run all phases headless and report.

This is the §1.8 counterpart to ``traj_5f_slide_grasp.py``. Instead of solving one
K+1-step trajectory offline, it drives a single ``TendonHandController`` through
the phases, N control ticks each:

  Phase 0  ``PreGrasp``        servo to a collision-free hover posture above the object
  Phase 1  ``SupportContact``  fingertips onto the table, in opposed half-spaces
  Phase 2  ``ObjectApproach``  slide along the table onto the ellipsoid proxy
  Phase 3  ``ObjectServo``     lift off as needed, servo on the exact geometry

The phases are different CONSTRAINT SETS over the same single-state graph, not
time windows: ``set_phase()`` swaps the constraints and keeps the converged robot
state, so each phase warm-starts from the last.

Per phase it reports the equality/goal violations the controller tracks
(``phase_violations()``) plus an INDEPENDENT geometric check -- fingertip surface
gaps, finger-object clearance and cross-finger clearance, computed here from the
solved poses rather than read back out of the solver -- so a constraint the graph
thinks it satisfied cannot hide a penetration.

Run from the crest-sparse/ root so the import resolves to the installed
extension rather than the in-tree .so:

    python -m python.tests.tendon_hand.ctrl_5f_phases
    python -m python.tests.tendon_hand.ctrl_5f_phases --step-anchor length
    python -m python.tests.tendon_hand.ctrl_5f_phases --primitive coin \
        --contact-fingers 1,0,0,0,1
"""

import argparse
import itertools
import sys
import time

import numpy as np

from .config import disc_node_indices, proximal_disc_flags, tip_node_index
from .scene import get_primitive_specs, primitive_surface_gap
from .solvers import (
    HandControllerSolver, HandSolveParams, capabilities, free_space_start_pose,
    resolve_table_origin)


# Penetration tolerance (m). Same threshold the collision demos use.
PASS_TOL = 1e-4

PHASE_NAMES = {0: "pre-grasp positioning", 1: "support contact",
               2: "object approach", 3: "object servo"}

# Phase-0 advance thresholds (m, rad): "close enough to the hover posture".
# The rotation tolerance is the tracking equilibrium, not an arbitrary number:
# the target and step priors balance at a residual set by their sigma ratio, and
# ~0.04 rad (2.4 deg) is where the default pair settles. Tightening it below that
# just burns ticks that cannot close.
PREGRASP_POS_TOL = 2e-3
PREGRASP_ROT_TOL = 5e-2

# Units for the violation families phase_violations() reports. Phase 0's are NOT
# metres, so a single hardcoded suffix would mislabel them.
VIOLATION_UNITS = {"pregrasp_rot": "rad", "pregrasp_tension": "N"}


def parse_mask(text, n=5):
    """``"1,0,0,0,1"`` -> ``[True, False, False, False, True]``."""
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(
            f"expected {n} comma-separated flags, got {len(parts)}")
    return [p not in ("0", "false", "False") for p in parts]


def sphere_positions(solver, result):
    """Per finger, the world position of every collision sphere, tagged with its
    proximal flag and whether it is the designated contact node."""
    frame = result.frames[-1]
    out = []
    for (name, cfg) in solver.configs:
        fm = frame[name].marginals
        nodes = disc_node_indices(cfg)
        prox = proximal_disc_flags(cfg, solver.params.num_proximal_discs)
        tip = tip_node_index(cfg)
        out.append([(n, np.array(fm.rod.states[n].pose.mean)[:3, 3], bool(p), n == tip)
                    for n, p in zip(nodes, prox)])
    return out


def geometric_report(solver, result, plane_origin, plane_normal, contact_mask):
    """Independent geometry check on the solved poses. Returns
    ``(worst_object, worst_finger_finger, worst_table)`` clearances in metres,
    each >= 0 when nothing penetrates."""
    p = solver.params
    spec = solver.spec
    center, R = solver.object_center, solver.object_rotation
    r_col = p.collision_radius
    n_hat = np.asarray(plane_normal, float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    spheres = sphere_positions(solver, result)

    # Fingertip surface gaps (the thing phases 2-3 drive to zero).
    print("  tip surface gaps (target ~0 in phases 2-3):")
    for (name, _), tip_r, entries in zip(solver.configs, solver.tip_radii, spheres):
        pos = next(pos for _n, pos, _pr, is_tip in entries if is_tip)
        gap = primitive_surface_gap(R.T @ (pos - center), spec) - tip_r
        tag = "" if contact_mask[solver.finger_names.index(name)] else "  (no contact)"
        print(f"    [{name:>6}] {gap:+.5f} m{tag}")

    # Finger-object: every non-contact sphere must stay outside the object.
    worst_obj = np.inf
    for entries in spheres:
        for _n, pos, _pr, is_tip in entries:
            if is_tip:
                continue
            worst_obj = min(
                worst_obj,
                primitive_surface_gap(R.T @ (pos - center), spec) - r_col)

    # Finger-finger: skip node-0 (root) spheres and proximal-proximal pairs,
    # matching the exclusions the C++ factors apply.
    worst_ff = np.inf
    for ia, ib in itertools.combinations(range(len(spheres)), 2):
        for na, pa, proxa, _ta in spheres[ia]:
            if na == 0:
                continue
            for nb, pb, proxb, _tb in spheres[ib]:
                if nb == 0 or (proxa and proxb):
                    continue
                worst_ff = min(worst_ff, np.linalg.norm(pa - pb) - 2.0 * r_col)

    # Finger-table: every non-root sphere above the plane. The designated contact
    # sphere is exempt in phases 1-2, where an equality pins it TO the plane.
    # Phase 0 is deliberately NOT exempt: it has no support equality, so its tips
    # are held to the inequality like every other sphere (Eq 1.97's "for all j").
    exempt_contact = solver.params.phase in (1, 2)
    worst_table = np.inf
    for entries, touching in zip(spheres, contact_mask):
        for n, pos, _pr, is_tip in entries:
            if n == 0:
                continue
            if is_tip and touching and exempt_contact:
                continue
            worst_table = min(
                worst_table,
                float(np.dot(pos - np.asarray(plane_origin, float), n_hat)) - r_col)

    print(f"  worst finger-object clearance: {worst_obj:+.5f} m")
    print(f"  worst finger-finger gap:       {worst_ff:+.5f} m")
    print(f"  worst finger-table clearance:  {worst_table:+.5f} m")
    return worst_obj, worst_ff, worst_table


def report_pregrasp_target(solver):
    """Print how the phase-0 target was derived.

    Worth the lines: the hand-frame axes are MEASURED from forward kinematics
    (§1.8's "wrist approach axis is its local +z" does not hold for this mount),
    and a mis-derived orientation is otherwise silent -- the solve converges
    happily onto a wrong pose.
    """
    T, info = solver.pregrasp_target
    print(f"  target: {info['source']}   h_clear={info['h_clear']:.4f} m"
          f"   n_hat={np.round(info['n_hat'], 3)}"
          f"   m_hat={np.round(info['m_hat'], 3)}")
    if info["source"] == "derived":
        print(f"    base-frame axes  palm a={np.round(info['a_hat'], 3)}"
              f"  fingers g={np.round(info['g_hat'], 3)}"
              f"  thumb-ward s={np.round(info['s_hat'], 3)}")
        print(f"    contact centroid p_bar={np.round(info['p_bar'], 4)} m"
              f"   hover point={np.round(info['hover_point'], 4)}")
    print(f"    t_pre={np.round(T[:3, 3], 4)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primitive", default="mid_sphere_ellipsoid",
                        choices=sorted(get_primitive_specs()))
    parser.add_argument("--contact-fingers", type=parse_mask, default=None,
                        metavar="1,1,1,1,1",
                        help="Per-finger contact mask (index,middle,ring,pinky,thumb). "
                             "Unchecked fingers keep collision avoidance but are "
                             "never driven onto the table or the object -- use this "
                             "for a pinch on a small object.")
    parser.add_argument("--step-anchor", default="tension",
                        choices=["tension", "length", "both"],
                        help="What anchors a tick to the measured state. 'length' "
                             "is the hardware-faithful mode (Eq 1.13 analogue).")
    parser.add_argument("--ticks", type=int, default=6,
                        help="Control ticks per phase.")
    parser.add_argument("--pregrasp-ticks", type=int, default=20,
                        help="Control ticks for phase 0. More than the contact "
                             "phases get: phase 0 is a first-order servo whose "
                             "per-tick step is set by the sigma ratio (~10 %% of "
                             "the remaining error at the defaults).")
    parser.add_argument("--advance-on-converge", action="store_true",
                        help="End phase 0 as soon as it is within "
                             f"{PREGRASP_POS_TOL} m / {PREGRASP_ROT_TOL} rad of "
                             "the target, instead of always running the full "
                             "tick budget.")
    parser.add_argument("--al-iters", type=int, default=4,
                        help="AL outer iterations per tick (the outer loop is "
                             "amortized across ticks, so this is small).")
    parser.add_argument("--phases", default="0,1,2,3",
                        help="Which phases to run, in order.")
    parser.add_argument("--start-in-collision", action="store_true",
                        help="Start at an identity base pose instead of lifting "
                             "clear of the table first. Reproduces the old "
                             "initial condition (fingers ~37 mm through the "
                             "table); phase 0 cannot recover from it.")
    args = parser.parse_args()

    if not capabilities()["controller"]:
        print("FAIL: the installed _crest_sparse has no TendonHandController.\n"
              "      Rebuild the extension (pip install . from crest-sparse/) and "
              "re-run from the crest-sparse/ root.")
        return 1

    p = HandSolveParams()
    p.primitive = args.primitive
    p.collision = True
    p.table = True
    p.step_anchor = args.step_anchor
    p.ctrl_al_iters = args.al_iters
    if args.contact_fingers is not None:
        p.contact_fingers = args.contact_fingers

    # Start in free space. An identity base pose buries the fingers ~37 mm
    # through the table, and phase 0 cannot dig out of that: the collision
    # inequalities dominate the merit function and the inner LM rejects every
    # step. Positioning the hand is phase 0's job -- being IN collision at t=0 is
    # not a problem it is meant to solve.
    if not args.start_in_collision:
        p.wrist_pose, start_info = free_space_start_pose(p)
        print(f"free-space start: lift {start_info['lift']:.4f} m along the "
              f"support normal   table{start_info['table_clearance']:+.4f} m   "
              f"object{start_info['object_clearance']:+.4f} m")

    print(f"Section 1.8 phased controller | primitive={p.primitive} | "
          f"anchor={p.step_anchor} | {args.ticks} ticks/phase "
          f"({args.pregrasp_ticks} for phase 0)")
    print(f"contact fingers: {p.contact_fingers}")

    solver = HandControllerSolver(p)
    # The same origin the controller's env was built with, so this check measures
    # against the plane the solver actually enforced.
    plane_origin = resolve_table_origin(p, solver.spec, solver.object_center)

    ok = True
    for phase in [int(s) for s in args.phases.split(",")]:
        print(f"\n=== Phase {phase}: {PHASE_NAMES[phase]} "
              f"{'=' * 30}")
        solver.set_phase(phase)

        if phase == 0:
            if not capabilities()["pregrasp"]:
                print("  SKIP: the installed extension predates the PreGrasp phase.")
                continue
            report_pregrasp_target(solver)

        result = None
        budget = args.pregrasp_ticks if phase == 0 else args.ticks
        for tick in range(budget):
            t0 = time.time()
            result = solver.step()
            dt_ms = (time.time() - t0) * 1000.0
            viol = dict(solver.phase_violations())
            viol_txt = "  ".join(
                f"{n}={v:.2e}{VIOLATION_UNITS.get(n, 'm')}"
                for n, v in viol.items()) or "(none)"
            print(f"  tick {tick:>2}  {dt_ms:7.1f} ms  "
                  f"iters={result.meta.iterations:<3} "
                  f"err={result.meta.error:11.4g}  {viol_txt}")
            if (phase == 0 and args.advance_on_converge
                    and viol.get("pregrasp_pos", np.inf) <= PREGRASP_POS_TOL
                    and viol.get("pregrasp_rot", np.inf) <= PREGRASP_ROT_TOL):
                print(f"  converged on the pre-grasp target after {tick + 1} ticks")
                break

        w_obj, w_ff, w_tab = geometric_report(
            solver, result, plane_origin, p.plane_normal, p.contact_fingers)
        phase_ok = (w_obj >= -PASS_TOL and w_ff >= -PASS_TOL and w_tab >= -PASS_TOL)
        ok = ok and phase_ok
        print(f"  phase {phase}: {'ok' if phase_ok else 'PENETRATION'}")

    print("\nFactor error summary (type, count, total_error):")
    for name, count, err in solver._controller.get_factor_error_summary()[:12]:
        print(f"  {err:12.4g}  x{count:<5} {name}")

    print("\nRESULT:", "PASS (no penetration)" if ok else "FAIL (penetration present)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
