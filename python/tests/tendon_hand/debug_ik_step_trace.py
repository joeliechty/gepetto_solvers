"""Headless replay of the visualizer's *Step* button, with the full solve log.

``viz_interactive.py`` can drive an IK solve one Augmented Lagrangian outer
iteration at a time (the *Solve steps* folder: Step / Auto solve), but the only
thing it can show per iteration is three numbers in a markdown box. This runs the
identical :class:`~.solvers.HandIKStepper` loop with no viser attached and prints
everything the solve knows:

* the resolved configuration -- scene, base pose, tendon priors, AL settings, and
  what the installed binding actually supports;
* the factor graph's error broken down by factor family, before and after;
* per outer iteration: AL state (mu, cost, violation, converged/stalled), the
  inner-LM tail, per-finger fingertip gaps, the worst finger-object and
  finger-finger collision clearance, and wall time;
* per outer iteration, the solved KINEMATIC STATE -- the achieved hand base pose
  (against the commanded one), and per finger the tendon tensions Q, the tendon
  lengths L and the fingertip pose; ``--nodes`` expands that to every rod node;
* an aggregated AL outer-loop table over the whole run, plus the failure-mode
  classification from :mod:`debug_al_trace`.

Defaults reproduce the visualizer's own start state -- the default hand base pose
(``solvers.DEFAULT_WRIST_XYZ`` / ``DEFAULT_WRIST_RPY``, i.e. the palm-down hover
the GUI opens on), all five fingertips driven onto the default analytic object
(``HandSolveParams.primitive``, the half-buried mid sphere), Section 1.5
collision avoidance ON and the support plane OFF. Nothing here mutates solver
behaviour: it is a read-only harness like :mod:`debug_al_trace`, which reports one
*whole* solve where this reports every step of one.

Run from the ``crest-sparse/`` repo root in the ``crest_py11`` env, so the import
resolves to the installed binding rather than the in-tree ``.so``::

    python -m python.tests.tendon_hand.debug_ik_step_trace

    # write the log to results/ as well as the terminal
    python -m python.tests.tendon_hand.debug_ik_step_trace --log

    # add the C++ outer-loop stderr trace and the per-step inner-LM tail
    python -m python.tests.tendon_hand.debug_ik_step_trace --verbose --inner

    # a pinch instead of a whole-hand grasp, no collision spheres
    python -m python.tests.tendon_hand.debug_ik_step_trace \
        --contact-fingers 1,0,0,0,1 --no-collision
"""

import argparse
import itertools
import os
import time
from types import SimpleNamespace

import numpy as np

from .config import disc_node_indices, proximal_disc_flags, tip_node_index
from .debug_al_trace import (_arr as trace_array, classify, print_inner_tail,
                             print_outer_trace)
from .scene import (GRASP_FLEXOR_TENSION, TABLE_NORMAL, get_primitive_specs,
                    primitive_surface_gap)
from .solvers import (DEFAULT_WRIST_RPY, DEFAULT_WRIST_XYZ, NUM_FINGERS,
                      HandIKStepper, HandSolveParams, capabilities,
                      resolve_table_origin, wrist_pose_from_xyzrpy)
from .utils import PlannerLogger, log_planner_parameters


# Taken from HandSolveParams rather than named here: the mid analytic sphere is
# both the params default and what the visualizer's object dropdown opens on, so
# reading it keeps this harness a faithful repro if that default ever moves.
DEFAULT_PRIMITIVE = HandSolveParams().primitive

# Penetration tolerance (m), matching the collision demos and ctrl_5f_phases.
PASS_TOL = 1e-4


def parse_mask(text, n=NUM_FINGERS):
    """``"1,0,0,0,1"`` -> ``[True, False, False, False, True]``."""
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(
            f"expected {n} comma-separated flags, got {len(parts)}")
    return [p not in ("0", "false", "False", "off", "no") for p in parts]


# ---------------------------------------------------------------------------
# Configuration reporting.
# ---------------------------------------------------------------------------

def print_capabilities():
    """What the INSTALLED binding supports. First thing to check when a run
    behaves unlike the same settings in the GUI: an in-tree ``.so`` that lags the
    C++ source silently drops fields (see ``solvers._set_if``)."""
    caps = capabilities()
    print("binding capabilities:")
    for name in sorted(caps):
        print(f"  {name:>16}: {'yes' if caps[name] else 'NO'}")


def print_scene(stepper, params):
    """The geometry the constraints are written against, resolved."""
    T = np.asarray(params.wrist_pose, float)
    spec = stepper.spec
    print("scene:")
    print(f"  primitive        : {params.primitive} ({spec['type']})")
    print(f"  object center    : {np.array2string(stepper.object_center, precision=4)}")
    print(f"  contact fingers  : "
          f"{', '.join(n for n, on in zip(stepper.finger_names, params.contact_fingers) if on) or 'none'}")
    print(f"  tip radii (m)    : "
          f"{np.array2string(np.asarray(stepper.tip_radii), precision=4)}")
    print("hand base pose (wrist prior mean):")
    print(f"  translation (m)  : {np.array2string(T[:3, 3], precision=4)}")
    for row in np.array2string(T[:3, :3], precision=4).splitlines():
        print(f"  rotation         : {row}" if row.startswith("[[")
              else f"                     {row}")
    print(f"  sigma pos / rot  : {params.sigma_wrist_pos:.3g} m / "
          f"{params.sigma_wrist_rot:.3g} rad")
    print("environment:")
    print(f"  collision        : {'on' if params.collision else 'off'}"
          + (f"  radius={params.collision_radius:.4f} m  "
             f"sigma={params.collision_sigma:.3g}  "
             f"cull={params.cull_margin}" if params.collision else ""))
    if params.table:
        origin = resolve_table_origin(params, stepper.spec, stepper.object_center)
        print(f"  table            : on   origin="
              f"{np.array2string(np.asarray(origin), precision=4)}  "
              f"normal={np.array2string(np.asarray(params.plane_normal), precision=3)}")
    else:
        print("  table            : off")
    print("augmented lagrangian:")
    print(f"  mu0={params.al_mu:g}  rate={params.al_rate:g}  "
          f"(1 outer iteration per step, duals warm-started across steps)")


def print_factor_errors(stepper, header):
    """Per-factor-family error at the current values.

    The single most useful thing when a solve refuses to move: it says which
    part of the graph the inner LM is actually spending its budget on. A
    constraint family carrying 1e-2 against rod-physics families carrying 1e6 is
    not going to be what the optimizer solves."""
    rows = stepper.factor_errors()
    if not rows:
        # The graph is built INSIDE solve(), not in the constructor, so this is
        # empty until the first step -- which is why it is reported after step 1
        # rather than at the initial guess.
        print(f"{header}: (empty -- no graph yet, or a binding without "
              "get_factor_error_summary)")
        return
    total = sum(e for _n, _c, e in rows) or 1.0
    print(f"{header}:")
    print(f"  {'factor family':>34}  {'count':>7}  {'error':>14}  {'%':>7}")
    print("  " + "-" * 68)
    for name, count, err in sorted(rows, key=lambda r: -r[2]):
        print(f"  {name:>34}  {count:>7}  {err:>14.6g}  {100.0 * err / total:>6.2f}%")
    print(f"  {'TOTAL':>34}  {sum(c for _n, c, _e in rows):>7}  {total:>14.6g}")


# ---------------------------------------------------------------------------
# Per-step geometry (independent of what the solver thinks it achieved).
# ---------------------------------------------------------------------------

def sphere_positions(stepper, result):
    """Per finger, ``[(node, world_position, is_proximal, is_tip)]`` for every
    collision sphere -- the same ``disc_pose_idx`` walk the renderer draws."""
    frame = result.frames[0]
    out = []
    for name, cfg in stepper.configs:
        fm = frame[name].marginals
        prox = proximal_disc_flags(cfg, stepper.params.num_proximal_discs)
        tip = tip_node_index(cfg)
        out.append([(n, np.asarray(fm.rod.states[n].pose.mean, float)[:3, 3],
                     bool(p), n == tip)
                    for n, p in zip(disc_node_indices(cfg), prox)])
    return out


def clearances(stepper, result):
    """``(worst_object, worst_finger_finger)`` clearance in metres, computed here
    from the solved poses rather than read back out of the solver, so a
    constraint the graph believes it satisfied cannot hide a penetration.

    Fingertips are excluded from the object check: they are being driven ONTO the
    surface, so their gap is the contact residual (reported per finger by
    :func:`print_gaps`), not a collision. Finger-finger applies the same
    exclusions the C++ factors do -- no node-0 pairs, no proximal-proximal pairs.
    """
    spec, center, R = stepper.spec, stepper.object_center, stepper.object_rotation
    r_col = stepper.params.collision_radius
    spheres = sphere_positions(stepper, result)

    worst_obj = np.inf
    for entries in spheres:
        for _n, pos, _prox, is_tip in entries:
            if is_tip:
                continue
            worst_obj = min(worst_obj,
                            primitive_surface_gap(R.T @ (pos - center), spec) - r_col)

    worst_ff = np.inf
    for ia, ib in itertools.combinations(range(len(spheres)), 2):
        for na, pa, proxa, _ta in spheres[ia]:
            if na == 0:
                continue
            for nb, pb, proxb, _tb in spheres[ib]:
                if nb == 0 or (proxa and proxb):
                    continue
                worst_ff = min(worst_ff, float(np.linalg.norm(pa - pb)) - 2.0 * r_col)
    return worst_obj, worst_ff


def rpy_from_R(R):
    """ZYX (yaw-pitch-roll) angles from a rotation matrix -- the inverse of
    ``solvers.euler_to_R``, so a pose printed here can be typed straight back
    into ``--wrist`` or the visualizer's sliders."""
    R = np.asarray(R, float)
    pitch = float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])))
    if abs(abs(pitch) - np.pi / 2) < 1e-8:      # gimbal lock: yaw absorbs roll
        return float(np.arctan2(-R[1, 2], R[1, 1])), pitch, 0.0
    return (float(np.arctan2(R[2, 1], R[2, 2])), pitch,
            float(np.arctan2(R[1, 0], R[0, 0])))


def base_pose(stepper, result):
    """The achieved hand base pose ``T_base``, recovered from the first finger's
    node-0 pose as ``T_0 o hand_base_offset^-1``.

    Exact, not an estimate -- the reparameterization defines ``T_0`` that way.
    Worth printing separately from the commanded ``params.wrist_pose``: the
    difference between the two IS how much the wrist prior gave up, which is the
    thing to look at when a solve cannot reach the object."""
    frame = result.frames[0]
    node0 = np.asarray(
        frame[stepper.finger_names[0]].marginals.rod.states[0].pose.mean, float)
    return node0 @ np.linalg.inv(
        np.asarray(stepper.configs[0][1].hand_base_offset, float))


def _fmt(v, prec=4):
    return np.array2string(np.asarray(v, float), precision=prec,
                           suppress_small=True, max_line_width=200)


def print_kinematic_state(stepper, result, indent="    ", nodes=False):
    """The solved kinematic state: base pose, and per finger the tendon tensions
    Q, the tendon lengths L and the fingertip pose.

    These are the state vector the solve actually optimizes (tensions drive the
    rod, lengths are what a motor would command), so a step whose AL numbers
    barely move but whose Q/L are still shifting is a different situation from
    one where the whole state is frozen -- and only this readout tells them
    apart. ``nodes`` additionally dumps every rod node, which is the full
    kinematic state rather than its endpoints.
    """
    T = base_pose(stepper, result)
    frame = result.frames[0]
    print(f"{indent}kinematic state:")
    print(f"{indent}  base T   : pos {_fmt(T[:3, 3])}  rpy {_fmt(rpy_from_R(T[:3, :3]))}")
    cmd = np.asarray(stepper.params.wrist_pose, float)
    print(f"{indent}             (commanded pos {_fmt(cmd[:3, 3])}, "
          f"moved {np.linalg.norm(T[:3, 3] - cmd[:3, 3]) * 1e3:.3f} mm)")
    for name in stepper.finger_names:
        fm = frame[name].marginals
        tip = np.asarray(fm.rod.states[-1].pose.mean, float)
        print(f"{indent}  [{name:>6}] Q(N) {_fmt(fm.tensions.mean, 4)}")
        print(f"{indent}           L(m) {_fmt(fm.tendon_lengths, 5)}")
        print(f"{indent}           tip  pos {_fmt(tip[:3, 3], 5)}  "
              f"rpy {_fmt(rpy_from_R(tip[:3, :3]))}")
        if nodes:
            print(f"{indent}           rod nodes ({len(fm.rod.states)}), "
                  f"position | stress:")
            for i, st in enumerate(fm.rod.states):
                pos = np.asarray(st.pose.mean, float)[:3, 3]
                print(f"{indent}             {i:>3}  {_fmt(pos, 5)}  "
                      f"|s|={np.linalg.norm(np.asarray(st.stress.mean, float)):.4g}")


def print_gaps(result):
    """Per-finger fingertip surface gap; the fingers asked to touch are what the
    solve is scored on, the rest are printed for context only."""
    gaps = result.surface_gaps(0)
    contact = set(result.contact_names())
    cells = [f"{name}={gaps[name]:+.5f}" + ("" if name in contact else "*")
             for name in result.finger_names]
    print(f"    tip gaps (m): {'  '.join(cells)}"
          + ("   (* not a contact finger)"
             if len(contact) < len(result.finger_names) else ""))


# ---------------------------------------------------------------------------
# The stepped run.
# ---------------------------------------------------------------------------

def step_through(stepper, args):
    """Drive the AL outer loop one iteration per call, logging each one.

    Returns ``(rows, status, result)`` where ``rows`` is the per-step record the
    summary table and the aggregated AL trace are built from."""
    rows = []
    result = None
    print("\n" + "=" * 72)
    print(f"STEPPING (up to {args.max_steps} AL outer iterations, one per step)")
    print("=" * 72)

    for _ in range(args.max_steps):
        t0 = time.perf_counter()
        result = stepper.step()
        wall_ms = (time.perf_counter() - t0) * 1e3
        status = stepper.status()
        meta = result.meta
        gap = result.worst_gap(0)
        obj, ff = clearances(stepper, result)

        print(f"\n-- step {status.steps:>3} --  {status.state}")
        print(f"    AL         : mu={status.mu:.6g}  cost={status.cost:.6g}  "
              f"violation={status.violation:.6g}")
        print(f"    inner LM   : iterations={meta.iterations}  "
              f"error={meta.error:.6g}  "
              f"solver={getattr(meta, 'total_time_ms', float('nan')):.1f} ms  "
              f"wall={wall_ms:.1f} ms")
        print(f"    worst gap  : {gap * 1e3:+.4f} mm  (over the contact fingers)")
        print_gaps(result)
        print(f"    clearance  : object {obj * 1e3:+.3f} mm   "
              f"finger-finger {ff * 1e3:+.3f} mm"
              + ("" if stepper.params.collision
                 else "   (collision OFF -- reported, not enforced)"))
        if args.kinematics:
            print_kinematic_state(stepper, result, nodes=args.nodes)
        if args.inner:
            print_inner_tail(meta, tail=args.inner_tail)
        if status.steps == 1:
            # The earliest this can be read: the graph is assembled inside
            # solve(), so there is nothing to break down before the first step.
            print_factor_errors(stepper, "    factor errors after step 1")

        rows.append({"step": status.steps, "state": status.state, "mu": status.mu,
                     "cost": status.cost, "violation": status.violation,
                     "gap": gap, "object": obj, "finger_finger": ff,
                     "inner_iters": meta.iterations, "error": meta.error,
                     "ms": wall_ms})
        if status.done:
            break

    return rows, stepper.status(), result


def print_summary_table(rows):
    print("\n" + "=" * 72)
    print("PER-STEP SUMMARY")
    print("=" * 72)
    print(f"  {'step':>4}  {'state':>9}  {'mu':>11}  {'cost':>13}  "
          f"{'violation':>12}  {'gap mm':>9}  {'obj mm':>9}  {'inner':>6}  {'ms':>7}")
    print("  " + "-" * 100)
    for r in rows:
        print(f"  {r['step']:>4}  {r['state']:>9}  {r['mu']:>11.4g}  "
              f"{r['cost']:>13.6g}  {r['violation']:>12.4g}  "
              f"{r['gap'] * 1e3:>9.4f}  {r['object'] * 1e3:>9.3f}  "
              f"{r['inner_iters']:>6}  {r['ms']:>7.1f}")


def print_verdict(rows, status, args):
    """The aggregated AL outer-loop table and the failure-mode classification.

    A stepped solve's ``meta`` describes only the LAST step, so the trace arrays
    are re-assembled here from the per-step record and handed to the same
    :func:`debug_al_trace.classify` a one-shot solve uses -- the aggregated
    series IS the outer loop the one-shot solve would have run internally."""
    agg = SimpleNamespace(
        al_iteration_mus=[r["mu"] for r in rows],
        al_iteration_costs=[r["cost"] for r in rows],
        al_iteration_violations=[r["violation"] for r in rows])

    print("\n" + "=" * 72)
    print("AGGREGATED AL OUTER LOOP (one row per step)")
    print("=" * 72)
    print_outer_trace(agg)

    last = rows[-1]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    stop = {"converged": "converged (violation and cost inside tolerance)",
            "stalled": "STALLED -- a step changed nothing; mu still grows, so "
                       "stepping again can unwedge it",
            "running": f"hit the --max-steps cap ({args.max_steps}) still running"}
    print(f"  stopped because : {stop.get(status.state, status.state)}")
    print(f"  steps taken     : {status.steps}")
    print(f"  worst tip gap   : {last['gap'] * 1e3:+.4f} mm")
    print(f"  object clearance: {last['object'] * 1e3:+.3f} mm"
          + ("" if last["object"] >= -PASS_TOL
             else f"   PENETRATION (> {PASS_TOL * 1e3:.1f} mm tolerance)"))
    print(f"  finger-finger   : {last['finger_finger'] * 1e3:+.3f} mm")
    print(f"  total solve time: {sum(r['ms'] for r in rows) / 1e3:.2f} s")
    print(f"  >>> {classify(agg, last['gap'], args.max_steps)}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def build_params(args):
    """The visualizer's own defaults, with collision on and the table off."""
    p = HandSolveParams()
    p.primitive = args.primitive
    p.wrist_pose = wrist_pose_from_xyzrpy(args.wrist[:3], args.wrist[3:])
    p.sigma_wrist_pos = args.sigma_wrist_pos
    p.sigma_wrist_rot = args.sigma_wrist_rot
    p.passive_tension = args.passive
    p.flexor_tensions = [args.flexor] * NUM_FINGERS
    if args.contact_fingers is not None:
        p.contact_fingers = args.contact_fingers
    p.al_mu, p.al_rate = args.al_mu, args.al_rate
    p.al_iters = args.max_steps          # reported only; the stepper caps at 1/step
    p.collision = args.collision
    p.collision_radius = args.collision_radius
    p.collision_sigma = args.collision_sigma
    p.cull_margin = args.cull_margin
    # The point of this harness: no support plane. Phase-1-style table contact
    # belongs to the controller (ctrl_5f_phases.py), not to a stepped IK solve.
    p.table = False
    p.plane_normal = np.array(TABLE_NORMAL, float)
    return p


def build_parser():
    specs = sorted(get_primitive_specs().keys())
    ap = argparse.ArgumentParser(
        description="Step a tendon-hand IK solve one AL outer iteration at a "
                    "time and log everything about it.")
    ap.add_argument("--primitive", choices=specs, default=DEFAULT_PRIMITIVE)
    ap.add_argument("--wrist", type=float, nargs=6,
                    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                    default=list(DEFAULT_WRIST_XYZ) + list(DEFAULT_WRIST_RPY),
                    help="hand base pose (m, rad; ZYX euler). Default is the "
                         "shared solvers.DEFAULT_WRIST_* pose the visualizer "
                         "opens on.")
    ap.add_argument("--sigma-wrist-pos", dest="sigma_wrist_pos", type=float,
                    default=1e-4)
    ap.add_argument("--sigma-wrist-rot", dest="sigma_wrist_rot", type=float,
                    default=1e-3)
    ap.add_argument("--passive", type=float, default=0.5)
    ap.add_argument("--flexor", type=float, default=GRASP_FLEXOR_TENSION,
                    help="uniform per-finger flexor tension (N)")
    ap.add_argument("--contact-fingers", dest="contact_fingers", type=parse_mask,
                    default=None, metavar="I,M,R,P,T",
                    help="per-finger contact flags in index,middle,ring,pinky,"
                         "thumb order (default: all five)")
    ap.add_argument("--al-mu", dest="al_mu", type=float, default=1.0)
    ap.add_argument("--al-rate", dest="al_rate", type=float, default=2.0)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=40,
                    help="cap on AL outer iterations (the GUI's 'max steps')")
    ap.add_argument("--no-collision", dest="collision", action="store_false",
                    default=True,
                    help="drop the Section 1.5 collision spheres (they are ON "
                         "by default here, unlike in the GUI)")
    ap.add_argument("--collision-radius", dest="collision_radius", type=float,
                    default=0.003)
    ap.add_argument("--collision-sigma", dest="collision_sigma", type=float,
                    default=1e-4)
    ap.add_argument("--cull-margin", dest="cull_margin", type=float, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="set CREST_AL_VERBOSE=1 for the C++ outer-loop stderr trace")
    ap.add_argument("--inner", action="store_true",
                    help="print the inner-LM iteration tail after every step")
    ap.add_argument("--inner-tail", dest="inner_tail", type=int, default=8)
    ap.add_argument("--no-kinematics", dest="kinematics", action="store_false",
                    default=True,
                    help="skip the per-step kinematic state dump (base pose, "
                         "per-finger tendon tensions / lengths, tip poses)")
    ap.add_argument("--nodes", action="store_true",
                    help="dump every rod node's position and stress as part of "
                         "the kinematic state (145 nodes per step -- verbose)")
    ap.add_argument("--dump-config", action="store_true", default=True,
                    help="dump the resolved HandSolveParams (on by default)")
    ap.add_argument("--no-dump-config", dest="dump_config", action="store_false")
    ap.add_argument("--log", action="store_true",
                    help="tee stdout to results/debug_ik_step_<primitive>_<stamp>.log")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verbose:
        os.environ["CREST_AL_VERBOSE"] = "1"

    logger = None
    if args.log:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
        logger = PlannerLogger(f"debug_ik_step_{args.primitive}",
                               log_dir=os.path.abspath(log_dir), timestamp=True)
    try:
        if not capabilities()["ik_stepping"]:
            print("This harness needs a _crest_sparse with "
                  "TendonHandSolver.reset_al_duals (the same build the GUI's "
                  "Step button needs). Rebuild the extension.")
            return 1

        params = build_params(args)
        print("=" * 72)
        print("STEPPED IK SOLVE -- headless replay of the visualizer's Step button")
        print("=" * 72)
        print_capabilities()

        t0 = time.perf_counter()
        stepper = HandIKStepper(params)
        print(f"\ngraph built in {(time.perf_counter() - t0) * 1e3:.0f} ms")
        print_scene(stepper, params)
        if args.dump_config:
            log_planner_parameters(params)

        rows, status, result = step_through(stepper, args)
        if not rows:
            print("no steps taken (--max-steps 0?)")
            return 1
        print_factor_errors(stepper, "\nfactor errors at the final iterate")
        if not args.kinematics:
            # The per-step dump is off, but the state the run ENDED in is the one
            # number-for-number record of what was solved, so it is never skipped.
            print("\nfinal kinematic state:")
            print_kinematic_state(stepper, result, indent="  ", nodes=args.nodes)
        print_summary_table(rows)
        print_verdict(rows, status, args)
        return 0
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
