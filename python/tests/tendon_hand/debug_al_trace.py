"""Augmented-Lagrangian trace dumper for the tendon-hand solvers.

A *read-only* diagnostic harness that wraps the existing solver classes
(:mod:`solvers`) and surfaces the per-iteration AL diagnostics the interactive
visualizer throws away -- so you can see *why* a solve stalls instead of only
its final ``iters``/``err``.

Nothing here changes solver behaviour or any defaults: it just builds a
``HandSolveParams``, runs one solve (or a parameter sweep), and prints the
``SolutionMetadata`` trace arrays that the C++ side already fills in:

    al_iteration_mus / al_iteration_costs / al_iteration_violations  (outer AL)
    iteration_errors / iteration_step_norms / iteration_trust_region (inner LM)

Run it from the ``crest-sparse/`` repo root in the ``crest_py11`` env so the
import resolves to the installed binding, not the in-tree ``.so``::

    # single solve, C++ outer-loop stderr trace on
    python -m python.tests.tendon_hand.debug_al_trace --mode IK \
        --primitive coin --verbose

    # sweep the wrist-position prior and watch the final violation move
    python -m python.tests.tendon_hand.debug_al_trace --mode IK \
        --primitive big_sphere --sweep sigma_wrist_pos 1e-4 1e-3 1e-2 1e-1 1e0

    # inner-tolerance false-stagnation check on the planner
    python -m python.tests.tendon_hand.debug_al_trace --mode Planner \
        --primitive big_sphere --table --sweep al_inner_tol 1e-2 1e-3 1e-4 0

Companion to the debug plan in
``~/.claude/plans/my-viser-visualizer-shows-whimsical-widget.md``; the
failure-signature table there is what :func:`classify` echoes.
"""

import argparse
import math
import os

import numpy as np

from .solvers import SOLVERS, HandSolveParams
from .scene import get_primitive_specs, TABLE_NORMAL
from .utils import log_planner_parameters, PlannerLogger


# Same ZYX (yaw-pitch-roll) convention the visualizer uses for the wrist slider
# (viz_interactive._euler_to_R); duplicated here to keep this headless (no viser).
def _euler_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _wrist_pose(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _euler_to_R(*rpy)
    T[:3, 3] = xyz
    return T


# Params that are plain scalars we allow --sweep / --set to override by name.
# (Kept explicit so a typo'd knob name fails loudly instead of silently.)
_SCALAR_KNOBS = {
    "sigma_wrist_pos", "sigma_wrist_rot", "passive_tension", "tip_wrench_sigma",
    "al_mu", "al_rate", "al_iters", "K", "dt", "gp_wrist", "gp_tense", "gp_len",
    "start_flexor", "al_inner_tol", "al_abs_cost_tol", "collision_radius",
    "collision_sigma", "num_proximal_discs", "cull_margin",
}
_INT_KNOBS = {"al_iters", "K", "num_proximal_discs"}


def _coerce(name, raw):
    """Coerce a CLI string to the knob's type (int knobs -> int, else float)."""
    if name in _INT_KNOBS:
        return int(float(raw))
    return float(raw)


def _contact_mask(raw):
    """Parse ``--contact-fingers 1,1,0,0,1`` into the per-finger bool list.

    Not a --set/--sweep knob (those are scalars only); a masked-off finger simply
    contributes no contact constraint, which is the cheapest way to ask whether a
    stall is one unreachable finger dragging the whole AL solve down."""
    flags = [f.strip() for f in raw.split(",") if f.strip()]
    if len(flags) != 5:
        raise SystemExit(
            f"--contact-fingers: expected 5 comma-separated flags, got {len(flags)}")
    return [f not in ("0", "false", "False", "off", "no") for f in flags]


# ---------------------------------------------------------------------------
# Trace formatting.
# ---------------------------------------------------------------------------

def _arr(meta, name):
    """Fetch a metadata trace array as a plain float list ([] if absent/empty)."""
    v = getattr(meta, name, None)
    if v is None:
        return []
    return [float(x) for x in v]


def print_outer_trace(meta):
    """The AL outer-loop table: one row per outer iteration."""
    mus = _arr(meta, "al_iteration_mus")
    costs = _arr(meta, "al_iteration_costs")
    viols = _arr(meta, "al_iteration_violations")
    n = max(len(mus), len(costs), len(viols))
    if n == 0:
        print("  (no AL outer-loop trace -- kinematic/unconstrained solve)")
        return
    print(f"  {'iter':>4}  {'mu':>12}  {'cost':>14}  {'violation':>14}")
    print("  " + "-" * 50)

    def g(lst, i):
        return lst[i] if i < len(lst) else float("nan")

    for i in range(n):
        print(f"  {i:>4}  {g(mus, i):>12.4g}  {g(costs, i):>14.6g}  "
              f"{g(viols, i):>14.6g}")


def print_inner_tail(meta, tail=8):
    """Last few inner-LM iterations -- tells 'inner maxed out' from 'sloppy exit'."""
    errs = _arr(meta, "iteration_errors")
    steps = _arr(meta, "iteration_step_norms")
    tr = _arr(meta, "iteration_trust_region")
    n = max(len(errs), len(steps), len(tr))
    if n == 0:
        # The AL path skips the LM/Dogleg iterate-loop that fills these arrays;
        # the per-outer inner-iteration counts live in the C++ --verbose trace.
        print("  (no inner-LM iterate trace on the AL path -- use --verbose and "
              "read its 'uopt_iters' column for inner iterations per outer step)")
        return
    lo = max(0, n - tail)
    print(f"  inner-LM tail (last {n - lo} of {n} iters):")
    print(f"  {'k':>4}  {'error':>14}  {'step_norm':>14}  {'trust_region':>14}")
    print("  " + "-" * 54)

    def g(lst, i):
        return lst[i] if i < len(lst) else float("nan")

    for i in range(lo, n):
        print(f"  {i:>4}  {g(errs, i):>14.6g}  {g(steps, i):>14.6g}  "
              f"{g(tr, i):>14.6g}")


def classify(meta, worst_gap, al_iters_cap):
    """Heuristic pointer into the plan's failure-signature table. A *hint*, not a
    verdict -- always eyeball the outer trace and the render together.

    Distinguishes the three constrained-solve outcomes that matter:
      * exited *before* the al_iters cap with violation still high  -> stagnation
        (the mode-2 inner-tol signature);
      * ran *to* the cap with the violation still visibly dropping   -> just needs
        more outer iters (not a stall);
      * ran to the cap with the violation *frozen*                   -> mode-1/3
        prior-dominance / unreachable geometry.
    """
    mus, vio = _arr(meta, "al_iteration_mus"), _arr(meta, "al_iteration_violations")
    if not vio:
        return "kinematic / unconstrained -- no AL constraint to stall"
    n_outer = len(vio)
    final_mu = mus[-1] if mus else float("nan")
    final_vio = vio[-1]
    gap_mm = worst_gap * 1e3
    # Relative drop over the last few outer iters: is the violation still moving?
    ref = vio[max(0, n_outer - 4)]
    rel_drop = (ref - final_vio) / ref if ref > 0 else 0.0
    still_moving = rel_drop > 0.1
    hit_cap = n_outer >= al_iters_cap

    if final_vio < 1e-4 and gap_mm < 0.5:
        return f"HEALTHY -- violation {final_vio:.2g}, worst_gap {gap_mm:.2f} mm"
    if not hit_cap and not still_moving:
        return ("mode (2) inner-tol false-stagnation? -- exited early at "
                f"{n_outer}/{al_iters_cap} outer iters, violation {final_vio:.2g} "
                "still high and flat; try al_inner_tol <= 1e-3")
    if still_moving:
        return (f"PROGRESSING (not stalled) -- violation {final_vio:.2g} still "
                f"dropping ({rel_drop * 100:.0f}% over last iters), worst_gap "
                f"{gap_mm:.2f} mm; raise al_iters to let it finish")
    return ("mode (1)/(3) prior-dominance or unreachable -- ran to cap, mu "
            f"{final_mu:.2g} but violation frozen at {final_vio:.2g} "
            f"(worst_gap {gap_mm:.2f} mm); loosen the wrist prior / GP or check "
            "reachability")


# ---------------------------------------------------------------------------
# Solve helpers.
# ---------------------------------------------------------------------------

def _build_params(args):
    p = HandSolveParams()
    p.record_iterations = True  # this harness exists to read the AL trace
    p.primitive = args.primitive
    p.wrist_pose = _wrist_pose(args.wrist[:3], args.wrist[3:])
    p.sigma_wrist_pos = args.sigma_wrist_pos
    p.sigma_wrist_rot = args.sigma_wrist_rot
    p.passive_tension = args.passive
    if args.flexor is not None:
        p.flexor_tensions = [args.flexor] * 5
    if args.contact_fingers is not None:
        p.contact_fingers = _contact_mask(args.contact_fingers)
    p.al_mu, p.al_rate, p.al_iters = args.al_mu, args.al_rate, args.al_iters
    p.al_inner_tol = args.al_inner_tol
    p.K, p.dt, p.gp_wrist, p.gp_tense = args.K, args.dt, args.gp_wrist, args.gp_tense
    p.start_flexor = args.start_flexor
    p.collision = args.collision
    p.collision_sigma = args.collision_sigma
    p.cull_margin = args.cull_margin
    p.table = args.table
    p.plane_normal = np.array(TABLE_NORMAL, float)
    p.plane_avoidance = args.plane_avoidance
    if args.table and args.k_touch is not None:
        p.k_touch = args.k_touch
    # Apply any --set overrides last (they win over the dedicated flags).
    for name, raw in args.set or []:
        if name not in _SCALAR_KNOBS:
            raise SystemExit(f"--set: unknown knob '{name}' (allowed: "
                             f"{', '.join(sorted(_SCALAR_KNOBS))})")
        setattr(p, name, _coerce(name, raw))
    return p


def _terminal_gap(result):
    """Worst fingertip surface gap at the final frame (IK/planner terminal)."""
    return result.worst_gap(len(result.frames) - 1)


def run_one(args, params, header=None):
    """Solve once and print the full trace + classification. Returns the result."""
    Solver = SOLVERS[args.mode]
    result = Solver(params).solve()
    meta = result.meta
    gap = _terminal_gap(result)
    if header:
        print(header)
    print(f"  final: iters={meta.iterations}  error={meta.error:.4g}  "
          f"worst_gap={gap * 1e3:.3f} mm  "
          f"solve={getattr(meta, 'total_time_ms', float('nan')):.0f} ms")
    print_outer_trace(meta)
    if args.inner:
        print_inner_tail(meta)
    print(f"  >>> {classify(meta, gap, params.al_iters)}")
    return result


def run_sweep(args):
    """Re-solve for each value of args.sweep[0], printing a one-line summary row."""
    knob = args.sweep[0]
    if knob not in _SCALAR_KNOBS:
        raise SystemExit(f"--sweep: unknown knob '{knob}' (allowed: "
                         f"{', '.join(sorted(_SCALAR_KNOBS))})")
    values = [_coerce(knob, v) for v in args.sweep[1:]]
    if not values:
        raise SystemExit("--sweep needs at least one value after the knob name")

    print(f"\nSWEEP {knob} over {values}  (mode={args.mode}, "
          f"primitive={args.primitive}, table={args.table}, "
          f"collision={args.collision})")
    print(f"  {knob:>16}  {'iters':>6}  {'error':>12}  {'worst_gap':>10}  "
          f"{'final_mu':>10}  {'final_vio':>12}")
    print("  " + "-" * 78)
    for v in values:
        params = _build_params(args)
        setattr(params, knob, v)
        result = SOLVERS[args.mode](params).solve()
        meta = result.meta
        gap = _terminal_gap(result)
        mus = _arr(meta, "al_iteration_mus")
        vio = _arr(meta, "al_iteration_violations")
        fmu = mus[-1] if mus else float("nan")
        fvi = vio[-1] if vio else float("nan")
        print(f"  {v:>16.6g}  {meta.iterations:>6}  {meta.error:>12.4g}  "
              f"{gap * 1e3:>8.3f}mm  {fmu:>10.3g}  {fvi:>12.4g}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_set(pair):
    if "=" not in pair:
        raise argparse.ArgumentTypeError("--set expects NAME=VALUE")
    name, raw = pair.split("=", 1)
    return (name.strip(), raw.strip())


def build_parser():
    specs = sorted(get_primitive_specs().keys())
    ap = argparse.ArgumentParser(
        description="Dump the AL outer/inner trace for a tendon-hand solve.")
    ap.add_argument("--mode", choices=list(SOLVERS), default="IK")
    ap.add_argument("--primitive", choices=specs, default="big_sphere")
    ap.add_argument("--wrist", type=float, nargs=6,
                    metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                    default=[0.0] * 6, help="wrist start pose (m, rad; ZYX euler)")
    # Priors / tensions.
    ap.add_argument("--sigma-wrist-pos", dest="sigma_wrist_pos", type=float, default=1e-4)
    ap.add_argument("--sigma-wrist-rot", dest="sigma_wrist_rot", type=float, default=1e-3)
    ap.add_argument("--passive", type=float, default=0.5)
    ap.add_argument("--flexor", type=float, default=None,
                    help="uniform per-finger flexor tension (default: solver default)")
    ap.add_argument("--contact-fingers", dest="contact_fingers", default=None,
                    metavar="I,M,R,P,T",
                    help="per-finger contact flags in index,middle,ring,pinky,thumb "
                         "order (e.g. 1,0,0,0,1 for a pinch); default: all contact")
    # AL.
    ap.add_argument("--al-mu", dest="al_mu", type=float, default=1.0)
    ap.add_argument("--al-rate", dest="al_rate", type=float, default=2.0)
    ap.add_argument("--al-iters", dest="al_iters", type=int, default=40)
    ap.add_argument("--al-inner-tol", dest="al_inner_tol", type=float, default=1e-2)
    # Planner.
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gp-wrist", dest="gp_wrist", type=float, default=1e-2)
    ap.add_argument("--gp-tense", dest="gp_tense", type=float, default=1.0)
    ap.add_argument("--start-flexor", dest="start_flexor", type=float, default=0.5)
    # Collision / table.
    ap.add_argument("--collision", action="store_true")
    ap.add_argument("--collision-sigma", dest="collision_sigma", type=float, default=1e-4)
    ap.add_argument("--cull-margin", dest="cull_margin", type=float, default=None)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--no-plane-avoidance", dest="plane_avoidance",
                    action="store_false", default=True)
    ap.add_argument("--k-touch", dest="k_touch", type=int, default=None)
    # Overrides / sweep / output.
    ap.add_argument("--set", type=_parse_set, action="append", metavar="NAME=VALUE",
                    help="override any scalar knob by name (repeatable)")
    ap.add_argument("--sweep", nargs="+", metavar=("KNOB", "VALUE"),
                    help="sweep a scalar knob over the listed values")
    ap.add_argument("--verbose", action="store_true",
                    help="set CREST_AL_VERBOSE=1 for the C++ outer-loop stderr trace")
    ap.add_argument("--inner", action="store_true",
                    help="also print the inner-LM iteration tail")
    ap.add_argument("--dump-config", action="store_true",
                    help="dump the resolved params before solving")
    ap.add_argument("--log", action="store_true",
                    help="tee stdout to results/debug_al_<mode>_<primitive>.log")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.verbose:
        os.environ["CREST_AL_VERBOSE"] = "1"

    logger = None
    if args.log:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
        logger = PlannerLogger(f"debug_al_{args.mode}_{args.primitive}",
                               log_dir=os.path.abspath(log_dir), timestamp=True)
    try:
        if args.sweep:
            run_sweep(args)
        else:
            params = _build_params(args)
            if args.dump_config:
                log_planner_parameters(params)
            hdr = (f"\nSOLVE mode={args.mode} primitive={args.primitive} "
                   f"table={args.table} collision={args.collision}")
            run_one(args, params, header=hdr)
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
