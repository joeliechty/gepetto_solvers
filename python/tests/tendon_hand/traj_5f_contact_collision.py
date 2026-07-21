"""Five-finger hand grasp *trajectory* WITH Section 1.5 collision avoidance.

The trajectory analogue of ``ik_5f_contact_collision.py``: plan a
K+1-step grasp trajectory (terminal tip contact, GP priors on wrist pose and
tendon tensions — the ``traj_5f_contact`` setup) with
``config.attach_collision`` layered on, so every plannable step k >= 1 carries
the AL inequality constraints (Eq 1.57-1.58): sphere-to-SDF keeping each
finger's disc spheres out of the object, and sphere-to-sphere keeping distinct
fingers apart (proximal-proximal and node-0 pairs excluded, matching the C++
factors). The start step k=0 is a pinned measurement and is not constrained --
it may legitimately begin in collision.

Reports, per trajectory step, the worst finger-object clearance and the worst
cross-finger sphere gap (the terminal contact tips are excluded from the
object check at k=K only — their contact factor pins them tangent), plus the
terminal tip contact gaps. PASS requires no penetration at ANY step and the
terminal contact to hold.

Optionally solves the same trajectory with collision OFF first (--baseline) to
show what the constraints actually prevented.

Run (from the ``crest-sparse/`` directory):
    python -m python.tests.tendon_hand.traj_5f_contact_collision big_sphere --no-viz
    # small/fast: one finger, five steps
    python -m python.tests.tendon_hand.traj_5f_contact_collision big_sphere --no-viz --num-fingers 1 -K 5
"""

import os
import argparse
import itertools
import time

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index, attach_collision, disc_node_indices, proximal_disc_flags)
from .scene import (
    OBJECT_CENTER, get_primitive_specs, primitive_surface_gap,
    GRASP_FLEXOR_TENSION, GRASP_SPHERE_CENTER, TENDON_NAMES)
from .utils import FingerTraj
from .._plotting.trajectory_plotter import (
    plot_trajectory, plot_hand_wrist_trajectory)
from ..tendon_finger.utils import (
    PlannerLogger, log_planner_parameters, log_prior_table,
    log_conditioning_report, report_al_iterations)

PASS_TOL = 1e-4        # max allowed penetration (m)
CONTACT_TOL = 5e-4     # max allowed |terminal tip gap| (m)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primitive", nargs="?", default="big_sphere",
                        choices=["big_sphere", "capsule", "sphere", "cylinder", "cube"])
    parser.add_argument("-K", "--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-fingers", type=int, default=0,
                        help="Use only the first N digits (0 = all five). "
                             "1 isolates finger-object collision (no pairs).")
    parser.add_argument("--collision-radius", type=float, default=0.003)
    parser.add_argument("--collision-sigma", type=float, default=1.0,
                        help="Constraint row scaling (1.0 = same as contact rows).")
    # --- Wrist GP process noise, expressed directly as the allowable per-step
    #     displacement std (m / rad). With the GP's identity state transition the
    #     between-step twist IS the displacement, so sigma == the micro-adjustment
    #     the wrist may make each dt with near-zero Mahalanobis penalty. Built into
    #     gp_wrist_Qc as sigma^2 / dt so the C++ Covariance(Qc * dt) yields exactly
    #     sigma^2 per step (see plan_trajectory). The default funds this test's
    #     actual ~2.5 cm approach over K=5 steps (0.05 m/s average) — sigma is
    #     permission, not command, and any tighter value's precision (1/sigma^2)
    #     beats the AL's achievable mu so the terminal contact silently loses.
    parser.add_argument("--gp-wrist-pos", type=float, default=0.05,
                        help="Per-step wrist *position* GP std (m). Size it as "
                             "approach_distance/K: this test's start is ~2.5 cm "
                             "from tangency, so K=5 needs ~0.05 (verified PASS; "
                             "0.02 and below out-muscle the AL contact, whose "
                             "achievable mu ~8e3 < the GP precision 1/sigma^2 — "
                             "see wrist-gp-stiffness-vs-al-mu). Use 0.002-0.004 "
                             "only for true micro-adjustment starts (mm away).")
    parser.add_argument("--gp-wrist-rot", type=float, default=0.0075,
                        help="Per-step wrist *rotation* GP std (rad). 0.005-0.01.")
    parser.add_argument("--gp-tense", type=float, default=1.0)
    # --- k=0 hand-pose prior: absolute accuracy of the end-effector relative to
    #     the perception-located object. Rigid-arm FK is repeatable but absolute
    #     volumetric accuracy is rarely sub-mm, so loosening these lets the AL
    #     solver rigidly shift the whole trajectory a few mm to meet the surface
    #     instead of stretching the GP springs / cranking mu.
    parser.add_argument("--sigma-wrist-pos", type=float, default=0.004,
                        help="k=0 wrist *position* prior std (m). 0.003-0.005 "
                             "reflects real perception+FK volumetric accuracy.")
    parser.add_argument("--sigma-wrist-rot", type=float, default=0.02,
                        help="k=0 wrist *rotation* prior std (rad). 0.015-0.03 "
                             "(~1-1.7 deg).")
    # --- Object-pose prior: perception tolerance of an SDF built from a point
    #     cloud. The object pose is a variable with this prior, so loosening it
    #     lets the solver trade a small object shift against hand motion when
    #     resolving contact/collision.
    parser.add_argument("--sigma-object-pos", type=float, default=0.003,
                        help="Object-pose *position* prior std (m). 0.002-0.005 "
                             "= realistic point-cloud SDF registration accuracy.")
    parser.add_argument("--sigma-object-rot", type=float, default=0.03,
                        help="Object-pose *rotation* prior std (rad). 0.017-0.05 "
                             "(1-3 deg).")
    parser.add_argument("--start-flexor", type=float, default=0.5)
    parser.add_argument("--al-mu", type=float, default=1.0)
    parser.add_argument("--al-rate", type=float, default=2.0)
    parser.add_argument("--al-iters", type=int, default=40)
    parser.add_argument("--al-inner-tol", type=float, default=1e-3,
                        help="Initial inner-LM relative tolerance (inexact AL); "
                             "tightens ~1/mu down to 1e-5. 0 = full-precision "
                             "inner solves every outer iteration. 1e-2 is ~2x "
                             "faster on graphs that survive it, but on larger "
                             "graphs (K=10 x 5 fingers) the sloppy inner solves "
                             "produce no measurable outer progress and the AL "
                             "loop FALSELY exits on stagnation at ~mm-scale "
                             "violations; 1e-3 converges wherever 0 does at "
                             "~equal cost.")
    parser.add_argument("--al-abs-cost-tol", type=float, default=1e12,
                        help="Outer AL absolute cost threshold. Default (huge) "
                             "stops on constraint violation alone — the PASS "
                             "criteria (penetration, terminal gap) ARE the "
                             "violations. Set 1e-5 (GTSAM default) to keep "
                             "polishing the soft-prior cost until stagnation "
                             "(~2x slower, slightly smoother trajectories).")
    parser.add_argument("--cull-margin", type=float, default=0.02,
                        help="Cull finger-finger collision pairs whose initial "
                             "gap exceeds this (m). Negative = keep all pairs. "
                             "The per-step report still checks every pair.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also solve with collision OFF for comparison "
                             "(doubles the runtime).")
    parser.add_argument("--debug-iterations", action="store_true",
                        help="Record the solver's per-outer-iteration progress "
                             "(one snapshot per Augmented-Lagrangian iteration) "
                             "and print the cost/violation/mu trace plus an "
                             "AL-convergence figure. Enables the deeper solve-"
                             "landscape logging (priors table + conditioning).")
    parser.add_argument("--sample-interval", type=int, default=1,
                        help="With --debug-iterations, keep a trajectory "
                             "snapshot every N outer iterations (default 1).")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--save-figures", "-SF", action="store_true",
                        help="Render the trajectory animation off-screen to "
                             "results/<experiment>/ as frames + a GIF.")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def experiment_label(args):
    n = args.num_fingers if args.num_fingers > 0 else 5
    return f"collision_traj_{args.primitive}_K{args.steps}_F{n}"


def build_configs(args, dims, vdb_path, object_pose, tip_radii, collision):
    """Fresh per-solve configs: terminal tip contact always, collision optional."""
    configs = get_default_hand_configs(dims)
    radii = list(tip_radii)
    if args.num_fingers > 0:
        configs = configs[:args.num_fingers]
        radii = radii[:args.num_fingers]

    # Perception tolerance of the point-cloud SDF (Pose3 tangent: rot-first).
    object_pose_cov = np.diag([args.sigma_object_rot ** 2] * 3 +
                              [args.sigma_object_pos ** 2] * 3)

    for (_, cfg), tip_radius in zip(configs, radii):
        env_i = crest_sparse.EnvironmentConfig()
        env_i.load_sdf(vdb_path)
        env_i.object_pose_mean = object_pose
        env_i.object_pose_cov = object_pose_cov
        env_i.object_pose_per_step = False
        env_i.contact_node_radius = tip_radius
        env_i.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env_i

    if collision:
        attach_collision(configs, vdb_path, object_pose,
                         radius=args.collision_radius,
                         sigma=args.collision_sigma,
                         cull_margin=args.cull_margin)
    return configs, radii


def _log_solve_landscape(args, plan_config, num_tendons, tensions_cov,
                         tip_wrench_cov, start_cov, object_pose):
    """Dump every knob that shapes this solve: the full planner config, then a
    priors/constraint-weights table (std + information weight per prior) so the
    trade-offs the optimizer sees are legible at a glance."""
    log_planner_parameters(plan_config, extras={
        "primitive": args.primitive,
        "num_fingers": args.num_fingers if args.num_fingers > 0 else 5,
        "num_tendons": num_tendons,
        "object_pose": object_pose,
    })

    # Every soft prior / constraint noise model, with its weight. GP Qc entries
    # are process-noise covariances (scaled by dt inside the C++ GP factor), not
    # raw prior variances, but their diagonal is the tuning knob and its
    # precision is what trades off against the other priors on the same variable.
    entries = [
        {"name": "wrist pose @ k=0 (pos)", "factor": "hand-pose prior, Eq 1.40",
         "sigma": args.sigma_wrist_pos,
         "note": "perception+FK volumetric accuracy; loose enough to rigidly "
                 "shift the trajectory a few mm onto the object surface"},
        {"name": "wrist pose @ k=0 (rot)", "factor": "hand-pose prior, Eq 1.40",
         "sigma": args.sigma_wrist_rot},
        {"name": "GP wrist per-step (pos)", "factor": "BetweenFactor<Pose3>, Eq 1.41/1.42",
         "sigma": args.gp_wrist_pos,
         "note": "allowable wrist displacement per dt (identity transition => "
                 "sigma is the per-step micro-adjustment; smaller = stiffer)"},
        {"name": "GP wrist per-step (rot)", "factor": "BetweenFactor<Pose3>, Eq 1.41/1.42",
         "sigma": args.gp_wrist_rot},
        {"name": "GP tension Qc", "factor": "BetweenFactor<Vec>, Eq 1.11",
         "var": args.gp_tense},
        {"name": "tension prior (passive)", "factor": "background tension prior",
         "var": tensions_cov[0, 0],
         "note": "5 passive tendons pinned; ratio vs flexor sets the boundary "
                 "layer (see grasp-traj-tension-gp-boundary-layer memory)"},
        {"name": "tension prior (flexor)", "factor": "background tension prior",
         "var": tensions_cov[-1, -1], "note": "loose => flexor free to close"},
        {"name": "start tension @ k=0", "factor": "measured k=0 state",
         "var": start_cov[0, 0]},
        {"name": "tip wrench prior", "factor": "external tip-wrench prior",
         "cov": tip_wrench_cov},
        {"name": "object pose (pos)", "factor": "SDF object-pose prior",
         "sigma": args.sigma_object_pos,
         "note": "point-cloud SDF registration tolerance; solver may trade a "
                 "small object shift against hand motion"},
        {"name": "object pose (rot)", "factor": "SDF object-pose prior",
         "sigma": args.sigma_object_rot},
        {"name": "collision constraint", "factor": "AL inequality, Eq 1.57/1.58",
         "sigma": args.collision_sigma,
         "note": f"row scaling; r_col={args.collision_radius} m, "
                 f"cull_margin={args.cull_margin} m"},
    ]
    log_prior_table(entries)

    print("AL / solver knobs:")
    print(f"  linear_solver={plan_config.base.linear_solver_type}  "
          f"mu0={args.al_mu} rate={args.al_rate} max_iters={args.al_iters}")
    print(f"  inner_rel_tol_initial={args.al_inner_tol}  "
          f"abs_cost_tol={args.al_abs_cost_tol}")
    print("=" * 72)


def plan_trajectory(args, configs, label, object_pose=None, log_landscape=False):
    num_tendons = configs[0][1].num_tendons

    plan_config = crest_sparse.TendonHandTrajectoryPlannerConfig()
    plan_config.K = args.steps
    plan_config.dt = args.dt
    plan_config.wrist_pose = np.eye(4)
    plan_config.sigma_wrist_pos = args.sigma_wrist_pos
    plan_config.sigma_wrist_rot = args.sigma_wrist_rot
    # Pose3 tangent ordering is [rot, rot, rot, pos, pos, pos]. The C++ GP factor
    # uses Covariance(gp_wrist_Qc * dt), so dividing the target per-step variance
    # by dt here makes the effective per-step covariance exactly sigma^2 (dt
    # cancels) -- i.e. --gp-wrist-{pos,rot} ARE the per-step displacement stds.
    qc_rot = args.gp_wrist_rot ** 2 / args.dt
    qc_pos = args.gp_wrist_pos ** 2 / args.dt
    plan_config.gp_wrist_Qc = np.diag([qc_rot, qc_rot, qc_rot,
                                       qc_pos, qc_pos, qc_pos])
    plan_config.gp_tense_Qc = args.gp_tense * np.eye(num_tendons)
    plan_config.gp_len_Qc = np.zeros((0, 0))
    # Cholesky, explicitly: the AL path coerces any QR string to Cholesky anyway
    # (inequality AntiFactors linearize to negated Hessians QR cannot eliminate —
    # the inner LM would silently stall; see SolverBase.cpp), so configure what
    # actually runs rather than logging a misleading QR label.
    plan_config.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
    plan_config.base.al_initial_mu = args.al_mu
    plan_config.base.al_mu_increase_rate = args.al_rate
    plan_config.base.al_max_iterations = args.al_iters
    plan_config.base.al_inner_rel_tol_initial = args.al_inner_tol
    plan_config.base.al_abs_cost_tol = args.al_abs_cost_tol
    # Capture each AL outer iteration (cost/violation/mu trace + snapshots) so we
    # can visualize what the solver is doing step by step.
    if args.debug_iterations:
        plan_config.base.record_iterations = True
        plan_config.base.iteration_sample_interval = args.sample_interval

    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, GRASP_FLEXOR_TENSION])
    tensions_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    start_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, args.start_flexor])
    start_cov = np.diag([1e-6] * num_tendons)

    if log_landscape:
        _log_solve_landscape(args, plan_config, num_tendons, tensions_cov,
                             tip_wrench_cov, start_cov,
                             object_pose if object_pose is not None else np.eye(4))

    planner = crest_sparse.TendonHandTrajectoryPlanner(configs, plan_config)
    print(f"[{label}] built planner: {planner.num_fingers()} fingers, "
          f"K={args.steps} steps.")

    all_tensions = [crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
                    for _ in configs]
    all_tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                        for _ in configs]
    all_start = [crest_sparse.VectorXGaussian(start_mean, start_cov)
                 for _ in configs]

    t0 = time.time()
    result = planner.plan(all_tensions, all_tip_wrenches, start_tensions=all_start)
    dt_s = time.time() - t0
    print(f"[{label}] planned in {dt_s:.1f} s | iters={result.meta.iterations} | "
          f"error={result.meta.error:.4g}")
    return planner, result


def per_step_collision_report(args, configs, result, spec, object_pose):
    """Per-step worst finger-object clearance / cross-finger gap, with the same
    exclusions as the C++ factors (no node-0 spheres, no proximal-proximal
    pairs, terminal contact tip excluded from the object check at k=K only).

    Returns (worst_obj_over_steps, worst_ff_over_steps) over the *plannable*
    steps k >= 1. The start step k=0 is a pinned measurement and carries no
    collision constraints (it may legitimately begin in collision — e.g. a
    straight finger through the grasp sphere); it is printed for context but
    excluded from the pass criterion.
    """
    object_rotation = object_pose[:3, :3]
    object_center = object_pose[:3, 3]
    r = args.collision_radius
    K = args.steps

    print("\n step | worst obj clearance | worst finger-finger gap  (want >= 0)")
    print("------+---------------------+------------------------")
    worst_obj_all, worst_ff_all = np.inf, np.inf
    for k, hand_m in enumerate(result.trajectory):
        spheres = []  # per finger: (node, pos, proximal, is_tip)
        for (_, cfg), fm in zip(configs, hand_m.fingers):
            tip_idx = tip_node_index(cfg)
            entries = []
            for n, p in zip(disc_node_indices(cfg), proximal_disc_flags(cfg)):
                if n == 0:
                    continue
                pos = np.array(fm.rod.states[n].pose.mean)[:3, 3]
                entries.append((n, pos, bool(p), n == tip_idx))
            spheres.append(entries)

        worst_obj = np.inf
        for entries in spheres:
            for n, pos, _p, is_tip in entries:
                if k == K and is_tip:
                    continue  # terminal contact factor owns the tip at k=K
                local = object_rotation.T @ (pos - object_center)
                worst_obj = min(worst_obj,
                                primitive_surface_gap(local, spec) - r)

        worst_ff = np.inf
        for ia, ib in itertools.combinations(range(len(spheres)), 2):
            for na, pa, proxa, _ta in spheres[ia]:
                for nb, pb, proxb, _tb in spheres[ib]:
                    if proxa and proxb:
                        continue
                    worst_ff = min(worst_ff,
                                   np.linalg.norm(pa - pb) - 2.0 * r)

        ff_str = f"{worst_ff:+.5f} m" if np.isfinite(worst_ff) else "    (n/a)"
        tag = ("  <- start (unconstrained)" if k == 0
               else ("  <- goal" if k == K else ""))
        print(f"  {k:>3} |        {worst_obj:+.5f} m |        {ff_str}{tag}")
        if k > 0:  # k=0 is a pinned measurement, not a plannable step
            worst_obj_all = min(worst_obj_all, worst_obj)
            worst_ff_all = min(worst_ff_all, worst_ff)

    return worst_obj_all, worst_ff_all


def terminal_contact_report(configs, tip_radii, result, spec, object_pose):
    object_rotation = object_pose[:3, :3]
    object_center = object_pose[:3, 3]
    print("\nTerminal (k=K) tip contact gaps (target ~0):")
    worst = 0.0
    for (name, _), tip_radius, fm in zip(configs, tip_radii,
                                         result.trajectory[-1].fingers):
        tip_pos = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
        tip_local = object_rotation.T @ (tip_pos - object_center)
        gap = primitive_surface_gap(tip_local, spec) - tip_radius
        worst = max(worst, abs(gap))
        print(f"  [{name:>6}] surface gap {gap:+.5f} m (r={tip_radius:.4f})")
    return worst


def main():
    args = parse_args()
    results_dir = os.path.join("results", experiment_label(args))
    os.makedirs(results_dir, exist_ok=True)
    logger = PlannerLogger("collision_trajectory", log_dir=results_dir,
                           timestamp=False)
    try:
        _main(args, results_dir)
    finally:
        logger.close()


def _main(args, results_dir):
    spec = get_primitive_specs()[args.primitive]
    object_center = (GRASP_SPHERE_CENTER
                     if args.primitive in ("big_sphere", "capsule")
                     else OBJECT_CENTER)
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[:3, :3] = object_rotation
    object_pose[:3, 3] = object_center

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{args.primitive} (run from python/).")

    dims = load_hand_dimensions()
    tip_radii = default_hand_tip_radii(dims)

    # --- Optional collision-OFF baseline (the plain grasp trajectory) ---
    if args.baseline:
        print(f"[1/2] collision OFF baseline:")
        configs_off, radii_off = build_configs(
            args, dims, vdb_path, object_pose, tip_radii, collision=False)
        _planner_off, result_off = plan_trajectory(args, configs_off, "OFF")
        obj_off, ff_off = per_step_collision_report(
            args, configs_off, result_off, spec, object_pose)
        print(f"[OFF] worst-over-steps: object {obj_off:+.5f} m, "
              f"finger-finger {ff_off:+.5f} m")

    # --- Collision ON ---
    print(f"[{'2/2' if args.baseline else '1/1'}] collision ON:")
    configs, radii = build_configs(
        args, dims, vdb_path, object_pose, tip_radii, collision=True)
    print(f"  primitive={args.primitive} K={args.steps} fingers={len(configs)} "
          f"r_col={args.collision_radius} sigma_col={args.collision_sigma} "
          f"al=({args.al_mu}, {args.al_rate}, {args.al_iters})")
    planner, result = plan_trajectory(
        args, configs, "ON", object_pose=object_pose, log_landscape=True)

    worst_obj, worst_ff = per_step_collision_report(
        args, configs, result, spec, object_pose)
    worst_gap = terminal_contact_report(configs, radii, result, spec, object_pose)

    print(f"\nWORST over plannable steps (k>=1): object clearance {worst_obj:+.5f} m | "
          f"finger-finger gap {worst_ff:+.5f} m | terminal |gap| {worst_gap:.5f} m")
    collision_ok = worst_obj >= -PASS_TOL and worst_ff >= -PASS_TOL
    contact_ok = worst_gap <= CONTACT_TOL
    print("RESULT:", "PASS" if (collision_ok and contact_ok) else
          f"FAIL (collision_ok={collision_ok}, contact_ok={contact_ok})")

    # --- Solve-landscape diagnostics: which factor types dominate the residual,
    #     and how well-conditioned is the linearized system at the solution. ---
    print("\nFactor error summary (type, count, total_error):")
    for name, count, err in planner.get_factor_error_summary()[:10]:
        print(f"  {err:12.4g}  x{count:<4} {name}")

    log_conditioning_report(planner)

    # Per-outer-iteration AL trace + convergence figure (only with --debug-iterations).
    if args.debug_iterations:
        report_al_iterations(result, results_dir, experiment_label(args))

    # --- Save the state / trajectory figures (headless-safe; always saved). ---
    # One rich per-finger state plot each (tensions, lengths, disc kinematics,
    # internal/external wrenches) reusing the single-finger plot_trajectory, plus
    # one shared wrist-pose trajectory figure (same as the grasp trajectory test).
    exp_label = experiment_label(args)
    finger_names = [name for name, _ in configs]
    print("\nSaving trajectory figures...")
    for i, name in enumerate(finger_names):
        finger_traj = FingerTraj([hand_m.fingers[i] for hand_m in result.trajectory])
        plot_trajectory(
            finger_traj, tendon_names=TENDON_NAMES, show=False,
            save_path=os.path.join(results_dir, f"{exp_label}_states_{name}.png"))
    plot_hand_wrist_trajectory(
        result, configs[0][1].hand_base_offset, dt=args.dt, show=False,
        save_path=os.path.join(results_dir, f"{exp_label}_wrist.png"))

    if args.no_viz:
        print(f"Saved experiment results to {results_dir}/")
        return

    # --- Trajectory playback ---
    primitives = [dict(spec["plot"](object_center), color="goldenrod",
                       opacity=0.35)]

    class _FingerSol:
        def __init__(self, marginals, meta):
            self.marginals = marginals
            self.meta = meta

    def _solutions(hand_m):
        return {name: _FingerSol(fm, result.meta)
                for name, fm in zip(finger_names, hand_m.fingers)}

    if args.save_figures:
        # Single-window off-screen frames -> GIF (the multi-view plotter is for
        # interactive inspection; the frames pipeline is single-window).
        from .._plotting.tendon_hand_plotter import TendonHandPlotter
        plotter = TendonHandPlotter(
            finger_names, plot_backbone_ellipsoids=False,
            camera_azimuth=165, camera_elevation=20,
            camera_focal_point=list(object_center), camera_distance=0.5,
            primitives=primitives,
            save_frames_dir_name="frames", frames_base_dir=results_dir)
        for k, hand_m in enumerate(result.trajectory):
            print(f"Rendering step {k}/{args.steps}")
            plotter.update(_solutions(hand_m))
        plotter.plotter.save_video(fps=args.fps, name=experiment_label(args))
        print(f"Saved experiment results to {results_dir}/")
        return

    from .._plotting.tendon_hand_plotter import TendonHandMultiViewPlotter
    plotter = TendonHandMultiViewPlotter(
        finger_names, plot_backbone_ellipsoids=False,
        camera_focal_point=list(object_center), camera_distance=0.5,
        primitives=primitives)
    while True:
        for k, hand_m in enumerate(result.trajectory):
            plotter.update(_solutions(hand_m))
            time.sleep(max(args.dt, 0.15))
        again = input("Replay trajectory? [y/N] ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
