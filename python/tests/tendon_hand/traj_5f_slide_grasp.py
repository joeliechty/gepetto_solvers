"""Five-finger hand slide-and-grasp *trajectory* (Section 1.6).

Adds a world-fixed support plane (a "table") to the collision-aware grasp
trajectory of ``traj_5f_contact_collision.py`` and, optionally, the human-inspired
sliding approach of Section 1.6.2. The table is analytic (an origin point + an
outward normal), so no VDB grid is needed — it plugs straight into the AL
collision/contact machinery via ``PlaneCollisionGapFactor``: as an inequality for
avoidance (Eq 1.59) and, on the designated sliding node, as a single-residual
equality on the sphere center for contact. (That equality replaces §1.6's
five-residual ``PlaneWitnessContactFactor``, Eq 1.61-1.64, whose free contact
point four of its rows existed only to pin.)

Two optional inputs drive it, both defaulting to "off" so the script degrades to
the plain collision grasp:

  * ``--table`` (on by default here): configure the plane on every finger via
    ``config.attach_table``. With no ``--k-touch`` this enforces the plane purely
    as *collision* at every plannable step — a free-space approach to the object
    that keeps the fingers off the table.

  * ``--k-touch K``: split the horizon into the three §1.6.2 phases about the
    contact-initialization step. Phase 1 (``0 < k < K``) is the free-space
    approach (table collision); Phases 2-3 (``k >= K``) place the fingertips on
    the table and let them slide toward the object (table sliding equality), with
    the terminal object grasp still active at k=K.

Backwards-compat: pass ``--no-table`` to recover exactly ``traj_5f_contact_collision``.

Run (from the ``crest-sparse/`` directory):
    # free-space approach to the object above a table (no sliding)
    python -m python.tests.tendon_hand.traj_5f_slide_grasp big_sphere --no-viz
    # full slide-and-grasp: sliding starts at step 4
    python -m python.tests.tendon_hand.traj_5f_slide_grasp big_sphere --no-viz --k-touch 4
    # small/fast sanity: one finger, five steps
    python -m python.tests.tendon_hand.traj_5f_slide_grasp big_sphere --no-viz --num-fingers 1 -K 5
"""

import os
import argparse
import time

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index, disc_node_indices, attach_collision, attach_table)
from .scene import (
    OBJECT_CENTER, get_primitive_specs, configure_object_surface,
    GRASP_SPHERE_CENTER, GRASP_FLEXOR_TENSION, TABLE_NORMAL, table_plot_spec,
    TENDON_NAMES)
from .utils import FingerTraj
from .._plotting.trajectory_plotter import (
    plot_trajectory, plot_hand_wrist_trajectory)
from .utils import PlannerLogger, log_conditioning_report

# Reuse the collision-grasp planner + reports verbatim (they are table-agnostic).
from . import traj_5f_contact_collision as base

PASS_TOL = 1e-4        # max allowed penetration (m)
CONTACT_TOL = 5e-4     # max allowed |terminal tip gap| (m)
SLIDE_TOL = 1e-3       # max allowed |fingertip distance-to-plane| in the slide phase (m)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primitive", nargs="?", default="big_sphere",
                        choices=["big_sphere", "capsule", "sphere", "cylinder", "cube",
                                 "coin", "credit_card", "pen"])
    parser.add_argument("-K", "--steps", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--num-fingers", type=int, default=0,
                        help="Use only the first N digits (0 = all five).")

    # --- Section 1.6 table / slide-and-grasp controls ---
    table_grp = parser.add_mutually_exclusive_group()
    table_grp.add_argument("--table", dest="table", action="store_true",
                           help="Configure the support plane (default).")
    table_grp.add_argument("--no-table", dest="table", action="store_false",
                           help="Disable the table -> plain collision grasp "
                                "(== traj_5f_contact_collision).")
    parser.set_defaults(table=True)
    parser.add_argument("--k-touch", type=int, default=None,
                        help="Contact-initialization step (Section 1.6.2). "
                             "Unset = free-space approach with table collision "
                             "only; set = fingertips slide on the table for k >= "
                             "k_touch. Ignored without --table.")
    parser.add_argument("--table-normal", type=float, nargs=3, default=None,
                        metavar=("NX", "NY", "NZ"),
                        help="Outward table normal (default +Z).")
    parser.add_argument("--table-origin", type=float, nargs=3, default=None,
                        metavar=("OX", "OY", "OZ"),
                        help="Table origin point (default: tangent to the "
                             "object's underside along -normal).")

    # --- Collision knobs (mirror traj_5f_contact_collision) ---
    parser.add_argument("--collision-radius", type=float, default=0.003)
    parser.add_argument("--collision-sigma", type=float, default=1.0)
    parser.add_argument("--cull-margin", type=float, default=0.02)

    # --- Wrist / GP / start knobs (consumed by base.plan_trajectory) ---
    parser.add_argument("--gp-wrist-pos", type=float, default=0.05)
    parser.add_argument("--gp-wrist-rot", type=float, default=0.0075)
    parser.add_argument("--gp-tense", type=float, default=1.0)
    parser.add_argument("--sigma-wrist-pos", type=float, default=0.004)
    parser.add_argument("--sigma-wrist-rot", type=float, default=0.02)
    parser.add_argument("--sigma-object-pos", type=float, default=0.003)
    parser.add_argument("--sigma-object-rot", type=float, default=0.03)
    parser.add_argument("--start-flexor", type=float, default=0.5)

    # --- AL solver knobs (consumed by base.plan_trajectory) ---
    parser.add_argument("--al-mu", type=float, default=1.0)
    parser.add_argument("--al-rate", type=float, default=2.0)
    parser.add_argument("--al-iters", type=int, default=40)
    parser.add_argument("--al-inner-tol", type=float, default=1e-3)
    parser.add_argument("--al-abs-cost-tol", type=float, default=1e12)
    parser.add_argument("--debug-iterations", action="store_true")
    parser.add_argument("--sample-interval", type=int, default=1)

    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--save-figures", "-SF", action="store_true",
                        help="Render the trajectory animation off-screen to "
                             "results/<experiment>/ as frames + a video.")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.k_touch is not None and not args.table:
        parser.error("--k-touch requires --table (the sliding constraints need a plane).")
    if args.k_touch is not None and not (0 < args.k_touch <= args.steps):
        parser.error(f"--k-touch must be in (0, K]={args.steps}; got {args.k_touch}.")
    return args


def experiment_label(args):
    n = args.num_fingers if args.num_fingers > 0 else 5
    mode = ("slide" if args.k_touch is not None
            else ("table" if args.table else "notable"))
    return f"slide_grasp_{args.primitive}_{mode}_K{args.steps}_F{n}"


def _object_radius(spec):
    if "radius" in spec:
        return float(spec["radius"])
    if "half_extents" in spec:
        return float(max(spec["half_extents"]))
    if "semi_axes" in spec:
        return float(max(spec["semi_axes"]))
    return 0.05


def resolve_table(args, spec, object_center):
    """Resolve the (origin, normal) of the support plane for this scene. Default
    normal is +Z; default origin is tangent to the object's underside so the
    object 'rests' on the table and the hand is in the free (+normal) half-space."""
    normal = (np.asarray(args.table_normal, dtype=float)
              if args.table_normal is not None else np.array(TABLE_NORMAL, float))
    normal = normal / np.linalg.norm(normal)
    if args.table_origin is not None:
        origin = np.asarray(args.table_origin, dtype=float)
    else:
        origin = np.asarray(object_center, float) - _object_radius(spec) * normal
    return origin, normal


def build_configs(args, dims, spec, objects_dir, object_pose, tip_radii,
                  plane_origin, plane_normal):
    """Object terminal contact + Section 1.5 collision + (optional) Section 1.6
    table. The object surface is an analytic hyper-ellipsoid (Section 1.6.3) or a
    baked SDF grid. Returns (configs, tip_radii) for this solve."""
    configs = get_default_hand_configs(dims)
    radii = list(tip_radii)
    if args.num_fingers > 0:
        configs = configs[:args.num_fingers]
        radii = radii[:args.num_fingers]

    object_pose_cov = np.diag([args.sigma_object_rot ** 2] * 3 +
                              [args.sigma_object_pos ** 2] * 3)
    for (_, cfg), tip_radius in zip(configs, radii):
        env_i = crest_sparse.EnvironmentConfig()
        configure_object_surface(env_i, spec, objects_dir, args.primitive)
        env_i.object_pose_mean = object_pose
        env_i.object_pose_cov = object_pose_cov
        env_i.object_pose_per_step = False
        env_i.contact_node_radius = tip_radius
        env_i.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env_i

    # attach_collision only sets the collision fields on the existing contact env
    # (built above), so it never re-loads the surface; vdb_path is unused here.
    attach_collision(configs, None, object_pose,
                     radius=args.collision_radius, sigma=args.collision_sigma,
                     cull_margin=args.cull_margin)

    if args.table:
        attach_table(configs, plane_origin, plane_normal,
                     avoidance=True, tip_radii=radii)
    return configs, radii


def per_step_table_report(args, configs, result, plane_origin, plane_normal, tip_radii):
    """Per-step table interaction. For every step prints:
      * worst table clearance over the (non-tip) collision spheres  (want >= 0),
      * the fingertip distance-to-plane minus tip radius            (~0 while sliding).
    Phase tags follow --k-touch. Returns (worst_clearance_over_phase1,
    worst_abs_slide_residual_over_phase23)."""
    n_hat = np.asarray(plane_normal, float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    p0 = np.asarray(plane_origin, float)
    r = args.collision_radius
    K = args.steps
    kt = args.k_touch

    def phase(k):
        if k == 0:
            return "start"
        if kt is None:
            return "free"
        return "approach" if k < kt else ("slide" if k < K else "grasp")

    print("\n step | phase    | worst table clearance | tip dist-to-plane  (want: clr>=0, slide~0)")
    print("------+----------+-----------------------+-------------------")
    worst_clr, worst_slide = np.inf, 0.0
    for k, hand_m in enumerate(result.trajectory):
        clr = np.inf
        tip_dists = []
        for (_, cfg), r_tip, fm in zip(configs, tip_radii, hand_m.fingers):
            tip_idx = tip_node_index(cfg)
            for n in disc_node_indices(cfg):
                if n == 0:
                    continue
                pos = np.array(fm.rod.states[n].pose.mean)[:3, 3]
                sdf = float((pos - p0).dot(n_hat))       # signed dist to plane
                if n == tip_idx:
                    tip_dists.append(sdf - r_tip)
                else:
                    clr = min(clr, sdf - r)
        ph = phase(k)
        tip_str = (f"{np.mean(tip_dists):+.5f} m" if tip_dists else "   (n/a)")
        clr_str = f"{clr:+.5f} m" if np.isfinite(clr) else "   (n/a)"
        print(f"  {k:>3} | {ph:>8} |         {clr_str} |        {tip_str}")
        if ph in ("free", "approach") and np.isfinite(clr):
            worst_clr = min(worst_clr, clr)
        if ph in ("slide", "grasp") and tip_dists:
            worst_slide = max(worst_slide, max(abs(d) for d in tip_dists))
    return worst_clr, worst_slide


def _main(args, results_dir):
    spec = get_primitive_specs()[args.primitive]
    object_center = (GRASP_SPHERE_CENTER
                     if args.primitive in ("big_sphere", "capsule")
                     or spec["type"] == "ellipsoid"
                     else OBJECT_CENTER)
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[:3, :3] = object_rotation
    object_pose[:3, 3] = object_center

    # Object surface: analytic hyper-ellipsoid (Section 1.6.3) or baked SDF grid
    # (configure_object_surface, invoked in build_configs, validates the asset).
    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")

    dims = load_hand_dimensions()
    tip_radii = default_hand_tip_radii(dims)
    plane_origin, plane_normal = resolve_table(args, spec, object_center)

    mode = ("slide-and-grasp (k_touch=%d)" % args.k_touch
            if args.k_touch is not None
            else ("free-space approach + table collision" if args.table
                  else "no table (plain collision grasp)"))
    print(f"[slide_grasp] primitive={args.primitive} K={args.steps} "
          f"fingers={args.num_fingers or 5} mode: {mode}")
    if args.table:
        print(f"  table: origin={np.round(plane_origin, 4)} "
              f"normal={np.round(plane_normal, 3)}")

    configs, radii = build_configs(
        args, dims, spec, objects_dir, object_pose, tip_radii,
        plane_origin, plane_normal)

    planner, result = plan_trajectory(args, configs)

    # Object / finger-finger collision (reuse the table-agnostic base report).
    worst_obj, worst_ff = base.per_step_collision_report(
        args, configs, result, spec, object_pose)
    worst_gap = base.terminal_contact_report(configs, radii, result, spec, object_pose)

    worst_clr, worst_slide = (np.inf, 0.0)
    if args.table:
        worst_clr, worst_slide = per_step_table_report(
            args, configs, result, plane_origin, plane_normal, radii)

    print(f"\nWORST over plannable steps: object {worst_obj:+.5f} m | "
          f"finger-finger {worst_ff:+.5f} m | terminal |gap| {worst_gap:.5f} m")
    collision_ok = worst_obj >= -PASS_TOL and worst_ff >= -PASS_TOL
    contact_ok = worst_gap <= CONTACT_TOL
    table_ok = True
    if args.table:
        table_clr_ok = (not np.isfinite(worst_clr)) or worst_clr >= -PASS_TOL
        slide_ok = args.k_touch is None or worst_slide <= SLIDE_TOL
        table_ok = table_clr_ok and slide_ok
        print(f"TABLE: worst free-approach clearance "
              f"{worst_clr if np.isfinite(worst_clr) else float('nan'):+.5f} m | "
              f"worst slide residual {worst_slide:.5f} m")
    ok = collision_ok and contact_ok and table_ok
    print("RESULT:", "PASS" if ok else
          f"FAIL (collision={collision_ok}, contact={contact_ok}, table={table_ok})")

    print("\nFactor error summary (type, count, total_error):")
    for name, count, err in planner.get_factor_error_summary()[:12]:
        print(f"  {err:12.4g}  x{count:<4} {name}")
    log_conditioning_report(planner)

    # --- Figures (headless-safe; always saved). ---
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

    # --- Trajectory playback (object + table drawn). ---
    primitives = [dict(spec["plot"](object_center), color="goldenrod", opacity=0.35)]
    if args.table:
        primitives.append(dict(table_plot_spec(plane_origin, plane_normal),
                               color="slategray", opacity=0.30))

    class _FingerSol:
        def __init__(self, marginals, meta):
            self.marginals = marginals
            self.meta = meta

    def _solutions(hand_m):
        return {name: _FingerSol(fm, result.meta)
                for name, fm in zip(finger_names, hand_m.fingers)}

    if args.save_figures:
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


def plan_trajectory(args, configs):
    """Build + run the trajectory planner. Mirrors base.plan_trajectory's knob
    assembly (kept inline rather than reused so we can also set the one field it
    does not know about: the Section 1.6 k_touch phase split)."""
    num_tendons = configs[0][1].num_tendons

    plan_config = crest_sparse.TendonHandTrajectoryPlannerConfig()
    plan_config.K = args.steps
    plan_config.dt = args.dt
    plan_config.wrist_pose = np.eye(4)
    plan_config.sigma_wrist_pos = args.sigma_wrist_pos
    plan_config.sigma_wrist_rot = args.sigma_wrist_rot
    qc_rot = args.gp_wrist_rot ** 2 / args.dt
    qc_pos = args.gp_wrist_pos ** 2 / args.dt
    plan_config.gp_wrist_Qc = np.diag([qc_rot, qc_rot, qc_rot, qc_pos, qc_pos, qc_pos])
    plan_config.gp_tense_Qc = args.gp_tense * np.eye(num_tendons)
    plan_config.gp_len_Qc = np.zeros((0, 0))
    plan_config.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
    plan_config.base.al_initial_mu = args.al_mu
    plan_config.base.al_mu_increase_rate = args.al_rate
    plan_config.base.al_max_iterations = args.al_iters
    plan_config.base.al_inner_rel_tol_initial = args.al_inner_tol
    plan_config.base.al_abs_cost_tol = args.al_abs_cost_tol
    if args.debug_iterations:
        plan_config.base.record_iterations = True
        plan_config.base.iteration_sample_interval = args.sample_interval
    # Section 1.6: the slide-and-grasp phase split (None = free-space).
    plan_config.k_touch = args.k_touch

    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, GRASP_FLEXOR_TENSION])
    tensions_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    start_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, args.start_flexor])
    start_cov = np.diag([1e-6] * num_tendons)

    planner = crest_sparse.TendonHandTrajectoryPlanner(configs, plan_config)
    print(f"[plan] built planner: {planner.num_fingers()} fingers, "
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
    print(f"[plan] planned in {dt_s:.1f} s | iters={result.meta.iterations} | "
          f"error={result.meta.error:.4g}")
    return planner, result


def main():
    args = parse_args()
    results_dir = os.path.join("results", experiment_label(args))
    os.makedirs(results_dir, exist_ok=True)
    logger = PlannerLogger("slide_grasp_trajectory", log_dir=results_dir,
                           timestamp=False)
    try:
        _main(args, results_dir)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
