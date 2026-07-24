"""Plan a *trajectory* of five-finger grasps (PDF Section 1.4).

This is the trajectory-planning counterpart of ``ik_5f_contact.py``.
Where that test does a single kinematic solve, here we plan a K+1-step trajectory
whose control actions are the shared wrist pose and each finger's tendon tensions.
The steps are tied together with Gaussian-process temporal priors:

  * wrist pose    : BetweenFactor<Pose3>           (Eq 1.41/1.42)
  * tendon tension: per-finger BetweenFactor<Vec>  (Eq 1.11)
  * tendon length : per-finger BetweenFactor<Vec>  (Eq 1.13, optional)

Boundary (Section 1.4.2): the start step (k=0) carries a *loose* hand-pose prior
so the wrist may reposition; the terminal step (k=K) carries the per-finger SDF
contact constraints (contact-as-goal). The solve runs on the Augmented
Lagrangian path because of the terminal contact.

Everything a run produces (log, per-finger state figures, wrist-trajectory
figure, animation frames + GIF) lands in ``results/<experiment>/``.

Run (from the ``python/`` directory):
    # interactive 3D animation + saved state/wrist figures
    python -m tests.tendon_hand.traj_5f_contact big_sphere
    # headless: render the animation off-screen to a GIF and save all figures
    python -m tests.tendon_hand.traj_5f_contact big_sphere -SF
"""

import os
import argparse
import time

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index)
from .scene import (
    OBJECT_CENTER, get_primitive_specs, primitive_surface_gap,
    configure_object_surface, GRASP_FLEXOR_TENSION, GRASP_SPHERE_CENTER,
    TENDON_NAMES)
from .utils import FingerTraj
from .._plotting.trajectory_plotter import plot_trajectory, plot_hand_wrist_trajectory
from .._plotting.al_convergence_plotter import plot_al_convergence
from ..tendon_finger.utils import PlannerLogger, log_planner_parameters


def experiment_label(args):
    """Per-experiment label, e.g. grasp_traj_big_sphere_K10 — names the results
    directory that collects this run's log, figures, frames and GIF."""
    return f"grasp_traj_{args.primitive}_K{args.steps}"


def _add_object_mesh(pv_plotter, spec, center):
    """Render the shared contact object into the pyvista window (best-effort)."""
    import pyvista as pv
    t = spec["type"]
    if t == "sphere":
        mesh = pv.Sphere(radius=spec["radius"], center=center)
    elif t == "cylinder":
        mesh = pv.Cylinder(center=center, direction=(0, 0, 1),
                           radius=spec["radius"], height=spec["height"])
    elif t == "capsule":
        mesh = pv.Capsule(center=center, direction=(0, 0, 1),
                          radius=spec["radius"], cylinder_length=spec["height"])
    elif t == "cube":
        hx, hy, hz = spec["half_extents"]
        mesh = pv.Cube(center=center, x_length=2 * hx, y_length=2 * hy, z_length=2 * hz)
    elif t == "ellipsoid":
        a, b, c = (float(v) for v in spec["semi_axes"])
        mesh = pv.ParametricEllipsoid(a, b, c).translate(center, inplace=False)
    else:
        return
    pv_plotter.add_mesh(mesh, color="goldenrod", opacity=0.35)


class _FingerSol:
    """Adapter exposing (.marginals, .meta) as the hand plotter expects."""
    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta


def _report_al_iterations(result, results_dir, exp_label):
    """Print the per-outer-iteration AL trace and save the convergence figure.

    Headless-safe (only saves the PNG). Returns True if a trace was present.
    """
    costs = list(result.meta.al_iteration_costs)
    viols = list(result.meta.al_iteration_violations)
    mus = list(result.meta.al_iteration_mus)
    if not costs:
        print("\n[debug-iterations] no AL iteration trace was recorded "
              "(record_iterations off, or the solve took the non-AL path).")
        return False

    print("\nAL outer-iteration trace:")
    print(" iter |      cost |  violation |        mu")
    print("------+-----------+------------+-----------")
    for i, (c, v, mu) in enumerate(zip(costs, viols, mus)):
        print(f"  {i:>3} | {c:9.4g} | {v:10.4g} | {mu:9.4g}")

    save_path = os.path.join(results_dir, f"{exp_label}_al_convergence.png")
    plot_al_convergence(costs, viols, mus,
                        title=f"{exp_label} — AL convergence",
                        save_path=save_path, show=False)
    print(f"Saved AL convergence figure to {save_path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a K-step five-finger tendon-hand grasp trajectory with "
                    "GP priors on the wrist pose and finger tensions.")
    parser.add_argument("primitive", nargs="?", default="big_sphere",
                        choices=["big_sphere", "capsule", "sphere", "cylinder", "cube",
                                 "coin", "credit_card", "pen"])
    parser.add_argument("-K", "--steps", type=int, default=10,
                        help="Number of trajectory steps (creates K+1 states).")
    parser.add_argument("--dt", type=float, default=0.1, help="Step duration (s).")
    parser.add_argument("--gp-wrist", type=float, default=1e-2,
                        help="Wrist-pose GP process-noise scale (diag of gp_wrist_Qc). "
                             "Smaller => smoother/less wrist motion between steps.")
    parser.add_argument("--gp-tense", type=float, default=1.0,
                        help="Tension GP process-noise scale (diag of gp_tense_Qc).")
    parser.add_argument("--gp-len", type=float, default=0.0,
                        help="Length GP process-noise scale; 0 disables the length GP.")
    parser.add_argument("--sigma-wrist-pos", type=float, default=1e-4,
                        help="Wrist position prior std (m) at k=0. Tight by default so "
                             "the start wrist is pinned to the measured pose; loosen to "
                             "let the optimizer reposition the start wrist.")
    parser.add_argument("--sigma-wrist-rot", type=float, default=1e-3,
                        help="Wrist rotation prior std (rad) at k=0 (tight by default).")
    parser.add_argument("--start-flexor", type=float, default=0.5,
                        help="Measured flexor tension (N) at k=0. Default 0.5 = open hand; "
                             "the trajectory then closes toward the grasp flexion.")
    parser.add_argument("--al-mu", type=float, default=1.0, help="AL initial penalty mu")
    parser.add_argument("--al-rate", type=float, default=2.0, help="AL mu increase rate")
    parser.add_argument("--al-iters", type=int, default=40, help="AL max outer iterations")
    parser.add_argument("--al-inner-tol", type=float, default=1e-2,
                        help="Initial inner-LM relative tolerance (inexact AL); "
                             "tightens ~1/mu down to 1e-5. 0 = full-precision "
                             "inner solves every outer iteration (slow).")
    parser.add_argument("--al-abs-cost-tol", type=float, default=1e12,
                        help="Outer AL absolute cost threshold. Default (huge) "
                             "stops once the contact violation is small; set "
                             "1e-5 (GTSAM default) to keep polishing the "
                             "soft-prior cost until stagnation (~2x slower).")
    parser.add_argument("--debug-iterations", action="store_true",
                        help="Record and visualize the solver's per-iteration progress "
                             "(one snapshot per Augmented-Lagrangian outer iteration): "
                             "prints a cost/violation/mu table, saves an AL-convergence "
                             "figure, and (unless --no-viz) animates the terminal grasp "
                             "converging onto the object across iterations.")
    parser.add_argument("--sample-interval", type=int, default=1,
                        help="With --debug-iterations, keep a trajectory snapshot every N "
                             "outer iterations (default 1 = every iteration).")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip the interactive 3D view (figures are still saved).")
    parser.add_argument("--save-figures", "-SF", action="store_true",
                        help="Render the 3D animation off-screen and save it to "
                             "results/<experiment>/ as frame PNGs plus a GIF (no window).")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the saved animation GIF.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Everything this run produces lands in one per-experiment results directory.
    results_dir = os.path.join("results", experiment_label(args))
    os.makedirs(results_dir, exist_ok=True)

    logger = PlannerLogger("grasp_trajectory", log_dir=results_dir, timestamp=False)
    try:
        _main(args, results_dir)
    finally:
        logger.close()


def _main(args, results_dir):
    exp_label = experiment_label(args)

    spec = get_primitive_specs()[args.primitive]
    object_center = (GRASP_SPHERE_CENTER
                     if args.primitive in ("big_sphere", "capsule")
                     or spec["type"] == "ellipsoid"
                     else OBJECT_CENTER)
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[0:3, 0:3] = object_rotation
    object_pose[0:3, 3] = object_center

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")

    # --- Five fingers on the shared wrist (same builder + contact setup as the
    #     static grasp test). Each finger's tip node is driven onto the shared SDF. ---
    dims = load_hand_dimensions()
    configs = get_default_hand_configs(dims)
    tip_radii = default_hand_tip_radii(dims)
    finger_names = [name for name, _ in configs]

    for (_, cfg), tip_radius in zip(configs, tip_radii):
        env_i = crest_sparse.EnvironmentConfig()
        configure_object_surface(env_i, spec, objects_dir, args.primitive)
        env_i.object_pose_mean = object_pose
        env_i.object_pose_cov = 1e-8 * np.eye(6)
        env_i.object_pose_per_step = False
        env_i.contact_node_radius = tip_radius
        env_i.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env_i

    # --- Trajectory planner config ---
    num_tendons = configs[0][1].num_tendons  # 6 for the anatomical finger

    plan_config = crest_sparse.TendonHandTrajectoryPlannerConfig()
    plan_config.K = args.steps
    plan_config.dt = args.dt
    plan_config.wrist_pose = np.eye(4)
    plan_config.sigma_wrist_pos = args.sigma_wrist_pos
    plan_config.sigma_wrist_rot = args.sigma_wrist_rot
    plan_config.gp_wrist_Qc = args.gp_wrist * np.eye(6)
    plan_config.gp_tense_Qc = args.gp_tense * np.eye(num_tendons)
    plan_config.gp_len_Qc = (args.gp_len * np.eye(num_tendons)
                             if args.gp_len > 0.0 else np.zeros((0, 0)))
    plan_config.base.linear_solver_type = "MULTIFRONTAL_QR"
    plan_config.base.al_initial_mu = args.al_mu
    plan_config.base.al_mu_increase_rate = args.al_rate
    plan_config.base.al_max_iterations = args.al_iters
    plan_config.base.al_inner_rel_tol_initial = args.al_inner_tol
    plan_config.base.al_abs_cost_tol = args.al_abs_cost_tol
    # Capture each AL outer iteration (full-trajectory snapshot + cost/violation/mu
    # trace) so we can visualize what the solver is doing step by step.
    if args.debug_iterations:
        plan_config.base.record_iterations = True
        plan_config.base.iteration_sample_interval = args.sample_interval

    log_planner_parameters(plan_config, extras={
        "primitive": args.primitive,
        "object_pose": object_pose,
        "num_tendons": num_tendons,
        "object_surface": (f"ellipsoid semi_axes={spec['semi_axes']}"
                           if spec["type"] == "ellipsoid" else spec["vdb"]),
    })

    planner = crest_sparse.TendonHandTrajectoryPlanner(configs, plan_config)
    print(f"Built hand trajectory planner: {planner.num_fingers()} fingers, "
          f"K={args.steps} steps.")

    # Background / target tensions applied at k>=1: passive tendons pinned,
    # flexor (index 5) loose toward the grasp flexion; tight tip-wrench prior.
    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, GRASP_FLEXOR_TENSION])
    tensions_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    all_tensions = [crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
                    for _ in configs]
    all_tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                        for _ in configs]

    # Measured k=0 state: the known hand opening. Flexor at --start-flexor (0 =
    # open), passives at their background hold, all pinned tightly. This replaces
    # the background tension prior at k=0 so the hand starts in this configuration
    # (with the wrist tightly held at wrist_pose) and closes over the trajectory —
    # mirroring how the real planner receives the current wrist pose + tendon state.
    start_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, args.start_flexor])
    start_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    all_start_tensions = [crest_sparse.VectorXGaussian(start_mean, start_cov)
                          for _ in configs]

    t0 = time.time()
    result = planner.plan(all_tensions, all_tip_wrenches,
                          start_tensions=all_start_tensions)
    dt_ms = (time.time() - t0) * 1000.0

    K = args.steps
    print(f"Planned in {dt_ms:.1f} ms | iters={result.meta.iterations} | "
          f"error={result.meta.error:.4g} | {len(result.trajectory)} states")

    # Per-step summary: wrist proxy (finger-0 base node translation = T_wrist o
    # offset_0, so its motion tracks the wrist) and the worst tip surface gap.
    print("\n step |   wrist-proxy (finger0 base xyz)        | max tip gap (m)")
    print("------+-----------------------------------------+----------------")
    prev_base = None
    for k, hand_m in enumerate(result.trajectory):
        base_pose = np.array(hand_m.fingers[0].rod.states[0].pose.mean)
        base_xyz = base_pose[:3, 3]
        step_disp = (0.0 if prev_base is None
                     else float(np.linalg.norm(base_xyz - prev_base)))
        prev_base = base_xyz

        gaps = []
        for (_, _cfg), tip_radius, fm in zip(configs, tip_radii, hand_m.fingers):
            tip_pos = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
            tip_local = object_rotation.T @ (tip_pos - object_center)
            gaps.append(primitive_surface_gap(tip_local, spec) - tip_radius)
        tag = "  <- start" if k == 0 else ("  <- goal" if k == K else "")
        print(f"  {k:>3} | [{base_xyz[0]:+.4f} {base_xyz[1]:+.4f} {base_xyz[2]:+.4f}] "
              f"d={step_disp:.4f} | {max(abs(g) for g in gaps):+.5f}{tag}")

    # Terminal grasp check (should match the static test's near-zero gaps).
    print("\nTerminal (k=K) per-finger surface gaps:")
    for (name, _), tip_radius, fm in zip(configs, tip_radii,
                                         result.trajectory[K].fingers):
        tip_pos = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
        tip_local = object_rotation.T @ (tip_pos - object_center)
        gap = primitive_surface_gap(tip_local, spec) - tip_radius
        print(f"  [{name:>6}] tip {tip_pos}  |  surface gap {gap:+.5f} m  "
              f"(target ~0, r={tip_radius:.4f})")

    print("\nFactor error summary (type, count, total_error):")
    for name, count, err in planner.get_factor_error_summary()[:8]:
        print(f"  {err:12.4g}  x{count:<4} {name}")

    # --- Solver-iteration diagnostics (only when --debug-iterations). ---
    if args.debug_iterations:
        _report_al_iterations(result, results_dir, exp_label)

    # --- Save the state / trajectory figures (headless-safe; always saved). ---
    # One rich per-finger state plot each (tensions, lengths, disc kinematics,
    # internal/external wrenches) reusing the single-finger plot_trajectory, plus
    # one shared wrist-pose trajectory figure.
    print("\nSaving trajectory figures...")
    for i, name in enumerate(finger_names):
        finger_traj = FingerTraj([hand_m.fingers[i] for hand_m in result.trajectory])
        plot_trajectory(
            finger_traj, tendon_names=TENDON_NAMES, show=False,
            save_path=os.path.join(results_dir, f"{exp_label}_states_{name}.png"))
    plot_hand_wrist_trajectory(
        result, configs[0][1].hand_base_offset, dt=args.dt, show=False,
        save_path=os.path.join(results_dir, f"{exp_label}_wrist.png"))

    # --- 3D animation of the robot states across the trajectory. ---
    if args.no_viz:
        print(f"Saved experiment results to {results_dir}/")
        return

    from .._plotting.tendon_hand_plotter import TendonHandPlotter

    plotter_kwargs = dict(
        plot_backbone_ellipsoids=False,
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=list(object_center),
        camera_distance=0.5,
    )
    if args.save_figures:
        # Frames -> results/<exp>/frames/{k}.png, GIF -> results/<exp>/<exp>.gif.
        plotter_kwargs["save_frames_dir_name"] = "frames"
        plotter_kwargs["frames_base_dir"] = results_dir

    def _show_hand(hand_m):
        plotter.update({name: _FingerSol(fm, result.meta)
                        for name, fm in zip(finger_names, hand_m.fingers)})

    plotter = TendonHandPlotter(finger_names, **plotter_kwargs)
    title = (f"{args.primitive} solver iterations (terminal grasp)"
             if args.debug_iterations else f"{args.primitive} grasp trajectory")
    plotter.plotter.plotter.add_text(title, position="upper_left", font_size=12,
                                     name="title")
    _add_object_mesh(plotter.plotter.plotter, spec, object_center)

    # Segment boundaries into the shared frames/ stream: (gif_name, start, end).
    # The plotter's monotonic frame counter is never reset (mesh actors build
    # only on frame==0), so we slice one flat frame stream into several GIFs.
    segments = []

    if args.debug_iterations:
        # Watch the solver converge: for every trajectory step k, animate that
        # step across the recorded AL outer iterations (initial guess first).
        # Each snapshot is that iteration's full trajectory; we show step k of
        # it. One GIF per step, plus a final converged-trajectory playback.
        snapshots = [planner.get_initial_solution()]
        snapshots += list(planner.get_intermediate_solutions())

        for k in range(K + 1):
            print(f"Animating {len(snapshots)} solver snapshots (step {k}/{K})...")
            plotter.plotter.plotter.add_text(
                f"{args.primitive} step {k}/{K} convergence",
                position="upper_left", font_size=12, name="title")
            start = plotter.plotter.frame
            for snap in snapshots:
                _show_hand(snap.trajectory[k])
                if not args.save_figures:
                    time.sleep(max(args.dt, 0.15))
            _show_hand(result.trajectory[k])  # settle on converged step state
            segments.append((f"{exp_label}_step{k}", start, plotter.plotter.frame))

        # Final converged trajectory playback (steps 0..K over time).
        print(f"Animating converged trajectory ({K + 1} steps)...")
        plotter.plotter.plotter.add_text(
            f"{args.primitive} converged trajectory",
            position="upper_left", font_size=12, name="title")
        start = plotter.plotter.frame
        for hand_m in result.trajectory:
            _show_hand(hand_m)
            if not args.save_figures:
                time.sleep(args.dt)
        segments.append((f"{exp_label}_trajectory", start, plotter.plotter.frame))
    else:
        start = plotter.plotter.frame
        for k, hand_m in enumerate(result.trajectory):
            print(f"Displaying step {k}/{K}")
            _show_hand(hand_m)
            if not args.save_figures:
                time.sleep(args.dt)
        segments.append((exp_label, start, plotter.plotter.frame))

    if args.save_figures:
        for name, start, end in segments:
            plotter.plotter.save_video(fps=args.fps, name=name,
                                       frame_range=(start, end))
        print(f"Saved experiment results to {results_dir}/")
        return

    input("Press Enter to close...")


if __name__ == "__main__":
    main()
