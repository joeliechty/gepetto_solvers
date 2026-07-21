"""Plan a *point-to-point* five-finger hand trajectory (per-finger position goals).

This is the position-goal counterpart of ``traj_5f_contact.py``
and the hand-level analogue of ``tendon_finger/point_to_point_planning.py``. The
start conditions are identical to the grasp-trajectory test -- the shared wrist is
pinned at identity, the hand starts open (flexor slack), and the steps are tied
together with the same Gaussian-process temporal priors:

  * wrist pose    : BetweenFactor<Pose3>           (Eq 1.41/1.42)
  * tendon tension: per-finger BetweenFactor<Vec>  (Eq 1.11)
  * tendon length : per-finger BetweenFactor<Vec>  (Eq 1.13, optional)

The only difference is the terminal (k=K) goal. Instead of driving each fingertip
onto a shared SDF object (contact-as-goal), we give every finger an explicit
world-frame tip-position goal via the planner's ``goal_positions``. Each becomes a
soft ``PositionPriorFactor`` on that finger's tip node, so -- with no contact
configured -- the solve runs on the plain (non-Augmented-Lagrangian) path, exactly
like the single-finger point-to-point planner.

The goal positions below are a set of real, fully-converged fingertip positions
taken from the terminal step of ``results/grasp_traj_capsule_K10`` (surface gap
~0), so they are known to be reachable with the wrist at identity.

Everything a run produces (log, per-finger state figures, wrist-trajectory figure,
animation frames + GIF) lands in ``results/<experiment>/``.

Run (from the ``python/`` directory):
    # interactive 3D animation + saved state/wrist figures
    python -m tests.tendon_hand.traj_5f_point
    # headless: render the animation off-screen to a GIF and save all figures
    python -m tests.tendon_hand.traj_5f_point -SF
"""

import os
import argparse
import time

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index)
from .scene import GRASP_FLEXOR_TENSION, TENDON_NAMES
from .utils import FingerTraj
from .._plotting.trajectory_plotter import plot_trajectory, plot_hand_wrist_trajectory
from ..tendon_finger.utils import PlannerLogger, log_planner_parameters

# Per-finger world-frame tip-position goals (order = config order: index, middle,
# ring, pinky, thumb). These are the fully-converged terminal fingertip positions
# from results/grasp_traj_capsule_K10/grasp_trajectory.log (surface gap ~0 with the
# wrist at identity), so they are known-reachable point-to-point targets.
GOAL_POSITIONS = np.array([
    [-0.04813853,  0.12550637, -0.00818868],  # index
    [-0.06403976,  0.10639372, -0.02463397],  # middle
    [-0.04877054,  0.12505107, -0.04776692],  # ring
    [-0.04164243,  0.12383119, -0.06981800],  # pinky
    [-0.06472072,  0.06869271, -0.03323478],  # thumb
])


def experiment_label(args):
    """Per-experiment label, e.g. p2p_hand_K10 -- names the results directory that
    collects this run's log, figures, frames and GIF."""
    return f"p2p_hand_K{args.steps}"


class _FingerSol:
    """Adapter exposing (.marginals, .meta) as the hand plotter expects."""
    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a K-step five-finger tendon-hand point-to-point trajectory "
                    "with per-finger tip-position goals and GP priors on the wrist "
                    "pose and finger tensions.")
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
                             "the trajectory then closes toward the goal positions.")
    parser.add_argument("--goal-cov", type=float, default=1e-5,
                        help="Per-finger goal-position prior variance (diag of "
                             "goal_position_cov). Smaller => the terminal tips are "
                             "pulled more tightly onto the goal points.")
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

    logger = PlannerLogger("point_to_point_hand", log_dir=results_dir, timestamp=False)
    try:
        _main(args, results_dir)
    finally:
        logger.close()


def _main(args, results_dir):
    exp_label = experiment_label(args)

    # --- Five fingers on the shared wrist (same builder as the grasp test).
    #     No contact/SDF is attached: the terminal goal is a per-finger position. ---
    dims = load_hand_dimensions()
    configs = get_default_hand_configs(dims)
    tip_radii = default_hand_tip_radii(dims)
    finger_names = [name for name, _ in configs]

    if len(configs) != len(GOAL_POSITIONS):
        raise ValueError(
            f"GOAL_POSITIONS has {len(GOAL_POSITIONS)} rows but there are "
            f"{len(configs)} fingers; they must match (order index..thumb).")

    # --- Trajectory planner config (identical start conditions to the grasp test) ---
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

    # Per-finger terminal tip-position goals (world frame). This replaces the
    # SDF contact-as-goal of the grasp test; with no contact the solve stays on
    # the plain (non-AL) path, so no AL parameters are needed.
    plan_config.goal_positions = [GOAL_POSITIONS[i] for i in range(len(configs))]
    plan_config.goal_position_cov = args.goal_cov * np.eye(3)

    plan_config.base.linear_solver_type = "MULTIFRONTAL_QR"

    log_planner_parameters(plan_config, extras={
        "num_tendons": num_tendons,
        "goal_positions": GOAL_POSITIONS,
    })

    planner = crest_sparse.TendonHandTrajectoryPlanner(configs, plan_config)
    print(f"Built hand point-to-point planner: {planner.num_fingers()} fingers, "
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

    # Measured k=0 state: the known hand opening. Flexor at --start-flexor (open),
    # passives at their background hold, all pinned tightly (same as grasp test).
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

    # Per-step summary: wrist proxy (finger-0 base node translation tracks the
    # wrist) and the worst tip-to-goal distance across fingers.
    print("\n step |   wrist-proxy (finger0 base xyz)        | max tip->goal (m)")
    print("------+-----------------------------------------+------------------")
    prev_base = None
    for k, hand_m in enumerate(result.trajectory):
        base_pose = np.array(hand_m.fingers[0].rod.states[0].pose.mean)
        base_xyz = base_pose[:3, 3]
        step_disp = (0.0 if prev_base is None
                     else float(np.linalg.norm(base_xyz - prev_base)))
        prev_base = base_xyz

        dists = []
        for i, fm in enumerate(hand_m.fingers):
            tip_pos = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
            dists.append(float(np.linalg.norm(tip_pos - GOAL_POSITIONS[i])))
        tag = "  <- start" if k == 0 else ("  <- goal" if k == K else "")
        print(f"  {k:>3} | [{base_xyz[0]:+.4f} {base_xyz[1]:+.4f} {base_xyz[2]:+.4f}] "
              f"d={step_disp:.4f} | {max(dists):.5f}{tag}")

    # Terminal goal check (per-finger tip-to-goal distance should be ~0).
    print("\nTerminal (k=K) per-finger tip-to-goal distances:")
    for i, (name, _) in enumerate(configs):
        fm = result.trajectory[K].fingers[i]
        tip_pos = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
        dist = float(np.linalg.norm(tip_pos - GOAL_POSITIONS[i]))
        print(f"  [{name:>6}] tip {tip_pos}  goal {GOAL_POSITIONS[i]}  "
              f"|  dist {dist:.5f} m")

    print("\nFactor error summary (type, count, total_error):")
    for name, count, err in planner.get_factor_error_summary()[:8]:
        print(f"  {err:12.4g}  x{count:<4} {name}")

    # --- Save the state / trajectory figures (headless-safe; always saved). ---
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
        camera_focal_point=list(GOAL_POSITIONS.mean(axis=0)),
        camera_distance=0.5,
    )
    if args.save_figures:
        # Frames -> results/<exp>/frames/{k}.png, GIF -> results/<exp>/<exp>.gif.
        plotter_kwargs["save_frames_dir_name"] = "frames"
        plotter_kwargs["frames_base_dir"] = results_dir

    plotter = TendonHandPlotter(finger_names, **plotter_kwargs)
    plotter.plotter.plotter.add_text("hand point-to-point trajectory",
                                     position="upper_left", font_size=12)
    # Scatter the per-finger goal points as visual targets (best-effort).
    try:
        plotter.plotter.plotter.add_points(
            GOAL_POSITIONS, color="red", point_size=14, render_points_as_spheres=True)
    except Exception as exc:  # pragma: no cover - viz-only convenience
        print(f"(could not render goal points: {exc})")

    for k, hand_m in enumerate(result.trajectory):
        solutions = {name: _FingerSol(fm, result.meta)
                     for name, fm in zip(finger_names, hand_m.fingers)}
        print(f"Displaying step {k}/{K}")
        plotter.update(solutions)
        if not args.save_figures:
            time.sleep(args.dt)

    if args.save_figures:
        plotter.plotter.save_video(fps=args.fps, name=exp_label)
        print(f"Saved experiment results to {results_dir}/")
        return

    input("Press Enter to close...")


if __name__ == "__main__":
    main()
