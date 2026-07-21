"""Plan a *collision-free point-to-point* five-finger hand trajectory.

This is the collision-aware sibling of ``traj_5f_point.py``
and the point-goal sibling of ``traj_5f_contact_collision.py``.
The hand starts open at an identity wrist and is planned over K steps with:

  * per-finger world-frame **tip-position goals** at k=K -- soft
    ``PositionPriorFactor``s (the planner's ``goal_positions``/``goal_position_cov``),
    so the fingers reach toward *points* rather than onto a contact surface.
  * **collision avoidance** at every plannable step -- Section 1.5 AL inequality
    constraints keeping each finger out of the object and the fingers apart.
  * the usual Gaussian-process temporal priors tying the steps together (wrist
    pose BetweenFactor<Pose3>, per-finger tension BetweenFactor<Vec>).

Because collision turns the solve onto the Augmented-Lagrangian path, the linear
solver MUST be Cholesky (QR stalls on the AntiFactor's negated Hessian) -- unlike
the no-collision point-to-point test, which uses QR.

The goal points (``GRASP_GOALS``, shared via scene.py) are the
*collision-free* terminal fingertip positions from the collision+contact grasp
solve on the big sphere -- the hand wraps the sphere with every backbone node
held outside it, so these points are reachable with the whole finger clear of the
object (a free-space flexor curl instead spears the main fingers straight through
the sphere, which no collision-free posture can match).

Everything a run produces (log, per-finger state figures, wrist-trajectory
figure, AL-convergence figure, animation frames + GIF) lands in
``results/<experiment>/``.

Run (from the ``python/`` directory):
    python -m tests.tendon_hand.traj_5f_point_collision -SF
    python -m tests.tendon_hand.traj_5f_point_collision --no-viz
"""

import os
import argparse
import time

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, load_hand_dimensions, attach_collision)
from .scene import (
    get_primitive_specs, primitive_surface_gap, GRASP_SPHERE_CENTER,
    GRASP_GOALS, TENDON_NAMES)
from .utils import collision_report, FingerTraj
from .._plotting.trajectory_plotter import (
    plot_trajectory, plot_hand_wrist_trajectory)
from ..tendon_finger.utils import PlannerLogger, log_planner_parameters

# Target flexor tension at k>=1 (loose prior; the goal priors do the closing).
BACKGROUND_FLEXOR = 2.0


class _FingerSol:
    """Adapter exposing (.marginals, .meta) as the hand plotter expects."""
    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta


class _HandStepShim:
    """Wraps a per-step hand marginals as ``.marginals`` so collision_report(),
    which reads ``solution.marginals.fingers``, can be reused per step."""
    def __init__(self, hand_marginals):
        self.marginals = hand_marginals


def experiment_label(args):
    return f"collision_p2p_hand_{args.primitive}_K{args.steps}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primitive", nargs="?", default="big_sphere")
    parser.add_argument("-K", "--steps", type=int, default=10,
                        help="Number of trajectory steps (creates K+1 states).")
    parser.add_argument("--dt", type=float, default=0.1, help="Step duration (s).")
    parser.add_argument("--gp-wrist", type=float, default=1e-2,
                        help="Wrist-pose GP process-noise scale.")
    parser.add_argument("--gp-tense", type=float, default=1.0,
                        help="Tension GP process-noise scale.")
    parser.add_argument("--sigma-wrist-pos", type=float, default=1e-4)
    parser.add_argument("--sigma-wrist-rot", type=float, default=1e-3)
    parser.add_argument("--start-flexor", type=float, default=0.5,
                        help="Measured flexor tension (N) at k=0 (0.5 = open hand).")
    parser.add_argument("--goal-cov", type=float, default=1e-5,
                        help="Per-finger goal-position prior variance.")
    # Collision / AL knobs (mirror five_finger_hand_collision_trajectory_test.py).
    parser.add_argument("--collision-radius", type=float, default=0.003)
    parser.add_argument("--collision-sigma", type=float, default=1.0)
    parser.add_argument("--cull-margin", type=float, default=0.02,
                        help="Drop finger-finger sphere pairs whose initial gap "
                             "exceeds this margin (speedup; None-like<0 = keep all).")
    parser.add_argument("--al-mu", type=float, default=1.0)
    parser.add_argument("--al-rate", type=float, default=2.0)
    parser.add_argument("--al-iters", type=int, default=40)
    parser.add_argument("--num-fingers", type=int, default=0,
                        help="Use only the first N digits (0 = all five).")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--save-figures", "-SF", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.join("results", experiment_label(args))
    os.makedirs(results_dir, exist_ok=True)
    logger = PlannerLogger("collision_point_to_point_hand",
                           log_dir=results_dir, timestamp=False)
    try:
        _main(args, results_dir)
    finally:
        logger.close()


def _main(args, results_dir):
    exp_label = experiment_label(args)

    spec = get_primitive_specs()[args.primitive]
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[:3, :3] = object_rotation
    object_pose[:3, 3] = GRASP_SPHERE_CENTER

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{args.primitive} (run from python/).")

    dims = load_hand_dimensions()
    r = args.collision_radius

    # --- Reachable, collision-free grasp goal points (see GRASP_GOALS). ---
    n_use = args.num_fingers if args.num_fingers > 0 else len(GRASP_GOALS)
    goals = [GRASP_GOALS[i] for i in range(n_use)]
    print(f"Using {len(goals)} collision-free grasp goal points.")

    # --- Five fingers on the shared wrist, with collision attached. ---
    configs = get_default_hand_configs(dims)
    if args.num_fingers > 0:
        configs = configs[:args.num_fingers]
    finger_names = [name for name, _ in configs]
    num_tendons = configs[0][1].num_tendons

    if len(goals) != len(configs):
        raise ValueError(f"{len(goals)} goals for {len(configs)} fingers.")

    # Pin the object (sigma 1e-6/dof) so the fingers do all the moving.
    attach_collision(configs, vdb_path, object_pose,
                     radius=r, sigma=args.collision_sigma,
                     object_pose_cov=1e-12 * np.eye(6),
                     cull_margin=(args.cull_margin if args.cull_margin >= 0 else None))

    # --- Trajectory planner config. ---
    plan_config = crest_sparse.TendonHandTrajectoryPlannerConfig()
    plan_config.K = args.steps
    plan_config.dt = args.dt
    plan_config.wrist_pose = np.eye(4)
    plan_config.sigma_wrist_pos = args.sigma_wrist_pos
    plan_config.sigma_wrist_rot = args.sigma_wrist_rot
    plan_config.gp_wrist_Qc = args.gp_wrist * np.eye(6)
    plan_config.gp_tense_Qc = args.gp_tense * np.eye(num_tendons)
    plan_config.gp_len_Qc = np.zeros((0, 0))

    plan_config.goal_positions = [np.asarray(g, dtype=float) for g in goals]
    plan_config.goal_position_cov = args.goal_cov * np.eye(3)

    # Cholesky is REQUIRED with collision (AL inequality constraints); QR stalls.
    plan_config.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
    plan_config.base.al_initial_mu = args.al_mu
    plan_config.base.al_mu_increase_rate = args.al_rate
    plan_config.base.al_max_iterations = args.al_iters

    log_planner_parameters(plan_config, extras={
        "primitive": args.primitive,
        "num_fingers": len(configs),
        "num_tendons": num_tendons,
        "object_pose": object_pose,
    })

    planner = crest_sparse.TendonHandTrajectoryPlanner(configs, plan_config)
    print(f"Built collision point-to-point planner: {planner.num_fingers()} "
          f"fingers, K={args.steps} steps.")

    # Background/target tensions at k>=1: passives pinned, flexor loose (so the
    # goal priors AND collision can drive it). Loose flexor is essential: a tight
    # prior makes penetration the merit minimum and the AL loop quits at iters=1.
    tensions_mean = np.array([0.5] * (num_tendons - 1) + [BACKGROUND_FLEXOR])
    tensions_cov = np.diag([1e-6] * (num_tendons - 1) + [1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    all_tensions = [crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
                    for _ in configs]
    all_tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                        for _ in configs]

    # Measured k=0 state: open hand (flexor slack), all pinned tightly.
    start_mean = np.array([0.5] * (num_tendons - 1) + [args.start_flexor])
    start_cov = np.diag([1e-6] * num_tendons)
    all_start_tensions = [crest_sparse.VectorXGaussian(start_mean, start_cov)
                          for _ in configs]

    t0 = time.time()
    result = planner.plan(all_tensions, all_tip_wrenches,
                          start_tensions=all_start_tensions)
    dt_s = time.time() - t0
    K = args.steps
    print(f"Planned in {dt_s:.1f} s | iters={result.meta.iterations} | "
          f"error={result.meta.error:.4g} | {len(result.trajectory)} states")

    # --- Per-step summary: object clearance + worst tip->goal distance. ---
    print("\n step | worst obj clearance | worst finger-finger gap | max tip->goal")
    print("------+---------------------+-------------------------+--------------")
    for k, hand_m in enumerate(result.trajectory):
        obj_c, ff_g = collision_report(configs, _HandStepShim(hand_m),
                                       spec, object_pose, r)
        dists = []
        for i, fm in enumerate(hand_m.fingers):
            tip = np.array(fm.rod.states[-1].pose.mean)[:3, 3]
            dists.append(float(np.linalg.norm(tip - goals[i])))
        tag = "  <- start" if k == 0 else ("  <- goal" if k == K else "")
        print(f"  {k:>3} | {obj_c:+.5f} m         | {ff_g:+.5f} m             "
              f"| {max(dists):.5f}{tag}")

    # Worst over plannable steps (k>=1; k=0 is the unconstrained measured start).
    worst_obj = min(collision_report(configs, _HandStepShim(hm), spec,
                                     object_pose, r)[0]
                    for hm in result.trajectory[1:])
    print("\nTerminal (k=K) per-finger tip-to-goal distances:")
    term_dists = []
    for i, (name, _) in enumerate(configs):
        tip = np.array(result.trajectory[K].fingers[i].rod.states[-1].pose.mean)[:3, 3]
        d = float(np.linalg.norm(tip - goals[i]))
        term_dists.append(d)
        print(f"  [{name:>6}] dist {d:.5f} m")

    clearance_ok = worst_obj >= -1e-4
    # Goals are collision-free grasp tips; a collision-constrained tip reaches
    # them to within ~the collision radius.
    reach_tol = r + 3e-3
    reach_ok = max(term_dists) <= reach_tol
    print(f"\nWORST plannable object clearance (k>=1): {worst_obj:+.5f} m")
    print("RESULT:", "PASS" if (clearance_ok and reach_ok) else "FAIL",
          f"(collision-free: {clearance_ok}, reached within "
          f"{reach_tol*1e3:.1f} mm: {reach_ok})")

    print("\nFactor error summary (type, count, total_error):")
    for name, count, err in planner.get_factor_error_summary()[:8]:
        print(f"  {err:12.4g}  x{count:<4} {name}")

    # --- Save state / wrist figures (headless-safe; always saved). ---
    print("\nSaving trajectory figures...")
    for i, name in enumerate(finger_names):
        finger_traj = FingerTraj([hm.fingers[i] for hm in result.trajectory])
        plot_trajectory(
            finger_traj, tendon_names=TENDON_NAMES, show=False,
            save_path=os.path.join(results_dir, f"{exp_label}_states_{name}.png"))
    plot_hand_wrist_trajectory(
        result, configs[0][1].hand_base_offset, dt=args.dt, show=False,
        save_path=os.path.join(results_dir, f"{exp_label}_wrist.png"))

    if args.no_viz:
        print(f"Saved experiment results to {results_dir}/")
        return

    from .._plotting.tendon_hand_plotter import TendonHandPlotter

    plotter_kwargs = dict(
        plot_backbone_ellipsoids=False,
        camera_azimuth=165, camera_elevation=20,
        camera_focal_point=list(GRASP_SPHERE_CENTER),
        camera_distance=0.5,
    )
    if args.save_figures:
        plotter_kwargs["save_frames_dir_name"] = "frames"
        plotter_kwargs["frames_base_dir"] = results_dir

    plotter = TendonHandPlotter(finger_names, **plotter_kwargs)
    plotter.plotter.plotter.add_text("collision-free point-to-point trajectory",
                                     position="upper_left", font_size=12)
    try:
        plotter.plotter.plotter.add_points(
            np.array(goals), color="red", point_size=14,
            render_points_as_spheres=True)
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
