import os
import numpy as np
import time
import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter
from .._plotting.trajectory_plotter import plot_trajectory
from ..tendon_finger.config import get_6tendon_config
from .utils import PlannerLogger, log_planner_parameters
# Shared primitive registry + analytic surface-gap helper (see scene.py).
from .scene import OBJECT_CENTER, get_primitive_specs, primitive_surface_gap
import argparse


def experiment_label(args):
    """Per-experiment label, e.g. point_to_contact_sphere_sdf_K10. Used to name
    the results directory that collects this run's log, frames, and figures."""
    return (f"point_to_contact_{args.primitive}_{args.contact_mode}"
            f"_K{args.traj_steps}")


def main():
    args = parse_args()
    if args.contact_mode == "analytic-sphere" and args.primitive != "sphere":
        raise SystemExit("--contact-mode analytic-sphere is only valid with the sphere primitive.")

    # Everything this run produces (log, animation frames, GIF, tension plot)
    # lands in a single per-experiment results directory.
    results_dir = os.path.join("results", experiment_label(args))
    os.makedirs(results_dir, exist_ok=True)

    # timestamp=False: the results dir already encodes the experiment params, so
    # re-running the same config overwrites its log rather than accumulating one
    # per run.
    logger = PlannerLogger("point_to_contact", log_dir=results_dir, timestamp=False)
    try:
        _main(args, results_dir)
    finally:
        logger.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plan a tendon-finger trajectory that ends in contact with a "
                    "primitive object, against either its SDF or (for a sphere) an "
                    "analytic sphere primitive.")
    parser.add_argument("primitive", nargs="?", default="sphere",
                        choices=["sphere", "cylinder", "cube"],
                        help="Object primitive to plan contact against.")
    parser.add_argument("--contact-mode", default="sdf",
                        choices=["sdf", "analytic-sphere"],
                        help="sdf: SDF witness contact (any primitive). "
                             "analytic-sphere: closed-form sphere-sphere contact "
                             "(sphere primitive only).")
    parser.add_argument("--traj_steps", "-K", type=int, default=10, help="Number of timesteps")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip the interactive 3D animation (for headless runs / tuning).")
    parser.add_argument("--save-figures", "-SF", action="store_true",
                        help="Render the trajectory animation off-screen and save it to "
                             "results/<experiment>/ as a sequence of frame PNGs plus a GIF "
                             "(no interactive window).")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the saved animation video.")
    parser.add_argument("--use-hand-base", action="store_true",
                        help="Enable the Section 4 hand-base reparameterization: node 0 "
                             "becomes T_0 = T_base o T_offset (offset = Identity here, so "
                             "the geometry and results match the legacy node-0 path).")
    parser.add_argument("--al-mu", type=float, default=1e4, help="AL initial penalty mu")
    parser.add_argument("--al-rate", type=float, default=2.0, help="AL mu increase rate")
    parser.add_argument("--al-iters", type=int, default=40, help="AL max outer iterations")
    return parser.parse_args()


def _main(args, results_dir):
    spec = get_primitive_specs()[args.primitive]
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[0:3, 0:3] = object_rotation
    object_pose[0:3, 3] = OBJECT_CENTER

    # 1. Setup base model config
    model_config = get_6tendon_config()
    num_tendons = model_config.num_tendons  # 6

    # Section 4 hand-base reparameterization. With offset = Identity the finger
    # base T_0 = T_base, so this is a pure correctness/equivalence check against
    # the legacy node-0 path. The planner gives each step its own hand-base
    # variable, anchored per step (Eq 40); the GP-on-T_base of Eq 41-42 is the
    # multi-finger extension and is not needed for this anchored single finger.
    if args.use_hand_base:
        model_config.use_hand_base = True
        model_config.hand_base_offset = np.eye(4)

    num_discs = model_config.num_discs
    num_between_nodes = model_config.num_between_nodes
    num_nodes = num_discs + (num_discs - 1) * num_between_nodes
    tip_node_index = num_nodes - 1
    # Disc nodes only — cheaper than collision-checking every interior node.
    disc_node_indices = [i * (num_between_nodes + 1) for i in range(num_discs)]
    tip_radius = 0.003

    # 2. Setup planner config
    planner_config = crest_sparse.TrajectoryPlannerConfig()
    planner_config.model_config = model_config
    # Leave linear_solver_type at the default MULTIFRONTAL_QR (as the working
    # sdf_3dof_contact_kinematics_test does) — QR is more stable than CHOLESKY on
    # the ill-conditioned witness-point contact system.
    planner_config.model_config.base.delta_initial = 1.0
    planner_config.model_config.base.max_iterations = 500

    # Augmented Lagrangian schedule for the hard terminal contact constraint.
    # The planner's large static objective (~hundreds of rod/prior factors)
    # dwarfs the contact penalty at low mu, so AL's relative-convergence check
    # trips after a handful of outer steps. Starting mu high (~1e4) makes the
    # contact penalty bite in the first inner solve and drives the tip onto the
    # surface (~20 micron terminal gap); mu=1 stalls the tip ~7 cm short.
    planner_config.model_config.base.al_initial_mu = args.al_mu
    planner_config.model_config.base.al_mu_increase_rate = args.al_rate
    planner_config.model_config.base.al_max_iterations = args.al_iters

    planner_config.K = args.traj_steps
    planner_hz = 5
    planner_config.dt = 1.0 / planner_hz

    bg_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    bg_sigmas = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e1])
    planner_config.background_tensions_mean = bg_mean
    planner_config.background_tensions_sigmas = bg_sigmas

    planner_config.gp_tense_Qc = np.eye(num_tendons) * 1e-2
    planner_config.gp_len_Qc = np.eye(num_tendons) * 1e-4

    planner_config.tension_limit_alpha = 10.0
    planner_config.tension_limit_q_min = 0.0
    planner_config.active_tendon_indices = [5]

    # Start position (same as point_to_point_planning).
    planner_config.start_position = np.array([1.42977609e-02, 1.35008944e-01, 0.0])
    planner_config.start_position_cov = np.eye(3) * 1e-6

    # No goal_position / goal_pose — contact-as-goal supersedes them (Eq 30).

    # Warm-start: bias k=K toward a flexed configuration so the planner
    # initializes its rod state along the natural curl manifold. The flexor
    # (index 5) cov is loose so the optimizer refines its terminal value freely.
    planner_config.goal_tensions = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    planner_config.goal_tensions_cov = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")

    log_extras = {
        "planner_hz": planner_hz,
        "num_tendons": num_tendons,
        "num_discs": num_discs,
        "num_nodes": num_nodes,
        "tip_node_index": tip_node_index,
        "tip_radius": tip_radius,
        "primitive": args.primitive,
        "contact_mode": args.contact_mode,
        "object_pose": object_pose,
    }

    # 3. Configure the terminal contact (and, for SDF mode, the collision running
    #    cost) according to the chosen contact mode.
    if args.contact_mode == "sdf":
        vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
        if not os.path.exists(vdb_path):
            raise FileNotFoundError(
                f"{vdb_path} not found. Generate it with "
                f"python -m tests._objects.make_{args.primitive} (run from python/).")
        env = crest_sparse.EnvironmentConfig()
        env.load_sdf(vdb_path)
        env.object_pose_mean = object_pose
        env.object_pose_cov = 1e-8 * np.eye(6)   # rigidly anchored
        env.object_pose_per_step = False
        # Collision constraints on the disc nodes (disc 7 == the tip node), as
        # Section 1.5 AL inequalities (c_pen <= 0) at every step. The planner
        # auto-skips the constraint on the terminal contact node at k=K so it
        # can't fight the contact factor; intermediate steps still keep the
        # finger out of the object. Sigma 1.0 whitens the collision rows the
        # same as the contact constraint rows.
        env.collision_sigma = 1.0
        env.collision_node_indices = disc_node_indices
        env.collision_node_radii = [0.003] * len(disc_node_indices)
        # Terminal contact: tip sphere lands tangent to the SDF surface. Hard AL
        # equality constraint on the 5-residual SdfWitnessContactFactor.
        env.target_contact_node = tip_node_index
        env.contact_node_radius = tip_radius
        planner_config.environment = env
        log_extras["vdb_path"] = vdb_path
        log_planner_parameters(planner_config, environment=env, extras=log_extras)
    else:  # analytic-sphere
        sc = crest_sparse.SpherePrimitiveContactConfig()
        sc.finger_node_index = tip_node_index
        sc.finger_node_radius = tip_radius
        sc.sphere_center = OBJECT_CENTER
        sc.sphere_radius = spec["radius"]
        sc.sphere_pose_cov = 1e-8 * np.eye(6)
        sc.witness = True   # 5-residual witness form, mirrors the SDF contact
        planner_config.sphere_contact = sc
        log_planner_parameters(planner_config, extras=log_extras)

    # 4. Plan. Contact is a hard equality constraint solved by the Augmented
    #    Lagrangian optimizer (auto-enabled by the contact config).
    print(f"Building factor graph ({args.primitive}, {args.contact_mode})...")
    planner = crest_sparse.TendonFingerTrajectoryPlanner(planner_config)

    def tip_gap_at(step):
        """Signed gap between the tip sphere surface and the object surface at a
        trajectory step (analytic, independent of the solver)."""
        tip_pose = np.array(planner_result.trajectory[step].rod.states[-1].pose.mean)
        tip_pos = tip_pose[:3, 3]
        tip_local = object_rotation.T @ (tip_pos - OBJECT_CENTER)
        return primitive_surface_gap(tip_local, spec) - tip_radius, tip_pos

    print("Planning contact trajectory (Augmented Lagrangian)...")
    start_time = time.time()
    planner_result = planner.plan()
    result = planner_result
    elapsed = time.time() - start_time

    per_step = [tip_gap_at(k)[0] for k in range(planner_config.K + 1)]
    worst_k = int(np.argmin(per_step))
    print(f"  iters={result.meta.iterations} | error={result.meta.error:.4g} | "
          f"gap[K]={per_step[-1]:+.6f} | worst_gap={per_step[worst_k]:+.6f}@k={worst_k}")
    print(f"Solved in {elapsed:.2f}s | "
          f"build={result.meta.build_time_ms:.0f}ms opt={result.meta.optimize_time_ms:.0f}ms "
          f"marginals={result.meta.marginalize_time_ms:.0f}ms")

    summary = planner.get_factor_error_summary()
    print("\nTop factor types by total error:")
    for name, count, total in summary[:8]:
        print(f"  {total:11.4g}  ({count:5d} factors)  {name}")

    # 5. Verify terminal contact + no intermediate penetration.
    gap_K, tip_world = tip_gap_at(-1)
    flexor_K = result.trajectory[-1].tensions.mean[5]
    print("\nContact check:")
    print(f"  Tip world pos:        {tip_world}")
    print(f"  Terminal surface gap: {gap_K:+.6f} m  (target ~0)")
    print(f"  Worst per-step gap:   {per_step[worst_k]:+.6f} m at k={worst_k}")
    print(f"  Worst penetration:    {max(0.0, -per_step[worst_k]) * 1000:.3f} mm")
    print(f"  Flexor tension @K:    {flexor_K:.3f}")

    # 6. Animate
    tendon_names = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]
    exp_label = experiment_label(args)

    if args.no_viz:
        plot_trajectory(result, tendon_names=tendon_names,
                        save_path=os.path.join(results_dir, f"{exp_label}_tensions.png"))
        return

    plotter_kwargs = dict(
        plot_backbone_frames=True,
        plot_tip_force=True,
        plot_backbone_ellipsoids=True,
        contact_node_index=tip_node_index,
        contact_node_radius=tip_radius,
        primitives=[dict(spec["plot"](OBJECT_CENTER), color="goldenrod", opacity=0.35)],
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.1, 0],
    )

    if args.save_figures:
        # Frames -> results/<exp_label>/frames/{k}.png, video -> results/<exp_label>/<exp_label>.gif.
        plotter_kwargs["save_frames_dir_name"] = "frames"
        plotter_kwargs["frames_base_dir"] = results_dir

    plotter = TendonFingerPlotter(**plotter_kwargs)

    for k, marginals in enumerate(result.trajectory):
        print(f"Displaying Step {k}/{planner_config.K}  flexor={marginals.tensions.mean[5]:.3f}")

        class MockSolution:
            pass
        sol = MockSolution()
        sol.marginals = marginals
        sol.meta = result.meta
        plotter.update(sol)
        if not args.save_figures:
            time.sleep(planner_config.dt)

    if args.save_figures:
        plotter.plotter.save_video(fps=args.fps, name=exp_label)
        plot_trajectory(result, tendon_names=tendon_names,
                        save_path=os.path.join(results_dir, f"{exp_label}_tensions.png"))
        print(f"Saved experiment results to {results_dir}/")
        return

    input("Press Enter to close...")
    plot_trajectory(result, tendon_names=tendon_names,
                    save_path=os.path.join(results_dir, f"{exp_label}_tensions.png"))


if __name__ == "__main__":
    main()
