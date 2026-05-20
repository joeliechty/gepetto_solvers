import os
import numpy as np
import time
import pyvista as pv
import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter
from .._plotting.trajectory_plotter import plot_trajectory
from .config import get_6tendon_config
from .utils import PlannerLogger, log_planner_parameters
import argparse


def main():
    logger = PlannerLogger("point_to_contact")
    try:
        _main()
    finally:
        logger.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_steps", "-K", type=int, default=10, help="Number of timesteps")
    return parser.parse_args()

def _main():

    args = parse_args()
    # 1. Setup base model config
    model_config = get_6tendon_config()
    num_tendons = model_config.num_tendons  # 6

    # The finger has num_discs=9 with num_between_nodes=3 -> 33 rod nodes total.
    num_discs = model_config.num_discs
    num_between_nodes = model_config.num_between_nodes
    num_nodes = num_discs + (num_discs - 1) * num_between_nodes
    tip_node_index = num_nodes - 1
    # Disc nodes only — cheaper than collision-checking every interior node.
    disc_node_indices = [i * (num_between_nodes + 1) for i in range(num_discs)]

    # 2. Setup planner config
    planner_config = crest_sparse.TrajectoryPlannerConfig()
    planner_config.model_config = model_config
    planner_config.model_config.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
    # planner_config.model_config.base.optimizer_type = "LM"

    planner_config.model_config.base.delta_initial = 1.0
    # GTSAM's default per-call cap is 100; on the contact problem every
    # continuation stage hits that cap mid-descent. Bump so we can tell
    # whether the optimizer is iter-limited vs actually converged.
    planner_config.model_config.base.max_iterations = 500
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
    # initializes its rod state along the natural curl manifold rather than
    # straight-from-rest. Without this the optimizer can't cross the rod-stress
    # hump from a straight initial guess and converges to a local minimum with
    # the tip 5-8 cm off the sphere (see point_to_contact_*.log K-sweep).
    # get_initial_values() interpolates t_k = t_start + (k/K)*(t_goal - t_start)
    # across timesteps, so this also seeds intermediate k's tendon tensions.
    # The cov is tight on passive tendons (lock them at 0.5) and loose on the
    # flexor so the optimizer can refine its terminal value freely.
    planner_config.goal_tensions = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    planner_config.goal_tensions_cov = np.diag(
        [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])

    # 3. Build the environment (cylinder SDF object)
    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, "sphere.vdb"))

    # Sphere: radius 0.025 m (see _objects/make_sphere.py).  No rotation needed
    # (sphere is symmetric).  Centered at the p2p goal position so we know the
    # finger can reach it.
    object_pose = np.eye(4)
    object_pose[0:3, 3] = [6.02088876e-02, 3.77734425e-02, 0.0]

    env = crest_sparse.EnvironmentConfig()
    env.load_sdf(vdb_path)
    env.object_pose_mean = object_pose
    env.object_pose_cov = 1e-8 * np.eye(6)   # rigidly anchored
    env.object_pose_per_step = False

    # Collision running cost on the disc nodes AND the tip. The tip needs its
    # own collision sphere because the terminal contact factor's e1/e2 rows
    # (Eq 26) are surface-equality with no side preference — a tip *inside*
    # the object can satisfy them by placing p_c behind the tip. The
    # collision factor's one-sided hinge supplies the missing sign at every
    # k (intermediate + terminal). collision_sigma tightened from 1e-3 to
    # 1e-4 so info per row (1e8) sits above the final-stage contact factor
    # (1e6) and below the start anchor (1e12).
    tip_radius = 0.003
    env.collision_epsilon = 0.002             # 2 mm safety margin
    env.collision_sigma = 1e-4
    env.collision_node_indices = disc_node_indices + [tip_node_index]
    env.collision_node_radii = [0.003] * len(disc_node_indices) + [tip_radius]

    # Terminal contact: tip sphere must land on the cylinder's surface.
    # The first contact_cov here seeds the planner; later stages tighten it
    # via set_contact_cov() in the continuation loop below.
    # contact_cov is 3x3: rows 0-1 weight the dummy-point equality
    # (e1, e2 in SdfContactFactor); row 2 weights the tip-side non-penetration
    # hinge e3.
    env.target_contact_node = tip_node_index
    env.contact_node_radius = tip_radius
    contact_cov_stages = [1e-3, 1e-4, 1e-5, 1e-6]
    env.contact_cov = np.diag([contact_cov_stages[0]] * 3)

    planner_config.environment = env

    log_planner_parameters(
        planner_config,
        environment=env,
        extras={
            "planner_hz": planner_hz,
            "num_tendons": num_tendons,
            "num_discs": num_discs,
            "num_between_nodes": num_between_nodes,
            "num_nodes": num_nodes,
            "tip_node_index": tip_node_index,
            "disc_node_indices": disc_node_indices,
            "tip_radius": tip_radius,
            "contact_cov_stages": contact_cov_stages,
            "object_pose": object_pose,
            "vdb_path": vdb_path,
        },
    )

    # 4. Plan with contact-cov continuation. Solving cold at the tight cov
    # collapses Dogleg's trust region (huge residual / σ at iter 0). Warm-
    # starting from looser stages avoids that — values_ is reused across
    # plan() calls, only the contact factor's covariance changes.
    print("Building factor graph...")
    planner = crest_sparse.TendonFingerTrajectoryPlanner(planner_config)

    sphere_world_radius = 0.025  # matches _objects/sphere.vdb
    obj_pose_inv = np.linalg.inv(object_pose)

    def tip_sdf_at_step(traj, k):
        tip_pose = np.array(traj[k].rod.states[-1].pose.mean)
        tip_w = tip_pose[0:3, 3]
        tip_in_obj_ = (obj_pose_inv @ np.append(tip_w, 1.0))[:3]
        return np.linalg.norm(tip_in_obj_) - sphere_world_radius

    print("Planning contact trajectory (with continuation)...")
    start_time = time.time()
    for stage, cov_val in enumerate(contact_cov_stages):
        planner.set_contact_cov(np.diag([cov_val] * 3))
        result = planner.plan()
        per_step = [tip_sdf_at_step(result.trajectory, k)
                    for k in range(planner_config.K + 1)]
        worst_k = int(np.argmin(per_step))
        print(f"  stage {stage}: contact_cov={cov_val:.0e} | "
              f"iters={result.meta.iterations} | error={result.meta.error:.4g} | "
              f"tip_sdf[K]={per_step[-1]:+.5f} | "
              f"worst_sdf={per_step[worst_k]:+.5f}@k={worst_k}")
    elapsed = time.time() - start_time

    print(f"Solved in {elapsed:.2f}s | iters={result.meta.iterations} | error={result.meta.error:.4g}")
    print(f"  build={result.meta.build_time_ms:.0f}ms  opt={result.meta.optimize_time_ms:.0f}ms  "
          f"marginals={result.meta.marginalize_time_ms:.0f}ms")

    # Per-factor-type residual breakdown — diagnoses which factor type
    # contributes the bulk of result.meta.error. A well-conditioned graph
    # should not have a single factor type dominating by orders of magnitude.
    summary = planner.get_factor_error_summary()
    print(f"\nTop factor types by total error (sum of factor->error(values)):")
    for name, count, total in summary[:8]:
        print(f"  {total:11.4g}  ({count:5d} factors)  {name}")

    # 5. Verify the terminal tip ends up on the sphere surface and no
    #    intermediate step penetrates.
    def tip_sdf_at(step):
        tip_pose = np.array(result.trajectory[step].rod.states[-1].pose.mean)
        tip_w = tip_pose[0:3, 3]
        tip_in_obj_ = (obj_pose_inv @ np.append(tip_w, 1.0))[:3]
        return np.linalg.norm(tip_in_obj_) - sphere_world_radius, tip_w

    sdf_at_tip, tip_world = tip_sdf_at(-1)
    tip_in_obj = (obj_pose_inv @ np.append(tip_world, 1.0))[:3]
    print(f"\nContact check:")
    print(f"  Tip world pos:   {tip_world}")
    print(f"  Tip in obj frame:{tip_in_obj}")
    print(f"  SDF at tip:      {sdf_at_tip:.5f} m  (target = contact radius {tip_radius:.5f})")
    print(f"  Residual:        {sdf_at_tip - tip_radius:+.5f} m")

    # Per-step penetration check. sdf - tip_radius < 0 means the tip sphere
    # is intersecting the object.
    per_step = [tip_sdf_at(k)[0] for k in range(planner_config.K + 1)]
    worst_step = int(np.argmin(per_step))
    worst_sdf = per_step[worst_step]
    print(f"  Worst per-step SDF (tip center to surface): {worst_sdf:+.5f} m at k={worst_step}")
    print(f"  Worst tip-sphere penetration depth: {max(0.0, tip_radius - worst_sdf) * 1000:.3f} mm")

    # 6. Animate
    # Visualize the same disc collision spheres the planner is now actually
    # constraining.
    viz_collision_indices = disc_node_indices + [tip_node_index]
    viz_collision_radii = [0.003] * len(disc_node_indices) + [tip_radius]

    common_plotter_kwargs = dict(
        single_plot_mode=False,
        plot_backbone_frames=True,
        plot_tip_force=True,
        plot_backbone_ellipsoids=True,
        collision_node_indices=viz_collision_indices,
        collision_node_radii=viz_collision_radii,
        contact_node_index=tip_node_index,
        contact_node_radius=tip_radius,
        camera_focal_point=[0, 0.1, 0],
    )

    side_plotter = TendonFingerPlotter(
        camera_azimuth=180,
        camera_elevation=-90,
        **common_plotter_kwargs,
    )
    top_plotter = TendonFingerPlotter(
        camera_azimuth=180,
        camera_elevation=0,
        **common_plotter_kwargs,
    )
    bottom_plotter = TendonFingerPlotter(
        camera_azimuth=180,
        camera_elevation=90,
        **common_plotter_kwargs,
    )
    front_plotter = TendonFingerPlotter(
        camera_azimuth=90,
        camera_elevation=0,
        **common_plotter_kwargs,
    )
    plotters = [side_plotter, top_plotter, bottom_plotter, front_plotter]

    # Show the target sphere in both scenes.
    sphere_radius = 0.025
    for p in plotters:
        sphere_mesh = pv.Sphere(radius=sphere_radius, center=object_pose[0:3, 3])
        p.plotter.plotter.add_mesh(sphere_mesh, color='cadmiumyellow', opacity=0.5, smooth_shading=True)

    for k, marginals in enumerate(result.trajectory):
        print(f"Displaying Step {k}/{planner_config.K}")
        print(f"  Tensions mean: {marginals.tensions.mean}")
        print(f"  Tendon lengths: {marginals.tendon_lengths}")
        print(f"  Tip pose mean:\n{marginals.rod.states[-1].pose.mean}")

        class MockSolution:
            pass
        sol = MockSolution()
        sol.marginals = marginals
        sol.meta = result.meta

        for p in plotters:
            p.update(sol)
        time.sleep(planner_config.dt)

    input("Press Enter to close...")

    # 7. Plot full trajectory state
    tendon_names = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]
    plot_trajectory(result, tendon_names=tendon_names, save_path="point_to_contact_trajectory.png")


if __name__ == "__main__":
    main()
