import os
import numpy as np
import time
import pyvista as pv
import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter
from .._plotting.trajectory_plotter import plot_trajectory
from .config import get_6tendon_config



def main():
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
    planner_config.K = 10
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

    # 3. Build the environment (cylinder SDF object)
    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, "sphere.vdb"))

    # Sphere: radius 0.005 m (see _objects/make_sphere.py).  No rotation needed
    # (sphere is symmetric).  Centered at the p2p goal position so we know the
    # finger can reach it.
    object_pose = np.eye(4)
    object_pose[0:3, 3] = [6.02088876e-02, 3.77734425e-02, 0.0]

    env = crest_sparse.EnvironmentConfig()
    env.load_sdf(vdb_path)
    env.object_pose_mean = object_pose
    env.object_pose_cov = 1e-8 * np.eye(6)   # rigidly anchored
    env.object_pose_per_step = False

    # Collision running cost on the disc nodes.
    env.collision_epsilon = 0.002             # 2 mm safety margin
    env.collision_sigma = 1e-3
    # env.collision_node_indices = disc_node_indices
    # env.collision_node_radii = [0.003] * len(disc_node_indices)
    env.collision_node_indices = []
    env.collision_node_radii = []

    # Terminal contact: tip sphere must land on the cylinder's surface.
    tip_radius = 0.003
    env.target_contact_node = tip_node_index
    env.contact_node_radius = tip_radius
    env.contact_cov = np.diag([1e-6, 1e-6])

    planner_config.environment = env

    # 4. Plan
    print("Building factor graph...")
    planner = crest_sparse.TendonFingerTrajectoryPlanner(planner_config)

    print("Planning contact trajectory...")
    start_time = time.time()
    result = planner.plan()
    elapsed = time.time() - start_time

    print(f"Solved in {elapsed:.2f}s | iters={result.meta.iterations} | error={result.meta.error:.4g}")
    print(f"  build={result.meta.build_time_ms:.0f}ms  opt={result.meta.optimize_time_ms:.0f}ms  "
          f"marginals={result.meta.marginalize_time_ms:.0f}ms")

    # 5. Verify the terminal tip ends up on the cylinder surface.
    #    The cylinder's local-frame SDF is zero on its surface; tip-center distance
    #    to the surface should equal the configured contact node radius.
    final_tip_pose = np.array(result.trajectory[-1].rod.states[-1].pose.mean)
    tip_world = final_tip_pose[0:3, 3]
    tip_in_obj = np.linalg.inv(object_pose) @ np.append(tip_world, 1.0)
    tip_in_obj = tip_in_obj[:3]
    # Sphere SDF: distance from center minus radius.
    sdf_at_tip = np.linalg.norm(tip_in_obj) - 0.005  # sphere radius 0.005 m
    print(f"\nContact check:")
    print(f"  Tip world pos:   {tip_world}")
    print(f"  Tip in obj frame:{tip_in_obj}")
    print(f"  SDF at tip:      {sdf_at_tip:.5f} m  (target = contact radius {tip_radius:.5f})")
    print(f"  Residual:        {sdf_at_tip - tip_radius:+.5f} m")

    # 6. Animate
    plotter = TendonFingerPlotter(
        single_plot_mode=False,
        plot_backbone_frames=True,
        plot_tip_force=True,
        plot_backbone_ellipsoids=True,
        camera_azimuth=180,
        camera_elevation=-90,
        camera_focal_point=[0, 0.1, 0]
    )

    # Show the target sphere in the scene.
    sphere_radius = 0.005
    sphere_mesh = pv.Sphere(radius=sphere_radius, center=object_pose[0:3, 3])
    plotter.plotter.plotter.add_mesh(sphere_mesh, color='cadmiumyellow', opacity=0.5, smooth_shading=True)

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

        plotter.update(sol)
        time.sleep(planner_config.dt)

    input("Press Enter to close...")

    # 7. Plot full trajectory state
    tendon_names = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]
    plot_trajectory(result, tendon_names=tendon_names, save_path="point_to_contact_trajectory.png")


if __name__ == "__main__":
    main()
