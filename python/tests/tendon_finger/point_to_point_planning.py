import numpy as np
import time
import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter
from .._plotting.trajectory_plotter import plot_trajectory
from .config import get_6tendon_config

def main():
    # 1. Setup Base Model Config
    model_config = get_6tendon_config()
    num_tendons = model_config.num_tendons  # 6

    # 2. Setup Planner Config
    planner_config = crest_sparse.TrajectoryPlannerConfig()
    planner_config.model_config = model_config
    planner_config.model_config.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"
    planner_config.model_config.base.delta_initial = 1.0
    planner_config.K = 10       # K+1 = 21 time steps
    planner_config.dt = 0.1

    # Background Tensions: passive tendons (0-4) are tight, active tendon (5) is free
    bg_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    bg_sigmas = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e6])
    planner_config.background_tensions_mean = bg_mean
    planner_config.background_tensions_sigmas = bg_sigmas

    # Initial State
    planner_config.start_tensions = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    planner_config.start_tensions_cov = np.eye(num_tendons) * 1e-8

    # # Goal State (Target Pose for the tip)
    # goal_pose = np.eye(4)
    # goal_pose[0:3, 3] = [0.05, 0.1, 0.0]  # Example translation target
    # planner_config.goal_pose = goal_pose
    # planner_config.goal_pose_cov = np.eye(6) * 1e-4

    # Goal tension
    planner_config.goal_tensions = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.5])
    planner_config.goal_tensions_cov = np.eye(num_tendons) * 1e-4

    # GP Prior Covariance (smoothness of tension changes over time)
    planner_config.gp_Qc = np.eye(num_tendons) * 1e-2

    # Tension limit barrier: only active tendon (index 5) needs the barrier
    planner_config.tension_limit_alpha = 10.0
    planner_config.tension_limit_q_min = 0.0
    planner_config.active_tendon_indices = [5]

    # 3. Initialize and Run Planner
    print("Building and solving factor graph trajectory...")
    planner = crest_sparse.TendonFingerTrajectoryPlanner(planner_config)

    start_time = time.time()
    result = planner.plan()
    elapsed = time.time() - start_time

    print(f"Solved in {elapsed:.2f}s | iters={result.meta.iterations} | error={result.meta.error:.4g}")
    print(f"  build={result.meta.build_time_ms:.0f}ms  opt={result.meta.optimize_time_ms:.0f}ms  "
          f"marginals={result.meta.marginalize_time_ms:.0f}ms")

    # 4. Visualization
    plotter = TendonFingerPlotter(
        single_plot_mode=False,
        plot_backbone_frames=True,
        plot_tip_force=True,
        plot_backbone_ellipsoids=True,
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.1, 0]
    )

    # Animate the resulting trajectory
    for k, marginals in enumerate(result.trajectory):
        print(f"Displaying Step {k}/{planner_config.K}")
        print(f"  Tensions mean: {marginals.tensions.mean}")
        print(f"  Tip pose mean:\n{marginals.rod.states[-1].pose.mean}")

        class MockSolution:
            pass
        sol = MockSolution()
        sol.marginals = marginals
        sol.meta = result.meta

        plotter.update(sol)
        time.sleep(0.5)

    input("Press Enter to close...")

    # 5. Plot full trajectory state
    tendon_names = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]
    plot_trajectory(result, tendon_names=tendon_names, save_path="point_to_point_trajectory.png")

if __name__ == "__main__":
    main()
