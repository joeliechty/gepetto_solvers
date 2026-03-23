"""
Demo script: Two-finger grasp of a rolling cylinder using SDF contact.

This demonstrates:
1. Setting up a two-finger hand (index + thumb)
2. Loading an SDF object (cylinder) with rolling constraints
3. Simulating a pinch grasp where the cylinder can roll/rotate

Usage:
    python -m python.tests.tendon_hand.test_roll_cylinder

Requirements:
    - A cylinder SDF in OpenVDB format (.vdb). You can generate one using
      OpenVDB's levelSetSphere or levelSetCapsule tools, or export from
      Blender/Houdini.
"""
import sys
import os
import numpy as np
import time

import crest_sparse
from .._plotting.tendon_hand_plotter import TendonHandPlotter
from .config import get_hand_config


def compute_pinch_tensions(finger_name, frame_idx, num_tendons=6):
    """Compute tensions for a pinching motion.

    Both fingers flex inward together to pinch an object.
    """
    background_tension = 0.2
    tensions = np.full(num_tendons, background_tension)

    # Gradual increase to pinch, then hold
    phase = 0.02 * frame_idx
    pinch_force = min(2.0, 0.5 + 0.01 * frame_idx)  # Ramp up to 2.5N

    # Tendon 5 is the flexor (at 180 deg)
    tensions[5] = background_tension + pinch_force * (0.5 * np.sin(phase) + 0.5)

    return tensions


def main(vdb_path=None, PLOT=True):
    """Run the cylinder rolling demo.

    Parameters
    ----------
    vdb_path : str, optional
        Path to the cylinder SDF file (.vdb). If None, runs without object
        contact (for testing the setup).
    PLOT : bool
        Whether to show visualization.
    """
    print("Setting up two-finger grasp demo...")
    start_time = time.time()

    # 1. Setup two opposing fingers (Index and Thumb facing each other)
    configs = get_hand_config(num_fingers=1, thumb_side="left")  # Index + left thumb
    finger_names = [name for name, _ in configs]
    print(f"Fingers: {finger_names}")

    # Create hand solver
    hand_solver_config = crest_sparse.TendonHandSolverConfig()
    hand_solver = crest_sparse.TendonHandSolver(configs, hand_solver_config)

    print(f"Initialized hand solver with {hand_solver.num_fingers()} fingers in "
          f"{time.time() - start_time:.2f} seconds.")

    # 2. Setup the Rolling Cylinder Object (if VDB path provided)
    if vdb_path is not None and os.path.exists(vdb_path):
        print(f"Loading SDF from: {vdb_path}")

        # Rolling constraint sigmas (order: RotX, RotY, RotZ, X, Y, Z)
        # Cylinder long axis is Y, rolls along X
        locked = 1e-4   # Very stiff (locked DOF)
        free = 1e4      # Very soft (free DOF)
        rolling_sigmas = np.array([
            locked,  # RotX: locked (no tipping)
            free,    # RotY: free (can spin on long axis)
            locked,  # RotZ: locked (no tipping)
            free,    # X: free (can roll forward/back)
            locked,  # Y: locked (no sliding sideways)
            locked   # Z: locked (stays on surface)
        ])

        # Place cylinder between the two fingers
        initial_pose = np.eye(4)
        initial_pose[0:3, 3] = [0.0, 0.06, 0.0]  # In front of palm, centered

        # Set the object with SDF contact
        hand_solver.set_object(vdb_path, initial_pose, rolling_sigmas)
        print("Object loaded and contact model initialized.")
    else:
        if vdb_path is not None:
            print(f"WARNING: VDB file not found: {vdb_path}")
        print("Running without object contact (testing hand motion only).")

    # 3. Setup visualization
    if PLOT:
        plotter = TendonHandPlotter(
            finger_names,
            single_plot_mode=False,
            plot_backbone_ellipsoids=False,
            camera_azimuth=180,
            camera_elevation=15,
            camera_focal_point=[0, 0.06, 0],
            camera_distance=0.3,
            window_size=(1200, 1200),
            plot_collision_spheres=True,
        )

    # 4. Setup noise models
    num_tendons = 6
    tensions_cov = (1e-2) ** 2 * np.eye(num_tendons)
    tip_wrench_mean = np.zeros(6)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    print(f"Setup complete in {time.time() - start_time:.2f} seconds.")
    print("Starting simulation loop...")

    # 5. Simulation Loop
    loop_start = time.time()
    num_iters = 500

    for i in range(num_iters):
        if i % 50 == 0:
            print(f"Iteration {i}/{num_iters}")

        # Collect tensions and tip wrenches for all fingers
        all_tensions = []
        all_tip_wrenches = []

        for finger_idx, name in enumerate(finger_names):
            tensions_mean = compute_pinch_tensions(name, i)
            tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
            tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

            all_tensions.append(tensions)
            all_tip_wrenches.append(tip_wrench)

        # Solve - this now includes SDF contact factors if object was set
        solution = hand_solver.solve(all_tensions, all_tip_wrenches)

        # Update visualization
        if PLOT:
            solutions = {}
            for name, finger_marginals in zip(finger_names, solution.marginals.fingers):
                finger_solution = crest_sparse.TendonRobotSolution()
                finger_solution.marginals = finger_marginals
                finger_solution.meta = solution.meta
                solutions[name] = finger_solution
            plotter.update(solutions)

    elapsed = time.time() - loop_start
    print(f"Completed {num_iters} iterations in {elapsed:.2f} seconds "
          f"({num_iters/elapsed:.1f} Hz)")


if __name__ == "__main__":
    # You can pass a VDB path as command line argument
    vdb_path = sys.argv[1] if len(sys.argv) > 1 else None

    main(
        vdb_path=vdb_path,
        PLOT=True
    )
