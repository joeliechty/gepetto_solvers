import sys
import numpy as np
import time

import crest_sparse
from .._plotting.tendon_hand_plotter import TendonHandPlotter

from .config import get_hand_config


def compute_tensions(finger_name, finger_idx, frame_idx, mode, num_tendons=6):
    """Compute tendon tensions for one finger at one timestep.

    Parameters
    ----------
    finger_name : str
        Name of the finger (e.g. "index", "thumb_right").
    finger_idx : int
        Order index used for wave phase offset.
    frame_idx : int
        Animation frame counter.
    mode : str
        "sync" -- all fingers flex together.
        "wave" -- sequential wave from index to pinky.
    num_tendons : int
        Number of tendons per finger (default 6).
    """
    background_tension = 0.5
    tensions = np.full(num_tendons, background_tension)

    if mode == "sync":
        phase = 0.01 * frame_idx + np.pi * 5./6.
    elif mode == "wave":
        phase = 0.01 * frame_idx + np.pi * 5./6. + 0.5 * finger_idx
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Thumbs get a slight extra phase offset
    if "thumb" in finger_name:
        phase += 0.5

    # Tendon 5 is the active flexor (at 180 deg), oscillates 0.5 to 2.5 N
    tensions[5] = background_tension + 1.50 * (np.cos(phase) + 1)

    return tensions


def main(mode="wave", PLOT=False):

    print(f"Running tendon hand test in '{mode}' mode. Press Ctrl+C to exit.")
    start_time = time.time()
    configs = get_hand_config(num_fingers=4, thumb_side="left")
    finger_names = [name for name, _ in configs]

    end_time = time.time()
    print(f"Loaded configs for {len(configs)} fingers in {end_time - start_time:.2f} seconds.")
    start_time = end_time

    # Create combined hand solver with all finger configurations
    hand_solver_config = crest_sparse.TendonHandSolverConfig()
    hand_solver = crest_sparse.TendonHandSolver(configs, hand_solver_config)

    end_time = time.time()
    print(f"Initialized hand solver with {hand_solver.num_fingers()} fingers in {end_time - start_time:.2f} seconds.")
    start_time = end_time

    plotter1 = TendonHandPlotter(
        finger_names,
        single_plot_mode=False,
        plot_backbone_ellipsoids=False,
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.08, 0],
        camera_distance=0.6,
        window_size=(1200, 1200),
        plot_collision_spheres=True,
    )

    plotter2 = TendonHandPlotter(
        finger_names,
        single_plot_mode=False,
        plot_backbone_ellipsoids=False,
        camera_azimuth=15,
        camera_elevation=20,
        camera_focal_point=[0, 0.08, 0],
        camera_distance=0.6,
        window_size=(1200, 1200),
        plot_collision_spheres=True,
    )

    num_tendons = 6
    tensions_cov = (1e-2) ** 2 * np.eye(num_tendons)
    tip_wrench_mean = np.zeros(6)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    end_time = time.time()
    print(f"Initialized plotters and covariance matrices in {end_time - start_time:.2f} seconds.")
    start_time = end_time

    # inital solve with no tensions to show initial pose
    all_tensions = [crest_sparse.VectorXGaussian(np.zeros(num_tendons), tensions_cov) for _ in finger_names]
    all_tip_wrenches = [crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov) for _ in finger_names]
    initial_solution = hand_solver.solve(all_tensions, all_tip_wrenches)

    for i in range(1000):
        if i % 50 == 0:
            print(f"Iteration {i}/1000")

        # Collect tensions and tip wrenches for all fingers
        all_tensions = []
        all_tip_wrenches = []

        for finger_idx, name in enumerate(finger_names):
            tensions_mean = compute_tensions(name, finger_idx, i, mode)
            tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
            tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

            all_tensions.append(tensions)
            all_tip_wrenches.append(tip_wrench)

        # Single solve for all fingers at once
        solution = hand_solver.solve(all_tensions, all_tip_wrenches)

        # Convert solution to dict format for plotter
        # Wrap each finger's marginals back into a Solution-like object
        solutions = {}
        for name, finger_marginals in zip(finger_names, solution.marginals.fingers):
            finger_solution = crest_sparse.TendonRobotSolution()
            finger_solution.marginals = finger_marginals
            finger_solution.meta = solution.meta
            solutions[name] = finger_solution

        if PLOT:
            plotter1.update(solutions)
            plotter2.update(solutions)

    end_time = time.time()
    print(f"Completed 1000 iterations in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":

    main(
        # mode="sync",
        mode="wave",
        PLOT=True
        )
