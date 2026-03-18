import sys
import numpy as np

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
        phase = 0.01 * frame_idx - np.pi
    elif mode == "wave":
        phase = 0.01 * frame_idx - np.pi + 0.5 * finger_idx
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Thumbs get a slight extra phase offset
    if "thumb" in finger_name:
        phase += 0.5

    # Tendon 5 is the active flexor (at 180 deg), oscillates 0.5 to 2.5 N
    tensions[5] = background_tension + 1.0 * (np.cos(phase) + 1)

    return tensions


def main(mode="wave"):
    configs = get_hand_config(num_fingers=4, thumb_side="left")
    finger_names = [name for name, _ in configs]

    solvers = {name: crest_sparse.TendonRobotSolver(cfg) for name, cfg in configs}

    plotter1 = TendonHandPlotter(
        finger_names,
        single_plot_mode=False,
        plot_backbone_ellipsoids=False,
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.08, 0],
        camera_distance=0.6,
    )

    plotter2 = TendonHandPlotter(
        finger_names,
        single_plot_mode=False,
        plot_backbone_ellipsoids=False,
        camera_azimuth=15,
        camera_elevation=20,
        camera_focal_point=[0, 0.08, 0],
        camera_distance=0.6,
    )

    num_tendons = 6
    tensions_cov = (1e-2) ** 2 * np.eye(num_tendons)
    tip_wrench_mean = np.zeros(6)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    for i in range(1000):
        solutions = {}

        for finger_idx, name in enumerate(finger_names):
            tensions_mean = compute_tensions(name, finger_idx, i, mode)

            tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
            tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

            solutions[name] = solvers[name].solve(tensions, tip_wrench, None)

        plotter1.update(solutions)
        plotter2.update(solutions)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "wave"
    main(mode=mode)
