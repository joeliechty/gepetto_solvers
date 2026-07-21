"""Live kinematics visualization of a full tendon *hand* (four fingers + thumb).

This is the multi-finger analogue of ``tendon_finger/kinematics_test.py``: it
animates the whole hand while sweeping (a) the per-finger flexor tendon tension
and (b) the shared wrist base-pose prior, so you can see where all the fingers
are and how they move relative to each other as the wrist moves.

Finger morphology (bone lengths, palm origins/angles) comes from the
``gepetto_core`` default hand configuration when that package is installed, and
falls back to hard-coded ``DEFAULT_HAND_DIMENSIONS`` otherwise, so this runs
standalone. No contact is attached -> a pure-kinematics solve driven by tensions.

The wrist prior mean starts at ``TendonHandSolverConfig.wrist_pose`` but is
re-commanded each frame through ``solver.set_wrist_pose(T)``, which re-aims the
shared wrist prior *without* rebuilding the solver. Because the solver retains
its ``values_`` across ``solve()`` calls, each frame **warm-starts** from the
previous solution rather than cold-starting from a straight hand — only the
first solve pays the cold-start cost, and every subsequent frame is a small
nudge that converges in a handful of iterations even at high flexor tension.

Run (from the ``python/`` directory):
    python -m tests.tendon_hand.fk_5f_sweep
"""

import time

import numpy as np

import crest_sparse

from .config import get_default_hand_configs
from .._plotting.tendon_hand_plotter import TendonHandPlotter


def _wrist_pose(i):
    """Commanded wrist pose at iteration ``i``: a small translation + tilt sweep."""
    # Sinusoidal translation (meters) and tilt (radians) so the hand visibly
    # translates and reorients while the fingers curl.
    tx = 0.02 * np.sin(0.01 * i)
    tz = 0.02 * np.sin(0.013 * i + np.pi / 2)
    tilt = np.deg2rad(20.0) * np.sin(0.007 * i)

    c, s = np.cos(tilt), np.sin(tilt)
    T = np.eye(4)

    # Tilt about the world X axis.
    T[:3, :3] = np.array([[1.0, 0.0, 0.0],
                          [0.0, c, -s],
                          [0.0, s,  c]])
    T[:3, 3] = [tx, 0.0, tz]
    return T


class _FingerSol:
    """Duck-typed per-finger solution the TendonHandPlotter consumes."""
    pass


def main():
    configs = get_default_hand_configs()
    finger_names = [name for name, _ in configs]
    num_tendons = configs[0][1].num_tendons  # 6

    hand_config = crest_sparse.TendonHandSolverConfig()
    hand_config.base.linear_solver_type = "MULTIFRONTAL_QR"
    # Only the first frame is a cold start (straight hand -> curled at full
    # tension); every frame after warm-starts from the previous solution, so this
    # headroom is really only needed once.
    hand_config.base.max_iterations = 500
    # Start the wrist prior at the frame-0 commanded pose; subsequent frames
    # re-aim it via solver.set_wrist_pose() without rebuilding.
    hand_config.wrist_pose = _wrist_pose(0)
    # Tight prior: the wrist rigidly follows the commanded pose.
    hand_config.sigma_wrist_pos = 1e-4
    hand_config.sigma_wrist_rot = 1e-3

    plotter = TendonHandPlotter(
        finger_names,
        plot_backbone_ellipsoids=False,
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0.0, 0.05, 0.0],
        camera_distance=0.5,
    )

    # Moderate, uniform prior on every tendon so each tension variable is well
    # constrained (a tight-passive/loose-flexor prior is underdetermined without
    # contact -> IndeterminantLinearSystem on the tension variable Q).
    tensions_cov = (1e-2) ** 2 * np.eye(num_tendons)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    # Passive tendons held at a background tension; the flexor (index 5) oscillates
    # to curl the finger. A per-finger phase offset makes the digits curl in a
    # wave so you can see them move relative to one another.
    background_tension = 0.5
    flexor_amplitude = 0.75  # peak flexor ~= background + 2*amplitude = 2.0 N
    finger_phases = np.linspace(0.0, np.pi, len(configs))

    # Build the solver ONCE. Rebuilding each frame would discard the retained
    # solution and force a straight-hand cold start every time; instead we keep
    # this instance and re-command the wrist each frame (warm-started sweep).
    solver = crest_sparse.TendonHandSolver(configs, hand_config)

    start_time = time.time()
    num_iters = 10000

    for i in range(num_iters):
        # --- Command the shared wrist pose (warm-start; no solver rebuild) ---
        wrist_pose = _wrist_pose(i)
        solver.set_wrist_pose(wrist_pose)

        # --- Per-finger tendon tensions (staggered flexor sweep) ---
        all_tensions = []
        flexors = []
        for phase in finger_phases:
            tensions_mean = np.full(num_tendons, background_tension)
            flexor = background_tension + flexor_amplitude * (np.cos(0.01 * i - np.pi + phase) + 1.0)
            tensions_mean[5] = flexor
            flexors.append(flexor)
            all_tensions.append(
                crest_sparse.VectorXGaussian(tensions_mean, tensions_cov))
        all_tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                            for _ in configs]

        solution = solver.solve(all_tensions, all_tip_wrenches)

        # --- Feed the plotter one shim per finger ---
        solutions = {}
        for name, fm in zip(finger_names, solution.marginals.fingers):
            s = _FingerSol()
            s.marginals = fm
            s.meta = solution.meta
            solutions[name] = s
        plotter.update(solutions)

        if i % 50 == 0:
            tip = np.array(solution.marginals.fingers[0].rod.states[-1].pose.mean)
            print(f"Iteration {i}/{num_iters} | iters={solution.meta.iterations} "
                  f"err={solution.meta.error:.3g}")
            print(f"  Flexor tensions: {np.round(flexors, 2)}")
            print(f"  Wrist pos: {wrist_pose[:3, 3]}")
            print(f"  {finger_names[0]} tip pos: {tip[:3, 3]}")

    end_time = time.time()
    print(f"Completed {num_iters} iterations in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
