import os
import time
import numpy as np

import crest_sparse
from .._plotting.state_estimation_plotter import plot_estimation_sweep
from .._plotting.tendon_finger_plotter import TendonFingerPlotter
from .config import get_6tendon_config


class _MockMeta:
    def __init__(self, total_ms=0.0):
        self.iterations = 0
        self.error = 0.0
        self.build_time_ms = 0.0
        self.optimize_time_ms = total_ms
        self.marginalize_time_ms = 0.0
        self.extract_time_ms = 0.0
        self.total_time_ms = total_ms


class _MockSolution:
    pass


def load_planner_trajectory(npz_path, control_hz):
    """Load interpolated tendon-length trajectory saved by point_to_point_planning.py."""
    data = np.load(os.path.expanduser(npz_path))
    lengths = data["trajectory"]  # (N, num_tendons)
    t = np.arange(len(lengths)) / control_hz
    return t, lengths


def make_bend_signal(bend_hz, end_angle_deg, t_total):
    """Synthesize a linearly-ramped bend sensor reading from 0° to end_angle_deg."""
    t = np.arange(0.0, t_total, 1.0 / bend_hz)
    angles = np.deg2rad(np.linspace(0.0, end_angle_deg, len(t)))
    return t, angles


def merge_event_streams(t_lengths, lengths, t_bend, bend_angles):
    """
    Merge asynchronous length and bend events into a single time-sorted stream.
    Returns list of tuples (timestamp, kind, payload) where kind ∈ {"length","bend"}.
    """
    events = []
    for ti, L in zip(t_lengths, lengths):
        events.append((float(ti), "length", L))
    for ti, a in zip(t_bend, bend_angles):
        events.append((float(ti), "bend", float(a)))
    events.sort(key=lambda e: e[0])
    return events


def build_estimator_config(num_tendons):
    """Base solver config + GP smoothness priors for the estimator."""
    base = get_6tendon_config()
    base.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY"  # FOR APPLE

    ec = crest_sparse.TendonFingerEstimatorConfig()
    ec.base_config = base

    # Background tension prior — same values used by the planner:
    # passive tendons (0-4) held at 0.5 N, active tendon (5) left free.
    bg_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    bg_sigmas = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e6])
    ec.background_tensions_mean = bg_mean
    ec.background_tensions_cov = np.diag(bg_sigmas ** 2)

    # GP smoothness on tensions (matches planner).
    ec.gp_tense_Qc = np.eye(num_tendons) * 1e-2
    # Smoothness on tendon lengths (same order of magnitude as planner).
    ec.gp_len_Qc = np.eye(num_tendons) * 1e-5
    # Light smoothness on disc poses.
    ec.gp_pose_Qc = np.eye(6) * 1e-6
    return ec


def run_single_sweep(estimator_config, t_lengths, lengths,
                     bend_hz, end_angle_deg, bend_sigma, length_sigma,
                     plotter=None):
    """Run the iterative solver for one (bend_hz, end_angle_deg) configuration."""
    num_tendons = lengths.shape[1]
    t_total = float(t_lengths[-1]) + 1e-9
    t_bend, bend_angles = make_bend_signal(bend_hz, end_angle_deg, t_total)
    events = merge_event_streams(t_lengths, lengths, t_bend, bend_angles)

    solver = crest_sparse.TendonFingerIterativeSolver(
        config=estimator_config,
        bend_sigma=bend_sigma,
    )

    length_cov = np.eye(num_tendons) * (length_sigma ** 2)
    length_cov[-1, -1] = 1e-8  # very confident measurement for active tendon length

    ts_recorded, lengths_est, tip_pos, tip_pos_cov = [], [], [], []

    t_start = time.time()
    for ts, kind, payload in events:
        if kind == "length":
            meas = crest_sparse.VectorXGaussian(payload.astype(np.float64), length_cov)
            t_step = time.perf_counter()
            solver.step(timestamp_sec=ts, lengths_meas=meas)
            step_ms = (time.perf_counter() - t_step) * 1e3
            # Record state after each length event (aligned with planner 100 Hz grid).
            marginals = solver.get_current_marginals()
            tip_T = marginals.rod.states[-1].pose.mean
            tip_cov6 = marginals.rod.states[-1].pose.cov
            ts_recorded.append(ts)
            lengths_est.append(np.array(marginals.tendon_lengths))
            tip_pos.append(tip_T[:3, 3].copy())
            tip_pos_cov.append(tip_cov6[3:6, 3:6].copy())  # translation block of tangent cov

            if plotter is not None:
                sol = _MockSolution()
                sol.marginals = marginals
                sol.meta = _MockMeta(total_ms=step_ms)
                plotter.update(sol)
        else:  # "bend"
            solver.step(timestamp_sec=ts, measured_bend=payload)

    elapsed = time.time() - t_start

    return {
        "t": np.asarray(ts_recorded),
        "tendon_lengths": np.asarray(lengths_est),
        "tip_pos": np.asarray(tip_pos),
        "tip_pos_cov": np.asarray(tip_pos_cov),
        "t_bend": t_bend,
        "bend_angles": bend_angles,
        "wall_time_sec": elapsed,
    }


def main(bend_freqs_hz=[200, 300, 400],
         end_angles_deg=[0, 45, 90],
         traj_path="~/git_repos/underactuated_hand/interpolated_trajectory.npz",
         control_hz=100,
         bend_sigma=1e-2,
         length_sigma=1e-4,
         save_path="state_estimation_sweep.png"):

    print(f"Loading planner trajectory from {traj_path} ...")
    t_lengths, lengths = load_planner_trajectory(traj_path, control_hz)
    print(f"  Loaded {len(lengths)} samples @ {control_hz} Hz "
          f"(t_end = {t_lengths[-1]:.3f}s, num_tendons = {lengths.shape[1]})")

    num_tendons = lengths.shape[1]
    estimator_config = build_estimator_config(num_tendons)

    plotter = TendonFingerPlotter(
        plot_tip_force=False,
        plot_backbone_frames=True,
        plot_backbone_ellipsoids=True,
        camera_azimuth=180,
        camera_elevation=-90,
        camera_focal_point=[0, 0.1, 0],
    )

    results = {}
    for bend_hz in bend_freqs_hz:
        for end_angle in end_angles_deg:
            print(f"\n--- Running estimator: bend={bend_hz}Hz, end_angle={end_angle}° ---")
            out = run_single_sweep(
                estimator_config, t_lengths, lengths,
                bend_hz=bend_hz, end_angle_deg=end_angle,
                bend_sigma=bend_sigma, length_sigma=length_sigma,
                plotter=plotter,
            )
            print(f"  wall time: {out['wall_time_sec']:.2f}s  "
                  f"({len(out['t'])} recorded states)")
            tip_final = out["tip_pos"][-1]
            tip_std = np.sqrt(np.diag(out["tip_pos_cov"][-1]))
            print(f"  final tip pos: [{tip_final[0]:+.4f}, {tip_final[1]:+.4f}, {tip_final[2]:+.4f}] m")
            print(f"  final tip ±σ:  [{tip_std[0]:.2e}, {tip_std[1]:.2e}, {tip_std[2]:.2e}] m")
            results[(bend_hz, end_angle)] = out

    print("\nPlotting sweep results ...")
    plot_estimation_sweep(results, lengths, t_lengths, save_path=save_path)


if __name__ == "__main__":
    bend_freqs_hz=[200]
    end_angles_deg=[0, -45, -90]

    time.sleep(5.0)

    main(bend_freqs_hz=bend_freqs_hz, end_angles_deg=end_angles_deg)
