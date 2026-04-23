import numpy as np
import matplotlib.pyplot as plt


def _freq_color(bend_hz, all_freqs):
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_freqs)))
    return cmap[all_freqs.index(bend_hz)]


def _angle_linestyle(end_angle_deg, all_angles):
    styles = ["-", "--", ":", "-."]
    return styles[all_angles.index(end_angle_deg) % len(styles)]


def plot_estimation_sweep(results, planned_lengths, t_lengths,
                          tendon_names=None, show=True, save_path=None):
    """
    Plot state estimation results across a sweep of (bend_hz, end_angle_deg) configurations.

    Parameters
    ----------
    results : dict[(bend_hz, end_angle_deg)] -> dict with keys:
        - "t":               (M,) timestamps at which marginals were recorded
        - "tendon_lengths":  (M, num_tendons) estimated lengths
        - "tip_pos":         (M, 3) estimated tip position
        - "tip_pos_cov":     (M, 3, 3) tip position covariance
        - "t_bend":          (Mb,) timestamps of bend sensor readings
        - "bend_angles":     (Mb,) injected bend sensor readings (rad)
    planned_lengths : (N, num_tendons) planned tendon lengths from the npz
    t_lengths : (N,) timestamps of planned lengths
    """
    freqs = sorted({k[0] for k in results})
    angles = sorted({k[1] for k in results})
    num_tendons = planned_lengths.shape[1]
    if tendon_names is None:
        tendon_names = [f"T{i}" for i in range(num_tendons)]

    # --- Figure 1: Tendon length tracking ---
    fig1, axes1 = plt.subplots(num_tendons, 1, figsize=(12, 2.0 * num_tendons), sharex=True)
    if num_tendons == 1:
        axes1 = [axes1]
    for i, ax in enumerate(axes1):
        ax.plot(t_lengths, planned_lengths[:, i], color="black",
                linewidth=1.2, alpha=0.6, label="Planned" if i == 0 else None)
        for (bend_hz, end_angle), d in results.items():
            ax.plot(d["t"], d["tendon_lengths"][:, i],
                    color=_freq_color(bend_hz, freqs),
                    linestyle=_angle_linestyle(end_angle, angles),
                    linewidth=0.9, alpha=0.8,
                    label=f"{bend_hz}Hz/{end_angle}°" if i == 0 else None)
        ax.set_ylabel(f"{tendon_names[i]} (m)")
        ax.grid(True, alpha=0.3)
    axes1[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes1[-1].set_xlabel("Time (s)")
    fig1.suptitle("Estimated Tendon Lengths vs. Planned", fontsize=12, fontweight="bold")
    fig1.tight_layout()

    # --- Figure 2: Bend sensor input streams ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    for (bend_hz, end_angle), d in results.items():
        ax2.plot(d["t_bend"], np.rad2deg(d["bend_angles"]),
                 color=_freq_color(bend_hz, freqs),
                 linestyle=_angle_linestyle(end_angle, angles),
                 linewidth=0.9, alpha=0.8,
                 label=f"{bend_hz}Hz, end={end_angle}°")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Bend angle (deg)")
    ax2.set_title("Synthetic Knuckle Bend Sensor Input")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, ncol=3)

    # --- Figure 3: Tip position with uncertainty bands ---
    pos_labels = ["X (m)", "Y (m)", "Z (m)"]
    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for col, lbl in enumerate(pos_labels):
        ax = axes3[col]
        for (bend_hz, end_angle), d in results.items():
            color = _freq_color(bend_hz, freqs)
            ls = _angle_linestyle(end_angle, angles)
            mean = d["tip_pos"][:, col]
            sigma = np.sqrt(np.maximum(d["tip_pos_cov"][:, col, col], 0.0))
            ax.plot(d["t"], mean, color=color, linestyle=ls, linewidth=1.0,
                    label=f"{bend_hz}Hz/{end_angle}°" if col == 0 else None)
            ax.fill_between(d["t"], mean - sigma, mean + sigma,
                            color=color, alpha=0.08)
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    axes3[0].legend(fontsize=7, ncol=3, loc="best")
    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle("Estimated Tip Position (±1σ shaded)", fontsize=12, fontweight="bold")
    fig3.tight_layout()

    if save_path is not None:
        base = save_path.rsplit(".", 1)[0]
        ext = save_path.rsplit(".", 1)[1] if "." in save_path else "png"
        fig1.savefig(f"{base}_lengths.{ext}", dpi=200, bbox_inches="tight")
        fig2.savefig(f"{base}_bend.{ext}",    dpi=200, bbox_inches="tight")
        fig3.savefig(f"{base}_tip.{ext}",     dpi=200, bbox_inches="tight")
        print(f"Saved estimation sweep plots to: {base}_{{lengths,bend,tip}}.{ext}")

    if show:
        plt.show()

    return fig1, fig2, fig3
