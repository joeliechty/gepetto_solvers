import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation


def _extract_trajectory_data(trajectory):
    """Extract arrays from a list of marginals."""
    K = len(trajectory)
    tc = trajectory[0].tendon_config
    disc_pose_idx = list(tc.disc_pose_idx)
    num_discs = tc.num_discs

    # Tensions: (K, num_tendons)
    tensions = np.array([m.tensions.mean for m in trajectory])

    # Disc positions and orientations: (K, num_discs, 3) and (K, num_discs, 3)
    disc_positions = np.zeros((K, num_discs, 3))
    disc_euler_xyz = np.zeros((K, num_discs, 3))
    for k, m in enumerate(trajectory):
        for d in range(num_discs):
            T = m.rod.states[disc_pose_idx[d]].pose.mean
            disc_positions[k, d] = T[:3, 3]
            disc_euler_xyz[k, d] = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)

    # Internal wrenches at disc nodes: (K, num_discs, 6) — [moments(0:3), forces(3:6)]
    internal_wrenches = np.zeros((K, num_discs, 6))
    for k, m in enumerate(trajectory):
        for d in range(num_discs):
            internal_wrenches[k, d] = m.rod.states[disc_pose_idx[d]].wrench.mean

    # External wrenches at disc nodes: (K, num_discs, 6)
    external_wrenches = np.zeros((K, num_discs, 6))
    for k, m in enumerate(trajectory):
        for d in range(num_discs):
            external_wrenches[k, d] = m.external_wrenches[d].mean

    return {
        "tensions": tensions,
        "disc_positions": disc_positions,
        "disc_euler_xyz": disc_euler_xyz,
        "internal_wrenches": internal_wrenches,
        "external_wrenches": external_wrenches,
        "num_discs": num_discs,
        "num_tendons": tensions.shape[1],
        "K": K,
    }


def plot_trajectory(result, tendon_names=None, show=True, save_path=None):
    """
    Plot the full hand state across all trajectory steps.

    Parameters
    ----------
    result : trajectory planner result
        result.trajectory is a list of marginals, one per time step.
    tendon_names : list of str, optional
        Names for each tendon. Defaults to ["T0", "T1", ...].
    show : bool
        Whether to call plt.show() at the end.
    """
    data = _extract_trajectory_data(result.trajectory)
    K = data["K"]
    steps = np.arange(K)
    num_discs = data["num_discs"]
    num_tendons = data["num_tendons"]

    if tendon_names is None:
        tendon_names = [f"T{i}" for i in range(num_tendons)]

    disc_labels = [f"D{d}" for d in range(num_discs)]

    # Color maps for disc lines and tendon lines
    disc_cmap = plt.cm.viridis(np.linspace(0, 1, num_discs))
    tendon_cmap = plt.cm.tab10(np.linspace(0, 1, max(num_tendons, 2)))

    fig = plt.figure(figsize=(18, 28))
    fig.suptitle("Tendon Finger Trajectory", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ------------------------------------------------------------------ #
    # Row 0: Tendon tensions (full width)                                 #
    # ------------------------------------------------------------------ #
    ax_tensions = fig.add_subplot(gs[0, :])
    for i in range(num_tendons):
        ax_tensions.plot(steps, data["tensions"][:, i], label=tendon_names[i],
                         color=tendon_cmap[i], linewidth=1.5)
    ax_tensions.set_title("Tendon Tensions")
    ax_tensions.set_xlabel("Step")
    ax_tensions.set_ylabel("Tension (N)")
    ax_tensions.legend(ncol=num_tendons, fontsize=8, loc="upper right")
    ax_tensions.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Row 1: Disc tip positions (x, y, z)                                 #
    # ------------------------------------------------------------------ #
    pos_labels = ["X (m)", "Y (m)", "Z (m)"]
    for col, lbl in enumerate(pos_labels):
        ax = fig.add_subplot(gs[1, col])
        for d in range(num_discs):
            ax.plot(steps, data["disc_positions"][:, d, col],
                    color=disc_cmap[d], linewidth=1.2,
                    label=disc_labels[d] if col == 0 else None)
        ax.set_title(f"Disc Position {lbl}")
        ax.set_xlabel("Step")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    fig.axes[1].legend(ncol=2, fontsize=7, loc="best")

    # ------------------------------------------------------------------ #
    # Row 2: Disc tip orientations (roll, pitch, yaw)                     #
    # ------------------------------------------------------------------ #
    euler_labels = ["Roll (°)", "Pitch (°)", "Yaw (°)"]
    for col, lbl in enumerate(euler_labels):
        ax = fig.add_subplot(gs[2, col])
        for d in range(num_discs):
            ax.plot(steps, data["disc_euler_xyz"][:, d, col],
                    color=disc_cmap[d], linewidth=1.2)
        ax.set_title(f"Disc Orientation {lbl}")
        ax.set_xlabel("Step")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Rows 3–4: Internal wrenches (forces then moments) at disc nodes     #
    # ------------------------------------------------------------------ #
    int_force_labels = ["Internal Fx (N)", "Internal Fy (N)", "Internal Fz (N)"]
    int_moment_labels = ["Internal Mx (N·m)", "Internal My (N·m)", "Internal Mz (N·m)"]

    for col, lbl in enumerate(int_force_labels):
        ax = fig.add_subplot(gs[3, col])
        for d in range(num_discs):
            ax.plot(steps, data["internal_wrenches"][:, d, 3 + col],
                    color=disc_cmap[d], linewidth=1.2,
                    label=disc_labels[d] if col == 0 else None)
        ax.set_title(lbl)
        ax.set_xlabel("Step")
        ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)
    fig.axes[7].legend(ncol=2, fontsize=7, loc="best")

    for col, lbl in enumerate(int_moment_labels):
        ax = fig.add_subplot(gs[4, col])
        for d in range(num_discs):
            ax.plot(steps, data["internal_wrenches"][:, d, col],
                    color=disc_cmap[d], linewidth=1.2)
        ax.set_title(lbl)
        ax.set_xlabel("Step")
        ax.set_ylabel("Moment (N·m)")
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Rows 5–6: External wrenches (forces then moments) at disc nodes     #
    # ------------------------------------------------------------------ #
    ext_force_labels = ["External Fx (N)", "External Fy (N)", "External Fz (N)"]
    ext_moment_labels = ["External Mx (N·m)", "External My (N·m)", "External Mz (N·m)"]

    for col, lbl in enumerate(ext_force_labels):
        ax = fig.add_subplot(gs[5, col])
        for d in range(num_discs):
            ax.plot(steps, data["external_wrenches"][:, d, 3 + col],
                    color=disc_cmap[d], linewidth=1.2,
                    label=disc_labels[d] if col == 0 else None)
        ax.set_title(lbl)
        ax.set_xlabel("Step")
        ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)
    fig.axes[13].legend(ncol=2, fontsize=7, loc="best")

    for col, lbl in enumerate(ext_moment_labels):
        ax = fig.add_subplot(gs[6, col])
        for d in range(num_discs):
            ax.plot(steps, data["external_wrenches"][:, d, col],
                    color=disc_cmap[d], linewidth=1.2)
        ax.set_title(lbl)
        ax.set_xlabel("Step")
        ax.set_ylabel("Moment (N·m)")
        ax.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved trajectory plot to: {save_path}")

    if show:
        plt.show()

    return fig
