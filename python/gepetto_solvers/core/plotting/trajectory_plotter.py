import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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

    # Tendon lengths: (K, num_tendons)
    tendon_lengths = np.array([m.tendon_lengths for m in trajectory])

    return {
        "tensions": tensions,
        "tendon_lengths": tendon_lengths,
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

    fig = plt.figure(figsize=(18, 32))
    # fig.suptitle("Tendon Finger Trajectory", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(8, 3, figure=fig, hspace=0.55, wspace=0.35)

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
    # Row 1: Tendon lengths (full width)                                  #
    # ------------------------------------------------------------------ #
    ax_lengths = fig.add_subplot(gs[1, :])
    for i in range(num_tendons):
        ax_lengths.plot(steps, data["tendon_lengths"][:, i], label=tendon_names[i],
                        color=tendon_cmap[i], linewidth=1.5)
    ax_lengths.set_title("Tendon Lengths")
    ax_lengths.set_xlabel("Step")
    ax_lengths.set_ylabel("Length (m)")
    ax_lengths.legend(ncol=num_tendons, fontsize=8, loc="upper right")
    ax_lengths.grid(True, alpha=0.3)

    # ------------------------------------------------------------------ #
    # Row 2: Disc tip positions (x, y, z)                                 #
    # ------------------------------------------------------------------ #
    pos_labels = ["X (m)", "Y (m)", "Z (m)"]
    for col, lbl in enumerate(pos_labels):
        ax = fig.add_subplot(gs[2, col])
        for d in range(num_discs):
            ax.plot(steps, data["disc_positions"][:, d, col],
                    color=disc_cmap[d], linewidth=1.2,
                    label=disc_labels[d] if col == 0 else None)
        ax.set_title(f"Disc Position {lbl}")
        ax.set_xlabel("Step")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    fig.axes[2].legend(ncol=2, fontsize=7, loc="best")

    # ------------------------------------------------------------------ #
    # Row 2: Disc tip orientations (roll, pitch, yaw)                     #
    # ------------------------------------------------------------------ #
    euler_labels = ["Roll (°)", "Pitch (°)", "Yaw (°)"]
    for col, lbl in enumerate(euler_labels):
        ax = fig.add_subplot(gs[3, col])
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
        ax = fig.add_subplot(gs[4, col])
        for d in range(num_discs):
            ax.plot(steps, data["internal_wrenches"][:, d, 3 + col],
                    color=disc_cmap[d], linewidth=1.2,
                    label=disc_labels[d] if col == 0 else None)
        ax.set_title(lbl)
        ax.set_xlabel("Step")
        ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)
    fig.axes[8].legend(ncol=2, fontsize=7, loc="best")

    for col, lbl in enumerate(int_moment_labels):
        ax = fig.add_subplot(gs[5, col])
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
        ax = fig.add_subplot(gs[6, col])
        for d in range(num_discs):
            ax.plot(steps, data["external_wrenches"][:, d, 3 + col],
                    color=disc_cmap[d], linewidth=1.2,
                    label=disc_labels[d] if col == 0 else None)
        ax.set_title(lbl)
        ax.set_xlabel("Step")
        ax.set_ylabel("Force (N)")
        ax.grid(True, alpha=0.3)
    fig.axes[14].legend(ncol=2, fontsize=7, loc="best")

    for col, lbl in enumerate(ext_moment_labels):
        ax = fig.add_subplot(gs[7, col])
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


def plot_hand_wrist_trajectory(result, hand_base_offset, dt=None,
                               show=True, save_path=None):
    """Plot the shared wrist pose across a hand trajectory.

    The wrist variable is not exposed directly in the per-finger marginals, so we
    recover it from finger 0's base node (rod state 0), whose world pose is
    ``T_wrist o hand_base_offset``. Inverting the fixed offset gives ``T_wrist``.

    Parameters
    ----------
    result : hand trajectory planner result
        ``result.trajectory`` is a list of HandState (one per step),
        each with a ``.digits`` list of per-finger marginals.
    hand_base_offset : (4, 4) array
        Finger 0's ``hand_base_offset`` (config.hand_base_offset), the fixed SE(3)
        transform from the wrist to that finger's base node.
    dt : float, optional
        Step duration; when given the x-axis is time (s) rather than step index.
    """
    offset_inv = np.linalg.inv(np.asarray(hand_base_offset, dtype=float))
    K1 = len(result.trajectory)

    pos = np.zeros((K1, 3))
    eul = np.zeros((K1, 3))
    for k, hand_m in enumerate(result.trajectory):
        base0 = np.asarray(hand_m.digits[0].rod.states[0].pose.mean, dtype=float)
        T_wrist = base0 @ offset_inv
        pos[k] = T_wrist[:3, 3]
        eul[k] = Rotation.from_matrix(T_wrist[:3, :3]).as_euler("xyz", degrees=True)

    x = np.arange(K1) * (dt if dt else 1.0)
    xlabel = "Time (s)" if dt else "Step"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    pos_labels = ["x", "y", "z"]
    eul_labels = ["roll", "pitch", "yaw"]
    cmap = plt.cm.tab10(np.linspace(0, 1, 3))

    for i, lbl in enumerate(pos_labels):
        axes[0].plot(x, pos[:, i], "-o", color=cmap[i], markersize=4, label=lbl)
    axes[0].set_title("Wrist Position")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, lbl in enumerate(eul_labels):
        axes[1].plot(x, eul[:, i], "-o", color=cmap[i], markersize=4, label=lbl)
    axes[1].set_title("Wrist Orientation")
    axes[1].set_ylabel("Angle (°)")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Hand Wrist Trajectory", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved wrist trajectory plot to: {save_path}")
    if show:
        plt.show()

    return fig


def plot_trajectory_comparison(result, control_traj, planner_config, tendon_names=None, show=True, save_path=None):
    """
    Compare GP-interpolated tendon lengths against the discrete planned trajectory.

    Parameters
    ----------
    result : trajectory planner result
        result.trajectory is a list of marginals (K+1 steps).
    control_traj : np.ndarray, shape (N_control, num_tendons)
        High-rate interpolated tendon lengths from interpolate_gp_trajectory.
    planner_config : TrajectoryPlannerConfig
        Used for dt and K to reconstruct discrete time axis.
    tendon_names : list of str, optional
    show : bool
    save_path : str, optional
    """
    num_tendons = control_traj.shape[1]
    if tendon_names is None:
        tendon_names = [f"T{i}" for i in range(num_tendons)]

    K = planner_config.K
    dt = planner_config.dt
    control_hz = (len(control_traj) - 1) / (K * dt)
    control_dt = 1.0 / control_hz

    # Discrete planned time axis (K+1 points)
    discrete_times = np.array([k * dt for k in range(K + 1)])
    discrete_lengths = np.array([result.trajectory[k].tendon_lengths for k in range(K + 1)])

    # Interpolated time axis
    interp_times = np.arange(len(control_traj)) * control_dt

    cmap = plt.cm.tab10(np.linspace(0, 1, max(num_tendons, 2)))

    fig, axes = plt.subplots(num_tendons, 1, figsize=(10, 2.5 * num_tendons), sharex=True)
    if num_tendons == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(interp_times, control_traj[:, i], color=cmap[i], linewidth=1.2, label="Interpolated")
        ax.plot(discrete_times, discrete_lengths[:, i], "o--", color=cmap[i],
                markersize=6, linewidth=1.0, alpha=0.7, label="Planned")
        ax.set_ylabel("Length (m)")
        ax.set_title(tendon_names[i])
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Planned vs. GP-Interpolated Tendon Lengths", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison plot to: {save_path}")

    if show:
        plt.show()

    return fig
