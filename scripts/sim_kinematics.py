import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config
from utils import TipForceFunction, tensions_function, setup_plt
from benchmark import solve_kinematics_bvp
    

def get_single_trajetory(sim_time, poses_or_benchmark, do_plot=False, use_midpoint=True, frame_rate=30):
    if poses_or_benchmark == "benchmark":
        do_benchmark = True
        poses_between_discs = 0
    else:
        do_benchmark = False
        poses_between_discs = poses_or_benchmark
    
    config = get_base_config()
    config.poses_between_discs = poses_between_discs
    config.use_midpoint = use_midpoint

    simulator = TipForceSolver(config)

    num_steps = sim_time * frame_rate

    if do_plot:
        plotter = TendonRobotPlotter('kinematics_sim', save_frames_mode=True, d_azimuth=1.0)
    
    tip_positions = []
    solve_times = []

    tip_force_function = TipForceFunction(max_magnitude=0.1, force_rate_hz=0.1, seed=42)

    x_guess = None

    for i in range(num_steps):
        percent_complete = (i + 1) / num_steps * 100
        print(f"\r\nProgress: {percent_complete:.1f} %\n\n", end="")

        t = float(i) / float(frame_rate)

        tip_force = tip_force_function(t)
        tensions = tensions_function(t)

        solution = simulator.simulation_step(tensions, tip_force)

        if do_benchmark:
            p, x_guess = solve_kinematics_bvp(tensions, tip_force, config, solution.tendon_disc_config.local_holes, x_guess)
            tip_positions.append(p[-1])
            continue

        tip_positions.append(solution.backbone_pose_mean[-1][:3,3])
        solve_times.append(solution.total_time_ms)

        if do_plot:
            plotter.update(solution)
    
    if do_plot:
        plotter.plotter.close()

    return np.array(tip_positions), np.mean(solve_times)


def run_simulation(sim_time, poses_between_discs, trajectory_accurate, use_midpoint):

    trajectories = []
    mean_solve_times = []
    
    for poses_between_i in poses_between_discs:
        trajectory, mean_solve_time = get_single_trajetory(
            sim_time, poses_between_i, do_plot=(poses_between_i == 5), use_midpoint=use_midpoint)

        trajectories.append(trajectory)
        mean_solve_times.append(mean_solve_time)

    def rms_error(traj_a, traj_b):
        traj_a = np.asarray(traj_a)
        traj_b = np.asarray(traj_b)
        if traj_a.shape != traj_b.shape:
            raise ValueError(f"Trajectory shapes do not match: {traj_a.shape} vs {traj_b.shape}")
        diff = traj_a - traj_b
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    rms_diffs = []
    for trajectory in trajectories:
        rms_diff = rms_error(trajectory, trajectory_accurate)
        rms_diffs.append(rms_diff)

    config =get_base_config()
    errors_percent = 100 * np.array(rms_diffs) / config.rod_length

    return errors_percent, mean_solve_times


if __name__ == "__main__":
    sim_time = 60
    poses_between_discs = np.arange(11)

    trajectory_accurate = get_single_trajetory(sim_time, "benchmark")[0]
    error_percents_euler, mean_solve_times_euler = run_simulation(sim_time, poses_between_discs, trajectory_accurate, use_midpoint=False)
    error_percents_midpoint, mean_solve_times_midpoint = run_simulation(sim_time, poses_between_discs, trajectory_accurate, use_midpoint=True)
    
    setup_plt(height=4, grid=True)

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(poses_between_discs, error_percents_euler, 'o-', label="Euler")
    axes[0].plot(poses_between_discs, error_percents_midpoint, 'o-', label="Midpoint")
    axes[0].set_ylabel('RMS Position Error (% length)')
    axes[0].semilogy()
    axes[0].legend()

    axes[1].plot(poses_between_discs, mean_solve_times_euler, 'o-', label="Euler")
    axes[1].plot(poses_between_discs, mean_solve_times_midpoint, 'o-', label="Midpoint")
    axes[1].set_xlabel('Poses Between Each Disc')
    axes[1].set_ylabel('Mean Solve Time (ms)')

    fig.align_ylabels()
    fig.tight_layout()

    fig.savefig("figures/kinematics_sim.pdf", bbox_inches="tight")
