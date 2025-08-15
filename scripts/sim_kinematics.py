import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config
from utils import TipForceFunction, tensions_function, setup_plt, generate_trajectory
from benchmark import solve_kinematics_bvp
    

def run_trajectory_simulation(t, tensions, poses_between_discs, frame_rate=30):
    euler_solvers = []
    midpoint_solvers = []

    for poses_between_i in poses_between_discs:
        config = get_base_config()
        config.poses_between_discs = poses_between_i

        config.use_midpoint = False
        euler_solvers.append(TipForceSolver(config))

        config.use_midpoint = True
        midpoint_solvers.append(TipForceSolver(config))

    plotter = TendonRobotPlotter('kinematics_sim', save_frames_mode=True, d_azimuth=1.0)

    tip_positions_benchmark = []
    tip_positions_euler = [[] for _ in poses_between_discs] # (poses_between_i, step)
    tip_positions_midpoint = [[] for _ in poses_between_discs]
    solve_times_euler = [[] for _ in poses_between_discs]
    solve_times_midpoint = [[] for _ in poses_between_discs]

    tip_force_function = TipForceFunction(max_magnitude=0.1, force_rate_hz=0.1, seed=42)

    x_guess = None

    for i, (t_i, tensions_i) in enumerate(zip(t, tensions)):
        tip_force = tip_force_function(t_i)

        for idx, (solver_euler, solver_midpoint) in enumerate(zip(euler_solvers, midpoint_solvers)):
            sol_euler = solver_euler.simulation_step(tensions_i, tip_force)
            tip_positions_euler[idx].append(sol_euler.backbone_pose_mean[-1][:3, 3])
            solve_times_euler[idx].append(sol_euler.total_time_ms)

            sol_midpoint = solver_midpoint.simulation_step(tensions_i, tip_force)
            tip_positions_midpoint[idx].append(sol_midpoint.backbone_pose_mean[-1][:3, 3])
            solve_times_midpoint[idx].append(sol_midpoint.total_time_ms)

            if poses_between_discs[idx] == 3:
                plotter.update(sol_midpoint)

        percent_complete = i / len(t) * 100
        print(f"\r\nProgress: {percent_complete:.1f} %\n", end="")

        holes = sol_euler.tendon_disc_config.local_holes
        p, x_guess = solve_kinematics_bvp(tensions_i, tip_force, get_base_config(), holes, x_guess)
        tip_positions_benchmark.append(p[-1])

    plotter.plotter.close()

    tip_positions_benchmark = np.array(tip_positions_benchmark)
    tip_positions_euler = [np.array(tp) for tp in tip_positions_euler]
    tip_positions_midpoint = [np.array(tp) for tp in tip_positions_midpoint]
    mean_solve_times_euler = [np.mean(st) for st in solve_times_euler]
    mean_solve_times_midpoint = [np.mean(st) for st in solve_times_midpoint]

    return (tip_positions_benchmark,
            tip_positions_euler,
            mean_solve_times_euler,
            tip_positions_midpoint,
            mean_solve_times_midpoint)


def get_rms_percent_errors(traj_list, traj_benchmark):
    rms_errors = []
    for traj in traj_list:
        diff = traj - traj_benchmark
        rms_errors.append(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    config =get_base_config()
    rms_percent_errors = 100 * np.array(rms_errors) / config.rod_length

    return rms_percent_errors


def generate_waypoints_in_ellipsoid(num_points, center=(0, 0, 0.15), radii=(0.1, 0.1, 0.05)):
    waypoints = []
    for _ in range(num_points):
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        r = np.random.random() ** (1/3)
        point = r * direction * np.array(radii)
        waypoints.append(np.array(center) + point)
    return np.array(waypoints)


def multi_point_trajectory(t, waypoints, time_per_waypoint=3.0):
    num_segments = len(waypoints) - 1

    segment_index = min(int(t // time_per_waypoint), num_segments)
    next_index = min(segment_index + 1, len(waypoints) - 1)

    alpha = (t % time_per_waypoint) / time_per_waypoint
    return (1 - alpha) * waypoints[segment_index] + alpha * waypoints[next_index]


if __name__ == "__main__":
    sim_time = 30
    poses_between_discs = np.arange(11)

    time_per_waypoint = 3.0
    num_waypoints = int(sim_time / time_per_waypoint) + 1

    waypoints = generate_waypoints_in_ellipsoid(num_waypoints)

    def trajectory(t):
        return multi_point_trajectory(t, waypoints)
    
    t, tensions = generate_trajectory(trajectory, sim_time)
    traj_benchmark, traj_euler, mean_solve_times_euler, traj_midpoint, mean_solve_times_midpoint = run_trajectory_simulation(t, tensions, poses_between_discs)

    rms_errors_euler = get_rms_percent_errors(traj_euler, traj_benchmark)
    rms_errors_midpoint = get_rms_percent_errors(traj_midpoint, traj_benchmark)

    config = get_base_config()
    num_nodes = config.num_discs + (config.num_discs - 1) * poses_between_discs

    setup_plt(height=4, grid=True)

    fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(num_nodes, rms_errors_euler, 'o-', label="Euler")
    axes[0].plot(num_nodes, rms_errors_midpoint, 'o-', label="Midpoint")
    axes[0].set_ylabel('RMS Position Error (% length)')
    axes[0].semilogy()
    axes[0].legend()

    axes[1].plot(num_nodes, mean_solve_times_euler, 'o-', label="Euler")
    axes[1].plot(num_nodes, mean_solve_times_midpoint, 'o-', label="Midpoint")
    axes[1].set_xlabel('Number of Arclength Mesh Nodes')
    axes[1].set_ylabel('Mean Solve Time (ms)')

    fig.align_ylabels()
    fig.tight_layout()

    fig.savefig("figures/kinematics_sim.pdf", bbox_inches="tight")
