import copy

import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config
from utils import TipForceFunction, generate_waypoint_trajectory, setup_plt
from benchmark import solve_kinematics_bvp


def run_trajectory_simulation(sim_time, poses_between_discs):
    euler_solvers = []
    midpoint_solvers = []

    for poses_between_i in poses_between_discs:
        config = get_base_config()
        config.poses_between_discs = poses_between_i

        config.use_midpoint = False
        euler_solvers.append(TipForceSolver(config))

        config.use_midpoint = True
        midpoint_solvers.append(TipForceSolver(config))

    # plotter = TendonRobotPlotter('kinematics_sim', plot_tip_force=True, save_frames_mode=True)
    t, positions, tensions, waypoints = generate_waypoint_trajectory(sim_time, seed=42)
    tip_force_function = TipForceFunction(max_magnitude=2 * config.tip_force_prior_std, seed=42)

    traj_benchmark = []
    traj_euler = [[] for _ in poses_between_discs] # (poses_between_i, step)
    traj_midpoint = [[] for _ in poses_between_discs]
    solve_times_euler = [[] for _ in poses_between_discs]
    solve_times_midpoint = [[] for _ in poses_between_discs]

    x_guess = None

    for i, (t_i, p_i, tensions_i) in enumerate(zip(t, positions, tensions)):
        tip_force = tip_force_function(t_i)

        for idx, (solver_euler, solver_midpoint) in enumerate(zip(euler_solvers, midpoint_solvers)):
            sol_euler = solver_euler.simulation_step(tensions_i, tip_force)
            traj_euler[idx].append(sol_euler.backbone_pose_mean[-1][:3, 3])
            solve_times_euler[idx].append(sol_euler.total_time_ms)

            sol_midpoint = solver_midpoint.simulation_step(tensions_i, tip_force)
            traj_midpoint[idx].append(sol_midpoint.backbone_pose_mean[-1][:3, 3])
            solve_times_midpoint[idx].append(sol_midpoint.total_time_ms)

            # if idx == 3:
            #     plotter.update(sol_midpoint)
            
        percent_complete = i / len(t) * 100
        print(f"\n\nProgress: {percent_complete:.1f} %\n", flush=True)

        holes = sol_euler.tendon_disc_config.local_holes
        p, x_guess = solve_kinematics_bvp(tensions_i, tip_force, get_base_config(), holes, x_guess)
        traj_benchmark.append(p[-1])

    traj_benchmark = np.array(traj_benchmark)
    traj_euler = [np.array(tp) for tp in traj_euler]
    traj_midpoint = [np.array(tp) for tp in traj_midpoint]
    mean_solve_times_euler = [np.mean(st) for st in solve_times_euler]
    mean_solve_times_midpoint = [np.mean(st) for st in solve_times_midpoint]

    return (traj_benchmark,
            traj_euler,
            mean_solve_times_euler,
            traj_midpoint,
            mean_solve_times_midpoint)


def get_rms_percent_errors(traj_list, traj_benchmark):
    rms_errors = []
    for traj in traj_list:
        diff = traj - traj_benchmark
        rms_errors.append(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    config = get_base_config()
    rms_percent_errors = 100 * np.array(rms_errors) / config.rod_length

    return rms_percent_errors


if __name__ == "__main__":
    sim_time = 60
    poses_between_discs = np.arange(12)
    traj_benchmark, traj_euler, mean_solve_times_euler, traj_midpoint, mean_solve_times_midpoint = \
        run_trajectory_simulation(sim_time, poses_between_discs=poses_between_discs)

    rms_errors_euler = get_rms_percent_errors(traj_euler, traj_benchmark)
    rms_errors_midpoint = get_rms_percent_errors(traj_midpoint, traj_benchmark)

    config = get_base_config()
    num_nodes = config.num_discs + (config.num_discs - 1) * poses_between_discs

    setup_plt(height=2.0, grid=True)

    # fig, axes = plt.subplots(2, 1, sharex=True)

    # axes[0].plot(num_nodes, rms_errors_euler, 'ko--', markerfacecolor="orangered", label="Euler")
    # axes[0].plot(num_nodes, rms_errors_midpoint, 'ko-', markerfacecolor="blue", label="midpoint")
    # axes[0].set_ylabel('position error (RMS % length)')
    # axes[0].semilogy()
    # axes[0].legend()

    # axes[1].plot(num_nodes, mean_solve_times_euler, 'ko--', markerfacecolor="orangered")
    # axes[1].plot(num_nodes, mean_solve_times_midpoint, 'ko-', markerfacecolor="blue")
    # axes[1].set_xlabel('number of arclength nodes')
    # axes[1].set_ylabel('mean solve time (ms)')

    # fig.align_ylabels()

    plt.figure()

    plt.plot(num_nodes, rms_errors_euler, 'ko--', markerfacecolor="orangered", label="Euler")
    plt.plot(num_nodes, rms_errors_midpoint, 'ko-', markerfacecolor="blue", label="midpoint")
    plt.ylabel('RMS position error (% robot length)')
    plt.xlabel('number of discrete arclength nodes')
    plt.semilogy()
    plt.legend()


    plt.tight_layout()

    plt.savefig("figures/kinematics_sim_results.pdf", bbox_inches="tight")
