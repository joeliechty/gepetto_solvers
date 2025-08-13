import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config
from utils import TipForceFunction, tensions_function, setup_plt

    
def simulate_trajectory(sim_time, do_plot, save_frames_mode, poses_between_discs, use_midpoint, frame_rate=3):
    config = get_base_config()
    config.poses_between_discs = poses_between_discs
    config.use_midpoint = use_midpoint

    simulator = TipForceSolver(config)

    num_steps = sim_time * frame_rate

    if do_plot:
        plotter = TendonRobotPlotter('kinematics_sim', save_frames_mode=save_frames_mode, d_azimuth=2.0)
    
    tip_position = []
    tensions_all = []
    t_all = []

    tip_force_function = TipForceFunction(max_magnitude=0.1, force_rate_hz=0.1, seed=42)

    for i in range(num_steps):
        t = float(i) / float(frame_rate)

        tip_force = tip_force_function(t)
        tensions = tensions_function(t)

        solution = simulator.simulation_step(tensions, tip_force)

        tip_position.append(solution.backbone_pose_mean[-1][:3,3])
        tensions_all.append(tensions)
        t_all.append(t)

        if do_plot:
            plotter.update(solution)
    
    if do_plot:
        plotter.plotter.close()

    return np.array(tip_position)


def run_simulation(sim_time, poses_between_discs, trajectory_accurate, use_midpoint):

    trajectories = [
        simulate_trajectory(
            sim_time,
            do_plot=(i == 3),
            save_frames_mode=(i == 3),
            poses_between_discs=poses_between_i,
            use_midpoint=use_midpoint
        )
        for i, poses_between_i in enumerate(poses_between_discs)
    ]

    def rms_error(traj_a, traj_b):
        diff = traj_a - traj_b
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    rms_diffs = []
    for trajectory in trajectories:
        rms_diff = rms_error(trajectory, trajectory_accurate)
        rms_diffs.append(rms_diff)

    config = get_base_config()
    error_percent = 100 * np.array(rms_diffs) / config.rod_length

    return error_percent


if __name__ == "__main__":
    sim_time = 120
    poses_between_discs = np.arange(11)
    big_num_poses = 50

    trajectory_accurate = simulate_trajectory(sim_time, False, False, big_num_poses, use_midpoint=True)
    error_percents_euler = run_simulation(sim_time, poses_between_discs, trajectory_accurate, use_midpoint=False)
    error_percents_midpoint = run_simulation(sim_time, poses_between_discs, trajectory_accurate, use_midpoint=True)
    
    setup_plt(height=3, grid=True)

    plt.figure()
    plt.plot(poses_between_discs, error_percents_euler, 'o-', label="Euler")
    plt.plot(poses_between_discs, error_percents_midpoint, 'o-', label="Midpoint")
    plt.xlabel('Poses Between Each Disc')
    plt.ylabel('RMS Tip Position Error (% robot length)')
    plt.semilogy()
    plt.tight_layout()
    plt.legend()

    plt.savefig("figures/kinematics_sim.pdf", bbox_inches="tight")
