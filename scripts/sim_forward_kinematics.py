import numpy as np
import matplotlib.pyplot as plt

from tendon_robot import TipForceSolver
from plotting import TendonRobotPlotter
from config import get_base_config
from utils import TipForceFunction, tensions_function, setup_plt

    
def simulation(sim_time, do_plot, save_frames_mode, poses_between_discs, frame_rate=10):
    config = get_base_config()
    config.poses_between_discs = poses_between_discs

    simulator = TipForceSolver(config)

    num_steps = sim_time * frame_rate

    if do_plot:
        plotter = TendonRobotPlotter('Forward Kinematics', save_frames_mode=save_frames_mode)
    
    tip_position = []
    tensions_all = []
    t_all = []

    tip_force_function = TipForceFunction(max_magnitude=0.2, seed=42)

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


import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sim_time = 5
    poses_between_discs = np.arange(5)

    trajectories = [
        simulation(
            sim_time,
            do_plot=(i == 3),
            save_frames_mode=(i == 3),
            poses_between_discs=poses_between_i
        )
        for i, poses_between_i in enumerate(poses_between_discs)
    ]

    def rms_error(traj_a, traj_b):
        diff = traj_a - traj_b
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    rms_diffs = []
    for traj_low, traj_high in zip(trajectories, trajectories[1:]):
        rms_diff = rms_error(traj_low, traj_high)
        rms_diffs.append(rms_diff)

    # num_poses = config.num_discs + (config.num_discs - 1) * poses_between_discs
    config = get_base_config()
    error_percent = 100 * np.array(rms_diffs) / config.rod_length

    setup_plt(height=6)

    plt.figure()
    plt.plot(poses_between_discs[1:], error_percent, 'o-')
    plt.xlabel('Poses Between Each Disc')
    plt.ylabel('Change in RMS Tip Position (% robot length)')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("figures/kinematics_convergence.pdf", bbox_inches="tight")
