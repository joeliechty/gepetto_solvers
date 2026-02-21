import matplotlib.pyplot as plt
import numpy as np

import crest_sparse
from .._plotting.cosserat_rod_plotter import CosseratRodPlotter
from .._plotting.utils import setup_plt

from .config import get_base_config
from .benchmark import CosseratRodBaseline


def get_tip_wrench(t):
    f_xy = 1 * np.array([np.cos(0.3 * t), np.sin(0.3 * t)])
    f_z = 0.5 * np.sin(0.5 * t)

    m_xy = 1 * np.array([np.cos(0.2 * t), np.sin(0.2 * t)])
    m_z = 2 * np.sin(0.4 * t)

    return np.hstack((m_xy, m_z, f_xy, f_z))


def simulate_trajectory(num_nodes, use_midpoint=True, use_baseline=False, sim_time=180.0, frame_rate=3.0):
    config = get_base_config()
    config.use_midpoint = use_midpoint
    config.num_nodes = num_nodes

    if use_baseline:
        solver = CosseratRodBaseline(config)
    else:
        solver = crest_sparse.CosseratRodSolver(config)

    # Only plot one of the cases to see what the robot did
    plot = use_midpoint and num_nodes == 15

    if plot:
        plotter = CosseratRodPlotter(
            plot_wrenches=True,
            plot_base_wrench=False,
            plot_backbone_ellipsoids=False,
            moment_scale=0.07,
            save_frames_dir_name='midpoint_vs_euler',
            plot_backbone_frames=True, 
            camera_azimuth=60, 
            camera_distance=1.5, 
            camera_focal_point=np.array([0, 0, 0.25]))
    
    dt = 1.0 / frame_rate
    num_steps = int(sim_time * frame_rate)

    tip_wrench_cov = np.diag(np.hstack((1e-4 * np.ones(3), 1e-3 * np.ones(3))))**2

    tip_poses = []
    
    for step in range(num_steps + 1):
        t = step * dt
        tip_wrench = get_tip_wrench(t)

        if use_baseline:
            solution = solver.solve(tip_wrench)
            pose = solution['pose'][-1]
        else:
            wrench = crest_sparse.Vector6Gaussian(tip_wrench, tip_wrench_cov)
            solution = solver.solve(wrench, None, None)
            pose = solution.marginals .states[-1].pose.mean

        tip_poses.append(pose)

        if plot:
            plotter.update(solution) 

        progress = 100.0 * step / num_steps
        print(f"num_nodes: {num_nodes}, Progress: {progress:5.1f}%", end="\r")

    tip_poses = np.array(tip_poses)
    p = tip_poses[:,:3,3]
    R = tip_poses[:,:3,:3]

    return {'p': p, 'R': R}


def run_sims(num_nodes):
    baseline = simulate_trajectory(0, use_baseline=True)     

    midpoint_error, euler_error = [], []
    for n in num_nodes:
        midpoint = simulate_trajectory(n, use_midpoint=True)    
        euler = simulate_trajectory(n, use_midpoint=False)    

        midpoint_error.append(np.linalg.norm(midpoint['p'] - baseline['p'], axis=1).mean())
        euler_error.append(np.linalg.norm(euler['p'] - baseline['p'], axis=1).mean())
    
    return np.array(midpoint_error), np.array(euler_error)


def main():
    num_nodes = np.arange(5, 75, step=1)
    # num_nodes = np.arange(4, 74, step=10)
    midpoint_error, euler_error = run_sims(num_nodes)
    
    # Convert to percent rod length
    config = get_base_config()
    L = config.rod_length
    midpoint_percent = 100 * midpoint_error / L
    euler_percent = 100 * euler_error / L

    setup_plt(width=2.7,height=1.8)
    plt.figure()

    plt.semilogy(num_nodes, midpoint_percent, '-k', label="midpoint")
    plt.semilogy(num_nodes, euler_percent, ':k', label="euler")

    plt.xlabel("number of arclength nodes")
    plt.ylabel("tip position error (% length)")
    plt.legend(ncol=2, columnspacing=0.2, borderpad=0.0, borderaxespad=0.2, handlelength=1.0, handletextpad=0.2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/cosserat_num_nodes.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()