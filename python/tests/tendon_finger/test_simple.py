import numpy as np
import time

import crest_sparse
from .._plotting.tendon_robot_plotter import TendonRobotPlotter

from .config import get_base_config, build_K_inv_per_segment, build_disc_positions_and_segments
from .benchmark import TendonRobotSolver

def main():
    config = get_base_config()
    
    # Define finger structure with bone and joint lengths
    # Each bone gets discs at start/end, joints are sandwiched between discs
    bone_joint_spec = [
        ("bone", 0.05),     # metacarpal
        ("joint", 0.01),    # metacarpophalangeal (MCP) joint
        ("bone", 0.05),     # proximal phalanx
        ("joint", 0.01),    # proximal interphalangeal (PIP) joint
        ("bone", 0.03),     # middle phalanx
        ("joint", 0.01),    # distal interphalangeal (DIP) joint
        ("bone", 0.02),     # distal phalanx
    ]
    
    # Build disc positions and get segment types
    disc_positions, segment_types, num_discs, total_length = build_disc_positions_and_segments(bone_joint_spec)
    
    config.rod_length = total_length
    config.num_discs = num_discs
    config.disc_positions_normalized = disc_positions
    
    # Build per-segment stiffness with bone/joint pattern
    config.K_inv_per_segment = build_K_inv_per_segment(
        num_discs=config.num_discs,
        num_between_nodes=config.num_between_nodes,
        segment_types=segment_types,
        bone_stiffness_scale=1e-6  # Relaxed from 1e-6 to avoid ill-conditioning
    )
    
    solver = crest_sparse.TendonRobotSolver(config)
    
    dummy_solution = solver.solve(crest_sparse.Vector4Gaussian(np.zeros(4), np.eye(4)), crest_sparse.Vector6Gaussian(np.zeros(6), np.eye(6)), None)
    solver_baseline = TendonRobotSolver(config, dummy_solution.marginals.tendon_config.hole_locations)

    plotter = TendonRobotPlotter(single_plot_mode=False)

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    # Background tension for passive tendons (Newtons)
    background_tension = 0.5

    for i in range(1000):
        tensions_mean = np.zeros(4)
        # Tendon 0 (0°): constant background tension
        tensions_mean[0] = background_tension
        # Tendons 1 (90°): primary actuator - varies
        tensions_mean[1] = background_tension + 5.0 * (np.cos(0.01 * i - np.pi) + 1)
        # Tendons 2-3 (180°, 270°): constant background tension
        tensions_mean[2] = background_tension 
        tensions_mean[3] = background_tension 
        
        tip_wrench_mean = np.zeros(6)
        # tip_wrench_mean[5] = 0.1 * np.sin(0.1 * i)
        # tip_wrench_mean[3] = 0.2

        tensions = crest_sparse.Vector4Gaussian(tensions_mean, tensions_cov)
        tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)

        solution = solver.solve(tensions, tip_wrench, None)
        plotter.update(solution)

        # solution_baseline = solver_baseline.solve(tensions_mean, tip_wrench_mean[3:])
        # p_baseline = solution_baseline[-1]['p']
        # p_gt = solution.marginals.rod.states[-1].pose.mean[:3,3]

        # print("p_gt: ", p_gt)
        # print("p_baseline: ", p_baseline)
        # print(f"baseline error: {np.linalg.norm(p_baseline - p_gt)}")



if __name__ == "__main__":
    main()
