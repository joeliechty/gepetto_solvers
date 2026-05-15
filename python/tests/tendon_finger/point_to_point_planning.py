import os
import numpy as np
import time
import crest_sparse
from .._plotting.tendon_finger_plotter import TendonFingerPlotter
from .._plotting.trajectory_plotter import plot_trajectory, plot_trajectory_comparison
from .config import get_6tendon_config

def make_calc_phi(num_tendons):
    """Returns a function that generates the Identity transition matrix."""
    def calc_phi(dt):
        return np.eye(num_tendons)
    return calc_phi

def make_calc_q(Qc):
    """Returns a function that scales your GP covariance matrix by dt."""
    def calc_q(dt):
        return Qc * dt
    return calc_q

def interpolate_gp_state(tau, t_k, t_kp1, L_k, L_kp1, calc_phi, calc_q):
    tau = np.clip(tau, t_k, t_kp1)
    
    dt_full = t_kp1 - t_k
    dt_tau = tau - t_k
    dt_rem = t_kp1 - tau

    Phi_full = calc_phi(dt_full)
    Phi_tau = calc_phi(dt_tau)
    Phi_rem = calc_phi(dt_rem)

    Q_full = calc_q(dt_full) 
    Q_tau = calc_q(dt_tau)

    forward_prediction = Phi_tau @ L_k
    
    predicted_end_state = Phi_full @ L_k
    end_state_error = L_kp1 - predicted_end_state

    Q_full_inv = np.linalg.inv(Q_full)
    correction = Q_tau @ Phi_rem.T @ Q_full_inv @ end_state_error

    return forward_prediction + correction

def interpolate_gp_trajectory(model_config, planner_config, result, control_hz, save_path):
    control_dt = 1.0 / control_hz
    num_tendons = model_config.num_tendons

    # Create the helper functions using your existing config
    calc_phi = make_calc_phi(num_tendons)
    calc_q = make_calc_q(planner_config.gp_len_Qc)

    control_traj = []

    print(f"Interpolating from {int(1/planner_config.dt)}Hz up to {control_hz}Hz...")

    # Loop through consecutive discrete states in the solved trajectory
    for k in range(planner_config.K):
        t_k = k * planner_config.dt
        t_kp1 = (k + 1) * planner_config.dt
        
        # Get the solved tendon lengths at the boundaries
        L_k = np.array(result.trajectory[k].tendon_lengths)
        L_kp1 = np.array(result.trajectory[k+1].tendon_lengths)

        # Generate timestamps for this segment (excluding the end to avoid duplicates)
        segment_times = np.arange(t_k, t_kp1, control_dt)
        
        for tau in segment_times:
            L_tau = interpolate_gp_state(tau, t_k, t_kp1, L_k, L_kp1, calc_phi, calc_q)
            control_traj.append(L_tau)

    # Add the very last discrete state to cap off the trajectory
    control_traj.append(np.array(result.trajectory[-1].tendon_lengths))

    control_traj = np.array(control_traj)
    print(f"Generated {len(control_traj)} high-resolution control points.")
    save_interp_trajectory(control_traj, save_path)
    return control_traj

def save_interp_trajectory(control_traj, save_path):
    """
    Saves the interpolated traj in a data structure that can be loaded by the controller later
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, trajectory=control_traj)

def main():
    # 1. Setup Base Model Config
    model_config = get_6tendon_config()
    num_tendons = model_config.num_tendons  # 6

    # 2. Setup Planner Config
    planner_config = crest_sparse.TrajectoryPlannerConfig()
    planner_config.model_config = model_config
    planner_config.model_config.base.linear_solver_type = "MULTIFRONTAL_CHOLESKY" # FOR APPLE
    planner_config.model_config.base.delta_initial = 1.0
    planner_config.K = 10       # K+1 steps
    planner_hz = 5  # planning frequency
    planner_config.dt = 1.0 / planner_hz  # time step duration

    # Background Tensions: passive tendons (0-4) are tight, active tendon (5) is free
    bg_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0])
    bg_sigmas = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e6])
    planner_config.background_tensions_mean = bg_mean
    planner_config.background_tensions_sigmas = bg_sigmas

    # GP Prior Covariance (smoothness of tension changes over time)
    planner_config.gp_tense_Qc = np.eye(num_tendons) * 1e-2

    # GP Prior Covariance for tendon lengths (smoothness of length changes over time)
    planner_config.gp_len_Qc = np.eye(num_tendons) * 1e-5

    # Tension limit barrier: only active tendon (index 5) needs the barrier
    planner_config.tension_limit_alpha = 10.0
    planner_config.tension_limit_q_min = 0.0
    planner_config.active_tendon_indices = [5]

    ### START AND GOAL STATES ###
    # # Start tension
    # planner_config.start_tensions = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    # planner_config.start_tensions_cov = np.eye(num_tendons) * 1e-8

    # Start position
    planner_config.start_position = np.array([1.42977609e-02, 1.35008944e-01, 0.0])
    planner_config.start_position_cov = np.eye(3) * 1e-6

    # Goal position
    # planner_config.goal_position = np.array([6.16703767e-02, 4.02459538e-02, 0.0])
    planner_config.goal_position = np.array([6.02088876e-02, 3.77734425e-02, 0.0])
    planner_config.goal_position_cov = np.eye(3) * 1e-5

    zero_bend_length = 0.13723930740614093 # [m] length of actuation tendon for this robot in the fully straight configuration TODO: compute from geo later

    # # Goal State (Target Pose for the tip)
    # goal_pose = np.eye(4)
    # goal_pose[0:3, 3] = [0.05, 0.1, 0.0]  # Example translation target
    # planner_config.goal_pose = goal_pose
    # planner_config.goal_pose_cov = np.eye(6) * 1e-4

    # # Goal tension
    # planner_config.goal_tensions = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.5])
    # planner_config.goal_tensions_cov = np.eye(num_tendons) * 1e-4

    # 3. Initialize and Run Planner
    print("Building factor graph...")
    planner = crest_sparse.TendonFingerTrajectoryPlanner(planner_config)

    print("Planning trajectory...")
    start_time = time.time()
    result = planner.plan()
    elapsed = time.time() - start_time

    print(f"Solved in {elapsed:.2f}s | iters={result.meta.iterations} | error={result.meta.error:.4g}")
    print(f"  build={result.meta.build_time_ms:.0f}ms  opt={result.meta.optimize_time_ms:.0f}ms  "
          f"marginals={result.meta.marginalize_time_ms:.0f}ms")
    

    # 3.5 Interpolate trajectory at intermediate time points using GP interpolation
    control_hz = 100
    interp_save_path = os.path.expanduser("~/git_repos/underactuated_hand/interpolated_trajectory.npz")
    control_traj = interpolate_gp_trajectory(model_config, planner_config, result, control_hz, save_path=interp_save_path)

    # 4. Visualization
    plotter = TendonFingerPlotter(
        single_plot_mode=False,
        plot_backbone_frames=True,
        plot_tip_force=True,
        plot_backbone_ellipsoids=True,
        camera_azimuth=180,
        camera_elevation=-90,
        camera_focal_point=[0, 0.1, 0]
    )

    # tendon displacement trajecotry
    disp_traj = []

    # Animate the resulting trajectory
    for k, marginals in enumerate(result.trajectory):
        print(f"Displaying Step {k}/{planner_config.K}")
        print(f"  Tensions mean: {marginals.tensions.mean}")
        print(f"  Tendon lengths: {marginals.tendon_lengths}")
        print(f"  Tip pose mean:\n{marginals.rod.states[-1].pose.mean}")

        disp_traj.append(marginals.tendon_lengths[5] - zero_bend_length)  # relative to initial lengths (if it is negative then tendon is shortening and bending the finger)

        class MockSolution:
            pass
        sol = MockSolution()
        sol.marginals = marginals
        sol.meta = result.meta

        plotter.update(sol)
        time.sleep(planner_config.dt)  # Sleep to simulate real-time animation

    print("Tendon 5 displacement trajectory (relative to initial length):")
    print(disp_traj)

    input("Press Enter to close...")

    # 5. Plot full trajectory state
    tendon_names = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]
    plot_trajectory(result, tendon_names=tendon_names, save_path="point_to_point_trajectory.png")

    plot_trajectory_comparison(result, control_traj, planner_config,
                               tendon_names=tendon_names,
                               save_path="point_to_point_comparison.png")

if __name__ == "__main__":
    main()
