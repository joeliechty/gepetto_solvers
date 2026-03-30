import sys
import os
import numpy as np
import time
import crest_sparse
from .._plotting.tendon_hand_plotter import TendonHandPlotter
from .config import get_hand_config, _rotation_x, _rotation_y, _rotation_z

def solve_ik_grasp(solver, vdb_path, finger_names, target_pose, plotter=None):
    num_fingers = solver.num_fingers()
    num_tendons = 6

    # --- Covariance matrices ---
    # Lock object rigidly in place
    super_locked = 1e-6
    stiff_sigmas = np.array([super_locked] * 6)

    # give solver max uncertainty about the tensions so that they can be optimized to solve for the dummy point constraints
    free_tension_cov = (5.0) ** 2 * np.eye(num_tendons)

    tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), (1e-3)**2 * np.eye(6)) for _ in range(num_fingers)]
    tensions_free_in = [crest_sparse.VectorXGaussian(np.zeros(num_tendons), free_tension_cov) for _ in range(num_fingers)]

    # --- Setup ---
    print("\n--- INVERSE KINEMATICS GRASP SOLVER ---")
    print(f"Object Target Position: {target_pose[0:3, 3]}")

    # Initialize the object
    solver.set_object(vdb_path, target_pose, stiff_sigmas)

    # --- Solve ---
    print("Optimizing tendon tensions to establish contact...")
    start_time = time.time()
    solution = solver.solve(tensions_free_in, tip_wrenches)
    print(f"Inverse kinematics solution in {time.time() - start_time:.2f} seconds.")

    # extract the solution
    for f_idx, name in enumerate(finger_names):
        opt_t = solution.marginals.fingers[f_idx].tensions.mean
        print(f"{name.upper()}:")
        print(f"  Extensor Tension: {opt_t[0]:.3f} N")
        print(f"  Flexor Tension:   {opt_t[5]:.3f} N")
        print(f"  All Tensions:     {np.round(opt_t, 3)}")

    # --- Plotting ---
    if plotter is not None:
        solutions_dict = {}
        for name, finger_marginals in zip(finger_names, solution.marginals.fingers):
            finger_solution = crest_sparse.TendonRobotSolution()
            finger_solution.marginals = finger_marginals
            finger_solution.meta = solution.meta
            solutions_dict[name] = finger_solution
            
        plotter.update(solutions_dict, object_pose=target_pose)
        
        print("\nClose the plot window to exit.")
        # Keep script alive to view the plot
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def main(vdb_path=None):
    if vdb_path is None:
        raise ValueError("vdb_path is required")

    # ==========================================
    # 1. HAND CONFIGURATION
    # ==========================================
    # Choose 1-4 fingers and thumb_side="right", "left", "both", or None
    THUMB_SIDE = None
    FINGER_SPREAD_ANGLE_DEG = 30.0
    NUM_FINGERS = 4
    configs = get_hand_config(num_fingers=NUM_FINGERS, thumb_side=THUMB_SIDE, finger_spread_angle_deg=FINGER_SPREAD_ANGLE_DEG)
    finger_names = [name for name, _ in configs]

    hand_solver_config = crest_sparse.TendonHandSolverConfig()
    solver = crest_sparse.TendonHandSolver(configs, hand_solver_config)

    # ==========================================
    # 2. OBJECT CONFIGURATION
    # ==========================================
    target_pose = np.eye(4)
    # Place object slightly in front of the palm and offset to the right
    target_pose[0:3, 3] = [0.03, 0.12, 0.0] 
    # Rotate cylinder horizontally
    target_pose[0:3, 0:3] = _rotation_x(np.pi/2) @ _rotation_z(np.pi/2)

    # ==========================================
    # 3. PLOTTER
    # ==========================================
    plotter = TendonHandPlotter(
        finger_names=finger_names,
        vdb_path=vdb_path,
        plot_object=True,
        camera_azimuth=150,
        camera_elevation=20,
        camera_focal_point=[0, 0.08, 0],
        camera_distance=0.6,
        window_size=(1200, 1200),
        plot_collision_spheres=True,
        single_plot_mode=True # Good for single-shot IK renders
    )

    # Run the solver
    solve_ik_grasp(solver, vdb_path, finger_names, target_pose, plotter)


if __name__ == "__main__":
    vdb_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(vdb_path)