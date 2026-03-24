import sys
import os
import numpy as np
import time
import crest_sparse
from .._plotting.tendon_hand_plotter import TendonHandPlotter
from .config import get_opposing_finger_config, _rotation_x, _rotation_y, _rotation_z

def run_inference_loop(simulation_solver, belief_solver, vdb_path, goal_sequence, plotter=None):
    num_fingers = simulation_solver.num_fingers()
    num_tendons = 6

    # --- Covariance Definitions ---
    locked = 1e-4
    super_locked = 1e-6
    free = 1e4

    # object
    # Order: [Local RotX, Local RotY, Local RotZ, Local TransX, Local TransY, Local TransZ]
    rolling_sigmas = np.array([
        locked, # RotX (World -Y): Locked (don't tilt side-to-side)
        free,   # RotY (World -X): Free to spin around the World X axis!
        locked, # RotZ (World +Z): Locked (don't spin like a top)
        free,   # TransX (World +Z): Free to translate up and down along World Z!
        locked, # TransY (World -X): Locked (don't slide sideways out of the hand)
        locked  # TransZ (World -Y): Locked (don't move closer/further from palm)
    ])
    stiff_sigmas = np.array([locked, locked, locked, locked, locked, locked]) # no movement

    # tendon priors
    frozen_tension_cov = super_locked * np.eye(num_tendons) # no change in tension
    free_tension_cov = (5.0)**2 * np.eye(num_tendons) # no information about tension

    tip_wrenches = [crest_sparse.Vector6Gaussian(np.zeros(6), (1e-3)**2 * np.eye(6)) for _ in range(num_fingers)]

    # --- Initialization ---
    current_pose = np.eye(4)
    current_pose[0:3, 3] = [0.0, 0.15, 0.0]
    R = _rotation_x(np.pi/2) @ _rotation_z(np.pi/2) # Rotate 90 deg around X to lay cylinder on table, then 90 deg around Z so it rolls along world X
    current_pose[0:3, 0:3] = R

    print("\n--- GRASP RAMP-UP ---")
    # start with zero tension
    current_tensions = [np.zeros(num_tendons) for _ in range(num_fingers)]

    # Initialize the object in both solvers so it acts as a rigid wall during grasp
    simulation_solver.set_object(vdb_path, current_pose, stiff_sigmas)
    belief_solver.set_object(vdb_path, current_pose, stiff_sigmas)

    # ramp up tensions to establish initial grasp (flexor index 5)
    ramp_steps = 10
    base_tension = 0.2
    step_size = base_tension / ramp_steps

    for step in range(ramp_steps):
        # Evenly increase tension accross all tendons
        for f in range(num_fingers):
            current_tensions[f] += step_size

        # lock these tensions in for the buildup
        t_in = [crest_sparse.VectorXGaussian(t, frozen_tension_cov) for t in current_tensions]

        # solve the sim and the beliefe factorgraphs
        sim_solution = simulation_solver.solve(t_in, tip_wrenches)
        belief_solution = belief_solver.solve(t_in, tip_wrenches)

        current_pose = sim_solution.marginals.object_pose.mean
        
        if plotter is not None:
            solutions = {}
            finger_names = ["finger_left", "finger_right"]
            for name, finger_marginals in zip(finger_names, sim_solution.marginals.fingers):
                finger_solution = crest_sparse.TendonRobotSolution()
                finger_solution.marginals = finger_marginals
                finger_solution.meta = sim_solution.meta
                solutions[name] = finger_solution
            plotter.update(solutions, object_pose=current_pose)    

    print("\n--- STARTING INFERENCE LOOP ---")

    # Initial contact grasp (flexor index 5)
    current_tensions = [np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) for _ in range(num_fingers)]

    # --- Inference Loop ---
    for goal_idx, goal_x in enumerate(goal_sequence):
        print(f"\nMoving to goal {goal_idx+1}/{len(goal_sequence)}: x={goal_x:.3f}")

        # define the mathematical target pose for the current goal
        goal_pose = current_pose.copy()
        goal_pose[0, 3] = goal_x

        # ================================================================
        # STEP 1: PLANNING AS INFERENCE (Find Optimal Tensions)
        # ================================================================
        # Tell the graph that the object MUST exist at the goal with almost no uncertainty (stiff sigmas)
        belief_solver.set_object(vdb_path, goal_pose, stiff_sigmas)

        # Tell the graph that there is lots of uncertainty about the tensions (free tendon cov) - we want it to find the optimal tensions to reach the goal
        tensions_free_in = [crest_sparse.VectorXGaussian(t, free_tension_cov) for t in current_tensions]

        # Solve. Let GTSAM back-prop the contact forces through the SDF to find the 
        # optimal tendon tensions to reach the goal
        plan_solution = belief_solver.solve(tensions_free_in, tip_wrenches)

        # Extract the MAP optimal tensions
        optimal_tensions = []
        for f_idx in range(num_fingers):
            try:
                opt_t = plan_solution.marginals.fingers[f_idx].tensions.mean
                optimal_tensions.append(opt_t)
            except AttributeError:
                print("\n[C++ ACTION REQUIRED] Your Python bindings are missing the tension variables!")
                print("You must add the tension variables to `TendonHandMarginals` in your C++ code so Python can read them.")                
                return
        
        # ================================================================
        # STEP 2: EXECUTE & OBSERVE (State Estimation / System ID)
        # ================================================================
        # Tell sim factorgraph that object is free tp move according to predefined sigmas
        simulation_solver.set_object(vdb_path, current_pose, rolling_sigmas)

        # Tell sim factorgraph that optimal tensions are locked
        tensions_frozen_in = [crest_sparse.VectorXGaussian(t, frozen_tension_cov) for t in optimal_tensions]

        # solve the pysical simulation
        sim_solution = simulation_solver.solve(tensions_frozen_in, tip_wrenches)

        # observe where the object *actually* ended up
        current_pose = sim_solution.marginals.object_pose.mean
        actual_x = current_pose[0, 3]

        print(f"  Commanded Flexor Tensions: {optimal_tensions[0][5]:.2f}N (L), {optimal_tensions[1][5]:.2f}N (R)")
        print(f"  Actual Object X: {actual_x:.4f} (Error: {abs(goal_x - actual_x):.4f})")

        # Update loop state for next iteration
        current_tensions = optimal_tensions

        if plotter is not None:
            solutions = {}
            finger_names = ["finger_left", "finger_right"]
            for name, finger_marginals in zip(finger_names, sim_solution.marginals.fingers):
                finger_solution = crest_sparse.TendonRobotSolution()
                finger_solution.marginals = finger_marginals
                finger_solution.meta = sim_solution.meta
                solutions[name] = finger_solution
            plotter.update(solutions, object_pose=current_pose)
            

def main(vdb_path=None):
    configs = get_opposing_finger_config(finger_type="index", finger_separation = 0.04)
    finger_names = [name for name, _ in configs]

    hand_solver_config = crest_sparse.TendonHandSolverConfig()
    simulation_solver = crest_sparse.TendonHandSolver(configs, hand_solver_config)
    belief_solver = crest_sparse.TendonHandSolver(configs, hand_solver_config)

    plotter = TendonHandPlotter(
        finger_names=finger_names,
        vdb_path=vdb_path,
        plot_object=True,
        camera_azimuth=180,
        camera_elevation=15,
        camera_focal_point=[0, 0.10, 0],
        camera_distance=0.7,
        window_size=(1200, 1200),
        plot_collision_spheres=True,
        single_plot_mode=True
    )

    # goal_sequence = [0.01, 0.02, 0.01, 0.0, -0.01, -0.02, -0.01, 0.0]
    goal_sequence = []
    waypoints = [0.0, 0.01, 0.02, 0.01, 0.0, -0.01, 0.0]
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i+1]
        # Break each segment into 10 tiny steps
        steps = np.linspace(start, end, num=10, endpoint=False)
        goal_sequence.extend(steps.tolist())
    
    goal_sequence.append(waypoints[-1]) # Add final goal

    if vdb_path is not None:
        run_inference_loop(simulation_solver, belief_solver, vdb_path, goal_sequence, plotter)


if __name__ == "__main__":
    vdb_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(vdb_path)
