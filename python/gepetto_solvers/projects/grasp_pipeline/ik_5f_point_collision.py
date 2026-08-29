"""Collision-free *point-to-point* kinematic solve of the five-finger hand.

This is the position-goal counterpart of ``ik_5f_collision.py`` and the
single-shot analogue of ``traj_5f_point_collision.py``.
A single ``TendonHandSolver`` solve (tensions in, poses out) is driven by two
things at once:

  * per-finger world-frame **tip-position goals** -- soft ``PositionPriorFactor``s
    on each finger's tip node (the solver's ``goal_positions``/``goal_position_cov``
    config, mirrored from the trajectory planner). The hand starts open (flexor
    slack) and the goals pull each fingertip toward its target point.
  * **collision avoidance** -- Section 1.5 AL inequality constraints
    (sphere-to-SDF + cross-finger sphere-to-sphere) that keep every collision
    sphere out of the object and the fingers apart.

The goals want to reach *points*, and collision should keep the reaching from
penetrating the object -- a "plan-to-a-point, collision-free" static solve.

======================================================================
KNOWN LIMITATION (2026-07-20): this single-shot test currently FAILS, and
it is a *solver* limitation, not a bug in the point-goal wiring or the
scenario. The point-goal C++ feature works (with collision OFF the hand
reaches these goals sub-mm); the collision avoidance does not enforce from
a cold start.

Why: from a cold (straight-hand) start the goal-driven solve jumps straight
to the goal-optimal configuration, which for the big-sphere grasp threads the
finger backbones *through* the sphere (the main fingers spear it head-on --
only a wrap-around posture is collision-free, and the tangent-contact solve is
what finds that basin). The single-shot Augmented-Lagrangian loop then cannot
back the solution out: the tight goal prior (cov 1e-5) pins the primal, so the
inner LM returns the same penetrating point every outer iteration and the AL
multiplier grows with the solution never moving. Forcing the AL tolerances
(al_abs_cost_tol=1e12, al_rel_*_tol->0) does not help -- there is no
binding-and-reaching regime single-shot. Even collision-ONLY single-shot
(ik_5f_collision, no goals) only relieves a fraction of the
penetration.

The *trajectory* planner does not have this problem because it enforces
collision incrementally from a collision-free k=0 and never enters the
penetrating basin -- see traj_5f_point_collision.py,
which PASSES (collision-free + reaches these same goals). Making the single-shot
solver enforce collision would need a real fix: a mu schedule that keeps
collision active against the goal prior, or warm-starting the single-shot from
the trajectory's collision-free approach (TendonHandSolver retains values_
across solve() calls and exposes set_wrist_pose for exactly this).
======================================================================

Run (from the ``python/`` directory):
    python scripts/ik_5f_point_collision.py --no-viz
"""

import os
import argparse
import time

import numpy as np

import gepetto_solvers

from gepetto_solvers.core.objects import OBJECTS_DIR
from gepetto_solvers.core.hand.config import (
    get_default_hand_configs, load_hand_dimensions, attach_collision)
from gepetto_solvers.core.geometry.scene import (
    get_primitive_specs, primitive_surface_gap, GRASP_SPHERE_CENTER, GRASP_GOALS)
# Reuse the collision-clearance report (worst finger-object / finger-finger gap).
from gepetto_solvers.core.diagnostics import collision_report

def solve_hand(configs, args, goal_positions=None):
    """Single-shot hand solve. When ``goal_positions`` is given (one Vector3 per
    finger) the solver adds a soft tip-position prior per finger; otherwise the
    solve is driven purely by the tension priors (baseline)."""
    hand_config = gepetto_solvers.TendonHandSolverConfig()
    hand_config.wrist_pose = np.eye(4)
    hand_config.sigma_wrist_pos = 1e-4
    hand_config.sigma_wrist_rot = 1e-3
    # Cholesky is REQUIRED once collision (AL inequality constraints) is on: the
    # AntiFactor Lagrange term linearizes to a negated Hessian that QR cannot
    # consume (LM silently stalls); Cholesky sums information matrices natively.
    hand_config.base.linear_solver_type = args.linear_solver
    hand_config.base.max_iterations = 500
    hand_config.base.al_initial_mu = args.al_mu
    hand_config.base.al_mu_increase_rate = args.al_rate
    hand_config.base.al_max_iterations = args.al_iters
    # Absolute cost threshold defaults to 1e-5, which (with the AL penalty cost far
    # above it) lets the outer loop declare convergence the moment the *relative*
    # cost/violation change stalls -- so from a cold straight-hand start the solve
    # quits after ~2 iterations with the fingers still through the sphere. Lift it
    # (as the collision trajectory test does) so mu can actually ramp.
    hand_config.base.al_abs_cost_tol = 1e12
    hand_config.base.al_inner_rel_tol_initial = 0.01

    if goal_positions is not None:
        hand_config.goal_positions = [np.asarray(g, dtype=float)
                                      for g in goal_positions]
        hand_config.goal_position_cov = args.goal_cov * np.eye(3)

    solver = gepetto_solvers.TendonHandSolver(configs, hand_config)

    num_tendons = configs[0][1].num_tendons
    # Tight passive tendons, loose flexor (the collision-test pattern): the flexor
    # prior must be soft enough that the goal priors (and collision constraints)
    # can drive it. A tight uniform prior pins the tensions so hard that the merit
    # minimum keeps the hand open / keeps penetration and the AL outer loop quits
    # after one iteration. Variance 1e-1 is still a proper prior, so tensions stay
    # determinate even when every collision constraint is inactive.
    tensions_mean = np.full(num_tendons, 0.5)     # start open: flexor slack too
    tensions_cov = np.diag([1e-6] * (num_tendons - 1) + [1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    all_tensions = [gepetto_solvers.VectorXGaussian(tensions_mean, tensions_cov)
                    for _ in configs]
    all_tip_wrenches = [gepetto_solvers.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
                        for _ in configs]

    t0 = time.time()
    solution = solver.solve(all_tensions, all_tip_wrenches)
    dt_ms = (time.time() - t0) * 1000.0
    print(f"  solved in {dt_ms:.1f} ms | iters={solution.meta.iterations} | "
          f"error={solution.meta.error:.4g}")
    return solution


def tip_positions(configs, solution):
    """World-frame tip position of every finger in a hand solution."""
    tips = []
    for (_, _cfg), fm in zip(configs, solution.marginals.fingers):
        tips.append(np.array(fm.rod.states[-1].pose.mean)[:3, 3])
    return tips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primitive", nargs="?", default="big_sphere")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--collision-radius", type=float, default=0.003)
    parser.add_argument("--collision-sigma", type=float, default=1.0,
                        help="Constraint row scaling (1.0 = same as contact rows).")
    parser.add_argument("--goal-cov", type=float, default=1e-5,
                        help="Per-finger goal-position prior variance (diag of "
                             "goal_position_cov). Smaller => tips pulled tighter "
                             "onto the goal points.")
    parser.add_argument("--al-mu", type=float, default=1.0,
                        help="Initial AL penalty. Low (1.0) + a ramp lets the outer "
                             "loop run; a large mu0 makes AL 'converge' in a couple "
                             "of iterations before the penalty can push fingers out.")
    parser.add_argument("--al-rate", type=float, default=2.0)
    parser.add_argument("--al-iters", type=int, default=40)
    parser.add_argument("--linear-solver", default="MULTIFRONTAL_CHOLESKY")
    parser.add_argument("--num-fingers", type=int, default=0,
                        help="Use only the first N digits (0 = all five).")
    args = parser.parse_args()

    spec = get_primitive_specs()[args.primitive]
    object_rotation = np.asarray(spec.get("rotation", np.eye(3)), dtype=float)
    object_pose = np.eye(4)
    object_pose[:3, :3] = object_rotation
    object_pose[:3, 3] = GRASP_SPHERE_CENTER

    objects_dir = OBJECTS_DIR
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python scripts/objects/make_.py{args.primitive} (run from python/).")

    dims = load_hand_dimensions()
    r = args.collision_radius

    def build_configs():
        cfgs = get_default_hand_configs(dims)
        return cfgs[:args.num_fingers] if args.num_fingers > 0 else cfgs

    # Reachable, collision-free grasp goals (see GRASP_GOALS). Sliced to the
    # requested finger count.
    n_use = args.num_fingers if args.num_fingers > 0 else len(GRASP_GOALS)
    goals = [GRASP_GOALS[i] for i in range(n_use)]
    print(f"[1/2] using {len(goals)} collision-free grasp goal points.")

    # --- Collision ON + point goals ---
    print(f"[2/2] collision ON + point-to-point goals:")
    configs_on = build_configs()
    # Pin the object (sigma 1e-6/dof) so the fingers do all the moving; otherwise
    # the constraints can relieve penetration by shoving the object a fraction of
    # a mm off its commanded pose.
    attach_collision(configs_on, vdb_path, object_pose,
                     radius=r, sigma=args.collision_sigma,
                     object_pose_cov=1e-12 * np.eye(6))
    sol_on = solve_hand(configs_on, args, goal_positions=goals)

    # --- Report: collision clearance AND per-finger tip->goal distance ---
    obj_on, ff_on = collision_report(configs_on, sol_on, spec, object_pose, r)
    tips_on = tip_positions(configs_on, sol_on)
    dists = [float(np.linalg.norm(t - g)) for t, g in zip(tips_on, goals)]

    print(f"\nresults:")
    print(f"  worst finger-object clearance: {obj_on:+.5f} m  (want >= 0)")
    print(f"  worst finger-finger gap:       {ff_on:+.5f} m  (want >= 0)")
    print("\n  per-finger tip -> goal distance:")
    for (name, _), d in zip(configs_on, dists):
        print(f"    [{name:>6}] {d:.5f} m")
    print(f"  worst tip->goal distance:      {max(dists):.5f} m")

    clearance_ok = obj_on >= -1e-4 and ff_on >= -1e-4
    # The goals are collision-free grasp tips (~on the surface + collision radius),
    # so a collision-constrained tip can reach them to within ~the collision radius.
    reach_tol = r + 3e-3
    reach_ok = max(dists) <= reach_tol
    print(f"\nRESULT:", "PASS" if (clearance_ok and reach_ok) else "FAIL",
          f"(collision-free: {clearance_ok}, reached goals within "
          f"{reach_tol*1e3:.1f} mm: {reach_ok})")

    if args.no_viz:
        return

    from gepetto_solvers.core.plotting.tendon_hand_plotter import TendonHandMultiViewPlotter

    class _FingerSol:
        pass

    finger_names = [name for name, _ in configs_on]
    solutions = {}
    for name, fm in zip(finger_names, sol_on.marginals.fingers):
        s = _FingerSol()
        s.marginals = fm
        s.meta = sol_on.meta
        solutions[name] = s
    plotter = TendonHandMultiViewPlotter(
        finger_names,
        plot_backbone_ellipsoids=False,
        camera_focal_point=list(GRASP_SPHERE_CENTER),
        camera_distance=0.5,
        primitives=[dict(spec["plot"](GRASP_SPHERE_CENTER),
                         color="goldenrod", opacity=0.35)],
    )
    plotter.update(solutions)
    # Scatter the per-finger goal points as red target markers (best-effort).
    try:
        for view in plotter.plotters:
            view.plotter.add_points(np.array(goals), color="red", point_size=14,
                                    render_points_as_spheres=True)
    except Exception as exc:  # pragma: no cover - viz-only convenience
        print(f"(could not render goal points: {exc})")
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
