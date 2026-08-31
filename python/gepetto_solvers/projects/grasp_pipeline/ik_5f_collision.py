"""Collision-only kinematic solve of the five-finger hand (no contact).

The simplest exercise of Section 1.5 collision avoidance: a single-shot
``TendonHandSolver`` kinematic solve (as in ``fk_5f_sweep.py`` -- tensions
in, poses out, no contact constraints anywhere), with the big grasp sphere
placed at the flexed-fingertip locus and the flexor tension cranked high enough
that the unconstrained hand curls *through* the sphere.

The hand is solved twice:
  1. collision OFF -- baseline; reports how deeply the spheres penetrate.
  2. collision ON  -- AL inequality constraints (sphere-to-SDF + cross-finger
     sphere-to-sphere) should hold every collision sphere out of the object and
     the fingers apart.

Because there is NO contact constraint, this isolates the inequality-constraint
path end-to-end: has_collision() routing to the AL solver, the collision-only
EnvironmentConfig guards, and both collision factor types.

Run (from the ``python/`` directory):
    python scripts/ik_5f_collision.py --no-viz
"""

import argparse
import os
import time

import numpy as np

import gepetto_solvers
from gepetto_solvers.core.diagnostics import collision_report
from gepetto_solvers.core.geometry.scene import (
    GRASP_SPHERE_CENTER,
    get_primitive_specs,
)
from gepetto_solvers.core.hand.config import (
    attach_collision,
    get_default_hand_configs,
    load_hand_dimensions,
)
from gepetto_solvers.core.objects import OBJECTS_DIR

# Above GRASP_FLEXOR_TENSION (2 N, tips exactly on the sphere): curl the
# unconstrained fingers well into the sphere so collision has work to do.
FLEXOR_TENSION = 3.0


def solve_hand(configs, args):
    hand_config = gepetto_solvers.TendonHandSolverConfig()
    hand_config.wrist_pose = np.eye(4)
    hand_config.sigma_wrist_pos = 1e-4
    hand_config.sigma_wrist_rot = 1e-3
    # NOTE: must be a Cholesky-based solver when collision (inequality
    # constraints) is on. GTSAM's AL optimizer builds the inequality Lagrange-
    # multiplier term with an AntiFactor whose linearization is a *negated*
    # HessianFactor; QR elimination cannot consume negative-information Hessians
    # (LM silently stalls), while Cholesky sums information matrices natively.
    hand_config.base.linear_solver_type = args.linear_solver
    hand_config.base.max_iterations = 500
    hand_config.base.al_initial_mu = args.al_mu
    hand_config.base.al_mu_increase_rate = args.al_rate
    hand_config.base.al_max_iterations = args.al_iters

    solver = gepetto_solvers.TendonHandSolver(configs, hand_config)

    num_tendons = configs[0][1].num_tendons
    # Tight passive tendons, loose flexor (the grasp-test pattern): the flexor
    # prior must be soft enough that the collision constraints can actually
    # drive it. A tight uniform prior (kinematics_test style) pins the tensions
    # so hard that the merit minimum KEEPS the penetration (deviating the flexor
    # by 1 N costs ~(1/sigma)^2/2 >> the collision penalty), and the AL outer
    # loop "converges" after one iteration with the fingers still inside the
    # object. Variance 1e-1 is still a proper prior, so the tension stays
    # determinate even when every collision constraint is inactive.
    tensions_mean = np.full(num_tendons, 0.5)
    tensions_mean[5] = FLEXOR_TENSION
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primitive", nargs="?", default="big_sphere")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--collision-radius", type=float, default=0.003)
    parser.add_argument("--collision-sigma", type=float, default=1.0,
                        help="Constraint row scaling (1.0 = same as contact rows).")
    parser.add_argument("--al-mu", type=float, default=1e4)
    parser.add_argument("--al-rate", type=float, default=2.0)
    # 25 outer iterations: the inequality violation halves per outer iteration
    # (mu doubles), and from ~26 mm of unconstrained penetration it takes ~20
    # doublings to get below the 0.1 mm pass threshold; 15 stalls at ~0.12 mm.
    parser.add_argument("--al-iters", type=int, default=25)
    parser.add_argument("--linear-solver", default="MULTIFRONTAL_CHOLESKY")
    parser.add_argument("--num-fingers", type=int, default=0,
                        help="Use only the first N digits (0 = all five). "
                             "1 isolates finger-object collision (no pairs).")
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

    # --- 1. Baseline: collision OFF, no environment at all ---
    print(f"[1/2] collision OFF (flexor {FLEXOR_TENSION} N):")
    configs_off = get_default_hand_configs(dims)
    if args.num_fingers > 0:
        configs_off = configs_off[:args.num_fingers]
    sol_off = solve_hand(configs_off, args)
    obj_off, ff_off = collision_report(configs_off, sol_off, spec, object_pose, r)
    print(f"  worst finger-object clearance: {obj_off:+.5f} m")
    print(f"  worst finger-finger gap:       {ff_off:+.5f} m")

    # --- 2. Collision ON (collision-only envs; no contact) ---
    print("[2/2] collision ON:")
    configs_on = get_default_hand_configs(dims)
    if args.num_fingers > 0:
        configs_on = configs_on[:args.num_fingers]
    # Tight object prior: the object pose is an optimized variable, and with the
    # default 1e-8 covariance (sigma 1e-4/dof) the collision constraints can
    # relieve penetration by shoving the *object* a few tenths of a millimeter
    # off its commanded pose -- the constraints are then satisfied w.r.t. the
    # shifted object while this test measures clearance against the commanded
    # pose. Pin the object (sigma 1e-6/dof) so the fingers do all the moving.
    attach_collision(configs_on, vdb_path, object_pose,
                     radius=r, sigma=args.collision_sigma,
                     object_pose_cov=1e-12 * np.eye(6))
    sol_on = solve_hand(configs_on, args)
    obj_on, ff_on = collision_report(configs_on, sol_on, spec, object_pose, r)
    print(f"  worst finger-object clearance: {obj_on:+.5f} m")
    print(f"  worst finger-finger gap:       {ff_on:+.5f} m")

    print()
    print(f"object clearance: {obj_off:+.5f} -> {obj_on:+.5f}  "
          f"(want OFF < 0 < ON)")
    print(f"finger gap:       {ff_off:+.5f} -> {ff_on:+.5f}")
    ok = obj_on >= -1e-4 and ff_on >= -1e-4
    print("RESULT:", "PASS (no penetration with collision ON)"
          if ok else "FAIL (penetration present with collision ON)")

    if args.no_viz:
        return

    from gepetto_solvers.core.plotting.tendon_hand_plotter import (
        TendonHandMultiViewPlotter,
    )

    class _FingerSol:
        pass

    finger_names = [name for name, _ in configs_on]
    solutions = {}
    for name, fm in zip(finger_names, sol_on.marginals.fingers):
        s = _FingerSol()
        s.marginals = fm
        s.meta = sol_on.meta
        solutions[name] = s
    # Four windows around the hand (three azimuths 90 deg apart + near-top-down)
    # with the collision object rendered, so penetration is visible from any side.
    plotter = TendonHandMultiViewPlotter(
        finger_names,
        plot_backbone_ellipsoids=False,
        camera_focal_point=list(GRASP_SPHERE_CENTER),
        camera_distance=0.5,
        primitives=[dict(spec["plot"](GRASP_SPHERE_CENTER),
                         color="goldenrod", opacity=0.35)],
    )
    plotter.update(solutions)
    input("Press Enter to close...")


if __name__ == "__main__":
    main()
