"""Parameter sweep for the SDF witness-point tip-contact solve.

The 3-residual SdfContactFactor solve (sdf_3dof_contact_kinematics_test.py)
throws IndeterminantLinearSystem near the object pose O0. Two compounding
reasons:

  1. The object (sphere.vdb) is rotationally symmetric, so its SDF world normal
     is independent of the object's rotation -> the contact factor gives O0's
     3 rotation DoF ~zero information. They are held only by the pose prior.
  2. The inner LM sub-solver of the AL optimizer uses GTSAM defaults
     (lambda_initial=1e-5, diagonal_damping=False) which are tuned for starts
     near the optimum; on this stiff, far-from-contact start the first linear
     solve is ill-posed (see SolverBase.h:41-48).

This script sweeps the AL / LM knobs (and the object-pose prior tightness),
catches the exception per combo, and prints which settings solve and the
resulting signed surface gap. Run headless:

    python -m python.tests.tendon_finger.sdf_3dof_contact_param_search
"""
import os
import time
import itertools
import numpy as np

import crest_sparse

from .config import get_6tendon_config


# Mirrored across x=0 (X negated): the 180-deg CAD tendon-routing flip curls the
# flexor toward world -X, so the target sphere moves to the -X side too.
SPHERE_CENTER = np.array([-6.02088876e-02, 3.77734425e-02, 0.0])
SPHERE_RADIUS = 0.025
TIP_RADIUS = 0.003


def build_config(al_initial_mu, al_mu_increase_rate, al_max_iterations,
                 lambda_initial, diagonal_damping, max_iterations,
                 obj_pose_cov_scale):
    config = get_6tendon_config()

    num_nodes = config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    tip_node_index = num_nodes - 1

    objects_dir = os.path.join(os.path.dirname(__file__), "..", "_objects")
    vdb_path = os.path.normpath(os.path.join(objects_dir, "sphere.vdb"))

    object_pose = np.eye(4)
    object_pose[0:3, 3] = SPHERE_CENTER

    env = crest_sparse.EnvironmentConfig()
    env.load_sdf(vdb_path)
    env.object_pose_mean = object_pose
    env.object_pose_cov = obj_pose_cov_scale * np.eye(6)
    env.object_pose_per_step = False
    env.target_contact_node = tip_node_index
    env.contact_node_radius = TIP_RADIUS
    config.sdf_contact = env

    config.base.al_initial_mu = al_initial_mu
    config.base.al_mu_increase_rate = al_mu_increase_rate
    config.base.al_max_iterations = al_max_iterations
    config.base.lambda_initial = lambda_initial
    config.base.diagonal_damping = diagonal_damping
    config.base.max_iterations = max_iterations

    return config, tip_node_index


def run_one(params):
    config, _ = build_config(**params)
    solver = crest_sparse.TendonFingerSolver(config)

    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    tensions_cov = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)

    tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
    tip_wrench = crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)

    t0 = time.time()
    solution = solver.solve(tensions, tip_wrench, None)
    dt_ms = (time.time() - t0) * 1000.0

    tip_pose = np.array(solution.marginals.rod.states[-1].pose.mean)
    tip_pos = tip_pose[:3, 3]
    gap = np.linalg.norm(tip_pos - SPHERE_CENTER) - (TIP_RADIUS + SPHERE_RADIUS)
    return {
        "gap": gap,
        "iters": solution.meta.iterations,
        "error": solution.meta.error,
        "tension5": solution.marginals.tensions.mean[5],
        "dt_ms": dt_ms,
    }


def main():
    # Grid. Listed inner-to-outer; the LM damping knobs (lambda_initial,
    # diagonal_damping) are the prime suspects, so they vary fastest.
    grid = {
        "al_initial_mu":       [1.0, 10.0, 100.0],
        "al_mu_increase_rate": [2.0],
        "al_max_iterations":   [40],
        "max_iterations":      [100],
        "obj_pose_cov_scale":  [1e-8, 1e-6],
        "lambda_initial":      [1e-5, 1e-2, 1.0],
        "diagonal_damping":    [False, True],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Sweeping {len(combos)} combinations...\n")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        tag = (f"mu0={params['al_initial_mu']:<6g} rate={params['al_mu_increase_rate']:<4g} "
               f"objcov={params['obj_pose_cov_scale']:<6g} "
               f"lam0={params['lambda_initial']:<6g} diag={str(params['diagonal_damping']):<5}")
        try:
            r = run_one(params)
            status = "OK   "
            detail = (f"gap={r['gap']:+.5f}m iters={r['iters']:<3d} "
                      f"err={r['error']:.3g} T5={r['tension5']:.2f} {r['dt_ms']:.0f}ms")
            results.append((abs(r["gap"]), params, r))
        except RuntimeError as e:
            msg = str(e).strip().splitlines()
            indet = any("Indeterminant" in m for m in msg)
            status = "FAIL "
            detail = "IndeterminantLinearSystem" if indet else " ".join(msg)[:80]
        print(f"[{status}] {tag} | {detail}")

    print("\n==== Converged combos, sorted by |gap| ====")
    if not results:
        print("None converged. The LM damping knobs did not resolve the O0 indeterminacy;")
        print("the next lever is structural (see script docstring / factor notes).")
        return
    results.sort(key=lambda x: x[0])
    for abs_gap, params, r in results[:15]:
        print(f"|gap|={abs_gap:.6f}m  gap={r['gap']:+.5f}  iters={r['iters']:<3d} "
              f"err={r['error']:.3g} | mu0={params['al_initial_mu']} "
              f"objcov={params['obj_pose_cov_scale']} lam0={params['lambda_initial']} "
              f"diag={params['diagonal_damping']}")


if __name__ == "__main__":
    main()
