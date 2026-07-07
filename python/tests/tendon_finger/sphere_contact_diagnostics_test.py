import argparse
import os
import time

import numpy as np

import crest_sparse
from .._plotting.solver_diagnostics_plotter import SolverDiagnosticsPlotter

from .config import get_6tendon_config


_DIAG_SAVE_PATH = "figures/sphere_contact_diagnostics.png"


def _make_3d_plotter(config, sphere_center, sphere_radius, tip_radius):
    """Imported lazily so the test still runs (with --no-3d) on machines where
    VTK/OpenGL fails to create a GLX context."""
    from .._plotting.tendon_finger_plotter import TendonFingerPlotter
    num_nodes = (
        config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    )
    tip_node_index = num_nodes - 1
    return TendonFingerPlotter(
        plot_backbone_frames=True,
        contact_node_index=tip_node_index,
        contact_node_radius=tip_radius,
        sphere_primitives=[
            {"center": sphere_center, "radius": sphere_radius,
             "color": "goldenrod", "opacity": 0.35}
        ],
        camera_azimuth=165,
        camera_elevation=20,
        camera_focal_point=[0, 0.1, 0],
    )


def _build_solver(record_iterations: bool, sample_interval: int):
    config = get_6tendon_config()

    # Mirrored across x=0 (X negated): the 180-deg CAD tendon-routing flip curls the
    # flexor toward world -X, so the target sphere moves to the -X side too.
    sphere_center = np.array([-6.02088876e-02, 3.77734425e-02, 0.0])
    sphere_radius = 0.025
    tip_radius = 0.003

    sc = crest_sparse.SpherePrimitiveContactConfig()
    sc.finger_node_index = -1
    sc.finger_node_radius = tip_radius
    sc.sphere_center = sphere_center
    sc.sphere_radius = sphere_radius
    sc.sphere_pose_cov = 1e-8 * np.eye(6)
    config.sphere_contact = sc

    config.base.record_iterations = record_iterations
    config.base.iteration_sample_interval = sample_interval

    solver = crest_sparse.TendonFingerSolver(config)
    return solver, config, sphere_center, sphere_radius, tip_radius


def _make_inputs():
    tensions_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 3.0])
    tensions_cov = np.diag([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-1])
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
    tip_wrench = crest_sparse.Vector6Gaussian(np.zeros(6), tip_wrench_cov)
    return tensions, tip_wrench


def _print_result(solution, sphere_center, tip_radius, sphere_radius):
    tip_pose = np.array(solution.marginals.rod.states[-1].pose.mean)
    tip_pos = tip_pose[:3, 3]
    gap = np.linalg.norm(tip_pos - sphere_center) - (tip_radius + sphere_radius)
    print(
        f"  iters={solution.meta.iterations} | error={solution.meta.error:.4g} | "
        f"total={solution.meta.total_time_ms:.1f} ms"
    )
    print(f"  tip position: {tip_pos}")
    print(f"  signed surface gap: {gap:+.5f} m  (target ~0)")
    print(f"  active tendon (5) tension: {solution.marginals.tensions.mean[5]:.3f}")


def _gather_diagnostics(solver, solution, diag):
    """Pull every diagnostic the solver exposes and stuff it into the plotter."""
    diag.record(
        solution=solution,
        factor_error_summary=solver.get_factor_error_summary(),
        factor_errors_by_type=solver.get_factor_errors_by_type(),
        initial_factor_summary=solver.get_initial_factor_error_summary(),
        hessian=solver.get_hessian_and_gradient()[0],
    )


# ---------------------------------------------------------------------------

def _save_diagnostics(diag):
    """Build the diagnostics figure and save it to disk (no window shown).

    Showing the matplotlib window before pyvista initializes can grab GUI/GL
    resources that conflict with VTK's GLX context, so we only save here and
    defer showing until after the pyvista plotter is up.
    """
    os.makedirs(os.path.dirname(_DIAG_SAVE_PATH), exist_ok=True)
    diag.build()
    diag.save(_DIAG_SAVE_PATH)


def run_final(solver, config, sphere_center, sphere_radius, tip_radius,
              show_3d: bool):
    """Single solve — final solution + full diagnostics."""
    tensions, tip_wrench = _make_inputs()

    t0 = time.time()
    solution = solver.solve(tensions, tip_wrench, None)
    dt_ms = (time.time() - t0) * 1000.0

    print(f"Solved in {dt_ms:.1f} ms")
    _print_result(solution, sphere_center, tip_radius, sphere_radius)

    n_errors = len(solution.meta.iteration_errors)
    if n_errors:
        print(f"  iteration_errors length: {n_errors}")
        print(f"  step_norms length:        {len(solution.meta.iteration_step_norms)}")
        print(f"  trust_region range: "
              f"[{min(solution.meta.iteration_trust_region):.3g}, "
              f"{max(solution.meta.iteration_trust_region):.3g}]")

    diag = SolverDiagnosticsPlotter(title="Sphere Contact Kinematics — Diagnostics")
    _gather_diagnostics(solver, solution, diag)
    _save_diagnostics(diag)

    if show_3d:
        plotter = _make_3d_plotter(config, sphere_center, sphere_radius, tip_radius)
        plotter.update(solution)
        plotter.update_initial(solver.get_initial_solution())

    diag.show(block=False)
    input("Press Enter to close...")


def run_step(solver, config, sphere_center, sphere_radius, tip_radius,
             show_3d: bool, step_delay_s: float = 0.05):
    """Animate through intermediate solutions from a single solve."""
    tensions, tip_wrench = _make_inputs()

    print(f"Running solver with record_iterations=True, "
          f"iteration_sample_interval={config.base.iteration_sample_interval} ...")
    t0 = time.time()
    solution = solver.solve(tensions, tip_wrench, None)
    dt_ms = (time.time() - t0) * 1000.0

    print(f"Solved in {dt_ms:.1f} ms")
    _print_result(solution, sphere_center, tip_radius, sphere_radius)

    intermediates = solver.get_intermediate_solutions()
    initial_sol = solver.get_initial_solution()
    print(f"Loaded {len(intermediates)} intermediate solution snapshots.")

    # Save diagnostics PNG first so the user still gets them even if 3D fails;
    # defer the interactive matplotlib window until after pyvista is up.
    diag = SolverDiagnosticsPlotter(title="Sphere Contact Kinematics — Step Diagnostics")
    _gather_diagnostics(solver, solution, diag)
    _save_diagnostics(diag)

    if show_3d:
        plotter = _make_3d_plotter(config, sphere_center, sphere_radius, tip_radius)
        plotter.update(solution)
        plotter.update_initial(initial_sol)
        for i, inter in enumerate(intermediates):
            print(f"  snapshot {i+1}/{len(intermediates)}", end="\r")
            plotter.update(inter)
            time.sleep(step_delay_s)
        print()
        plotter.update(solution)

    diag.show(block=False)
    input("Press Enter to close...")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["final", "step"],
        default="final",
        help=(
            "'final': run once and show final solution + diagnostics (default). "
            "'step': animate through intermediate solutions from the single solve."
        ),
    )
    parser.add_argument(
        "--sample-interval", type=int, default=1,
        help="Store a Values snapshot every N iterations (step mode). Default 1.",
    )
    parser.add_argument(
        "--no-3d", action="store_true",
        help=(
            "Skip the pyvista 3D plotter. Diagnostics PNG is still saved/shown. "
            "Use this if OpenGL/GLX fails on your machine."
        ),
    )
    args = parser.parse_args()

    record = True  # always track iterations for diagnostics
    sample_interval = args.sample_interval if args.mode == "step" else 0
    show_3d = not args.no_3d

    solver, config, sphere_center, sphere_radius, tip_radius = _build_solver(
        record_iterations=record,
        sample_interval=sample_interval,
    )

    if args.mode == "step":
        run_step(solver, config, sphere_center, sphere_radius, tip_radius,
                 show_3d=show_3d, step_delay_s=0.05)
    else:
        run_final(solver, config, sphere_center, sphere_radius, tip_radius,
                  show_3d=show_3d)


if __name__ == "__main__":
    main()
