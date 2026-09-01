"""Shared helpers for the tendon-hand scripts: run logging, planner/prior
parameter dumps, conditioning reports, and two small result adapters.

The logging half was consolidated here from ``tests/tendon_finger/utils.py``,
which used to carry a near-duplicate ``PlannerLogger`` -- half the hand scripts
imported it from there and half from here. There is one copy now.
"""

import itertools
import os
import sys
from datetime import datetime

import numpy as np

from .geometry.scene import primitive_surface_gap
from .hands.tendon_5f import (
    disc_node_indices,
    proximal_disc_flags,
)


class _Tee:
    """File-like object that writes to multiple streams (e.g. stdout + a file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class PlannerLogger:
    """
    Redirects stdout to both the terminal and a text file in the tendon_finger
    directory (or log_dir). With timestamp=True the file is uniquely named
    <planner_name>_<YYYYmmdd_HHMMSS>.log; with timestamp=False it is simply
    <planner_name>.log, so re-running the same experiment overwrites its log.
    """
    def __init__(self, planner_name, log_dir=None, timestamp=True):
        if log_dir is None:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(log_dir, exist_ok=True)
        if timestamp:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{planner_name}_{stamp}.log"
        else:
            filename = f"{planner_name}.log"
        self.path = os.path.join(log_dir, filename)
        self._file = open(self.path, "w")
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._file)
        print(f"Logging to: {self.path}")

    def close(self):
        if self._file is not None:
            sys.stdout = self._orig_stdout
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _format_value(v, indent=4):
    """Pretty-format a value for logging. Handles numpy arrays nicely."""
    if isinstance(v, np.ndarray):
        with np.printoptions(precision=6, suppress=True, linewidth=120):
            text = np.array2string(v)
        pad = " " * indent
        return ("\n" + pad).join(text.splitlines())
    return repr(v)


def _dump_attrs(obj, indent=2, _seen=None):
    """Recursively dump attributes of a config-like object."""
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return "<cycle>"
    _seen.add(oid)

    pad = " " * indent
    lines = []
    # Prefer __dict__ if present; otherwise fall back to dir().
    attrs = []
    if hasattr(obj, "__dict__") and obj.__dict__:
        attrs = list(obj.__dict__.keys())
    else:
        attrs = [a for a in dir(obj) if not a.startswith("_")]

    for name in sorted(attrs):
        if name.startswith("_"):
            continue
        try:
            v = getattr(obj, name)
        except Exception as e:
            lines.append(f"{pad}{name} = <error: {e}>")
            continue
        if callable(v):
            continue
        # Recurse into nested config-like objects (no __dict__ on pybind objs,
        # but they expose attributes via dir(); only recurse one or two levels
        # to keep output readable).
        if hasattr(v, "__dict__") and v.__dict__ and indent < 8:
            lines.append(f"{pad}{name}:")
            lines.append(_dump_attrs(v, indent + 2, _seen))
        else:
            lines.append(f"{pad}{name} = {_format_value(v, indent + 4)}")
    return "\n".join(lines)


def log_planner_parameters(planner_config, environment=None, extras=None):
    """Print planner + environment parameters in a readable block."""
    print("=" * 72)
    print("PLANNER CONFIG")
    print("=" * 72)
    print(_dump_attrs(planner_config))
    if environment is not None:
        print("=" * 72)
        print("ENVIRONMENT CONFIG")
        print("=" * 72)
        print(_dump_attrs(environment))
    if extras:
        print("=" * 72)
        print("EXTRA PARAMETERS")
        print("=" * 72)
        for k, v in extras.items():
            print(f"  {k} = {_format_value(v, 4)}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Solve-landscape diagnostics: priors + conditioning
# ---------------------------------------------------------------------------
#
# These help answer "what shapes the solve space, and how well-conditioned is
# it?" -- the prior table lists every soft prior / constraint noise model with
# its standard deviation and information weight (precision = 1/sigma^2, the
# quantity that actually scales that residual's contribution to the normal
# equations), and the conditioning report linearizes the solved graph to expose
# the Hessian's eigenvalue spread (condition number, near-null / gauge
# directions) and which factor types dominate the residual.


def _prior_std_precision(entry):
    """Return (std_vector, precision_vector) from a prior entry.

    An entry provides its noise via exactly one of:
        sigma : standard deviation (scalar or 1-D array)
        var   : variance          (scalar or 1-D array)
        cov   : covariance matrix  (diagonal is used)
    Precision is 1 / var, the weight each residual gets in the normal equations.
    """
    if "sigma" in entry:
        std = np.atleast_1d(np.asarray(entry["sigma"], dtype=float))
        var = std ** 2
    elif "var" in entry:
        var = np.atleast_1d(np.asarray(entry["var"], dtype=float))
        std = np.sqrt(var)
    elif "cov" in entry:
        cov = np.asarray(entry["cov"], dtype=float)
        var = np.atleast_1d(np.diag(cov) if cov.ndim == 2 else cov)
        std = np.sqrt(var)
    else:
        raise KeyError(f"prior entry {entry.get('name')!r} needs sigma/var/cov")
    with np.errstate(divide="ignore"):
        prec = np.where(var > 0, 1.0 / var, np.inf)
    return std, prec


def _fmt_range(v):
    """Compact scalar-or-range string for a per-component vector."""
    v = np.atleast_1d(v)
    lo, hi = float(np.min(v)), float(np.max(v))
    if np.isclose(lo, hi):
        return f"{lo:.3g}"
    return f"{lo:.3g}..{hi:.3g}"


def log_prior_table(entries, title="PRIORS / CONSTRAINT WEIGHTS"):
    """Print every soft prior / constraint noise model with its weight.

    ``entries`` is a list of dicts, each describing one prior:
        name   : short label (e.g. "wrist pose @ k=0")
        factor : the factor / PDF equation it corresponds to (optional)
        one of sigma / var / cov (see :func:`_prior_std_precision`)
        note   : free-form annotation (optional)

    For each we print the standard deviation (as a scalar or lo..hi range over
    components) and the information weight precision = 1/sigma^2. A tiny sigma is
    a stiff/near-hard prior (huge precision); a large sigma is a loose one. When
    two priors act on the same variable, their precision *ratio* is what sets the
    trade-off the optimizer sees -- the numbers here make those ratios legible.
    """
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"  {'prior':<26} {'std (sigma)':<16} {'precision 1/sig^2':<18} factor")
    print("  " + "-" * 84)
    for e in entries:
        std, prec = _prior_std_precision(e)
        name = e.get("name", "?")
        factor = e.get("factor", "")
        print(f"  {name:<26} {_fmt_range(std):<16} {_fmt_range(prec):<18} {factor}")
        if e.get("note"):
            print(f"  {'':<26} └─ {e['note']}")
    print("=" * 72)


def log_conditioning_report(diag_source, top_k=8, near_null_tol=1e-9,
                            max_dense_dim=4000):
    """Linearize the solved graph and report its conditioning + residual makeup.

    ``diag_source`` is any solved solver/planner exposing
    ``get_hessian_and_gradient()``, ``get_factor_error_summary()`` and
    ``get_initial_factor_error_summary()`` (TendonFingerSolver and, once its
    diagnostics are bound, HandTrajectoryPlanner both qualify).

    Reports, at the final solution:
      * Hessian size, gradient norm (residual gradient of the linearized cost);
      * eigenvalue spread of H (symmetric, via eigvalsh): smallest / largest and
        the 2-norm condition number lambda_max/lambda_min -- a large value or a
        cluster of near-zero eigenvalues flags gauge freedom / ill-conditioning
        that makes the solve slow or the covariance untrustworthy;
      * the count of near-null directions (|lambda| < ``near_null_tol``);
      * the top factor types by total error, at both the initial guess and the
        final solution, so you can see which factors dominate the landscape and
        how much the solve reduced each.
    """
    print("=" * 72)
    print("CONDITIONING / LANDSCAPE")
    print("=" * 72)

    try:
        H, g = diag_source.get_hessian_and_gradient()
        H = np.asarray(H, dtype=float)
        g = np.asarray(g, dtype=float)
    except Exception as exc:  # noqa: BLE001
        print(f"  Hessian unavailable ({exc.__class__.__name__}: {exc})")
        H = None

    if H is not None and H.size and H.shape[0] > max_dense_dim:
        print(f"  Hessian: {H.shape[0]}x{H.shape[1]}   |gradient| = "
              f"{np.linalg.norm(g):.4g}")
        print(f"  (dense eigendecomposition skipped: dim > {max_dense_dim}; "
              f"raise max_dense_dim to force it)")
        H = None
    if H is not None and H.size:
        # Symmetrize defensively; use the symmetric eigensolver.
        eig = np.linalg.eigvalsh(0.5 * (H + H.T))
        abs_eig = np.abs(eig)
        lam_min, lam_max = float(abs_eig.min()), float(abs_eig.max())
        cond = lam_max / lam_min if lam_min > 0 else np.inf
        near_null = int(np.count_nonzero(abs_eig < near_null_tol))
        print(f"  Hessian: {H.shape[0]}x{H.shape[1]}   |gradient| = "
              f"{np.linalg.norm(g):.4g}")
        print(f"  |eig| range: [{lam_min:.4g}, {lam_max:.4g}]   "
              f"cond(H) = {cond:.4g}")
        print(f"  smallest 5 |eig|: "
              f"{np.array2string(np.sort(abs_eig)[:5], precision=3)}")
        if near_null:
            print(f"  ** {near_null} near-null direction(s) (|eig| < "
                  f"{near_null_tol:g}) — gauge freedom / rank deficiency **")

    def _summary(getter, label):
        try:
            rows = getter()
        except Exception as exc:  # noqa: BLE001
            print(f"  {label} factor summary unavailable "
                  f"({exc.__class__.__name__}: {exc})")
            return
        print(f"\n  Factor error by type ({label}), top {top_k}:")
        print(f"    {'total_err':>12}  {'count':>6}  type")
        for name, count, err in list(rows)[:top_k]:
            print(f"    {err:12.4g}  {count:>6}  {name}")

    _summary(diag_source.get_initial_factor_error_summary, "initial guess")
    _summary(diag_source.get_factor_error_summary, "final solution")
    print("=" * 72)


def report_al_iterations(result, results_dir=None, exp_label=None):
    """Print the per-outer-iteration Augmented-Lagrangian trace (cost / violation
    / mu) and, when ``results_dir``/``exp_label`` are given and the plotter is
    importable, save an AL-convergence figure.

    Requires ``config.base.record_iterations = True`` on the solve. Returns True
    if a trace was present. Headless-safe (only ever saves a PNG).
    """
    costs = list(result.meta.al_iteration_costs)
    viols = list(result.meta.al_iteration_violations)
    mus = list(result.meta.al_iteration_mus)
    if not costs:
        print("\n[debug-iterations] no AL iteration trace was recorded "
              "(record_iterations off, or the solve took the non-AL path).")
        return False

    print("\nAL outer-iteration trace:")
    print(" iter |      cost |  violation |        mu")
    print("------+-----------+------------+-----------")
    for i, (c, v, mu) in enumerate(zip(costs, viols, mus)):
        print(f"  {i:>3} | {c:9.4g} | {v:10.4g} | {mu:9.4g}")

    if results_dir is not None and exp_label is not None:
        try:
            from .plotting.al_convergence_plotter import plot_al_convergence
            save_path = os.path.join(results_dir, f"{exp_label}_al_convergence.png")
            plot_al_convergence(costs, viols, mus,
                                title=f"{exp_label} — AL convergence",
                                save_path=save_path, show=False)
            print(f"Saved AL convergence figure to {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[debug-iterations] could not save AL figure "
                  f"({exc.__class__.__name__}: {exc})")
    return True


class FingerTraj:
    """Adapter exposing a single finger's per-step marginals as .trajectory, so
    the per-finger plot_trajectory() can be reused on one finger of the hand."""
    def __init__(self, trajectory):
        self.trajectory = trajectory


def collision_report(configs, solution, spec, object_pose, radius):
    """Worst finger-object clearance and cross-finger gap over the collision
    spheres (same exclusions as the C++ factors: no node-0 pairs, no
    proximal-proximal pairs; no contact node here since there is no contact)."""
    object_rotation = object_pose[:3, :3]
    object_center = object_pose[:3, 3]

    spheres = []
    for (_, cfg), fm in zip(configs, solution.marginals.digits):
        entries = []
        for n, p in zip(disc_node_indices(cfg), proximal_disc_flags(cfg)):
            pos = np.array(fm.rod.states[n].pose.mean)[:3, 3]
            entries.append((n, pos, bool(p)))
        spheres.append(entries)

    worst_obj = np.inf
    for entries in spheres:
        for _n, pos, _p in entries:
            local = object_rotation.T @ (pos - object_center)
            worst_obj = min(worst_obj, primitive_surface_gap(local, spec) - radius)

    worst_ff = np.inf
    for ia, ib in itertools.combinations(range(len(spheres)), 2):
        for na, pa, proxa in spheres[ia]:
            if na == 0:
                continue
            for nb, pb, proxb in spheres[ib]:
                if nb == 0 or (proxa and proxb):
                    continue
                worst_ff = min(worst_ff, np.linalg.norm(pa - pb) - 2.0 * radius)

    return worst_obj, worst_ff
        