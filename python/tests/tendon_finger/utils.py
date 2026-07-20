import os
import sys
from datetime import datetime
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import crest_sparse
from .config import get_base_config


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
    diagnostics are bound, TendonHandTrajectoryPlanner both qualify).

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
            from .._plotting.al_convergence_plotter import plot_al_convergence
            save_path = os.path.join(results_dir, f"{exp_label}_al_convergence.png")
            plot_al_convergence(costs, viols, mus,
                                title=f"{exp_label} — AL convergence",
                                save_path=save_path, show=False)
            print(f"Saved AL convergence figure to {save_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[debug-iterations] could not save AL figure "
                  f"({exc.__class__.__name__}: {exc})")
    return True


class GaussianProcessNoiseModel:
    def __init__(self, dim, frame_rate, total_time, tau=0.3, seed=None):
        self.dim = dim
        self.dt = 1.0 / frame_rate
        self.num_steps = int(total_time / self.dt)
        self.tau = tau
        self.rng = np.random.default_rng(seed)

        t = np.arange(self.num_steps) * self.dt
        ti, tj = np.meshgrid(t, t, indexing='ij')
        
        K = np.exp(-0.5 * (ti - tj)**2 / tau**2)
        K += 1e-8 * np.eye(self.num_steps)

        L = np.linalg.cholesky(K)

        self.samples = L @ self.rng.standard_normal((self.num_steps, dim))
        self.i = 0

    def step(self, cov):
        sample = self.samples[self.i]
        self.i += 1
        Lcov = np.linalg.cholesky(cov)
        return Lcov @ sample
    


def tensions_function(t):
    max_tensions = np.array([6.0, 2.0, 2.0, 2.0])
    rate_hz = 0.1

    tensions_rate_hz = np.array([1.0 * rate_hz, 1.1 * rate_hz, 1.2 * rate_hz, 1.3 * rate_hz])
    tensions = 0.5 * (1.0 - np.cos(2 * np.pi * tensions_rate_hz * t))

    return tensions * max_tensions


def pulse_function(t, rate_hz):
    return (0.5 * (1.0 - np.cos(2 * np.pi * rate_hz * t))) ** 3
    

class TipForceFunction:
    def __init__(self, max_magnitude=0.2, force_rate_hz=0.2, framerate=30, seed=None):
        self.max_magnitude = max_magnitude
        self.force_rate_hz = force_rate_hz
        self.steps_per_cycle = int(framerate / force_rate_hz)

        self.magnitude = 0
        self.direction = np.ones(3)

        self.step = 0

        self.rng = np.random.default_rng(seed)

    def sample_parameters(self):
        d = self.rng.normal(size=3)
        self.direction = d / np.linalg.norm(d)

    def __call__(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()

        pulse_scale = pulse_function(t, self.force_rate_hz)

        self.step += 1

        return pulse_scale * self.max_magnitude * self.direction
    

class BumpFunction:
    def __init__(self, num_forces):
        self.framerate = 30
        self.force_rate_hz = 0.1
        self.s = np.linspace(0, 1, num_forces)
        self.steps_per_cycle = self.framerate / self.force_rate_hz

        self.mu = 0
        self.sigma = 1
        self.mag = 0
        self.angle = 0

        self.sigma_range = [0.05, 0.1]
        self.mag_range = [0.02, 0.1]
        self.mu_rage = [0.2, 0.9]
        self.angle_range = [0.0, 2 * np.pi]
        self.step = 0

    def sample_parameters(self):
        self.mu = np.random.uniform(self.mu_rage[0], self.mu_rage[1])
        self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        self.mag = np.random.uniform(self.mag_range[0], self.mag_range[1])
        self.angle = np.random.uniform(self.angle_range[0], self.angle_range[1])

    def __call__(self, t):
        if self.step % self.steps_per_cycle == 0:
            self.sample_parameters()
        
        mag_t = (0.5 * (1.0 - np.cos(2 * np.pi * self.force_rate_hz * t))) ** 4

        self.step = self.step + 1

        return mag_t * self.get_max_forces()
    
    def get_max_forces(self):
        bump = 1 / (np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(- (self.s - self.mu)**2 / (2 * self.sigma**2))
        f_1d = self.mag * bump / np.sum(bump)

        f_x = f_1d * np.cos(self.angle)
        f_y = f_1d * np.sin(self.angle)
        f_z = np.zeros_like(f_y)

        return np.column_stack((f_x, f_y, f_z))


class DistLoadFunction:
    def __init__(self, num_forces, num_bumps=3):
        self.bump_functions = [BumpFunction(num_forces) for i in range(num_bumps)]
    
    def __call__(self, t):
        bumps_eval = [bump(t) for bump in self.bump_functions]
        return np.array(bumps_eval).sum(axis=0)

    def get_max_forces(self):
        max_bumps = [bump.get_max_forces() for bump in self.bump_functions]
        return np.array(max_bumps).sum(axis=0)
    
    def sample_parameters(self):
        [bump.sample_parameters() for bump in self.bump_functions]
    

class moving_savgol:
    def __init__(self, window_size=15, poly_order=2):
        self.buffer = deque(maxlen=window_size)
        self.poly_order = poly_order
        self.window_size = window_size

    def update(self, new_value):
        new_array = np.asarray(new_value)
        self.buffer.append(new_array)

        if len(self.buffer) < self.poly_order + 1:
            return new_array  # not enough points for a fit

        data = np.stack(self.buffer, axis=0)
        original_shape = data.shape[1:]
        t = np.arange(len(self.buffer))
        data_flat = data.reshape(len(self.buffer), -1)

        coeffs = np.polyfit(t, data_flat, min(self.poly_order, len(self.buffer) - 1))
        last_val = np.polyval(coeffs, t[-1])
        return last_val.reshape(original_shape)


def generate_trajectory(position_function, sim_time, damping=5e-2, frame_rate=30):
    config = get_base_config()
    solver = crest_sparse.TendonFingerSolver(config)

    num_steps = int(sim_time * frame_rate)
    tensions_min = 0.1 * np.ones(4)
    tensions_mean = tensions_min.copy()

    position_trajectory = []
    tensions_trajectory = []
    t = []

    tensions_cov = (1e-2) ** 2 * np.eye(4)
    tip_wrench_cov = (1e-3) ** 2 * np.eye(6)
    tip_wrench_mean = np.zeros(6)

    for i in range(num_steps):
        t_i = i / float(frame_rate)

        tensions = crest_sparse.VectorXGaussian(tensions_mean, tensions_cov)
        tip_wrench = crest_sparse.Vector6Gaussian(tip_wrench_mean, tip_wrench_cov)
        solution = solver.solve(tensions, tip_wrench, None)

        J_position = solution.marginals.J_pose_tensions[3:]
        pose = solution.marginals.rod.states[-1].pose.mean
        p = pose[:3, 3]
        R = pose[:3,:3]

        p_desired = position_function(t_i)
        p_error = R.T @ (p_desired - p)

        JTJ = J_position.T @ J_position
        A = JTJ + (damping**2) * np.eye(JTJ.shape[0])
        b = J_position.T @ p_error
        d_tensions = np.linalg.solve(A, b)

        tensions_mean = np.maximum(tensions_mean + d_tensions, tensions_min)

        position_trajectory.append(p_desired)
        tensions_trajectory.append(tensions_mean)
        t.append(t_i)

    return np.array(t), np.array(position_trajectory), np.array(tensions_trajectory)


def generate_waypoints(num_waypoints, center=(0, 0.175, 0.0), radii=(0.1, 0.05, 0.1), seed=None):
    rng = np.random.default_rng(seed)

    waypoints = []
    for _ in range(num_waypoints):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        r = rng.random() ** (1/3)
        point = r * direction * np.array(radii)
        waypoints.append(np.array(center) + point)

    return np.array(waypoints)
    

def generate_waypoint_trajectory(sim_time, frame_rate=30.0, time_per_waypoint=3.0, waypoints=None, seed=None):
    if waypoints is None:
        num_waypoints = int(sim_time / time_per_waypoint) + 1
        waypoints = generate_waypoints(num_waypoints, seed=seed)

    position_function = lambda t: waypoint_trajectory(t, waypoints)
    t, positions, tensions = generate_trajectory(position_function, sim_time, frame_rate=frame_rate)

    return t, positions, tensions, waypoints


def waypoint_trajectory(t, waypoints, time_per_waypoint=3.0):
    num_segments = len(waypoints) - 1

    segment_index = min(int(t // time_per_waypoint), num_segments)
    next_index = min(segment_index + 1, len(waypoints) - 1)

    alpha = (t % time_per_waypoint) / time_per_waypoint
    return (1 - alpha) * waypoints[segment_index] + alpha * waypoints[next_index]    
        