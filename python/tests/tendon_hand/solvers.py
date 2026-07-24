"""Reusable FK / IK / trajectory-planner solver classes for the tendon hand.

This factors the shared *build -> solve -> extract* skeleton that the demo
scripts (``ik_5f_contact.py``, ``fk_5f_sweep.py``, ``traj_5f_contact.py``)
each re-implement inline into three classes behind one common base
(:class:`HandSolverBase`), driven by a single :class:`HandSolveParams` struct
and returning a uniform :class:`HandResult`.

The point is a *unified* way to call the three solvers, so the interactive viser
visualizer (``viz_interactive.py``) -- and any future code -- can flip between
FK, IK and the trajectory planner without duplicating the setup boilerplate. The
existing demo scripts are left untouched; this module reuses their helpers
(``config.py`` / ``scene.py``) rather than replacing them.

Three flavours, matching the demos:

* :class:`HandFKSolver`      -- pure kinematics, no contact (``fk_5f_sweep.py``).
  Uses a *uniform* tension prior (a tight-passive/loose-flexor prior is
  underdetermined without contact) and keeps its solver so repeated solves
  warm-start via ``set_wrist_pose``.
* :class:`HandIKSolver`      -- single terminal grasp with per-finger SDF/analytic
  contact (``ik_5f_contact.py``); the C++ side routes to the Augmented Lagrangian.
* :class:`HandPlannerSolver` -- a K+1-step grasp trajectory with GP temporal priors
  (``traj_5f_contact.py`` / ``traj_5f_slide_grasp.py``).

Collision avoidance (Section 1.5) and the support-plane "table" (Section 1.6)
are opt-in via the params and applied to the IK/planner solves; FK stays a pure
kinematics solve (the renderer can still draw the spheres/table for reference).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import crest_sparse

from .config import (
    get_default_hand_configs, default_hand_tip_radii, load_hand_dimensions,
    tip_node_index, disc_node_indices, attach_collision, attach_table)
from .scene import (
    OBJECT_CENTER, GRASP_SPHERE_CENTER, GRASP_FLEXOR_TENSION, TABLE_NORMAL,
    get_primitive_specs, configure_object_surface, primitive_surface_gap)


# The _objects/ directory holding the baked .vdb SDF grids (relative to this file).
_OBJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "_objects")

# Anatomical hand digit count / config order: index, middle, ring, pinky, thumb.
NUM_FINGERS = 5

# The flexor tendon is index 5 in the 6-tendon anatomical routing (scene.TENDON_NAMES).
FLEXOR_IDX = 5


def _set_if(obj, name, value):
    """Set ``obj.name = value`` only if the binding exposes that field.

    The installed ``_crest_sparse`` extension can lag the C++ source: newer config
    fields (inexact-AL tolerances, slide-grasp ``k_touch``, the analytic-ellipsoid
    / table env fields) may be absent until the module is rebuilt. Guarding keeps
    the solvers working on the current binary and picks the fields up automatically
    once it is rebuilt."""
    if hasattr(obj, name):
        setattr(obj, name, value)
        return True
    return False


def capabilities():
    """What the *installed* binding supports, so callers (the visualizer) can gate
    unsupported controls instead of crashing on a stale build."""
    env = crest_sparse.EnvironmentConfig()
    pc = crest_sparse.TendonHandTrajectoryPlannerConfig()
    return {
        "ellipsoid": hasattr(env, "ellipsoid_semi_axes"),
        "table": hasattr(env, "plane_normal"),
        "collision_cull": hasattr(env, "collision_cull_margin"),
        "k_touch": hasattr(pc, "k_touch"),
    }


# ---------------------------------------------------------------------------
# Scene helpers (shared object placement, mirroring the demo scripts).
# ---------------------------------------------------------------------------

def default_object_center(primitive, spec):
    """Default world center for a primitive, matching the demo scripts: the big
    grasp sphere, capsule and analytic ellipsoids sit at the flexed-fingertip
    locus (``GRASP_SPHERE_CENTER``); the smaller primitives stay at ``OBJECT_CENTER``."""
    if primitive in ("big_sphere", "capsule") or spec["type"] == "ellipsoid":
        return np.array(GRASP_SPHERE_CENTER, dtype=float)
    return np.array(OBJECT_CENTER, dtype=float)


def object_extent_along(spec, normal):
    """Approximate object half-size along ``normal`` (m) -- used to seat a default
    support plane tangent to the object's underside. Only a default; the plane
    height is user-adjustable."""
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    t = spec["type"]
    if t == "sphere":
        return float(spec["radius"])
    if t in ("cylinder", "capsule"):
        # These primitives are rotated (Rx 90 deg) to stand their local Y axis
        # along world +Z: half-height along Z, radius laterally.
        cap = spec["radius"] if t == "capsule" else 0.0
        along_z = spec["height"] / 2.0 + cap
        return float(along_z if abs(n[2]) >= 0.5 else spec["radius"])
    if t == "cube":
        return float(np.abs(np.asarray(spec["half_extents"], float) * n).sum())
    if t == "ellipsoid":
        return float(np.abs(np.asarray(spec["semi_axes"], float) * n).sum())
    return 0.05


def resolve_scene(params):
    """Resolve (spec, center, rotation, 4x4 pose) for the object from the params,
    filling center/rotation from the primitive when left unset."""
    spec = get_primitive_specs()[params.primitive]
    center = (np.asarray(params.object_center, float)
              if params.object_center is not None
              else default_object_center(params.primitive, spec))
    rotation = (np.asarray(params.object_rotation, float)
                if params.object_rotation is not None
                else np.asarray(spec.get("rotation", np.eye(3)), float))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    return spec, center, rotation, pose


def resolve_table_origin(params, spec, object_center):
    """Resolve the support-plane origin: explicit ``params.plane_origin`` if set,
    else tangent to the object's underside along ``params.plane_normal``."""
    if params.plane_origin is not None:
        return np.asarray(params.plane_origin, float)
    n = np.asarray(params.plane_normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    return np.asarray(object_center, float) - object_extent_along(spec, n) * n


# ---------------------------------------------------------------------------
# Params / results.
# ---------------------------------------------------------------------------

@dataclass
class HandSolveParams:
    """Every knob the three solvers expose, with the demo-script defaults.

    Shared by FK / IK / planner; each solver reads only the fields it needs. The
    interactive visualizer mutates one instance of this from its GUI controls.
    """
    # --- Scene / object ---
    primitive: str = "big_sphere"
    object_center: Optional[np.ndarray] = None      # None => derive from primitive
    object_rotation: Optional[np.ndarray] = None     # None => primitive's rotation

    # --- Wrist start pose + prior ---
    wrist_pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    sigma_wrist_pos: float = 1e-4
    sigma_wrist_rot: float = 1e-3

    # --- Tensions (per-finger flexor + shared passive background) ---
    passive_tension: float = 0.5
    flexor_tensions: List[float] = field(
        default_factory=lambda: [GRASP_FLEXOR_TENSION] * NUM_FINGERS)
    tip_wrench_sigma: float = 1e-3

    # --- Augmented Lagrangian (IK / planner) ---
    al_mu: float = 1.0
    al_rate: float = 2.0
    al_iters: int = 40

    # --- Planner-only ---
    K: int = 10
    dt: float = 0.1
    gp_wrist: float = 1e-2
    gp_tense: float = 1.0
    gp_len: float = 0.0
    start_flexor: float = 0.5
    al_inner_tol: float = 1e-2
    al_abs_cost_tol: float = 1e12

    # --- Collision avoidance (Section 1.5, opt-in; IK / planner) ---
    collision: bool = False
    collision_radius: float = 0.003
    collision_sigma: float = 1e-4
    num_proximal_discs: int = 2
    cull_margin: Optional[float] = None

    # --- Support plane / "table" (Section 1.6, opt-in; IK / planner) ---
    table: bool = False
    plane_origin: Optional[np.ndarray] = None       # None => under the object
    plane_normal: np.ndarray = field(
        default_factory=lambda: np.array(TABLE_NORMAL, float))
    plane_avoidance: bool = True
    k_touch: Optional[int] = None                    # planner slide-grasp schedule


class _FingerSol:
    """Duck-typed per-finger solution the viser/pyvista renderers consume:
    exposes ``.marginals`` (a ``TendonFingerMarginals``) and ``.meta``. Same shim
    the demo scripts build inline (e.g. ``ik_5f_contact.py``)."""

    __slots__ = ("marginals", "meta")

    def __init__(self, marginals, meta):
        self.marginals = marginals
        self.meta = meta


def _make_frame(finger_names, hand_marginals, meta):
    """One render frame: ``{finger_name: _FingerSol}`` for a single hand state."""
    return {name: _FingerSol(fm, meta)
            for name, fm in zip(finger_names, hand_marginals.fingers)}


@dataclass
class HandResult:
    """Uniform result for all three solvers. ``frames`` has length 1 for FK/IK and
    K+1 for the planner, so a step-scrubber can index it the same way regardless."""
    frames: List[dict]
    meta: object
    spec: dict
    object_center: np.ndarray
    object_rotation: np.ndarray
    finger_names: List[str]
    tip_radii: List[float]

    def surface_gaps(self, k=0):
        """Per-finger fingertip surface gap (m, ~0 at contact) at frame ``k``,
        reusing the analytic ``primitive_surface_gap`` the demos report with."""
        frame = self.frames[k]
        gaps = {}
        for name, radius in zip(self.finger_names, self.tip_radii):
            fm = frame[name].marginals
            tip = np.asarray(fm.rod.states[-1].pose.mean)[:3, 3]
            local = self.object_rotation.T @ (tip - self.object_center)
            gaps[name] = primitive_surface_gap(local, self.spec) - radius
        return gaps

    def worst_gap(self, k=0):
        gaps = self.surface_gaps(k)
        return max(abs(g) for g in gaps.values()) if gaps else 0.0


# ---------------------------------------------------------------------------
# Solver base + the three flavours.
# ---------------------------------------------------------------------------

class HandSolverBase:
    """Shared setup for the tendon-hand solvers: builds the anatomical hand from
    ``gepetto_core`` dims and holds the resolved scene. Subclasses implement
    :meth:`solve`."""

    def __init__(self, params: Optional[HandSolveParams] = None):
        self.params = params or HandSolveParams()
        self.dims = load_hand_dimensions()
        self.configs = get_default_hand_configs(self.dims)
        self.tip_radii = default_hand_tip_radii(self.dims)
        self.finger_names = [name for name, _ in self.configs]
        self.spec, self.object_center, self.object_rotation, self.object_pose = \
            resolve_scene(self.params)

    # -- environment attachment (mutates self.configs in place) --

    def _attach_contact(self):
        """Per-finger contact env: shared object surface + this finger's tip node
        as the terminal contact (``ik_5f_contact.py`` block)."""
        for (_, cfg), radius in zip(self.configs, self.tip_radii):
            env = crest_sparse.EnvironmentConfig()
            configure_object_surface(env, self.spec, _OBJECTS_DIR,
                                     self.params.primitive)
            env.object_pose_mean = self.object_pose
            env.object_pose_cov = 1e-8 * np.eye(6)
            env.object_pose_per_step = False
            env.contact_node_radius = radius
            env.target_contact_node = tip_node_index(cfg)
            cfg.sdf_contact = env

    def _attach_collision(self):
        """Add Section 1.5 collision spheres onto each finger's (already attached)
        env. Reuses the contact env, so it works for SDF and ellipsoid objects
        alike (the vdb path is only used if a finger has no env yet)."""
        vdb = (None if self.spec["type"] == "ellipsoid"
               else os.path.normpath(os.path.join(_OBJECTS_DIR, self.spec["vdb"])))
        attach_collision(self.configs, vdb, self.object_pose,
                         radius=self.params.collision_radius,
                         sigma=self.params.collision_sigma,
                         num_proximal_discs=self.params.num_proximal_discs,
                         cull_margin=self.params.cull_margin)

    def _attach_table(self):
        """Attach the Section 1.6 support plane to every finger's env."""
        origin = resolve_table_origin(self.params, self.spec, self.object_center)
        attach_table(self.configs, origin, self.params.plane_normal,
                     avoidance=self.params.plane_avoidance,
                     tip_radii=self.tip_radii)

    # -- prior builders --

    def _tension_priors(self, cov):
        """One ``VectorXGaussian`` per finger: passive tendons at the background
        hold, flexor (index 5) at that finger's commanded tension."""
        priors = []
        for i, (_, cfg) in enumerate(self.configs):
            mean = np.full(cfg.num_tendons, self.params.passive_tension)
            mean[FLEXOR_IDX] = self.params.flexor_tensions[i]
            priors.append(crest_sparse.VectorXGaussian(mean, cov))
        return priors

    def _tip_wrenches(self):
        cov = self.params.tip_wrench_sigma ** 2 * np.eye(6)
        return [crest_sparse.Vector6Gaussian(np.zeros(6), cov) for _ in self.configs]

    def _result(self, frames, meta):
        return HandResult(frames, meta, self.spec, self.object_center,
                          self.object_rotation, self.finger_names, self.tip_radii)

    def solve(self) -> HandResult:  # pragma: no cover - abstract
        raise NotImplementedError


class HandFKSolver(HandSolverBase):
    """Pure-kinematics hand solve driven by tensions (no contact). Builds its
    ``TendonHandSolver`` once and re-commands the wrist each solve, so repeated
    calls warm-start from the previous solution (``fk_5f_sweep.py``)."""

    def __init__(self, params: Optional[HandSolveParams] = None):
        super().__init__(params)
        cfg = crest_sparse.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.max_iterations = 500
        self._solver = crest_sparse.TendonHandSolver(self.configs, cfg)

    def solve(self) -> HandResult:
        # Re-aim the shared wrist prior (warm start; no rebuild).
        self._solver.set_wrist_pose(self.params.wrist_pose)
        # Uniform prior on every tendon: a tight-passive/loose-flexor prior is
        # underdetermined without contact (IndeterminantLinearSystem on the
        # tension variable) -- see fk_5f_sweep.py.
        cov = (1e-2) ** 2 * np.eye(6)
        sol = self._solver.solve(self._tension_priors(cov), self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        return self._result([frame], sol.meta)


class HandIKSolver(HandSolverBase):
    """Single terminal grasp: each fingertip driven onto the shared object surface
    by a hard contact constraint (Augmented Lagrangian). ``ik_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_contact()
        if self.params.collision:
            self._attach_collision()
        if self.params.table:
            self._attach_table()

        cfg = crest_sparse.TendonHandSolverConfig()
        cfg.wrist_pose = self.params.wrist_pose
        cfg.sigma_wrist_pos = self.params.sigma_wrist_pos
        cfg.sigma_wrist_rot = self.params.sigma_wrist_rot
        cfg.base.linear_solver_type = "MULTIFRONTAL_QR"
        cfg.base.al_initial_mu = self.params.al_mu
        cfg.base.al_mu_increase_rate = self.params.al_rate
        cfg.base.al_max_iterations = self.params.al_iters

        solver = crest_sparse.TendonHandSolver(self.configs, cfg)
        # Tight passive / loose flexor: the optimizer drives contact through the
        # flexor while the passives stay pinned.
        cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
        sol = solver.solve(self._tension_priors(cov), self._tip_wrenches())
        frame = _make_frame(self.finger_names, sol.marginals, sol.meta)
        return self._result([frame], sol.meta)


class HandPlannerSolver(HandSolverBase):
    """A K+1-step grasp trajectory tied by GP temporal priors on the wrist pose and
    finger tensions, with terminal contact constraints. ``traj_5f_contact.py``."""

    def solve(self) -> HandResult:
        self._attach_contact()
        if self.params.collision:
            self._attach_collision()
        if self.params.table:
            self._attach_table()

        n = self.configs[0][1].num_tendons
        pc = crest_sparse.TendonHandTrajectoryPlannerConfig()
        pc.K = self.params.K
        pc.dt = self.params.dt
        pc.wrist_pose = self.params.wrist_pose
        pc.sigma_wrist_pos = self.params.sigma_wrist_pos
        pc.sigma_wrist_rot = self.params.sigma_wrist_rot
        pc.gp_wrist_Qc = self.params.gp_wrist * np.eye(6)
        pc.gp_tense_Qc = self.params.gp_tense * np.eye(n)
        pc.gp_len_Qc = (self.params.gp_len * np.eye(n)
                        if self.params.gp_len > 0.0 else np.zeros((0, 0)))
        pc.base.linear_solver_type = "MULTIFRONTAL_QR"
        pc.base.al_initial_mu = self.params.al_mu
        pc.base.al_mu_increase_rate = self.params.al_rate
        pc.base.al_max_iterations = self.params.al_iters
        # Inexact-AL tuning and slide-grasp scheduling exist only on newer builds.
        _set_if(pc.base, "al_inner_rel_tol_initial", self.params.al_inner_tol)
        _set_if(pc.base, "al_abs_cost_tol", self.params.al_abs_cost_tol)
        if self.params.table and self.params.k_touch is not None:
            _set_if(pc, "k_touch", self.params.k_touch)

        planner = crest_sparse.TendonHandTrajectoryPlanner(self.configs, pc)

        # Target tensions at k>=1 (tight passive / loose flexor), plus the measured
        # k=0 start (open hand at start_flexor, all pinned) that the trajectory
        # closes from.
        cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-1])
        start_cov = np.diag([1e-6] * n)
        starts = []
        for _, cfg in self.configs:
            sm = np.full(cfg.num_tendons, self.params.passive_tension)
            sm[FLEXOR_IDX] = self.params.start_flexor
            starts.append(crest_sparse.VectorXGaussian(sm, start_cov))

        result = planner.plan(self._tension_priors(cov), self._tip_wrenches(),
                              start_tensions=starts)
        frames = [_make_frame(self.finger_names, hm, result.meta)
                  for hm in result.trajectory]
        return self._result(frames, result.meta)


# Convenience registry the visualizer uses to switch modes.
SOLVERS = {
    "FK": HandFKSolver,
    "IK": HandIKSolver,
    "Planner": HandPlannerSolver,
}
