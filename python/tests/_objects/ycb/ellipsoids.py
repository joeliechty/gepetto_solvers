"""Approximate a mesh with a small set of ellipsoids.

The pipeline is the same for every backend:

1. Sample the mesh: interior points (voxel fill) drive the *clustering*, surface
   points drive the *fitting*. Keeping those separate matters -- voxel centers
   sit up to half a pitch outside the surface, so fitting to them inflates every
   ellipsoid (a unit sphere comes out 12% too big).
2. Split the surface points into `k` groups with one of the backends below.
3. Fit a minimum-volume enclosing ellipsoid (MVEE) to each group, trimming the
   worst outliers to hit a coverage target rather than enclosing everything.
4. Refine: reassign points to whichever ellipsoid is nearest in Mahalanobis
   distance and refit. The initial clustering does not know the final ellipsoid
   shapes, so this recovers a bit of quality.

Quality is reported as `excess_volume_ratio` (union volume / mesh volume, lower
is better, 1.0 is perfect) and the `surface_coverage` actually achieved.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree

Backend = Literal["gmm", "kmeans", "coacd"]
ProgressFn = Callable[[float, str], None]

BACKENDS: tuple[str, ...] = ("gmm", "kmeans", "coacd")


def _noop(fraction: float, message: str) -> None:
    del fraction, message


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------


@dataclass
class Ellipsoid:
    """An ellipsoid as center + semi-axis lengths + orientation.

    `rotation` columns are the semi-axis directions, so a point is inside when
    ``|| diag(1/radii) @ rotation.T @ (x - center) || <= 1``.
    """

    center: np.ndarray
    radii: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(3)
        self.radii = np.asarray(self.radii, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)

    @property
    def volume(self) -> float:
        return float(4.0 / 3.0 * np.pi * np.prod(self.radii))

    def mahalanobis(self, points: np.ndarray) -> np.ndarray:
        """Squared normalized distance; <= 1 means inside."""
        local = (np.atleast_2d(points) - self.center) @ self.rotation
        return np.sum((local / self.radii) ** 2, axis=-1)

    def contains(self, points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Inside test. The slack must exceed the MVEE convergence tolerance
        (1e-7): points on a converged boundary land a few 1e-7 outside, and a
        tighter threshold reports most of a perfect fit as uncovered."""
        return self.mahalanobis(points) <= 1.0 + tol

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """World-axis-aligned bounds. The ellipsoid can extend well past the
        point cloud it was fitted to, which is why union volume must be
        integrated over *this* box and not the mesh's."""
        half = np.sqrt(np.sum((self.rotation * self.radii) ** 2, axis=1))
        return self.center - half, self.center + half

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        return np.asarray(trimesh.transformations.quaternion_from_matrix(transform))

    def transform(self) -> np.ndarray:
        """4x4 placing a unit sphere onto this ellipsoid."""
        out = np.eye(4)
        out[:3, :3] = self.rotation * self.radii
        out[:3, 3] = self.center
        return out

    def as_mesh(self, subdivisions: int = 2) -> trimesh.Trimesh:
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        sphere.apply_transform(self.transform())
        return sphere

    def translated(self, offset: np.ndarray) -> "Ellipsoid":
        return Ellipsoid(self.center + np.asarray(offset, float), self.radii, self.rotation)

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "radii": self.radii.tolist(),
            "rotation": self.rotation.tolist(),
            "quaternion_wxyz": self.quaternion_wxyz.tolist(),
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Ellipsoid":
        return cls(
            np.asarray(data["center"], float),
            np.asarray(data["radii"], float),
            np.asarray(data["rotation"], float),
        )


# ---------------------------------------------------------------------------
# Minimum-volume enclosing ellipsoid
# ---------------------------------------------------------------------------


def mvee(
    points: np.ndarray, tol: float = 1e-3, max_iter: int = 2000, floor: float = 1e-5
) -> Ellipsoid:
    """Khachiyan's minimum-volume enclosing ellipsoid.

    Only the convex hull matters, and reducing to it first is exact -- it also
    takes a full k-sweep from ~40s to under a second on real YCB meshes.

    `tol` is deliberately loose. Cost is dominated by iterations to convergence
    (2.4s at 1e-7 vs 0.24s at 1e-3 on a real banana hull) while the volume
    differs by 0.2%, and the inflation step below restores exact enclosure
    regardless of how far the iteration got.
    """
    P = np.unique(np.atleast_2d(np.asarray(points, float)), axis=0)
    if len(P) < 4:
        return _degenerate_ellipsoid(P, floor)

    try:
        P = P[ConvexHull(P).vertices]
    except Exception:
        # Coplanar or otherwise degenerate cluster -- QHull refuses these.
        return _degenerate_ellipsoid(P, floor)

    n, d = P.shape
    Q = np.vstack([P.T, np.ones(n)])
    u = np.ones(n) / n
    try:
        for _ in range(max_iter):
            X = Q @ (u[:, None] * Q.T)
            M = np.einsum("ji,jk,ki->i", Q, np.linalg.inv(X), Q)
            j = int(np.argmax(M))
            step = (M[j] - d - 1.0) / ((d + 1.0) * (M[j] - 1.0))
            if not np.isfinite(step) or step < tol:
                break
            u *= 1.0 - step
            u[j] += step

        center = P.T @ u
        cov = P.T @ (u[:, None] * P) - np.outer(center, center)
        axes, sv, _ = np.linalg.svd(np.linalg.inv(cov) / d)
        radii = 1.0 / np.sqrt(sv)
    except np.linalg.LinAlgError:
        return _degenerate_ellipsoid(P, floor)

    if not (np.all(np.isfinite(radii)) and np.all(np.isfinite(center))):
        return _degenerate_ellipsoid(P, floor)

    ellipsoid = Ellipsoid(center, np.maximum(radii, floor), _proper(axes))

    # Khachiyan can exhaust max_iter on a large hull without fully converging,
    # leaving hull vertices measurably outside. Inflate to guarantee enclosure:
    # the ellipsoid is convex, so covering the hull covers every input point.
    worst = float(ellipsoid.mahalanobis(P).max())
    if np.isfinite(worst) and worst > 1.0:
        ellipsoid.radii = ellipsoid.radii * np.sqrt(worst)
    return ellipsoid


def _proper(rotation: np.ndarray) -> np.ndarray:
    """Force a right-handed rotation; SVD can hand back a reflection."""
    if np.linalg.det(rotation) < 0:
        rotation = rotation.copy()
        rotation[:, -1] *= -1.0
    return rotation


def _degenerate_ellipsoid(points: np.ndarray, floor: float) -> Ellipsoid:
    """Fallback for clusters MVEE cannot handle: too few points, or flat.

    Uses the principal axes of the points and the extent along each, floored so
    the result is always a valid non-zero ellipsoid. Real YCB scans hit this --
    023_wine_glass is only 84 faces.
    """
    P = np.atleast_2d(np.asarray(points, float))
    if len(P) == 0:
        return Ellipsoid(np.zeros(3), np.full(3, floor), np.eye(3))

    center = P.mean(axis=0)
    centered = P - center
    if len(P) >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=True)
        axes = _proper(vt.T)
    else:
        axes = np.eye(3)
    radii = np.abs(centered @ axes).max(axis=0) if len(P) >= 2 else np.zeros(3)
    return Ellipsoid(center, np.maximum(radii, floor), axes)


def trimmed_mvee(
    points: np.ndarray, coverage: float = 0.98, rounds: int = 3, floor: float = 1e-5
) -> Ellipsoid:
    """MVEE that may leave a fraction of the points outside.

    Strict MVEE is dominated by a handful of extreme points, so shaving the
    worst `1 - coverage` of them buys a much tighter ellipsoid for a little
    leakage. Each round re-ranks against the *current* ellipsoid.
    """
    P = np.atleast_2d(np.asarray(points, float))
    ellipsoid = mvee(P, floor=floor)
    if coverage >= 1.0 or len(P) < 8:
        return ellipsoid

    for _ in range(rounds):
        distances = ellipsoid.mahalanobis(P)
        threshold = np.quantile(distances, coverage)
        keep = distances <= threshold
        if keep.sum() < 4:
            break
        candidate = mvee(P[keep], floor=floor)
        if candidate.volume >= ellipsoid.volume:
            break
        ellipsoid = candidate
    return ellipsoid


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_points(
    mesh: trimesh.Trimesh, n_surface: int = 4000, divisions: int = 40
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (interior, surface, pitch).

    Interior points come from a filled voxel grid, which works even though YCB
    meshes are not watertight (`contains`-based sampling does not). Sparse scans
    can yield almost nothing, in which case clustering falls back to the surface.

    Surface points are area-sampled only. Folding in the raw mesh vertices looks
    harmless but is not: on a scanned mesh 3359 of the banana's 8368 vertices sit
    on the convex hull, and MVEE cost scales with hull size -- it made every fit
    ~100x slower for no measurable gain in fit quality.
    """
    surface = np.asarray(mesh.sample(n_surface), dtype=float)

    pitch = float(np.max(mesh.extents)) / max(divisions, 1)
    try:
        interior = np.asarray(mesh.voxelized(pitch=pitch).fill().points, dtype=float)
    except Exception:
        interior = np.empty((0, 3))

    if len(interior) < 32:
        interior = surface
    return interior, surface, pitch


def reference_volume(
    mesh: trimesh.Trimesh, interior: np.ndarray, pitch: float
) -> tuple[float, bool]:
    """Volume to compare the ellipsoid union against, and whether to trust it.

    `mesh.volume` is usable whenever the winding is consistent, even for the
    non-watertight YCB scans. A few objects are hollow shells rather than solids
    (023_wine_glass is 84 faces), where it is meaningless -- cross-checking
    against the voxel fill catches those so the UI can flag the ratio instead of
    reporting nonsense like 62x.
    """
    signed = float(mesh.volume)
    volume = abs(signed)

    if np.isfinite(volume) and volume > 1e-12:
        # A closed, consistently wound solid has positive signed volume. Negative
        # means the mesh is inverted or open -- 023_wine_glass reports -2.4e-6.
        reliable = signed > 0
        if reliable:
            # Even when positive, a volume that is a sliver of the convex hull
            # means the mesh is a shell rather than the solid we are approximating.
            try:
                hull = float(mesh.convex_hull.volume)
                if hull > 0 and volume / hull < 0.05:
                    reliable = False
            except Exception:
                pass
        return volume, reliable

    voxel = float(len(interior) * pitch**3) if (pitch > 0 and len(interior)) else float("nan")
    if np.isfinite(voxel) and voxel > 0:
        return voxel, False
    return float("nan"), False


# How closely a decimated hull must reproduce the full one's support function.
# 0.2 mm: a hundredth of the smallest object in the set (a 17 mm marble), and
# well under the millimetre at which a viewer could see one sink into a table.
SUPPORT_HULL_TOLERANCE = 2e-4


@functools.lru_cache(maxsize=2)
def _sphere_directions(subdivisions: int) -> np.ndarray:
    """Unit directions on a subdivided icosahedron, plus the six world axes.

    The axes are appended because they are the ones actually used: an object
    stands on a table whose normal is +Z, and a scene that has not rotated it
    asks for exactly that direction. An icosphere does not have a vertex there
    (an icosahedron's are at the permutations of ``(0, ±1, ±phi)``), so without
    them the one measurement everything depends on would be the interpolated one.
    """
    ico = np.asarray(trimesh.creation.icosphere(subdivisions=subdivisions).vertices,
                     dtype=float)
    return np.vstack([ico, np.eye(3), -np.eye(3)])


def support_hull(
    mesh: trimesh.Trimesh, tolerance: float = SUPPORT_HULL_TOLERANCE
) -> np.ndarray:
    """Convex-hull vertices of `mesh`, thinned to the ones that carry its shape.

    The full hull of a scanned mesh is far larger than what it describes: the
    peach's is 5836 vertices, the racquetball's 8085, because a smooth scan puts
    a vertex on every facet of a sphere. Committed alongside 94 fits that is
    4 MB of qhull output, so this keeps a subset chosen for the only thing the
    hull is read for -- its SUPPORT FUNCTION, "how far does this object reach
    along d", which is what seats it on a table and what measures its silhouette.

    Greedy: start from the vertices extreme along a coarse set of directions,
    then repeatedly add whichever vertex is missed worst, until no probe
    direction is off by more than `tolerance`. Typically ~170 vertices, and the
    error is one-sided -- a subset's hull is contained in the true hull, so a
    reach can be understated by up to `tolerance` and never overstated. An object
    seated on this hull may therefore sink 0.2 mm into its table; it can never
    float above it, which is the failure that would look wrong.
    """
    hull = np.asarray(mesh.convex_hull.vertices, dtype=float).reshape(-1, 3)
    if len(hull) == 0:
        return hull

    probe = _sphere_directions(4)          # 2562 directions + axes
    support = hull @ probe.T               # (V, K)
    exact = support.max(axis=0)

    keep = set(np.unique(np.argmax(hull @ _sphere_directions(2).T, axis=0)).tolist())
    for _ in range(len(hull)):
        index = np.fromiter(keep, dtype=int, count=len(keep))
        error = exact - support[index].max(axis=0)
        worst = int(np.argmax(error))
        if error[worst] <= tolerance:
            break
        keep.add(int(np.argmax(support[:, worst])))
    return hull[sorted(keep)]


# ---------------------------------------------------------------------------
# Clustering backends
# ---------------------------------------------------------------------------


def _labels_gmm(interior, surface, k, seed, scale):
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=seed,
        n_init=3,
        reg_covar=1e-10,
    )
    model.fit(interior / scale)
    return model.predict(surface / scale)


def _labels_kmeans(interior, surface, k, seed, scale):
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    model.fit(interior / scale)
    return model.predict(surface / scale)


def _labels_coacd(mesh, surface, k, seed):
    """Approximate convex decomposition, then nearest-part assignment.

    Unlike point clustering, this sees concavity -- the right tool for handles
    and hooks. `preprocess_mode="on"` is required because YCB meshes are not
    watertight. It is slow (seconds to tens of seconds), hence opt-in.
    """
    import coacd

    coacd.set_log_level("error")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        max_convex_hull=int(k),
        preprocess_mode="on",
        seed=int(seed),
    )
    if not parts:
        raise RuntimeError("CoACD returned no parts")

    vertices, owner = [], []
    for index, (part_vertices, _faces) in enumerate(parts):
        part_vertices = np.asarray(part_vertices, dtype=float)
        vertices.append(part_vertices)
        owner.append(np.full(len(part_vertices), index))
    vertices = np.vstack(vertices)
    owner = np.concatenate(owner)

    _, nearest = cKDTree(vertices).query(surface)
    return owner[nearest], len(parts)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class FitMetrics:
    k: int
    excess_volume_ratio: float
    surface_coverage: float
    union_volume: float
    mesh_volume: float
    volume_reliable: bool = True

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "excess_volume_ratio": self.excess_volume_ratio,
            "surface_coverage": self.surface_coverage,
            "union_volume": self.union_volume,
            "mesh_volume": self.mesh_volume,
            "volume_reliable": self.volume_reliable,
        }

    def summary(self) -> str:
        ratio = (
            f"{self.excess_volume_ratio:.2f}x volume"
            if self.volume_reliable
            else f"{self.excess_volume_ratio:.2f}x volume (unreliable: mesh is not a solid)"
        )
        return (
            f"{self.k} ellipsoid{'s' if self.k != 1 else ''} · {ratio} · "
            f"{self.surface_coverage * 100:.1f}% covered"
        )


@dataclass
class EllipsoidFit:
    """A decomposition, plus enough of the MESH to place the object in a scene.

    ``hull`` is the convex hull of the mesh the ellipsoids approximate, in the
    same displayed frame as the centers. It is carried because the union of the
    shells is a bound, not the object: a fit routinely reaches a centimetre or
    two past the real surface (the potted meat can, 16 mm; the chips can, 93 mm),
    so anything that asks "where does this object END" -- seating it on a table
    above all -- gets a badly wrong answer from the shells and the right one from
    the hull. Empty when the fit predates this field or was built without a mesh
    in hand; consumers fall back to the shells.
    """

    ellipsoids: list[Ellipsoid]
    metrics: FitMetrics
    backend: str
    coverage_target: float
    ground_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hull: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "coverage_target": self.coverage_target,
            "frame": {
                "note": (
                    "Centers are in the displayed frame: the mesh centered in XY "
                    "with its lowest point on z=0. Subtract ground_offset to get "
                    "coordinates in the original mesh frame."
                ),
                "ground_offset": np.asarray(self.ground_offset).tolist(),
            },
            "metrics": self.metrics.to_dict(),
            # Rounded to 10 um: this is a few hundred vertices per object and the
            # file is committed, so the digits past that are pure diff noise.
            "hull": np.round(np.asarray(self.hull, float).reshape(-1, 3), 5).tolist(),
            "ellipsoids": [e.to_dict() for e in self.ellipsoids],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EllipsoidFit":
        metrics = FitMetrics(**data["metrics"])
        return cls(
            [Ellipsoid.from_dict(e) for e in data["ellipsoids"]],
            metrics,
            data["backend"],
            data["coverage_target"],
            np.asarray(data.get("frame", {}).get("ground_offset", [0, 0, 0]), float),
            np.asarray(data.get("hull", []), float).reshape(-1, 3),
        )


def union_volume(
    ellipsoids: Sequence[Ellipsoid], n_samples: int = 80000, seed: int = 0
) -> float:
    """Monte Carlo volume of the union, integrated over the ellipsoids' own AABB."""
    if not ellipsoids:
        return 0.0
    corners = np.array([e.aabb() for e in ellipsoids])
    lo = corners[:, 0].min(axis=0)
    hi = corners[:, 1].max(axis=0)
    extent = hi - lo
    if np.any(extent <= 0):
        return 0.0

    samples = np.random.default_rng(seed).uniform(lo, hi, (n_samples, 3))
    inside = np.zeros(len(samples), dtype=bool)
    for ellipsoid in ellipsoids:
        inside |= ellipsoid.contains(samples)
    return float(inside.mean() * np.prod(extent))


def surface_coverage(ellipsoids: Sequence[Ellipsoid], surface: np.ndarray) -> float:
    if not ellipsoids:
        return 0.0
    covered = np.zeros(len(surface), dtype=bool)
    for ellipsoid in ellipsoids:
        covered |= ellipsoid.contains(surface)
    return float(covered.mean())


def _enforce_coverage(
    ellipsoids: list[Ellipsoid],
    surface: np.ndarray,
    target: float,
    max_rounds: int = 5,
) -> list[Ellipsoid]:
    """Grow ellipsoids until the union actually covers `target` of the surface.

    Trimming and refitting are both free to shrink an ellipsoid off points it
    used to own, so the achieved coverage can undershoot what was asked for.
    This hands each uncovered point to its nearest ellipsoid and refits that one
    strictly, which is what makes the coverage slider mean what it says --
    without it, asking for 100% enclosure returned 98-99%.
    """
    for _ in range(max_rounds):
        owned = [e.contains(surface) for e in ellipsoids]
        covered = np.any(owned, axis=0)
        if covered.mean() >= target - 1e-9:
            break

        missing = np.flatnonzero(~covered)
        distances = np.stack([e.mahalanobis(surface[missing]) for e in ellipsoids], axis=1)
        owner = distances.argmin(axis=1)

        updated: list[Ellipsoid] = []
        for index, ellipsoid in enumerate(ellipsoids):
            extra = missing[owner == index]
            if len(extra) == 0:
                updated.append(ellipsoid)
                continue
            points = np.vstack([surface[owned[index]], surface[extra]])
            updated.append(mvee(points))
        ellipsoids = updated
    return ellipsoids


def _refine(
    ellipsoids: list[Ellipsoid], surface: np.ndarray, coverage: float, iterations: int = 3
) -> list[Ellipsoid]:
    """Reassign points to the nearest ellipsoid and refit, a few times."""
    for _ in range(iterations):
        distances = np.stack([e.mahalanobis(surface) for e in ellipsoids], axis=1)
        labels = distances.argmin(axis=1)

        updated: list[Ellipsoid] = []
        for index in range(len(ellipsoids)):
            points = surface[labels == index]
            if len(points) >= 4:
                updated.append(trimmed_mvee(points, coverage))
        if not updated:
            return ellipsoids

        before = sum(e.volume for e in ellipsoids)
        after = sum(e.volume for e in updated)
        ellipsoids = updated
        if after >= before * 0.999:  # converged
            break
    return ellipsoids


def fit(
    mesh: trimesh.Trimesh,
    k: int,
    coverage: float = 0.98,
    backend: Backend = "gmm",
    seed: int = 0,
    refine: bool = True,
) -> EllipsoidFit:
    """Approximate `mesh` with `k` ellipsoids."""
    interior, surface, pitch = sample_points(mesh)
    scale = max(float(np.max(mesh.extents)), 1e-9)
    k = max(1, int(k))

    if k == 1:
        labels = np.zeros(len(surface), dtype=int)
        parts = 1
    elif backend == "coacd":
        labels, parts = _labels_coacd(mesh, surface, k, seed)
    elif backend == "kmeans":
        labels, parts = _labels_kmeans(interior, surface, k, seed, scale), k
    else:
        labels, parts = _labels_gmm(interior, surface, k, seed, scale), k

    ellipsoids: list[Ellipsoid] = []
    for index in range(parts):
        points = surface[labels == index]
        if len(points) >= 4:
            ellipsoids.append(trimmed_mvee(points, coverage))
    if not ellipsoids:
        ellipsoids = [trimmed_mvee(surface, coverage)]

    if refine and len(ellipsoids) > 1:
        ellipsoids = _refine(ellipsoids, surface, coverage)
    ellipsoids = _enforce_coverage(ellipsoids, surface, coverage)

    volume = union_volume(ellipsoids)
    mesh_volume, reliable = reference_volume(mesh, interior, pitch)
    metrics = FitMetrics(
        k=len(ellipsoids),
        excess_volume_ratio=float(volume / mesh_volume) if mesh_volume > 0 else float("nan"),
        surface_coverage=surface_coverage(ellipsoids, surface),
        union_volume=volume,
        mesh_volume=mesh_volume,
        volume_reliable=reliable,
    )
    return EllipsoidFit(ellipsoids, metrics, backend, coverage)


def auto_fit(
    mesh: trimesh.Trimesh,
    k_max: int = 10,
    coverage: float = 0.98,
    backend: Backend = "gmm",
    seed: int = 0,
    tolerance: float = 0.05,
    progress: ProgressFn = _noop,
) -> EllipsoidFit:
    """Sweep k and take the smallest one that is near the best result.

    The excess-volume curve is not monotonic -- splitting a round object makes it
    worse -- so this tracks a running best rather than stopping at the first
    increase. That is what keeps spherical objects at a single ellipsoid.
    """
    results: list[EllipsoidFit] = []
    for k in range(1, max(1, k_max) + 1):
        progress(k / max(k_max, 1), f"Fitting k={k}…")
        try:
            results.append(fit(mesh, k, coverage, backend, seed))
        except Exception:
            continue
    if not results:
        raise RuntimeError("no ellipsoid fit succeeded")

    ratios = [
        r.metrics.excess_volume_ratio
        if np.isfinite(r.metrics.excess_volume_ratio)
        else np.inf
        for r in results
    ]
    best = min(ratios)
    for result, ratio in zip(results, ratios):
        if ratio <= best * (1.0 + tolerance):
            return result
    return results[int(np.argmin(ratios))]


# ---------------------------------------------------------------------------
# Cache / export
# ---------------------------------------------------------------------------


def cache_path(root: Path, name: str, source: str) -> Path:
    return Path(root) / "ellipsoids" / f"{name}__{source}.json"


def export_json(out_dir: Path, name: str, source: str,
                fit_result: EllipsoidFit) -> Path:
    """Write one fit as a standalone file, separate from the multi-fit cache.

    This is the CHOSEN decomposition for an object -- one file, one fit, the
    thing downstream code consumes. It is deliberately not the same store as
    :func:`save_cached`, which keeps every (backend, k, coverage) combination
    that was ever tried under a signature key so re-picking one in the browser is
    instant. That cache is scratch and reproducible; an export is a decision, so
    it lives in the committed ``fits/`` directory (see ``data.FITS_DIR``).
    """
    path = Path(out_dir) / f"{name}__{source}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"object": name, "source": source, **fit_result.to_dict()}
    path.write_text(json.dumps(payload, indent=1))
    return path


def _signature(backend: str, k: int | None, coverage: float) -> str:
    return f"{backend}_k{'auto' if k is None else k}_c{coverage:.3f}"


def load_cached(
    root: Path, name: str, source: str, backend: str, k: int | None, coverage: float
) -> EllipsoidFit | None:
    path = cache_path(root, name, source)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
        entry = blob.get("fits", {}).get(_signature(backend, k, coverage))
        return EllipsoidFit.from_dict(entry) if entry else None
    except Exception:
        return None


def save_cached(
    root: Path,
    name: str,
    source: str,
    backend: str,
    k: int | None,
    coverage: float,
    fit_result: EllipsoidFit,
) -> Path:
    path = cache_path(root, name, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"object": name, "source": source, "fits": {}}
    if path.exists():
        try:
            blob = json.loads(path.read_text())
            blob.setdefault("fits", {})
        except Exception:
            pass
    blob["object"], blob["source"] = name, source
    blob["fits"][_signature(backend, k, coverage)] = fit_result.to_dict()
    path.write_text(json.dumps(blob, indent=1))
    return path
