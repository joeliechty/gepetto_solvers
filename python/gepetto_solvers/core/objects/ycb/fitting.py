"""The one fitting pipeline: mesh in, committed ellipsoid-set fit out.

Three callers want to fit a YCB object -- the browser's *Fit ellipsoids* button,
its ``--fit`` CLI, and the hand visualizer's fit-on-select -- and they must all
do it the SAME way. When they each spelled out the sequence themselves they
drifted, and a drifted copy is invisible: it still produces a fit, just a worse
one, and nothing says so. So the sequence lives here once and the callers pass
their settings in.

The sequence, and why each step is what it is:

* ``max_texture=1024`` matches the browser's texture default. Texture size does
  not affect geometry, but the mesh is CACHED under it, so a caller asking for a
  different size re-decodes the OBJ for no benefit.
* Ground and centre before fitting, and record the shift as ``ground_offset``, so
  exported centres are in the displayed frame and can be mapped back.
* Check ``load_cached`` first: the multi-fit cache is keyed by
  (backend, k, coverage), so re-picking a combination already tried is instant.
* ``auto_fit(k_max=10)`` when ``k`` is None. **k_max stays 10.** Lowering it is
  tempting for a batch, but the sweep returns the smallest k whose excess volume
  is within tolerance of the BEST k found, so a smaller ceiling changes which fit
  is chosen, not just how long the search takes.
* The mesh's convex hull rides along on the fit (``EllipsoidFit.hull``), because
  the shells are a BOUND on the object and a scene still has to know where the
  object itself ends -- see that field for what goes wrong without it.
* ``save_cached`` then ``export_json``: the cache is scratch, the export is the
  decision downstream code reads.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import ellipsoids as ye
from .data import FITS_DIR, YcbCache, ground_and_center, ground_offset

# The browser's own texture default. See the module docstring for why callers
# should not vary it casually.
DEFAULT_MAX_TEXTURE = 1024

# The automatic sweep's ceiling. Part of WHICH fit gets chosen, not merely how
# long the search runs -- see the module docstring.
DEFAULT_K_MAX = 10


def _noop(fraction: float, message: str) -> None:
    del fraction, message


def fit_object(
    cache: YcbCache,
    name: str,
    source: str,
    *,
    backend: str = "gmm",
    k: int | None = None,
    coverage: float = 0.98,
    k_max: int = DEFAULT_K_MAX,
    max_texture: int | None = DEFAULT_MAX_TEXTURE,
    export_dir: Path = FITS_DIR,
    use_cache: bool = True,
    progress: Callable[[float, str], None] = _noop,
) -> tuple[ye.EllipsoidFit, Path]:
    """Fetch, fit, cache and export one object. Returns ``(fit, export path)``.

    ``k=None`` sweeps k automatically (the default, and what every caller should
    use unless a specific decomposition has been chosen by eye).
    """
    progress(0.0, f"fetching from `{source}`…")
    raw = cache.load_mesh(name, source, max_texture=max_texture, progress=progress)
    offset = ground_offset(raw)
    mesh = ground_and_center(raw)

    result = (ye.load_cached(cache.root, name, source, backend, k, coverage)
              if use_cache else None)
    fresh = result is None
    if not fresh:
        progress(1.0, "using cached fit.")
    else:
        progress(0.05, f"fitting with `{backend}`…")
        if k is None:
            result = ye.auto_fit(mesh, k_max=k_max, coverage=coverage,
                                 backend=backend, progress=progress)
        else:
            result = ye.fit(mesh, k, coverage=coverage, backend=backend)
        result.ground_offset = offset

    # Unconditionally, cached fit or not: the hull is a property of the MESH
    # loaded above, not of the decomposition, so re-reading it here also fills it
    # in for a cache entry written before this field existed.
    result.hull = ye.support_hull(mesh)
    if fresh:
        ye.save_cached(cache.root, name, source, backend, k, coverage, result)

    path = ye.export_json(export_dir, name, source, result)
    return result, path
