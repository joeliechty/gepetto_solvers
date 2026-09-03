"""Baking a YCB object's exact form: its scanned mesh, as an OpenVDB grid.

The mesh path of :mod:`gepetto_solvers.core.objects.sdf`, plus the one thing that
is specific to these objects and easy to get wrong -- putting the scan in the
same frame as the ellipsoid fit that approximates it.

That frame is composed of two shifts, and neither is optional:

1. ``data.ground_and_center`` -- centre in XY, rest the lowest point on ``z=0``.
   This is the frame the browser fitted the ellipsoids in, recorded in each fit
   file's ``frame`` block.
2. ``ellipsoids.fit_recenter`` -- move the origin to the middle of the union's
   bounding box, which is where every other primitive in the registry puts its
   object-local origin.

Both are applied to the mesh here, and the second is imported rather than
recomputed, so the grid and the shells cannot end up describing an object in two
different places. A mismatch is silent: nothing errors, the approximation simply
sits a centimetre or two off the exact geometry, and a solve that hands off from
one to the other looks like the object jumped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import OBJECTS_DIR
from ..sdf import (
    DEFAULT_BAND_FLOOR,
    DEFAULT_MESH_EDGE_RADIUS,
    DEFAULT_VOXEL_SIZE,
    bake_mesh,
    write_grid,
)
from . import ellipsoids as ye
from .data import DEFAULT_CACHE, FITS_DIR, Catalog, YcbCache, ground_and_center

#: Where a baked YCB grid goes, relative to ``OBJECTS_DIR``. Matches the ``vdb``
#: key the specs name (``primitives.ycb_primitive_specs``).
YCB_GRID_SUBDIR = "ycb"


def fit_path(name: str) -> Path | None:
    """The committed fit for ``name``, or None. Fits are named
    ``<object>__<source>.json``, and the source is not known up front."""
    matches = sorted(Path(FITS_DIR).glob(f"{name}__*.json"))
    return matches[0] if matches else None


def grid_path(name: str, objects_dir=OBJECTS_DIR) -> Path:
    """Where ``name``'s baked grid lives."""
    return Path(objects_dir) / YCB_GRID_SUBDIR / f"{name}.vdb"


def object_frame_mesh(name: str, cache: YcbCache | None = None):
    """The scanned mesh for ``name``, moved into its OBJECT frame.

    Downloads it if the cache does not have it -- the scans are ~0.6 GB for the
    full set and deliberately not in version control.

    Raises rather than guessing when the object has no committed fit: the object
    frame is DEFINED by the fit (see this module's header), so a grid baked
    without one would be in a frame nothing else uses, and would place the object
    wrong in every scene that loaded it.
    """
    path = fit_path(name)
    if path is None:
        raise FileNotFoundError(
            f"no committed ellipsoid fit for {name!r} in {FITS_DIR}. A YCB "
            f"object's frame is defined by its fit, so there is nothing to bake "
            f"the grid RELATIVE TO -- fit it first with "
            f"`python scripts/objects/ycb_browser.py --fit {name}`")

    blob = json.loads(path.read_text())
    cache = cache or YcbCache(Catalog(), DEFAULT_CACHE)
    mesh = ground_and_center(
        cache.load_mesh(name, blob.get("source", "google_16k")))
    mesh.apply_translation(-ye.fit_recenter(blob["ellipsoids"]))
    return mesh


def bake_ycb(name: str, *, cache: YcbCache | None = None,
             voxel_size: float = DEFAULT_VOXEL_SIZE,
             band_halfwidth: float = DEFAULT_BAND_FLOOR,
             edge_radius: float = DEFAULT_MESH_EDGE_RADIUS,
             objects_dir=OBJECTS_DIR) -> Path:
    """Bake ``name``'s scan into its grid and write it. Returns the path.

    Holes are closed where ``trimesh`` can close them, and tolerated where it
    cannot. A YCB scan is essentially never watertight -- there is always a patch
    the scanner could not see, typically the underside -- and OpenVDB's sign
    flood fill handles that: measured on the potted meat can, the banana and the
    cracker box, all three come out with correct interiors despite every one
    failing ``is_watertight``. What is checked instead is the RESULT, by
    ``sdf._require_interior``, which asks whether the baked field has an inside
    at all rather than asking the mesh a proxy question it fails for unrelated
    reasons.
    """
    mesh = object_frame_mesh(name, cache=cache)
    if not mesh.is_watertight:
        mesh = mesh.copy()
        mesh.fill_holes()          # best effort; the result is what is checked

    grid = bake_mesh(np.asarray(mesh.vertices, dtype=np.float32),
                     np.asarray(mesh.faces, dtype=np.uint32),
                     voxel_size=voxel_size,
                     band_halfwidth=band_halfwidth,
                     edge_radius=edge_radius)
    return Path(write_grid(grid, grid_path(name, objects_dir), name=name))


def fitted_objects() -> list[str]:
    """Every object with a committed fit, i.e. everything the registry offers as
    a ``ycb:`` primitive and therefore everything :func:`bake_ycb` can bake."""
    return sorted({p.name.split("__")[0] for p in Path(FITS_DIR).glob("*__*.json")})
