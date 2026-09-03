"""Every object carries BOTH representations, and they describe one object.

The pipeline's later phases contact the exact geometry while collision stays on
the approximation, so an object missing either half is an object those phases
cannot be run on -- and the way that shows up, without these, is a phase greying
itself out for reasons nobody can see.

Split into three questions that fail differently:

* does the object NAME both forms (a fact about the registry, always checkable);
* is the exact one present on this machine (a fact about the checkout -- the
  grids are gitignored build output, so this is skipped when they are absent);
* do the two agree about where the object IS (the one that is silent when wrong).
"""

from __future__ import annotations

import numpy as np
import pytest

from _pkg import scene
from gepetto_solvers.core.objects import (
    has_exact_form,
    names_exact_form,
    vdb_path,
)

SPECS = scene.get_primitive_specs()
NAMES = sorted(SPECS)


@pytest.mark.parametrize("name", NAMES)
def test_every_object_has_an_ellipsoid_form(name):
    """``proxy_semi_axes`` answers for every primitive type, and it must: it is
    what the collision inequalities read in every phase."""
    axes = np.asarray(scene.proxy_semi_axes(SPECS[name]), float)
    assert axes.shape == (3,)
    assert (axes > 0).all(), f"{name} has a degenerate proxy: {axes}"


@pytest.mark.parametrize("name", NAMES)
def test_every_object_names_an_exact_form(name):
    """A spec names its grid whether or not the file is here. The two questions
    are separate on purpose -- an un-baked checkout must not look like a registry
    of objects that inherently cannot be contacted precisely."""
    assert names_exact_form(SPECS[name]), (
        f"{name} names no vdb, so nothing can contact its exact geometry")


@pytest.mark.parametrize("name", NAMES)
def test_a_baked_grid_shares_its_objects_frame(name):
    """THE SILENT ONE. The two representations are attached to one env and
    composed with one object pose, so if they disagree about the ORIGIN nothing
    raises -- the object simply has an approximation sitting somewhere near its
    exact geometry, and the phase 2 -> 3 hand-off looks like it teleported.

    Checked on the origin rather than the shape, because the origin is the part
    that can be wrong. Both forms are centred on the object's own bounding box by
    construction, so the grid's centre belongs at zero; a baker that skipped a
    shift lands it at that shift instead, which for a YCB object is centimetres
    (the chips can's recentre alone is 105 mm along z).

    Deliberately NOT checked here: whether the ellipsoid form bounds the exact
    one. It does not always, and that is a property of the fits rather than of
    the frames -- ``proxy_semi_axes`` gives an ellipsoid_set the half-extents of
    its bounding BOX, and an ellipsoid through the face centres does not contain
    the box's corners, so a boxy object's true surface legitimately pokes out.
    """
    spec = SPECS[name]
    if not has_exact_form(spec):
        pytest.skip("grid not baked on this machine "
                    "(python scripts/objects/setup_objects.py)")
    vdb = pytest.importorskip("openvdb", reason="reading a grid needs pyopenvdb")

    grid = vdb.readAll(vdb_path(spec))[0][0]
    lo, hi = grid.evalActiveVoxelBoundingBox()
    voxel = grid.transform.voxelSize()[0]
    centre = 0.5 * (np.asarray(lo, float) + np.asarray(hi, float)) * voxel

    # Scaled to the object, with a floor: the band is clipped slightly
    # differently on each side of an asymmetric shape, which moves the centre of
    # the ACTIVE region by a few millimetres without moving the surface at all.
    axes = np.asarray(scene.proxy_semi_axes(spec), float)
    tolerance = max(0.010, 0.05 * float(axes.max()))
    assert np.linalg.norm(centre) < tolerance, (
        f"{name}: the baked grid is centred {np.round(centre * 1000, 1)} mm off "
        f"the object origin, so it is probably not in the same frame as the "
        f"ellipsoid form")
