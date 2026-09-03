"""Baked grids have no sharp edges.

The property most likely to regress silently, because a sharp grid renders
perfectly and only misbehaves once a solve slides a fingertip along one of its
faces. A witness contact builds its whole contact frame from ``grad Phi``, which
turns through ~90 degrees across one voxel at a crease; the grasp-alignment
constraint differentiates that field a second time, so a crease hands it a shape
operator that diverges rather than one that is merely kinked.

``scripts/objects/make_cube.py`` has rounded the box primitives for this reason
since long before either constraint existed. What is new is that the same has to
hold for geometry with no closed form to shrink -- every scanned object -- which
is what :func:`sdf.fillet` does and what these check.
"""

from __future__ import annotations

import numpy as np
import pytest

from gepetto_solvers.core.objects import sdf as osdf

pytest.importorskip("openvdb", reason="baking a grid needs pyopenvdb")
trimesh = pytest.importorskip("trimesh", reason="the mesh path needs trimesh")

#: A box with 25/20/25 mm half-extents -- the `cube` primitive's shape, and the
#: simplest thing with a convex edge to round.
HALF = (0.025, 0.020, 0.025)
RADIUS = 0.004


@pytest.fixture(scope="module")
def vdb():
    return osdf.require_openvdb()


@pytest.fixture(scope="module")
def sharp_box(vdb):
    """A level set straight off a box mesh: every edge as sharp as it was."""
    box = trimesh.creation.box(extents=[2 * h for h in HALF])
    return vdb.FloatGrid.createLevelSetFromPolygons(
        np.asarray(box.vertices, np.float32),
        triangles=np.asarray(box.faces, np.uint32),
        transform=vdb.createLinearTransform(voxelSize=1e-3),
        halfWidth=30.0)


@pytest.fixture(scope="module")
def rounded_box(sharp_box, vdb):
    return osdf.fillet(sharp_box, RADIUS, vdb=vdb)


def _sampler(grid):
    """Trilinear world-space sampling, written out rather than taken from the
    C++ side: the point of a test readout is that it cannot inherit the sampler
    the code under test uses."""
    accessor = grid.getConstAccessor()
    transform = grid.transform

    def phi(point):
        c = transform.worldToIndex(tuple(float(x) for x in point))
        base = [int(np.floor(x)) for x in c]
        frac = [c[k] - base[k] for k in range(3)]
        total = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((1 - frac[0]) if dx == 0 else frac[0]) \
                        * ((1 - frac[1]) if dy == 0 else frac[1]) \
                        * ((1 - frac[2]) if dz == 0 else frac[2])
                    total += w * accessor.getValue(
                        (base[0] + dx, base[1] + dy, base[2] + dz))
        return total
    return phi


def test_the_fillet_cuts_the_corner_back_by_the_right_amount(sharp_box, rounded_box):
    """THE DECISIVE ONE, because it is a number the geometry fixes exactly.

    Rounding a 90-degree edge by ``r`` moves the edge line inward by
    ``r * (sqrt(2) - 1)``: the fillet's centre sits ``r`` from both faces, so the
    surface nearest the original edge is ``r * sqrt(2) - r`` away from it. The
    old corner point should therefore read exactly that far OUTSIDE the rounded
    solid, where it read zero on the sharp one."""
    edge = np.array([HALF[0], HALF[1], 0.0])       # the +x/+y vertical edge
    assert abs(_sampler(sharp_box)(edge)) < 5e-4, "the sharp box's edge is not on its surface"

    expected = RADIUS * (np.sqrt(2.0) - 1.0)
    assert _sampler(rounded_box)(edge) == pytest.approx(expected, abs=3e-4)


def test_the_fillet_leaves_the_flat_faces_where_they_were(sharp_box, rounded_box):
    """An opening removes material only near a convex edge. If the faces move,
    the object has been shrunk instead of rounded -- and every dimension a scene
    was set up against is then wrong by the fillet radius."""
    for grid in (sharp_box, rounded_box):
        phi = _sampler(grid)
        xs = np.linspace(0.020, 0.030, 201)
        crossing = float(np.interp(0.0, [phi([x, 0.0, 0.0]) for x in xs], xs))
        assert crossing == pytest.approx(HALF[0], abs=3e-4)


def test_the_fillet_makes_the_normal_turn_more_smoothly(sharp_box, rounded_box):
    """What the fillet is FOR: the normal field near an edge stops jumping.

    Measured as the worst angular step between adjacent samples on a ring
    sweeping around the edge -- which is what a fingertip sliding across it
    experiences, and what a Gauss-Newton step linearizes through."""
    def worst_step(grid, ring_radius=0.006, samples=64):
        phi = _sampler(grid)
        corner = np.array([HALF[0], HALF[1], 0.0])
        h = 1e-4
        normals = []
        for t in np.linspace(0.0, np.pi / 2, samples):
            p = corner + ring_radius * np.array([np.cos(t), np.sin(t), 0.0])
            g = np.array([(phi(p + e) - phi(p - e)) / (2 * h)
                          for e in (np.array([h, 0, 0]), np.array([0, h, 0]),
                                    np.array([0, 0, h]))])
            normals.append(g / (np.linalg.norm(g) or 1.0))
        normals = np.asarray(normals)
        dots = np.clip(np.sum(normals[1:] * normals[:-1], axis=1), -1.0, 1.0)
        return float(np.degrees(np.arccos(dots)).max())

    assert worst_step(rounded_box) < worst_step(sharp_box)


def test_an_unroundable_radius_is_refused(sharp_box, vdb):
    """Sub-voxel rounding measures the trilinear interpolant rather than the
    geometry, so it is refused rather than silently producing noise."""
    with pytest.raises(ValueError, match="smaller than"):
        osdf.fillet(sharp_box, 1e-4, vdb=vdb)


def test_a_creased_primitive_without_an_edge_radius_is_refused():
    """The analytic path inherits its fillets from the spec's ``edge_radius``, so
    a spec type with flat faces and no such radius must fail loudly. Otherwise a
    primitive added later bakes sharp and nothing says so until a phase-3 solve
    sticks on one of its edges."""
    spec = {"type": "cube", "half_extents": HALF}     # note: no edge_radius
    with pytest.raises(ValueError, match="edge_radius"):
        osdf.bake_analytic(spec)


def test_a_smooth_primitive_needs_no_edge_radius():
    """...and one that is C-infinity by construction is not asked for one."""
    osdf._require_smooth({"type": "ellipsoid", "semi_axes": (0.03, 0.03, 0.03)})
    osdf._require_smooth({"type": "sphere", "radius": 0.03})
