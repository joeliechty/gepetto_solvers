"""Baking the EXACT half of an object: its OpenVDB signed-distance grid.

Every object in the registry carries two representations. The ellipsoid form
``E_obj`` is derived or fitted and travels in the repository; the exact form is a
baked ``.vdb`` grid, which does not (it is build output, and there is a lot of
it). This module produces the second one, for both kinds of object the registry
holds:

===============================  ==========================================
:func:`bake_analytic`            a spec with a closed-form SDF
:func:`bake_mesh`                a spec whose truth is a scanned mesh
===============================  ==========================================

The two paths share :func:`fillet`, :func:`derive_band_halfwidth` and
:func:`write_grid`, which is where the properties a solve actually depends on
are decided. Both are driven by ``scripts/objects/setup_objects.py``; neither is
meant to be called ad hoc, because a grid baked with different settings than its
siblings is indistinguishable from a correct one until a solve behaves oddly.

**Frames.** A grid is baked in the object's OWN frame -- the same frame the
ellipsoid form is expressed in -- because the two are attached to one
``EnvironmentConfig`` and composed with one optimized object pose. Getting this
wrong does not produce an error; it produces an object whose exact geometry sits
a couple of centimetres from its approximation, so a solve that hands off from
one to the other appears to teleport. The mesh path is where the risk lives, and
it resolves the frame through exactly one helper --
:func:`~gepetto_solvers.core.objects.ycb.ellipsoids.fit_recenter` -- shared with
the code that builds the spec.
"""

from __future__ import annotations

import os

import numpy as np

from . import OBJECTS_DIR

#: Voxel edge (m). 1 mm is what every grid in this repository has been baked at,
#: and the constant that the finite-difference defaults in the contact factors
#: are keyed to: they step half a voxel, which on a coarser grid means a coarser
#: normal field for every solve that touches the object.
DEFAULT_VOXEL_SIZE = 1e-3

#: Floor on the stored band half-width (m). See :func:`derive_band_halfwidth`.
DEFAULT_BAND_FLOOR = 0.03

#: Fillet radius (m) for a grid baked from a MESH. Deliberately smaller than the
#: 5 mm the analytic box primitives use: 5 mm of rounding removes the features
#: that make several scanned objects graspable at all -- a marker's clip, the
#: tines of a fork -- where on a 25 mm box it only softens the corners.
DEFAULT_MESH_EDGE_RADIUS = 0.002

#: The largest fingertip in the repository (the Allegro pad, 14 mm), used by
#: :func:`derive_band_halfwidth` as the standoff a contact sphere's CENTRE keeps
#: from the surface. Read as a default rather than off a hand, because a grid is
#: a property of the object and must serve whichever hand reaches for it.
MAX_TIP_RADIUS = 0.014


def require_openvdb():
    """The ``openvdb`` Python module, or a pointed :class:`ImportError`.

    Not a dependency in ``pyproject.toml`` and cannot be: the Python bindings
    ship with the conda package, not on PyPI. Everything else in this repository
    treats a missing grid as "this object has no exact form" and carries on, so
    the one place that must complain is the baker itself.
    """
    try:
        import openvdb as vdb
    except ImportError:
        try:
            import pyopenvdb as vdb
        except ImportError as exc:
            raise ImportError(
                "baking an SDF needs the OpenVDB Python bindings, which come "
                "from conda rather than PyPI:\n"
                "    conda install -c conda-forge openvdb\n"
                "(conda_setup_py11_mac.sh installs them alongside the rest of "
                "the C++ toolchain)") from exc
    return vdb


# ---------------------------------------------------------------------------
# How much field to store.
# ---------------------------------------------------------------------------

def derive_band_halfwidth(spec, *, tip_radius=MAX_TIP_RADIUS, margin=0.01,
                          floor=DEFAULT_BAND_FLOOR, samples=512):
    """How far either side of the surface this object's grid must carry field.

    Outside the stored band an OpenVDB sampler returns the constant background
    with ZERO gradient, so a witness point out there gets no pull at all and the
    contact simply never closes. The band is therefore not a quality knob but a
    reachability one: it has to cover wherever a solve can first meet this grid.

    Which is a much smaller region than it used to be. The grids in this
    repository were baked with 60 mm of band because, before the staged pipeline
    existed, the SDF had to guide a finger in from free space. It no longer does:
    the approach is planned against ``E_obj``, and the grid is first touched at
    the hand-off, with the fingertips already resting on the ellipsoid. So what
    the band must span is the gap between the two surfaces --

        max over the proxy surface of (distance out to it from the true surface)
        + the contact sphere's radius + a margin

    -- which this measures directly by sampling the proxy and reading the
    object's own analytic distance at each sample. For a sphere, an ellipsoid or
    a well-fitted set that gap is ~0 and the answer is the floor; the box family
    is the worst case, its proxy being the minimum-volume enclosing ellipsoid at
    ``sqrt(3)`` times the half-extents.

    ``floor`` is what a mesh-backed object gets, since its truth is not
    analytically available here to measure against. It is also the floor for
    everything else, so no object ends up with a band too thin to be reached
    across.
    """
    from ..geometry.scene import primitive_surface_gap, proxy_semi_axes

    axes = np.asarray(proxy_semi_axes(spec), float)
    # A Fibonacci sphere, mapped onto the proxy: an even spread of directions
    # with no clustering at the poles, so the maximum below is not decided by
    # where the samples happen to bunch up.
    i = np.arange(samples) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / samples)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    unit = np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)

    gap = 0.0
    for point in unit * axes:
        try:
            gap = max(gap, float(primitive_surface_gap(point, spec)))
        except ValueError:
            # A spec type with no analytic distance: nothing to measure against,
            # so fall through to the floor, which is what a mesh gets anyway.
            return float(floor)
    return float(max(floor, gap + tip_radius + margin))


# ---------------------------------------------------------------------------
# Smoothing.
# ---------------------------------------------------------------------------

def fillet(grid, radius, vdb=None):
    """Round every CONVEX edge of a level set by ``radius`` (m), in place of the
    sharp crease the source geometry had. Returns a new grid.

    Sharp edges are not a cosmetic problem here. A witness contact builds its
    whole contact frame from ``grad Phi``, which flips through ~90 degrees across
    one voxel at a crease, so a fingertip sliding along a face reaches the edge
    and stalls against a normal that will not settle. The grasp-alignment
    constraint is worse off still: it differentiates the normal field a second
    time, and a crease hands it a shape operator that diverges rather than one
    that is merely kinked. ``scripts/objects/make_cube.py`` has rounded the box
    primitives for exactly this reason since long before either constraint
    existed.

    This is the morphological OPENING -- erode by ``radius``, then dilate back --
    generalized from that script's shrink-then-offset construction to geometry
    with no closed form to shrink. Each step is a MESH round trip: polygonize the
    offset isosurface, then re-derive a distance field from it. The re-derivation
    is the part that does the work and the part that is easy to leave out --
    shifting an isolevel down and straight back up again is a no-op, because the
    field still describes the original surface. Rebuilding from the eroded SHAPE
    is what forgets the corner.

    Round-tripping through meshes rather than editing voxel values directly is
    deliberate. A hand-shifted field is no longer normalized, its background no
    longer matches its band, and the discontinuity that leaves at the band edge
    produces a gradient far worse than the crease being removed. Every
    intermediate here is a level set OpenVDB built itself.

    Flat regions come back exactly where they started, so an object's dimensions
    are unchanged; only within ``radius`` of a convex edge does the surface
    become a fillet. CONCAVE creases are untouched -- an opening cannot round
    them, and the operation that could (a closing) fills any gap narrower than
    twice the radius, which on a real object means welding shut the hole in a mug
    handle and planning against something that is not there.
    """
    vdb = vdb or require_openvdb()
    if radius <= 0.0:
        return grid

    transform = grid.transform
    voxel = transform.voxelSize()[0]
    if radius < voxel:
        raise ValueError(
            f"fillet radius {radius} m is smaller than the {voxel} m voxel, so "
            "the erosion has nothing to resolve -- the result would be the "
            "original creases plus interpolation noise. Bake finer, or round by "
            "at least one voxel.")
    half_width = _band_voxels(grid, vdb)
    if radius > half_width * voxel:
        raise ValueError(
            f"fillet radius {radius} m exceeds the grid's {half_width * voxel} m "
            "band, so the offset isosurfaces it needs are not stored. Bake a "
            "wider band, or round by less.")

    # Erode, then dilate: two offsets in opposite directions, each rebuilt.
    eroded = _offset(grid, -radius, transform, half_width, vdb)
    return _offset(eroded, +radius, transform, half_width, vdb)


def _offset(grid, distance, transform, half_width, vdb):
    """A level set of the surface ``distance`` metres along ``grid``'s outward
    normal: polygonize that isosurface, rebuild a distance field from it.

    The rebuild is not incidental. Polygonizing at an isovalue alone would just
    name a surface the original field already contained; deriving fresh distances
    from the resulting SHAPE is what makes the offset a new object, corners and
    all."""
    points, triangles, quads = grid.convertToPolygons(isovalue=float(distance))
    if len(points) == 0:
        raise ValueError(
            f"offsetting by {distance} m left no surface -- the object has a "
            "feature thinner than twice the fillet radius. Round by less.")
    out = vdb.FloatGrid.createLevelSetFromPolygons(
        points, triangles=triangles, quads=quads,
        transform=transform, halfWidth=half_width)
    out.gridClass = vdb.GridClass.LEVEL_SET
    return out


def _band_voxels(grid, vdb):
    """The stored half-band of ``grid``, in VOXELS, for a rebuild that should
    keep it. Read off the background value, which a level set sets to its own
    band half-width -- so a rebuild cannot silently narrow the field that the
    reachability argument in :func:`derive_band_halfwidth` depends on."""
    voxel = grid.transform.voxelSize()[0]
    background = float(grid.background)
    if not np.isfinite(background) or background <= 0.0:
        return float(vdb.LEVEL_SET_HALF_WIDTH)
    return max(float(vdb.LEVEL_SET_HALF_WIDTH), background / voxel)


# ---------------------------------------------------------------------------
# The two bakers.
# ---------------------------------------------------------------------------

def bake_analytic(spec, *, voxel_size=DEFAULT_VOXEL_SIZE, band_halfwidth=None,
                  progress=None):
    """A grid for a spec with a closed-form SDF, sampled from
    :func:`~gepetto_solvers.core.geometry.scene.primitive_surface_gap`.

    That function is the one this repository already trusts to report an achieved
    contact gap independently of the solver, and it was written to mirror what
    the original per-primitive bakers wrote -- fillets included. Sampling it here
    rather than re-deriving each primitive's SDF is what keeps the baked grid and
    the independent readout the same surface: they cannot drift, because there is
    only one definition.

    Fillets come along with it for the box and cylinder, whose specs carry
    ``edge_radius``. A spec type with creases and no such radius RAISES rather
    than baking a sharp grid -- see :func:`fillet` for why a crease is a solve
    failure and not a cosmetic one.
    """
    from ..geometry.scene import primitive_surface_gap, proxy_semi_axes

    vdb = require_openvdb()
    _require_smooth(spec)

    if band_halfwidth is None:
        band_halfwidth = derive_band_halfwidth(spec)

    grid = vdb.FloatGrid(float(band_halfwidth))
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)
    accessor = grid.getAccessor()

    # proxy_semi_axes bounds every primitive type, so it bounds the surface too
    # (loosely for the box, exactly for the rest) -- a conservative box to sweep.
    reach = np.asarray(proxy_semi_axes(spec), float) + band_halfwidth
    n = np.ceil(reach / voxel_size).astype(int)

    total = int((2 * n[0] + 1) * (2 * n[1] + 1))
    done = 0
    for i in range(-n[0], n[0] + 1):
        for j in range(-n[1], n[1] + 1):
            for k in range(-n[2], n[2] + 1):
                point = np.array([i, j, k], float) * voxel_size
                value = primitive_surface_gap(point, spec)
                if abs(value) < band_halfwidth:
                    accessor.setValueOn((i, j, k), float(value))
            done += 1
            if progress is not None and done % 64 == 0:
                progress(done / total)
    grid.signedFloodFill()
    return grid


def bake_mesh(vertices, faces, *, voxel_size=DEFAULT_VOXEL_SIZE,
              band_halfwidth=DEFAULT_BAND_FLOOR,
              edge_radius=DEFAULT_MESH_EDGE_RADIUS):
    """A grid for a scanned mesh, already expressed in the OBJECT frame.

    ``vertices`` and ``faces`` go in verbatim: placing the mesh in the frame its
    ellipsoid form uses is the CALLER's job, because only the caller knows which
    fit the object's spec was built from. See this module's header on why that is
    the one thing worth being careful about, and
    :func:`~gepetto_solvers.core.objects.ycb.sdf.bake_ycb` for the one
    implementation of it.

    Always filleted, because a scan has no ``edge_radius`` to inherit and
    reproduces every sharp edge it was digitized with.
    """
    vdb = require_openvdb()
    transform = vdb.createLinearTransform(voxelSize=voxel_size)
    grid = vdb.FloatGrid.createLevelSetFromPolygons(
        np.asarray(vertices, dtype=np.float32),
        triangles=np.asarray(faces, dtype=np.uint32),
        transform=transform,
        halfWidth=max(3.0, band_halfwidth / voxel_size))
    _require_interior(grid, voxel_size)
    return fillet(grid, edge_radius, vdb=vdb)


def _require_interior(grid, voxel_size, min_voxels=2.0):
    """Raise unless the level set came out with a real inside.

    The property that actually matters, and the reason it is checked HERE rather
    than on the input mesh. A scan is essentially never watertight -- the YCB
    meshes have holes where the scanner could not see, and ``trimesh`` cannot
    close most of them -- but OpenVDB's sign flood fill copes with that perfectly
    well, so refusing an open mesh would reject almost every real object for a
    defect that does not affect the result.

    What genuinely breaks is a mesh so open that the fill cannot decide which
    side is inside. Then the field comes out non-negative everywhere: there is no
    solid, only a thin shell around a surface, and every contact constraint built
    on it pushes outward from nothing. That is what this detects, by asking the
    field the direct question -- how far inside does it ever get -- rather than
    asking the mesh a proxy question it fails for unrelated reasons.
    """
    lo, _hi = grid.evalMinMax()
    if lo > -min_voxels * voxel_size:
        raise ValueError(
            f"the level set has no interior (most negative value {lo * 1000:.2f} "
            f"mm, i.e. under {min_voxels} voxels deep), so OpenVDB's sign fill "
            f"could not tell inside from outside. The mesh is open enough that "
            f"it describes a surface rather than a solid -- pick another source "
            f"for this object, or repair the scan")


#: Spec types whose analytic SDF is C-infinity everywhere, so they need no
#: fillet. The rest have creases and must carry an ``edge_radius``.
_SMOOTH_TYPES = frozenset({"sphere", "ellipsoid", "capsule", "ellipsoid_set"})


def _require_smooth(spec):
    """Raise unless this spec's analytic SDF is free of sharp edges -- either
    inherently, or because it carries the ``edge_radius`` that rounds them.

    Guards the case the analytic path otherwise gets wrong SILENTLY: a new
    primitive with flat faces, added without a fillet radius, bakes a grid that
    looks right in a picture and stalls the first solve that slides a fingertip
    over one of its edges."""
    t = spec["type"]
    if t in _SMOOTH_TYPES:
        return
    if float(spec.get("edge_radius", 0.0)) > 0.0:
        return
    raise ValueError(
        f"the {t!r} primitive has C0 creases where its faces meet and no "
        f"'edge_radius' in its spec to round them. A gradient-based witness "
        f"contact stalls on those creases -- give the spec an edge_radius (the "
        f"box primitives use 0.005), which primitive_surface_gap already honours")


def bake_spec(name, spec, *, voxel_size=DEFAULT_VOXEL_SIZE, band_halfwidth=None,
              cache=None, progress=None):
    """Bake whichever exact form ``spec`` has, by the one rule that decides it.

    Three kinds of truth, in the order they are tested:

    1. a SCAN (``ycb:`` objects) -- the mesh is the object, so it goes through
       :func:`~gepetto_solvers.core.objects.ycb.sdf.bake_ycb`, which downloads it
       and places it in the frame the object's fit defines;
    2. a HULL (``megaminx``) -- the spec carries the real solid's vertices
       alongside an ellipsoid that only bounds it, so the solid is what gets
       baked. Skipping this would give the object an "exact" form identical to
       its approximation, which is exactly the distinction it exists to have;
    3. an ANALYTIC distance -- everything else, sampled from
       ``primitive_surface_gap``.

    Returns the written path.
    """
    from ..geometry.scene import get_primitive_specs  # noqa: F401  (cycle guard)

    if "ycb" in spec:
        from .ycb.sdf import bake_ycb
        return bake_ycb(spec["ycb"], cache=cache, voxel_size=voxel_size,
                        band_halfwidth=band_halfwidth or DEFAULT_BAND_FLOOR)

    path = os.path.join(OBJECTS_DIR, spec["vdb"])
    if band_halfwidth is None:
        band_halfwidth = derive_band_halfwidth(spec)

    hull = spec.get("hull_vertices")
    if hull is not None and len(hull):
        grid = _bake_hull(np.asarray(hull, float), voxel_size=voxel_size,
                          band_halfwidth=band_halfwidth)
    else:
        grid = bake_analytic(spec, voxel_size=voxel_size,
                             band_halfwidth=band_halfwidth, progress=progress)
    return write_grid(grid, path, name=name)


def _bake_hull(vertices, *, voxel_size, band_halfwidth,
               edge_radius=DEFAULT_MESH_EDGE_RADIUS):
    """A grid for a convex solid given by its vertices, via the convex hull.

    Used for the polyhedral primitives, whose spec carries the SOLID's vertices
    next to an ellipsoid that merely bounds it. Filleted like any other mesh:
    a dodecahedron's edges are as sharp as a scan's, and a witness contact stalls
    on them the same way."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(np.asarray(vertices, float))
    return bake_mesh(hull.points, hull.simplices, voxel_size=voxel_size,
                     band_halfwidth=band_halfwidth, edge_radius=edge_radius)


def write_grid(grid, path, name="surface"):
    """Write one grid to ``path``, creating the directory. Named because a reader
    takes the file's FIRST grid and an unnamed one is harder to identify later."""
    vdb = require_openvdb()
    grid.name = name
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vdb.write(str(path), grids=[grid])
    return path
