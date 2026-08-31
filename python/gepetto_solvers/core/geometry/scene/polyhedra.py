"""Exact polyhedron geometry -- currently the dodecahedron behind the megaminx.

A spec's ``hull_vertices`` is the real solid its analytic surface only bounds; the
viewer draws it, ``object_inplane_widths`` measures the silhouette on it, and
``object_extent_along`` seats the support plane on it.
"""

import numpy as np


def Rx(theta):
    """Rotation matrix about the X axis (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


# Regular dodecahedron: circumradius / inradius. For the canonical vertex set
# below the circumradius is sqrt(3) and the face planes sit at phi^2/sqrt(1+phi^2)
# = 1.376382, so the ratio is sqrt(3) * sqrt(1 + phi^2) / phi^2. A solid specified
# "across the flats" is therefore 1.2584x wider corner to corner -- the whole
# reason the enclosing sphere cannot also be the resting height.
DODECAHEDRON_CIRCUM_OVER_INRADIUS = 1.2584085


def dodecahedron_vertices(face_to_face):
    """The 20 vertices (m, object-local) of a regular dodecahedron measuring
    ``face_to_face`` between opposite faces, oriented FACE DOWN: one face lies
    flat in the -Z plane and its antipode flat in +Z, so the solid rests on a
    +Z-normal table the way a real one does.

    Built from the canonical vertex set -- (+-1,+-1,+-1) and the three cyclic
    (0, +-1/phi, +-phi) families -- which is vertex-aligned, not face-aligned:
    dropping it on a table as-is balances it on a corner. Rotating a face normal
    (the (+-phi,+-1,0) family, verified against the convex hull rather than
    assumed) onto +Z is what makes the orientation physical.

    The convex hull of these points IS the dodecahedron, so a renderer needs
    nothing further, and their support function is its exact half-width along any
    direction -- which is how :func:`object_extent_along` seats the table.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    cube = np.array([[sx, sy, sz] for sx in (1.0, -1.0)
                     for sy in (1.0, -1.0) for sz in (1.0, -1.0)])
    pair = np.array([[a, b] for a in (1.0, -1.0) for b in (1.0, -1.0)])
    zero = np.zeros(len(pair))
    verts = np.vstack([
        cube,                                                    # (+-1, +-1, +-1)
        np.stack([zero, pair[:, 0] / phi, pair[:, 1] * phi], 1),  # (0, +-1/phi, +-phi)
        np.stack([pair[:, 0] / phi, pair[:, 1] * phi, zero], 1),  # cyclic
        np.stack([pair[:, 0] * phi, zero, pair[:, 1] / phi], 1),  # cyclic
    ])

    # Take a face normal to +Z. Any of the 12 works: the solid is centrally
    # symmetric, so standing one face up lays its opposite face flat on the table.
    normal = np.array([phi, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    # An orthonormal basis whose third axis is the face normal; R = basis^T maps
    # the normal to +Z. Seed the first tangent off whichever axis the normal
    # leans on least, so the cross product never degenerates.
    seed = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(seed, normal)
    u /= np.linalg.norm(u)
    rotation = np.stack([u, np.cross(normal, u), normal])

    # Scale by inradius, not circumradius: face_to_face is measured across the
    # flats, and the canonical solid's face planes sit at 1.376382 from center.
    inradius = np.sqrt(3.0) / DODECAHEDRON_CIRCUM_OVER_INRADIUS
    return (verts @ rotation.T) * (0.5 * face_to_face / inradius)


# Size of the 12-sided Rubik's cube (megaminx), measured across the flats.
MEGAMINX_FACE_TO_FACE = 0.070


def _megaminx_spec(face_to_face):
    """The megaminx primitive: a regular dodecahedron the solver sees as its
    CIRCUMSCRIBED sphere.

    The sphere has to be the circumsphere, not the inscribed one, so no part of
    the real solid ever escapes the surface the contact/collision factors
    evaluate -- the fingers stop on the vertex shell instead of pressing into a
    face. What that costs is that the sphere is 1.2584x the half-height the solid
    actually stands at, so the shell dips ~9 mm below a table the solid is
    resting flat on. ``hull_vertices`` is what makes that correct rather than
    wrong: the table is seated on the SOLID's support function (see
    :func:`object_extent_along`), so the object sits at the height a
    face-down dodecahedron sits at and the proxy sphere sinks, instead of the
    solid being levitated onto its corner to keep the sphere tangent.
    """
    semi = 0.5 * face_to_face * DODECAHEDRON_CIRCUM_OVER_INRADIUS
    semi_axes = (semi, semi, semi)
    hull = dodecahedron_vertices(face_to_face)
    return {
        "type": "ellipsoid",
        "semi_axes": semi_axes,
        "hull_vertices": hull,
        "face_to_face": face_to_face,
        "plot": lambda c: {"type": "ellipsoid", "center": c,
                           "semi_axes": semi_axes, "hull_vertices": hull},
    }
