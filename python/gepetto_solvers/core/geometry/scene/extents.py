"""How big an object is along a direction, and which way it is longest.

:func:`object_extent_along` is the support half-width that seats the table;
:func:`object_inplane_widths` sweeps it over the support plane to get the
silhouette, and :func:`object_principal_inplane_axis` reads the Section 1.8
longest-in-plane axis off that sweep.
"""

import numpy as np


def proxy_semi_axes(spec):
    """Semi-axes (a, b, c) of a bounding hyper-ellipsoid for any primitive, in the
    OBJECT-LOCAL frame — the Section 1.7/1.8 pre-grasp proxy.

    §1.7 puts an ellipsoid *around* the object so the approach never sees a flat
    face or a sharp edge to stall on, so these enclose the primitive rather than
    hugging it. For the box that means the minimum-volume enclosing ellipsoid,
    whose axis-aligned semi-axes are ``sqrt(3) * half_extents`` (each corner
    ``(±hx, ±hy, ±hz)`` then satisfies ``x^T M x = 1``). Cylinder and capsule are
    modeled about their LOCAL Y axis, matching the ``make_*.py`` generators; the
    spec's ``rotation`` stands them up in the world afterwards.

    An ``ellipsoid`` primitive is already its own proxy. Raises for a spec whose
    type has no defined bound, rather than silently guessing.
    """
    t = spec["type"]
    if t == "ellipsoid":
        return np.asarray(spec["semi_axes"], dtype=float)
    if t == "ellipsoid_set":
        # Axis-aligned semi-axes of the box bounding the union. Genuinely a
        # BOUND, like the cube's sqrt(3) case: §1.7 wants the pre-grasp proxy to
        # enclose the object so the approach never sees a concave seam between two
        # members to stall on. Half the bounding box is the cheap such bound --
        # looser than an MVEE over the members, and unlike one it needs no solve.
        return 0.5 * np.asarray(spec["extents"], dtype=float)
    if t == "sphere":
        r = float(spec["radius"])
        return np.array([r, r, r])
    if t == "cylinder":
        r, h = float(spec["radius"]), float(spec["height"])
        return np.array([r, 0.5 * h, r])
    if t == "capsule":
        # Hemispherical caps extend the local-Y half-length by one radius.
        r, h = float(spec["radius"]), float(spec["height"])
        return np.array([r, 0.5 * h + r, r])
    if t == "cube":
        return np.sqrt(3.0) * np.asarray(spec["half_extents"], dtype=float)
    raise ValueError(f"no proxy ellipsoid defined for primitive type {t!r}")


def object_extent_along(spec, normal, rotation=None):
    """Object half-size along ``normal`` (m) -- used to seat a default support
    plane tangent to the object's underside, and (through
    :func:`object_inplane_widths`) to measure the object's in-plane silhouette.

    ``rotation`` is the object's world orientation. It matters for the analytic
    surfaces, whose geometry is stored in the OBJECT-LOCAL frame: a rotated
    object presents a different profile to the plane, and ignoring that seats the
    table at the wrong height. Most visible on a long thin object -- stand the
    screwdriver on end and its along-normal half-size goes from ~19 mm to ~106 mm.
    Passing None keeps the legacy axis-aligned reading.

    The baked-SDF primitives (cylinder/capsule/cube) are deliberately left on
    their existing special-cased handling: their specs carry a fixed ``rotation``
    the generators were baked with, which the branches below already account for.
    """
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) or 1.0)
    # Into the object's own frame, where the stored semi-axes/members live.
    n_local = (n if rotation is None
               else np.asarray(rotation, float).T @ n)
    t = spec["type"]
    hull = spec.get("hull_vertices")
    if hull is not None:
        # The spec carries the REAL solid the analytic surface only proxies (the
        # megaminx: a dodecahedron the factors see as its circumsphere; a YCB
        # object: the scanned mesh its ellipsoid set bounds). Seat the
        # table on the solid, since that is what the object rests on: how far the
        # lowest vertex reaches against n, which for a face-down solid is its
        # inradius. Using the proxy's half-width instead would hold a 70 mm
        # across-the-flats solid 88 mm tall -- balanced on a corner, which is a
        # pose it cannot physically hold. The proxy sphere then sinks into the
        # slab by the corner-vs-flat difference, which is correct: the sphere is
        # a bound on the solid, not the object.
        return float(np.max(-(np.asarray(hull, float) @ n_local)))
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
        # Support function ||diag(a) n||, the exact half-width along n. (For an
        # axis-aligned n -- the default +Z table -- this equals the L1 reading
        # this used to use, so no existing scene moves.)
        return float(np.linalg.norm(np.asarray(spec["semi_axes"], float) * n_local))
    if t == "ellipsoid_set":
        # Furthest any member reaches along n from the object origin: its center's
        # offset that way, plus its own reach. The reach of a rotated ellipsoid
        # along n is the support function ||diag(a) R^T n|| -- a norm, not the L1
        # sum the axis-aligned branch above uses. That distinction is not cosmetic
        # for these: a YCB fit's members are rotated to arbitrary angles, and L1
        # over-estimates by up to sqrt(3), which would seat the table centimetres
        # below a long thin object like the screwdriver.
        #
        # Max over members rather than sum, because they overlap by construction.
        # Signed center offset, not |offset|: the deepest member is the one whose
        # centre sits furthest AGAINST n, and taking the absolute value would let
        # a member on the far side masquerade as the lowest one.
        return max(
            float(-(np.asarray(m["center"], float) @ n_local)
                  + np.linalg.norm(np.asarray(m["semi_axes"], float)
                                   * (np.asarray(m["rotation"], float).T @ n_local)))
            for m in spec["members"])
    return 0.05


def inplane_basis(plane_normal):
    """Orthonormal ``(u, v)`` spanning the plane with normal ``plane_normal``,
    picked deterministically so repeated calls agree.

    ``u`` is seeded from whichever world axis the normal is LEAST aligned with
    (for the default +Z table that is +X), so the seed is never near-parallel to
    the normal and the Gram-Schmidt below is well conditioned. ``v = n x u``
    completes a right-handed frame, which makes sweeping ``cos(t) u + sin(t) v``
    a rotation about ``+n``.
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(n)))] = 1.0
    u = seed - (seed @ n) * n
    u = u / np.linalg.norm(u)
    return u, np.cross(n, u)


def object_inplane_widths(spec, rotation, plane_normal, *, n_angles=180):
    """``(dirs, widths)``: ``n_angles`` unit directions spanning the support
    plane over ``[0, pi)``, and the object's FULL width along each (m).

    This is the object's silhouette on the support plane, measured exactly. A
    shape's support width along a direction that LIES IN the plane is unchanged
    by projecting the shape onto that plane, so there is no projection step to
    do: the width along ``d`` is just ``h(d) + h(-d)`` of the solid itself,
    which is what :func:`object_extent_along` returns per direction. Half a turn
    is the whole sweep, since ``width(-d) == width(d)``.

    Unlike :func:`proxy_semi_axes`, this can see a direction that is not one of
    the object's own frame axes -- which is the point. A YCB fit's bounding box
    is taken in the frame the scan was exported in, so a screwdriver lying at
    27 degrees to that frame has its long axis quantised to +X and its
    elongation understated as 1.84; swept, it reads 6.12 at the true 27 degrees.
    The flat screwdriver reads 1.03 (isotropic!) against 6.23.

    The branches below are vectorised over directions because this runs per GUI
    frame: the same support functions as :func:`object_extent_along` (verified
    against it), but ~9 ms per YCB object becomes well under 1 ms. A spec
    carrying ``hull_vertices`` is measured on the SOLID, for the same reason
    :func:`object_extent_along` seats the table on it -- the silhouette that
    decides where §1.8 splits the support surface should be the object's, not
    that of shells overhanging it by a centimetre.
    """
    u, v = inplane_basis(plane_normal)
    theta = np.linspace(0.0, np.pi, int(n_angles), endpoint=False)
    dirs = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v

    hull = spec.get("hull_vertices")
    if hull is not None:
        H = np.asarray(hull, float)                                      # (V,3)
        U = dirs if rotation is None else dirs @ np.asarray(rotation, float)
        proj = H @ U.T                                                   # (V,K)
        return dirs, proj.max(axis=0) - proj.min(axis=0)

    if spec["type"] == "ellipsoid_set":
        members = spec["members"]
        C = np.array([m["center"] for m in members], dtype=float)        # (M,3)
        A = np.array([m["semi_axes"] for m in members], dtype=float)     # (M,3)
        R = np.array([m["rotation"] for m in members], dtype=float)      # (M,3,3)
        # Directions in the object frame, then in each member's own frame.
        U = dirs if rotation is None else dirs @ np.asarray(rotation, float)
        local = np.einsum("ki,mij->mkj", U, R)                           # (M,K,3)
        support = np.linalg.norm(A[:, None, :] * local, axis=2)          # (M,K)
        proj = C @ U.T                                                   # (M,K)
        return dirs, (proj + support).max(axis=0) - (proj - support).min(axis=0)

    widths = np.array([object_extent_along(spec, d, rotation)
                       + object_extent_along(spec, -d, rotation) for d in dirs])
    return dirs, widths


# Elongation (widest / narrowest in-plane width) below which an object has no
# meaningful long axis and object_principal_inplane_axis returns its fallback.
# Set above the scan noise of the round YCB objects -- tennis ball 1.02, soccer
# ball 1.03, apple 1.04 -- and below the shapes that genuinely have a long side
# to split along (mug 1.27, lemon 1.31, rubik's cube 1.37).
INPLANE_DEGENERACY_RATIO = 1.15


def object_principal_inplane_axis(spec, rotation, plane_normal, *,
                                  degeneracy_ratio=INPLANE_DEGENERACY_RATIO,
                                  fallback=None):
    """Unit in-plane direction along which the object is longest, as
    ``(e_long, ratio)``.

    §1.8 splits the support surface along "the longest axis of the object that is
    in-plane". This sweeps the object's silhouette on that plane
    (:func:`object_inplane_widths`) and returns the widest direction, with
    ``ratio`` = widest / narrowest width as the measure of how meaningful that
    choice is.

    MEASURED, NOT ASSUMED, because the answer is not the object's frame. This
    used to read the proxy ellipsoid's semi-axes (:func:`proxy_semi_axes`), which
    is an axis-aligned bounding box in the object's OWN frame, so it could only
    ever return one of three local axes and it reported near-1.0 ratios for
    anything lying at an angle to them. Both failures hit the same objects, and
    the fallback below then aliased them all onto one world direction::

        object                    proxy semi-axes        silhouette sweep
        ycb:044_flat_screwdriver   0 deg, 1.03 (!)       136 deg, 6.25
        ycb:043_phillips_screwdr.  0 deg, 1.84            27 deg, 6.29
        ycb:042_adjustable_wrench 90 deg, 1.45           119 deg, 3.48
        ycb:011_banana            90 deg, 1.66            64 deg, 2.85
        pen                        0 deg, 17.50           0 deg, 17.50
        credit_card                0 deg, 1.59            0 deg, 1.59
        sphere, coin, capsule,
        big/mid/small_sphere_ell.  1.00 (degenerate)      1.00 (degenerate)

    (Cylinder and capsule have their LONGEST axis out of plane — their spec
    rotation stands the local Y axis along world Z — so the sweep correctly finds
    them isotropic in-plane.) The one place the two metrics honestly disagree is
    the near-square ``cube`` primitive (half-extents 25/20/25 mm), whose widest
    direction really is its 39-degree diagonal rather than a side; on a shape
    that square the split direction hardly matters. Real boxes are ellipsoid sets
    and still land on their long side (cracker_box 88 deg, sugar_box 83 deg).

    DEGENERACY IS STILL COMMON (every ball, can and bowl), so this returns the
    ratio rather than making the caller guess. Below ``degeneracy_ratio``
    (:data:`INPLANE_DEGENERACY_RATIO`) the argmax is ellipsoid-fit noise -- a
    tennis ball measures 1.05, a soccer ball 1.10 -- so ``fallback`` is returned
    instead. That default is world **+Y**,
    which through ``m_hat = n_hat x e_long`` puts the opposition normal on -X:
    thumb on the -X side of the object, the other fingers on +X, which is the
    side the hand's default mount already reaches from (the thumb sits ~82 mm
    at -X of the object at the phase-0 posture). Choosing +Y over -Y is what
    makes the derived sign and :func:`~.solvers.orient_opposition_axis`'s
    posture-resolved sign agree there instead of fighting.
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    dirs, widths = object_inplane_widths(spec, rotation, n)
    widest = float(widths.max())
    narrowest = float(widths.min())
    ratio = float(widest / narrowest) if narrowest > 1e-12 else np.inf

    if ratio < degeneracy_ratio or widest < 1e-12:
        if fallback is None:
            fallback = np.array([0.0, 1.0, 0.0])
        e = np.asarray(fallback, dtype=float).reshape(3)
        e = e - (e @ n) * n
        ne = np.linalg.norm(e)
        if ne < 1e-9:
            raise ValueError(
                "object_principal_inplane_axis: the object is in-plane isotropic "
                "and the fallback axis is parallel to the plane normal; pass a "
                "fallback that lies in the support plane")
        return e / ne, ratio

    e = dirs[int(np.argmax(widths))]
    return e / np.linalg.norm(e), ratio
