"""Shared scene geometry and grasp constants for the tendon-hand scripts.

These object primitives, placement constants, and analytic surface-distance
helpers were previously defined inside the runnable demo scripts (chiefly
``sdf_3dof_contact_kinematics_test.py`` and ``five_finger_hand_grasp_test.py``)
and imported across siblings. They live here so the demos import shared code
instead of importing each other. Nothing in this module is runnable — it holds
only data and pure geometry.

Note: this is distinct from ``core/objects/``, which holds the ``make_*.py``
scripts that *bake* the SDF ``.vdb`` level-set files. The specs here must stay
consistent with the parameters those generators were run with.
"""

import functools
import json
import os

import numpy as np


# Default LogSumExp sharpness for ellipsoid-set objects, mirroring the C++
# EnvironmentConfig::ellipsoid_set_beta default. Distances are in metres, so the
# smooth min understates by up to ln(K)/beta -- 1.4 mm at K=4 here. Kept in the
# spec (not just on the env) because primitive_surface_gap has to reproduce the
# solver's residual, and it can only do that if it uses the same beta.
ELLIPSOID_SET_BETA = 1000.0

# Where ycb/browser.py exports the committed decompositions.
from gepetto_solvers.core.objects import YCB_FITS_DIR as _FITS_DIR
YCB_FITS_DIR = _FITS_DIR


# Object center, shared by all primitives: the p2p goal position used in the
# single-finger planner, mirrored across x=0 (X negated). The 6-tendon routing
# was rotated 180 deg about the finger axis to match the gepetto_core CAD
# convention, which flips the flexor curl from world +X to -X; the object moves
# with it. The SDF lives at the VDB local origin (see the _objects/make_*.py
# generators); we place it in the world by translating the object pose to this center.
OBJECT_CENTER = np.array([-6.02088876e-02, 3.77734425e-02, 0.0])


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


@functools.lru_cache(maxsize=1)
def ycb_primitive_specs():
    """Every committed YCB fit as an ``ellipsoid_set`` primitive, keyed ``ycb:<name>``.

    Reads ``_objects/ycb/fits/*.json`` -- the decompositions exported by
    ``scripts/objects/ycb_browser.py``. Returns ``{}`` when nothing has been fitted,
    so a checkout with an empty ``fits/`` simply has no YCB objects rather than
    failing to import.

    FRAME. The exported centers live in the browser's *display* frame: mesh
    centered in XY, lowest point resting on z=0. Every other primitive in this
    module puts its object-local origin at the object's own middle and lets
    ``object_pose_mean`` place that in the world, so the members are re-centered
    here on the midpoint of their own bounding box. Doing it once, at spec-build
    time, keeps the offset out of the factor path -- the C++ side sees member
    poses that are already correct relative to the object variable.

    ``recenter`` is kept on the spec so a renderer can put the *mesh* in the same
    frame as the shells; without it the two would be drawn a few cm apart.

    ``hull_vertices`` is the scanned mesh's convex hull, re-centered the same way,
    and it is what makes a YCB object sit ON a table rather than above or through
    it. The shells only BOUND the object -- a fit reaches past the real surface by
    16 mm on the potted meat can and 93 mm on the chips can -- so seating the
    plane on the lowest shell floats the object that far up. Seating it on the
    hull instead is exactly what the megaminx does with its dodecahedron
    (:func:`object_extent_along`): the object rests on its own underside and the
    proxy shells sink through the slab, which is the honest picture of a surface
    that was never the object in the first place. Absent for a fit exported
    before the hull was carried, which falls back to the old shell reading.

    ``grasp_subset`` is the authored list of member indices that are grasp
    TARGETS -- see :func:`grasp_subset_indices`. Present only when the fit names
    a proper subset, so its absence means "every shell, nothing to choose".
    """
    directory = os.path.normpath(YCB_FITS_DIR)
    if not os.path.isdir(directory):
        return {}

    specs = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, filename)) as handle:
                blob = json.load(handle)
            name = blob["object"]
            raw = blob["ellipsoids"]
            if not raw:
                continue

            # Bounding box of the union, from each member's own world AABB --
            # the same half-extent formula as ycb.ellipsoids.Ellipsoid.aabb().
            centers = np.array([m["center"] for m in raw], dtype=float)
            radii = np.array([m["radii"] for m in raw], dtype=float)
            rotations = np.array([m["rotation"] for m in raw], dtype=float)
            half = np.sqrt(np.sum((rotations * radii[:, None, :]) ** 2, axis=2))
            lo = (centers - half).min(axis=0)
            hi = (centers + half).max(axis=0)
            recenter = 0.5 * (lo + hi)

            members = [
                {"semi_axes": np.asarray(m["radii"], dtype=float),
                 "center": np.asarray(m["center"], dtype=float) - recenter,
                 "rotation": np.asarray(m["rotation"], dtype=float)}
                for m in raw
            ]
            hull = np.asarray(blob.get("hull", []), dtype=float).reshape(-1, 3)

            # The authored grasp subset, if this fit has one worth offering.
            # Out-of-range indices are dropped (a subset written against an
            # older decomposition), and a subset covering every member is
            # omitted entirely: there is nothing to choose between, and its
            # absence is what tells the caller not to offer the choice.
            subset = blob.get("grasp_subset")
            if subset is not None:
                subset = sorted({int(i) for i in subset if 0 <= int(i) < len(raw)})
                if len(subset) in (0, len(raw)):
                    subset = None

            specs[f"ycb:{name}"] = {
                "type": "ellipsoid_set",
                "ycb": name,
                "source": blob.get("source", ""),
                "beta": ELLIPSOID_SET_BETA,
                "members": members,
                "extents": hi - lo,
                "recenter": recenter,
                **({"grasp_subset": subset} if subset else {}),
                **({"hull_vertices": hull - recenter} if len(hull) else {}),
                "metrics": blob.get("metrics", {}),
                "plot": (lambda c, _m=members: {
                    "type": "ellipsoid_set", "center": c, "members": _m}),
            }
        except Exception:
            # One malformed export must not take the whole object list down --
            # the browser can rewrite it, and every other object still loads.
            continue
    return specs


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


# Registry of supported object primitives. "vdb" is the level-set file produced
# by the matching _objects/make_*.py script; the geometry fields must match the
# parameters those scripts were generated with. "plot" describes how the
# TendonFingerPlotter should render the primitive.
#
# YCB objects (ycb_primitive_specs) are merged in on top under "ycb:<name>" keys:
# real scanned objects approximated by an ellipsoid SET, for the cases a single
# hyper-ellipsoid cannot describe. They are data-driven, so which ones exist
# depends on what has been fitted into _objects/ycb/fits/.
def get_primitive_specs():
    return {**_builtin_primitive_specs(), **ycb_primitive_specs()}


def _builtin_primitive_specs():
    return {
        "sphere": {
            "type": "sphere",
            "vdb": "sphere.vdb",         # make_sphere.py (radius 0.025)
            "radius": 0.025,
            "plot": lambda c: {"type": "sphere", "center": c, "radius": 0.025},
        },
        "big_sphere": {
            # Larger sphere sized+located for the full anatomical-hand grasp: at
            # the flexed-fingertip locus (flexor ~2 N) all five tips land on its
            # surface. See _objects/make_big_sphere.py (radius 0.05). The grasp
            # test places it at its own GRASP_SPHERE_CENTER, not OBJECT_CENTER.
            "type": "sphere",
            "vdb": "big_sphere.vdb",
            "radius": 0.05,
            "plot": lambda c: {"type": "sphere", "center": c, "radius": 0.05},
        },
        "cylinder": {
            "type": "cylinder",
            "vdb": "cylinder.vdb",       # make_cylinder.py (radius 0.025, height 0.04, local Y axis)
            "radius": 0.025,
            "height": 0.04,
            # Rims filleted by this radius in the baked SDF (see make_cylinder.py)
            # so the gradient solver doesn't stick on the cap/side crease.
            "edge_radius": 0.005,
            # Rotate the (local Y-aligned) cylinder 90 deg about X so its axis is
            # vertical (world +Z). The finger moves in the z~0 plane, so it
            # contacts the curved side of this upright cylinder (radius 0.025 from
            # the center axis -- same reach as the sphere, so it's touchable).
            "rotation": Rx(np.pi / 2),
            "plot": lambda c: {"type": "cylinder", "center": c,
                               "radius": 0.025, "height": 0.04,
                               "direction": (0.0, 0.0, 1.0)},
        },
        "capsule": {
            # Capsule = cylinder with hemispherical caps; graspable-sized for
            # the full five-finger grasp (see make_capsule.py, radius 0.04,
            # cylinder length 0.07). Like the cylinder its local axis is Y, so
            # rotate 90 deg about X to stand it up along world +Z: the four
            # fingers wrap the curved side and the thumb opposes.
            "type": "capsule",
            "vdb": "capsule.vdb",        # make_capsule.py (radius 0.04, height 0.07, local Y axis)
            "radius": 0.04,
            "height": 0.07,
            "rotation": Rx(np.pi / 2),
            "plot": lambda c: {"type": "capsule", "center": c,
                               "radius": 0.04, "height": 0.07,
                               "direction": (0.0, 0.0, 1.0)},
        },
        "cube": {
            "type": "cube",
            # half_extents match the cylinder's footprint (radius 0.025 in X/Z,
            # half-height 0.02 in Y) so the finger contacts the flat +Y face the
            # same way it does the cylinder's flat cap.
            "vdb": "cube.vdb",           # make_cube.py (half_extents 0.025, 0.02, 0.025)
            "half_extents": (0.025, 0.02, 0.025),
            # Edges/corners filleted by this radius in the baked SDF (see
            # make_cube.py) so the gradient solver doesn't stick on the creases.
            "edge_radius": 0.005,
            "plot": lambda c: {"type": "cube", "center": c,
                               "extents": (0.05, 0.04, 0.05)},
        },
        # --- Analytic hyper-ellipsoid primitives (Section 1.6.3, Table 1.1) ---
        # These have no baked SDF; they are evaluated by the C++ ellipsoid
        # contact/collision factors. semi_axes = (a, b, c) => shape matrix
        # M = diag(a^-2, b^-2, c^-2). Thin axis is local Z so they lie flat on a
        # +Z table with no rotation. World orientation is carried by object_pose.
        "coin": {
            # Oblate spheroid (r >> h): a = b = r, c = h.
            "type": "ellipsoid",
            "semi_axes": (0.0121, 0.0121, 0.0009),
            "plot": lambda c: {"type": "ellipsoid", "center": c,
                               "semi_axes": (0.0121, 0.0121, 0.0009)},
        },
        "credit_card": {
            # Scalene ellipsoid (l > w >> h): a = l, b = w, c = h.
            "type": "ellipsoid",
            "semi_axes": (0.0428, 0.0270, 0.0004),
            "plot": lambda c: {"type": "ellipsoid", "center": c,
                               "semi_axes": (0.0428, 0.0270, 0.0004)},
        },
        "pen": {
            # Prolate spheroid (l >> r): a = l, b = c = r (long axis is local X).
            "type": "ellipsoid",
            "semi_axes": (0.0700, 0.0040, 0.0040),
            "plot": lambda c: {"type": "ellipsoid", "center": c,
                               "semi_axes": (0.0700, 0.0040, 0.0040)},
        },
        # --- Analytic "spheres" (degenerate hyper-ellipsoids, a = b = c) ---
        # Round objects handled by the C++ ellipsoid factors instead of a baked
        # SDF, so they can be grasped without generating a .vdb. Sized to bracket
        # the two SDF spheres: one matches big_sphere (0.05), one sits just under
        # the small sphere (0.025 -> 0.02), and one splits the difference (0.035).
        "big_sphere_ellipsoid": {
            "type": "ellipsoid",
            "semi_axes": (0.05, 0.05, 0.05),
            "plot": lambda c: {"type": "ellipsoid", "center": c,
                               "semi_axes": (0.05, 0.05, 0.05)},
        },
        "mid_sphere_ellipsoid": {
            "type": "ellipsoid",
            "semi_axes": (0.035, 0.035, 0.035),
            "plot": lambda c: {"type": "ellipsoid", "center": c,
                               "semi_axes": (0.035, 0.035, 0.035)},
        },
        "small_sphere_ellipsoid": {
            "type": "ellipsoid",
            "semi_axes": (0.02, 0.02, 0.02),
            "plot": lambda c: {"type": "ellipsoid", "center": c,
                               "semi_axes": (0.02, 0.02, 0.02)},
        },
        "megaminx": _megaminx_spec(MEGAMINX_FACE_TO_FACE),
    }


def primitive_surface_gap(p_local, spec):
    """Analytic signed distance from a point (in the object's local frame) to
    the primitive surface. Mirrors the SDFs in the _objects/make_*.py scripts so
    we can report the achieved contact gap independently of the solver."""
    ptype = spec["type"]
    if ptype == "sphere":
        return float(np.linalg.norm(p_local) - spec["radius"])
    if ptype == "cylinder":
        # Axis along Y, rims filleted by edge_radius (shrink bounds, offset out).
        er = spec.get("edge_radius", 0.0)
        r = spec["radius"] - er
        half_h = spec["height"] / 2.0 - er
        dist_xz = np.hypot(p_local[0], p_local[2])
        dx = dist_xz - r
        dy = abs(p_local[1]) - half_h
        out_dist = np.hypot(max(dx, 0.0), max(dy, 0.0))
        in_dist = min(max(dx, dy), 0.0)
        return float(out_dist + in_dist - er)
    if ptype == "capsule":
        # Distance to the Y-axis segment [-half_h, half_h] minus the radius.
        r = spec["radius"]
        half_h = spec["height"] / 2.0
        dy = p_local[1] - np.clip(p_local[1], -half_h, half_h)
        dist = np.sqrt(p_local[0] ** 2 + dy ** 2 + p_local[2] ** 2)
        return float(dist - r)
    if ptype == "cube":
        # Edges/corners filleted by edge_radius (shrink bounds, offset out).
        er = spec.get("edge_radius", 0.0)
        hx, hy, hz = spec["half_extents"]
        d = np.abs(p_local) - (np.array([hx, hy, hz]) - er)
        out_dist = np.linalg.norm(np.maximum(d, 0.0))
        in_dist = min(max(d[0], max(d[1], d[2])), 0.0)
        return float(out_dist + in_dist - er)
    if ptype == "ellipsoid":
        # Taubin first-order distance to x^T M x = 1 (Section 1.6.3, Eq 1.91),
        # matching the C++ EllipsoidCollisionGapFactor so the reported gap agrees
        # with what the solver drives to zero. M = diag(a^-2, b^-2, c^-2).
        return _taubin_gap(p_local, spec["semi_axes"])
    if ptype == "ellipsoid_set":
        # LogSumExp smooth min over the members (Section 1.2, Eq 1.11), which is
        # what EllipsoidSetCollisionGapFactor evaluates. It must be the smooth min
        # and not a hard one: the two differ by up to ln(K)/beta (1.4 mm at K=4,
        # beta=1000), and this number is compared against a solver residual that
        # carries exactly that bias.
        beta = float(spec.get("beta", ELLIPSOID_SET_BETA))
        d = np.array([_taubin_gap(_to_member_frame(p_local, m), m["semi_axes"])
                      for m in spec["members"]])
        d_min = d.min()
        # Shift by d_min so no exponent is positive -- same guard as the C++.
        return float(d_min - np.log(np.exp(-beta * (d - d_min)).sum()) / beta)
    raise ValueError(f"Unknown primitive type: {ptype!r}")


def _taubin_gap(x, semi_axes):
    """Taubin's first-order distance from ``x`` to ``sum((x_i/a_i)^2) = 1``."""
    a = np.asarray(semi_axes, dtype=float)
    m_diag = 1.0 / (a * a)
    x = np.asarray(x, dtype=float)
    Mx = m_diag * x
    g = np.linalg.norm(Mx)
    if g < 1e-9:
        g = 1e-9
    return float((x @ Mx - 1.0) / (2.0 * g))


def _to_member_frame(p_local, member):
    """Object-local point into one set member's own frame: ``R_k^T (p - t_k)``."""
    return np.asarray(member["rotation"], float).T @ (
        np.asarray(p_local, float) - np.asarray(member["center"], float))


def _primitive_surface_gradient(p_local, spec, h):
    grad = np.empty(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = h
        grad[i] = (primitive_surface_gap(p_local + step, spec)
                   - primitive_surface_gap(p_local - step, spec)) / (2.0 * h)
    return grad


def _primitive_surface_normal(p_local, spec, h=1e-6):
    """Outward unit normal at ``p_local``: the central-difference gradient of
    ``primitive_surface_gap``, which is a unit SDF gradient for every primitive here.

    The gradient vanishes on the medial axis (e.g. the exact center of the box, where
    opposite faces tie), so retry just off it before giving up: an equal nudge on all
    three axes breaks the symmetry and picks the genuinely nearest face, which keeps
    the projected foot point on the surface."""
    grad = _primitive_surface_gradient(p_local, spec, h)
    if np.linalg.norm(grad) < 1e-9:
        grad = _primitive_surface_gradient(p_local + h, spec, h)
    norm = np.linalg.norm(grad)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return grad / norm


def _ellipsoid_closest_point(x, semi_axes):
    """Exact closest point on the ellipsoid ``sum((x_i/a_i)^2) = 1`` to ``x``.

    Stationarity gives ``foot_i = a_i^2 x_i / (t + a_i^2)`` for the Lagrange
    multiplier ``t``, which is the unique root of the decreasing function
    ``f(t) = sum((a_i x_i / (t + a_i^2))^2) - 1`` on ``t > -min(a_i^2)``. Bisected
    rather than Newton-solved: unconditionally convergent, and 5 points per frame
    makes the cost irrelevant.

    Exact to machine precision except for a point *inside* the ellipsoid that lies
    exactly on a principal plane (some ``x_i == 0.0``), where the closest point is a
    tie broken by the epsilon below: the foot can then be off by up to ~0.4 mm on the
    flattest primitive here (``credit_card``). That needs a fingertip buried inside
    the object at an exact coordinate zero, so it does not arise in practice."""
    a2 = np.asarray(semi_axes, dtype=float) ** 2
    x = np.asarray(x, dtype=float).reshape(3)

    if np.linalg.norm(x) < 1e-12:
        # Dead center: every direction ties, so pick the nearest surface point --
        # the pole of the shortest semi-axis.
        i = int(np.argmin(a2))
        foot = np.zeros(3)
        foot[i] = np.sqrt(a2[i])
        return foot

    # Nudge exact zeros so f(t) -> +inf at the bracket's left edge (the degenerate
    # "point on a principal plane" case); 1e-12 m is far below any display scale.
    # Used for the foot point too, so a point *on* an axis still projects to the
    # correct pole instead of collapsing to the origin.
    x = np.where(np.abs(x) < 1e-12, 1e-12, x)

    def f(t):
        return np.sum(a2 * x * x / (t + a2) ** 2) - 1.0

    lo = -a2.min() + 1e-15
    hi = lo + 1.0
    while f(hi) > 0.0:
        hi = lo + 2.0 * (hi - lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return a2 * x / (0.5 * (lo + hi) + a2)


def primitive_surface_witness(p_local, spec, *, h=1e-6):
    """``(signed distance, closest surface point, outward unit normal there)`` for a
    point in the object's local frame -- ``primitive_surface_gap`` plus the *where*,
    so a viewer can draw the gap as a line that lands on the surface.

    Sphere / cylinder / capsule / cube: ``primitive_surface_gap`` is an exact SDF, so
    the foot point is one step along its gradient.

    Ellipsoid: ``primitive_surface_gap`` returns Taubin's *first-order* distance
    (matching the C++ ``EllipsoidCollisionGapFactor`` residual), which is accurate at
    contact but increasingly pessimistic further out -- a true 15 mm gap from the
    ``coin`` reads ~8 mm. For a distance the user reads off the screen in mm that is
    too wrong, so this solves for the exact closest point instead. The two agree to
    <0.1 mm near contact, where the solver actually operates, and diverge only in the
    far field. Reporting/rendering only: the solver's own witness points come from
    the C++ contact factors."""
    x = np.asarray(p_local, dtype=float).reshape(3)

    if spec["type"] == "ellipsoid":
        a = np.asarray(spec["semi_axes"], dtype=float)
        foot = _ellipsoid_closest_point(x, a)
        n = foot / (a * a)
        n = n / (np.linalg.norm(n) or 1.0)
        sign = 1.0 if np.sum((x / a) ** 2) > 1.0 else -1.0
        return float(sign * np.linalg.norm(x - foot)), foot, n

    if spec["type"] == "ellipsoid_set":
        # Nearest point on the nearest MEMBER, each solved exactly and mapped back
        # into the object frame. Deliberately a hard min, unlike
        # primitive_surface_gap's smooth one: this answers "where on the object is
        # the fingertip closest to", and a blended witness would sit off every
        # actual surface -- there is no point in between two ellipsoids to draw a
        # gap line to. The reported LENGTH still comes from this exact solve, so
        # near a seam it can differ from the solver's smooth-min residual by up to
        # ln(K)/beta; that is the standoff the smooth min buys, not an error.
        best = None
        for member in spec["members"]:
            a = np.asarray(member["semi_axes"], dtype=float)
            R = np.asarray(member["rotation"], dtype=float)
            t = np.asarray(member["center"], dtype=float)
            x_k = R.T @ (x - t)
            foot_k = _ellipsoid_closest_point(x_k, a)
            sign = 1.0 if np.sum((x_k / a) ** 2) > 1.0 else -1.0
            d_k = float(sign * np.linalg.norm(x_k - foot_k))
            if best is None or d_k < best[0]:
                n_k = foot_k / (a * a)
                n_k = n_k / (np.linalg.norm(n_k) or 1.0)
                best = (d_k, R @ foot_k + t, R @ n_k)
        return best

    d = primitive_surface_gap(x, spec)
    n = _primitive_surface_normal(x, spec, h)
    return float(d), x - d * n, n


def grasp_subset_indices(spec, use_subset=True):
    """Which members of ``spec`` a fingertip may be sent to -- ``[i, ...]``, or
    None meaning "all of them, no narrowing".

    THE one definition of that question. The env writes these into
    ``contact_ellipsoid_subset``, the gap reporting measures against the same
    members, and the renderer greys out the rest -- three readings that have to
    agree, or the picture and the number describe a grasp the solver never
    planned.

    None (rather than ``range(len(members))``) whenever there is no narrowing to
    do: the caller was not asking for the subset, the object has no authored one,
    or it is not an ellipsoid set at all. That distinction is load-bearing
    downstream -- an empty ``contact_ellipsoid_subset`` is what makes an env build
    the pre-existing graph, and it is what lets the visualizer tell "this object
    offers no choice" from "the choice is all shells".
    """
    if not use_subset or spec.get("type") != "ellipsoid_set":
        return None
    subset = spec.get("grasp_subset")
    if not subset:
        return None
    return [int(i) for i in subset]


def subset_spec(spec, indices):
    """``spec`` narrowed to ``indices``, for the PYTHON-side surface readings.

    ``primitive_surface_gap`` and the planar overlay walk ``members`` rather than
    taking a list of indices, so they need the narrowed spec; the C++ env takes
    the indices themselves. Same narrowing, two shapes -- which is why both come
    off :func:`grasp_subset_indices` instead of being derived independently.

    A shallow copy: only ``members`` is rebuilt, so ``extents``, ``recenter`` and
    ``hull_vertices`` keep describing the WHOLE object. They should -- narrowing
    the contact target does not move the object, shrink its bounding box, or lift
    it off the table.
    """
    if indices is None:
        return spec
    members = spec["members"]
    narrowed = dict(spec)
    narrowed["members"] = [members[i] for i in indices]
    return narrowed


def ellipsoid_members(spec):
    """The object as a list of analytic ellipsoids -- ``[(semi_axes, R, center), ...]``
    in the OBJECT frame -- or None for a primitive that has no such form.

    One entry for an ``ellipsoid`` or a ``sphere`` (both with an identity member
    pose), one per member for an ``ellipsoid_set``. None for ``cube`` /
    ``cylinder`` / ``capsule``: they have no closed-form ellipsoid cross-section,
    and the C++ planar factor takes an ellipsoid set, so there is nothing to hand
    it.

    A ``sphere`` is answered analytically whether or not it also carries a baked
    ``vdb`` grid -- the same approximation :meth:`HandResult.contact_witness`
    already makes, measuring the analytic look-alike rather than the .vdb, and it
    differs only inside the grid's edge fillets.

    This is the ellipsoid-set view of an object, which is exactly what
    :func:`attach_ellipsoid_set` writes into the env -- kept next to it so the two
    cannot disagree about what a spec's members are.
    """
    if spec["type"] == "ellipsoid":
        return [(np.asarray(spec["semi_axes"], dtype=float), np.eye(3), np.zeros(3))]
    if spec["type"] == "sphere":
        r = float(spec["radius"])
        return [(np.array([r, r, r]), np.eye(3), np.zeros(3))]
    if spec["type"] == "ellipsoid_set":
        return [(np.asarray(m["semi_axes"], dtype=float),
                 np.asarray(m["rotation"], dtype=float),
                 np.asarray(m["center"], dtype=float))
                for m in spec["members"]]
    return None


def plane_ellipse_section(semi_axes, rotation, center, plane_point, plane_normal,
                          *, num=96):
    """Sample the ellipse where a plane cuts one ellipsoid, or None if it misses.

    Everything is in ONE frame (the caller's -- the visualizer passes world, having
    already composed the object pose with the member's local pose): ``rotation`` and
    ``center`` place the ellipsoid, ``plane_point``/``plane_normal`` the plane.
    Returns an ``(num, 3)`` array of points on the intersection curve, closed
    (last point repeats the first) so it draws as a loop.

    This is the picture of ``G_planar`` in Eq 13 -- the 2D cross-section the in-plane
    distance is measured against. It is EXACT, unlike the factor's Taubin distance to
    it, which is the whole reason to draw it: the outline says where the cross-section
    really is, the factor's number says what the solver would think.

    Method: write points in the plane as ``p = c0 + u e1 + v e2`` and substitute into
    ``q^T M q = 1``. That gives a 2D conic ``[u v] Q [u v]^T + 2 [D E] [u v]^T + F``
    with ``Q`` positive definite (M is), so the section is an ellipse, an empty set,
    or a point. Completing the square gives its centre; eigen-decomposing ``Q`` gives
    its axes.
    """
    a = np.asarray(semi_axes, dtype=float).reshape(3)
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    c = np.asarray(center, dtype=float).reshape(3)
    q0 = np.asarray(plane_point, dtype=float).reshape(3) - c
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) or 1.0)

    M = R @ np.diag(1.0 / (a * a)) @ R.T

    # Any orthonormal in-plane basis will do -- the curve is the same set of points
    # whichever one is chosen; only the sampling phase changes.
    e1 = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    Q = np.array([[e1 @ M @ e1, e1 @ M @ e2],
                  [e2 @ M @ e1, e2 @ M @ e2]])
    b = np.array([e1 @ M @ q0, e2 @ M @ q0])
    F = float(q0 @ M @ q0) - 1.0

    p0 = -np.linalg.solve(Q, b)          # section centre, in (u, v)
    F0 = F + float(b @ p0)               # value at that centre
    if F0 >= 0.0:
        return None                      # the plane misses this ellipsoid

    evals, evecs = np.linalg.eigh(Q)
    if np.any(evals <= 0.0):
        return None                      # degenerate; nothing sensible to draw
    radii = np.sqrt(-F0 / evals)

    t = np.linspace(0.0, 2.0 * np.pi, num)
    uv = p0 + (evecs @ np.stack([radii[0] * np.cos(t), radii[1] * np.sin(t)])).T
    return c + q0 + uv[:, :1] * e1 + uv[:, 1:] * e2


def configure_object_surface(env, spec, objects_dir, primitive_name,
                             contact_subset=None):
    """Attach the object surface to a ``gepetto_solvers.EnvironmentConfig`` from a
    primitive spec: an analytic hyper-ellipsoid (Section 1.6.3, no VDB) or a
    baked SDF grid. Shared by the contact/collision demo scripts so both surface
    kinds are set up identically; leaves all other env fields untouched.

    ``contact_subset`` (:func:`grasp_subset_indices`) narrows which members of an
    ``ellipsoid_set`` the CONTACT equality may target. None = no narrowing, and
    it is inert for every other surface kind: a single ellipsoid or a baked SDF
    has no members to choose between."""
    if spec["type"] == "ellipsoid":
        env.ellipsoid_semi_axes = np.asarray(spec["semi_axes"], dtype=float)
        return
    if spec["type"] == "ellipsoid_set":
        attach_ellipsoid_set(env, spec, contact_subset=contact_subset)
        return
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python scripts/objects/make_{primitive_name}.py")
    env.load_sdf(vdb_path)


def attach_ellipsoid_set(env, spec, contact_subset=None):
    """Write an ``ellipsoid_set`` spec onto a ``gepetto_solvers.EnvironmentConfig``.

    Each member becomes an ``EllipsoidPrimitive`` whose ``local_pose`` is its
    constant pose in the OBJECT frame; the C++ side composes that with the one
    optimized object pose, so the set adds no variables of its own.

    ``contact_subset`` narrows the members the CONTACT equality may target, and
    ONLY those: the whole set is written either way, so the Eq 12 collision
    inequality still sees every shell. That asymmetry is the point of the
    feature -- the excluded members are the ones bounding the object rather than
    offering a handle, and they have to keep pushing the fingers out even while
    nothing is being sent to touch them.

    Raises on a binding that predates ``ellipsoid_set`` rather than degrading
    quietly. The usual ``_set_if`` treatment is wrong here: skipping this field
    does not lose a tuning knob, it leaves the env with NO object surface at all,
    and the solve then runs with the contact constraint silently missing. Callers
    that need to stay up on an old binding should gate on
    ``solvers.capabilities()["ellipsoid_set"]`` and not offer the object.

    A requested subset on a binding without ``contact_ellipsoid_subset`` raises
    for the same reason: the solve would run against every shell while the caller
    believed it had narrowed the target. Passing None asks for no narrowing, so
    it stays silent on an old binding -- that is the pre-existing behavior.
    """
    import gepetto_solvers

    if not hasattr(env, "ellipsoid_set"):
        raise AttributeError(
            "this gepetto_solvers build has no EnvironmentConfig.ellipsoid_set, so "
            f"the ellipsoid-set object {spec.get('ycb', '?')!r} cannot be built -- "
            "rebuild it (pip install . from the crest-sparse root)")
    if contact_subset and not hasattr(env, "contact_ellipsoid_subset"):
        raise AttributeError(
            "this gepetto_solvers build has no EnvironmentConfig."
            "contact_ellipsoid_subset, so the grasp subset for "
            f"{spec.get('ycb', '?')!r} cannot be applied -- rebuild it "
            "(pip install . from the crest-sparse root), or gate on "
            'solvers.capabilities()["grasp_subset"] and contact every shell')

    members = []
    for m in spec["members"]:
        primitive = gepetto_solvers.EllipsoidPrimitive()
        primitive.semi_axes = np.asarray(m["semi_axes"], dtype=float)
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(m["rotation"], dtype=float)
        pose[:3, 3] = np.asarray(m["center"], dtype=float)
        primitive.local_pose = pose
        members.append(primitive)
    env.ellipsoid_set = members
    env.ellipsoid_set_beta = float(spec.get("beta", ELLIPSOID_SET_BETA))
    if contact_subset:
        env.contact_ellipsoid_subset = [int(i) for i in contact_subset]


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


def configure_object_proxy_and_exact(env, spec, objects_dir, primitive_name,
                                     contact_subset=None):
    """Section 1.8 controller variant of :func:`configure_object_surface`: attach
    BOTH the bounding-ellipsoid proxy (phase 2's sliding target) and, when the
    primitive has one, the baked SDF of the exact geometry (phase 3's servoing
    target).

    The controller's C++ ``phase_env`` switches between them: phases 1-2 use the
    ellipsoid, and phase 3 zeroes ``ellipsoid_semi_axes`` so the factors fall
    through to ``sdf_grid``. An analytic-only primitive (coin, card, pen) has no
    exact geometry to switch to, so it keeps the ellipsoid in phase 3 and only
    the witness form changes — which is correct, since for those the ellipsoid
    *is* the object.

    ``contact_subset`` is accepted and forwarded so this stays a drop-in for
    :func:`configure_object_surface` (``config.attach_contact`` picks between the
    two and calls whichever it chose with one signature). It narrows only the
    set; the PROXY is a bound on the whole object and is never narrowed -- an
    approach that slid along a proxy shrunk to the grip would clip the parts of
    the object it is meant to steer around.
    """
    env.ellipsoid_semi_axes = proxy_semi_axes(spec)
    if spec["type"] == "ellipsoid":
        return
    if spec["type"] == "ellipsoid_set":
        # The set IS the exact geometry here -- there is no baked SDF to servo on
        # in phase 3 -- so attach it alongside the proxy. Note the C++ precedence
        # (set beats ellipsoid_semi_axes) means the set wins wherever both are
        # set, so the proxy above only takes effect in a phase that clears it.
        attach_ellipsoid_set(env, spec, contact_subset=contact_subset)
        return
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python scripts/objects/make_{primitive_name}.py")
    env.load_sdf(vdb_path)


# --- Full five-finger grasp scene (the "big_sphere" grasp target) ---

# Flexor tension (N) at which the anatomical fingertips land exactly on the big
# grasp sphere with an identity wrist. The big sphere was sized/placed for this
# flexion; the static and trajectory grasp scripts share it so results compare.
GRASP_FLEXOR_TENSION = 0.6

# Center of the big grasp sphere: the flexed-fingertip locus at
# GRASP_FLEXOR_TENSION. Used by the five-finger grasp/collision scripts; the
# other (single-finger-scale) primitives stay at OBJECT_CENTER.
GRASP_SPHERE_CENTER = np.array([-0.0221, 0.0885, -0.0160])

# Default outward normal of the Section 1.6 support plane ("table"): world +Z.
# The slide-and-grasp script places the plane tangent to the object's underside
# (origin = object_center - radius * TABLE_NORMAL) so the object rests on it and
# the hand works in the free (+normal) half-space.
TABLE_NORMAL = [0.0, 0.0, 1.0]

# Drawn size of that support plane, as ONE definition shared by every renderer
# (the trajectory viewer's table_plot_spec below and the interactive app's viser
# slab). A named constant rather than a per-function default because the slab is
# used as a physical LANDMARK when setting up real robot experiments -- its edge
# length is a number to be measured against, so it has to be stated once, be the
# same everywhere, and be reportable to the user.
#
# Constant regardless of what object is on it: the plane's POSITION is seated
# from the object (see solvers.auto_table_origin), its size never is.
TABLE_SPAN = 0.4          # m, edge of the square slab
TABLE_THICKNESS = 0.005   # m, thickness along the plane normal


def _table_plane_axis(plane_normal):
    """The cardinal axis the slab is thin along: the one the normal is most
    aligned with (exact for the default +Z table). Shared by every function here
    so the drawn box, its corner and its offset cannot disagree about which axis
    is 'up'."""
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    return int(np.argmax(np.abs(n))), n


def table_slab_center(plane_origin, plane_normal, *, thickness=TABLE_THICKNESS):
    """Center of the drawn slab for a plane through ``plane_origin``.

    Offset half a thickness to the FAR side of the plane, so the slab's TOP FACE
    is the constraint plane itself -- the surface objects are seated tangent to
    (:func:`solvers.auto_table_origin`) and the surface a real table would be
    measured from. Centering the box on the plane instead, as this used to,
    draws every seated object 2.5 mm sunk into the table and puts the visible
    surface half a thickness above where the solver's half-space actually is.
    """
    axis, n = _table_plane_axis(plane_normal)
    center = np.asarray(plane_origin, dtype=float).reshape(3).copy()
    center[axis] -= np.sign(n[axis]) * thickness / 2.0
    return center


def table_corner(plane_origin, plane_normal, *, span=TABLE_SPAN):
    """World position of the slab's minimum corner: the square's corner at the
    least coordinate along both in-plane cardinal axes (-X/-Y for the default +Z
    table), lying ON the plane -- i.e. on the top face.

    This is the scene's physical landmark. Real-robot setup needs a common point
    that both the model and the bench can be measured from, and a table corner is
    the one feature of this scene that exists in both. Derived from the same
    dominant-axis rule as the drawn box, so the frame drawn here cannot drift
    from the geometry it is a corner of.
    """
    axis, _n = _table_plane_axis(plane_normal)
    corner = np.asarray(plane_origin, dtype=float).reshape(3).copy()
    for i in range(3):
        if i != axis:
            corner[i] -= span / 2.0
    return corner


def table_plot_spec(plane_origin, plane_normal, *, span=TABLE_SPAN,
                    thickness=TABLE_THICKNESS):
    """A thin axis-aligned slab primitive for rendering the support plane in the
    trajectory viewer. Returns a ``build_primitive_mesh``-compatible dict
    ({"type": "box", "center", "extents"}). The slab is thin along whichever
    cardinal axis the normal is most aligned with (exact for the default +Z
    table) and hangs below the plane so its top face IS the plane (see
    :func:`table_slab_center`); it is only a visual aid — the solver uses the
    analytic half-space, not this mesh."""
    axis, _n = _table_plane_axis(plane_normal)
    extents = [span, span, span]
    extents[axis] = thickness
    return {"type": "box",
            "center": table_slab_center(plane_origin, plane_normal,
                                        thickness=thickness),
            "extents": tuple(extents)}


# Anatomical 6-tendon finger routing (index 5 = flexor), config order.
TENDON_NAMES = ["Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor"]

# Per-finger world-frame tip-position goals (order = config order: index, middle,
# ring, pinky, thumb). These are the *collision-free* terminal fingertip positions
# from the collision+contact grasp solve on the big sphere: the hand wraps the
# sphere with every backbone node held outside it, so unlike a free-space flexor
# curl (whose main fingers spear straight through the sphere) these points ARE
# reachable with the whole finger collision-free.
GRASP_GOALS = np.array([
    [+0.01058010, +0.10938996, +0.02336805],  # index
    [+0.01125694, +0.12307751, +0.01202090],  # middle
    [+0.01993645, +0.12410185, -0.01172549],  # ring
    [+0.02291420, +0.11488003, -0.03186456],  # pinky
    [+0.01562034, +0.08011573, +0.02589826],  # thumb
])
