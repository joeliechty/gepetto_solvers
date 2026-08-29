"""The primitive registry: what objects the scripts can be pointed at.

Baked-SDF primitives (``sphere``, ``big_sphere``, ``cylinder``, ``capsule``,
``cube``) name a ``.vdb`` grid; the analytic ones (``coin``, ``credit_card``,
``pen``, ``*_sphere_ellipsoid``, ``megaminx``) carry their geometry inline and
need no grid at all -- which is what lets the test suite run without them.
"""

import functools
import json
import os

import numpy as np

from .constants import ELLIPSOID_SET_BETA, YCB_FITS_DIR
from .polyhedra import (
    MEGAMINX_FACE_TO_FACE,
    Rx,
    _megaminx_spec,
)


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
