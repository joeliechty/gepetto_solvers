"""Shared scene geometry and grasp constants for the tendon-hand scripts.

These object primitives, placement constants, and analytic surface-distance
helpers were previously defined inside the runnable demo scripts (chiefly
``sdf_3dof_contact_kinematics_test.py`` and ``five_finger_hand_grasp_test.py``)
and imported across siblings. They live here so the demos import shared code
instead of importing each other. Nothing in this module is runnable — it holds
only data and pure geometry.

Note: this is distinct from ``tests/_objects/``, which holds the ``make_*.py``
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
YCB_FITS_DIR = os.path.join(os.path.dirname(__file__), "..", "_objects", "ycb", "fits")


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


@functools.lru_cache(maxsize=1)
def ycb_primitive_specs():
    """Every committed YCB fit as an ``ellipsoid_set`` primitive, keyed ``ycb:<name>``.

    Reads ``_objects/ycb/fits/*.json`` -- the decompositions exported by
    ``tests._objects.ycb.browser``. Returns ``{}`` when nothing has been fitted,
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
            specs[f"ycb:{name}"] = {
                "type": "ellipsoid_set",
                "ycb": name,
                "source": blob.get("source", ""),
                "beta": ELLIPSOID_SET_BETA,
                "members": members,
                "extents": hi - lo,
                "recenter": recenter,
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


def configure_object_surface(env, spec, objects_dir, primitive_name):
    """Attach the object surface to a ``crest_sparse.EnvironmentConfig`` from a
    primitive spec: an analytic hyper-ellipsoid (Section 1.6.3, no VDB) or a
    baked SDF grid. Shared by the contact/collision demo scripts so both surface
    kinds are set up identically; leaves all other env fields untouched."""
    if spec["type"] == "ellipsoid":
        env.ellipsoid_semi_axes = np.asarray(spec["semi_axes"], dtype=float)
        return
    if spec["type"] == "ellipsoid_set":
        attach_ellipsoid_set(env, spec)
        return
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{primitive_name} (run from the python/ dir).")
    env.load_sdf(vdb_path)


def attach_ellipsoid_set(env, spec):
    """Write an ``ellipsoid_set`` spec onto a ``crest_sparse.EnvironmentConfig``.

    Each member becomes an ``EllipsoidPrimitive`` whose ``local_pose`` is its
    constant pose in the OBJECT frame; the C++ side composes that with the one
    optimized object pose, so the set adds no variables of its own.

    Raises on a binding that predates ``ellipsoid_set`` rather than degrading
    quietly. The usual ``_set_if`` treatment is wrong here: skipping this field
    does not lose a tuning knob, it leaves the env with NO object surface at all,
    and the solve then runs with the contact constraint silently missing. Callers
    that need to stay up on an old binding should gate on
    ``solvers.capabilities()["ellipsoid_set"]`` and not offer the object.
    """
    import crest_sparse

    if not hasattr(env, "ellipsoid_set"):
        raise AttributeError(
            "this crest_sparse build has no EnvironmentConfig.ellipsoid_set, so "
            f"the ellipsoid-set object {spec.get('ycb', '?')!r} cannot be built -- "
            "rebuild it (pip install . from the crest-sparse root)")

    members = []
    for m in spec["members"]:
        primitive = crest_sparse.EllipsoidPrimitive()
        primitive.semi_axes = np.asarray(m["semi_axes"], dtype=float)
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(m["rotation"], dtype=float)
        pose[:3, 3] = np.asarray(m["center"], dtype=float)
        primitive.local_pose = pose
        members.append(primitive)
    env.ellipsoid_set = members
    env.ellipsoid_set_beta = float(spec.get("beta", ELLIPSOID_SET_BETA))


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


def object_principal_inplane_axis(spec, rotation, plane_normal, *,
                                  degeneracy_ratio=1.05, fallback=None):
    """Unit in-plane direction along which the object is longest, as
    ``(e_long, ratio)``.

    §1.8 splits the support surface along "the longest axis of the object that is
    in-plane". This takes the proxy ellipsoid's semi-axes
    (:func:`proxy_semi_axes`), maps each principal axis into the world with
    ``rotation``, projects out ``plane_normal``, and returns the direction with
    the largest surviving extent.

    DEGENERACY IS THE COMMON CASE, so this returns the ratio rather than making
    the caller guess. ``ratio`` is longest / second-longest in-plane extent;
    measured over the demo primitives with ``n_hat = +Z``::

        sphere, big_sphere, cylinder, capsule, coin,
        big/mid/small_sphere_ellipsoid ....... 1.00   (degenerate)
        cube ................................. 1.25
        credit_card .......................... 1.59
        pen .................................. 17.50

    (Cylinder and capsule have their LONGEST axis out of plane — their spec
    rotation stands the local Y axis along world Z — so projecting out ``n_hat``
    correctly leaves them isotropic in-plane.) Below ``degeneracy_ratio`` the
    argmax is numerical noise, so ``fallback`` is returned instead; it defaults to
    world +X projected into the plane, matching
    :func:`~.config.opposition_directions`' legacy default axis.
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    semi = proxy_semi_axes(spec)

    # Each principal axis, in the world, scaled by its semi-axis, with the
    # out-of-plane part removed: what is left is that axis's reach in the plane.
    inplane = [R[:, k] * float(semi[k]) for k in range(3)]
    inplane = [v - (v @ n) * n for v in inplane]
    extents = np.array([np.linalg.norm(v) for v in inplane])
    order = np.argsort(extents)[::-1]
    longest, second = extents[order[0]], extents[order[1]]
    ratio = float(longest / second) if second > 1e-12 else np.inf

    if ratio < degeneracy_ratio or longest < 1e-12:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0])
        e = np.asarray(fallback, dtype=float).reshape(3)
        e = e - (e @ n) * n
        ne = np.linalg.norm(e)
        if ne < 1e-9:
            raise ValueError(
                "object_principal_inplane_axis: the object is in-plane isotropic "
                "and the fallback axis is parallel to the plane normal; pass a "
                "fallback that lies in the support plane")
        return e / ne, ratio

    e = inplane[order[0]]
    return e / np.linalg.norm(e), ratio


def configure_object_proxy_and_exact(env, spec, objects_dir, primitive_name):
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
    """
    env.ellipsoid_semi_axes = proxy_semi_axes(spec)
    if spec["type"] == "ellipsoid":
        return
    if spec["type"] == "ellipsoid_set":
        # The set IS the exact geometry here -- there is no baked SDF to servo on
        # in phase 3 -- so attach it alongside the proxy. Note the C++ precedence
        # (set beats ellipsoid_semi_axes) means the set wins wherever both are
        # set, so the proxy above only takes effect in a phase that clears it.
        attach_ellipsoid_set(env, spec)
        return
    vdb_path = os.path.normpath(os.path.join(objects_dir, spec["vdb"]))
    if not os.path.exists(vdb_path):
        raise FileNotFoundError(
            f"{vdb_path} not found. Generate it with "
            f"python -m tests._objects.make_{primitive_name} (run from the python/ dir).")
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
