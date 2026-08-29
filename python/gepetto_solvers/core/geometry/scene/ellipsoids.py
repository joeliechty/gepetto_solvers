"""Ellipsoid decompositions: which shells exist, which may be touched, and how
they reach the environment config.

``grasp_subset`` narrows CONTACT only. ``attach_ellipsoid_set`` writes the whole
set to ``ellipsoid_set`` and the indices to ``contact_ellipsoid_subset``: the C++
reads the second at the contact sites while every free sphere keeps the first, so
an excluded shell goes on pushing the fingers out while nothing is sent to touch
it. Narrowing the contact target is a planning choice; narrowing the collision
set would be a lie about where the object is.
"""

import os

import numpy as np

from .constants import ELLIPSOID_SET_BETA
from .extents import proxy_semi_axes


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
