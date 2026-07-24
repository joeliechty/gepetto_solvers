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

import os

import numpy as np


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


# Registry of supported object primitives. "vdb" is the level-set file produced
# by the matching _objects/make_*.py script; the geometry fields must match the
# parameters those scripts were generated with. "plot" describes how the
# TendonFingerPlotter should render the primitive.
def get_primitive_specs():
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
        a = np.asarray(spec["semi_axes"], dtype=float)
        m_diag = 1.0 / (a * a)
        x = np.asarray(p_local, dtype=float)
        Mx = m_diag * x
        g = np.linalg.norm(Mx)
        if g < 1e-9:
            g = 1e-9
        return float((x @ Mx - 1.0) / (2.0 * g))
    raise ValueError(f"Unknown primitive type: {ptype!r}")


def configure_object_surface(env, spec, objects_dir, primitive_name):
    """Attach the object surface to a ``crest_sparse.EnvironmentConfig`` from a
    primitive spec: an analytic hyper-ellipsoid (Section 1.6.3, no VDB) or a
    baked SDF grid. Shared by the contact/collision demo scripts so both surface
    kinds are set up identically; leaves all other env fields untouched."""
    if spec["type"] == "ellipsoid":
        env.ellipsoid_semi_axes = np.asarray(spec["semi_axes"], dtype=float)
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
GRASP_FLEXOR_TENSION = 2.0

# Center of the big grasp sphere: the flexed-fingertip locus at
# GRASP_FLEXOR_TENSION. Used by the five-finger grasp/collision scripts; the
# other (single-finger-scale) primitives stay at OBJECT_CENTER.
GRASP_SPHERE_CENTER = np.array([-0.0221, 0.0885, -0.0160])

# Default outward normal of the Section 1.6 support plane ("table"): world +Z.
# The slide-and-grasp script places the plane tangent to the object's underside
# (origin = object_center - radius * TABLE_NORMAL) so the object rests on it and
# the hand works in the free (+normal) half-space.
TABLE_NORMAL = [0.0, 0.0, 1.0]


def table_plot_spec(plane_origin, plane_normal, *, span=0.3, thickness=0.005):
    """A thin axis-aligned slab primitive for rendering the support plane in the
    trajectory viewer. Returns a ``build_primitive_mesh``-compatible dict
    ({"type": "box", "center", "extents"}). The slab is thin along whichever
    cardinal axis the normal is most aligned with (exact for the default +Z
    table); it is only a visual aid — the solver uses the analytic half-space, not
    this mesh."""
    origin = np.asarray(plane_origin, dtype=float).reshape(3)
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    axis = int(np.argmax(np.abs(n)))       # dominant normal axis
    extents = [span, span, span]
    extents[axis] = thickness
    return {"type": "box", "center": origin, "extents": tuple(extents)}


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
