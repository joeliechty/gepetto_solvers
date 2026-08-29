"""Resolving a scene from params: where the object is, where the table sits, and
which way the opposition axis points.

:func:`orient_opposition_axis` is the one function here that is NOT a readout.
The sign of the axis IS the side assignment, and deriving it from the object
alone picks that by a coin flip; when it lands the wrong way up the constraint
asks the thumb and fingers to trade sides and the solve stalls without moving.
Every caller of :func:`default_half_space_axis` must pass the result through it.
"""

import numpy as np

from ..geometry.scene import (
    GRASP_SPHERE_CENTER,
    OBJECT_CENTER,
    get_primitive_specs,
    object_extent_along,
    object_principal_inplane_axis,
)
from ..hand.config import opposition_axis_from_object

# ---------------------------------------------------------------------------
# Scene helpers (shared object placement, mirroring the demo scripts).
# ---------------------------------------------------------------------------

def default_object_center(primitive, spec):
    """Default world center for a primitive, matching the demo scripts: the big
    grasp sphere, capsule and analytic ellipsoids sit at the flexed-fingertip
    locus (``GRASP_SPHERE_CENTER``); the smaller primitives stay at ``OBJECT_CENTER``."""
    if (primitive in ("big_sphere", "capsule")
            or spec["type"] in ("ellipsoid", "ellipsoid_set")):
        return np.array(GRASP_SPHERE_CENTER, dtype=float)
    return np.array(OBJECT_CENTER, dtype=float)


def resolve_scene(params):
    """Resolve (spec, center, rotation, 4x4 pose) for the object from the params,
    filling center/rotation from the primitive when left unset."""
    spec = get_primitive_specs()[params.primitive]
    center = (np.asarray(params.object_center, float)
              if params.object_center is not None
              else default_object_center(params.primitive, spec))
    rotation = (np.asarray(params.object_rotation, float)
                if params.object_rotation is not None
                else np.asarray(spec.get("rotation", np.eye(3)), float))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    return spec, center, rotation, pose


def auto_table_origin(params, spec, object_center):
    """The support-plane origin implied by the scene alone: the object seated on
    ``params.plane_normal`` at a burial fraction of ``params.table_burial``.

    ``table_burial`` is the fraction of the object's FULL along-normal extent
    lying below the plane, so the origin is

        c - (1 - 2 * burial) * half_extent * n_hat

    which is tangent to the underside at 0.0 (the object rests on the table) and
    through the centroid at 0.5 (half-buried). See
    :attr:`HandSolveParams.table_burial` for why half-buried is the default.

    Deliberately ignores ``params.plane_origin``. A GUI offering an ABSOLUTE
    plane height has to seed and re-seat its control from the scene's own answer;
    reading :func:`resolve_table_origin` for that would feed the control's own
    output back into itself, and any offset applied on top would compound on
    every call. Split out so both readings have exactly one definition of the
    seating rule.
    """
    n = np.asarray(params.plane_normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    # getattr: params-like objects predating table_burial keep the old geometry.
    burial = float(getattr(params, "table_burial", 0.0))
    # The object's world orientation, so a rotated object is seated on the
    # profile it actually presents to the plane. Falls back to the primitive's
    # own baked rotation when the caller has not overridden it.
    rotation = (params.object_rotation if params.object_rotation is not None
                else spec.get("rotation"))
    depth = (1.0 - 2.0 * burial) * object_extent_along(spec, n, rotation)
    return np.asarray(object_center, float) - depth * n


def resolve_table_origin(params, spec, object_center):
    """Resolve the TABLE origin: explicit ``params.plane_origin`` if set, else the
    scene's own seating rule (see :func:`auto_table_origin`).

    This is the physical support SURFACE -- the workspace table the robot, its
    URDF and the bench registration are all expressed against. It is what a
    renderer draws and what ``robot_plan`` corners against
    ``lbr_workspace_table_link``, and it is deliberately NOT the plane the factor
    graph constrains against: see :func:`resolve_constraint_plane_origin`.
    """
    if params.plane_origin is not None:
        return np.asarray(params.plane_origin, float)
    return auto_table_origin(params, spec, object_center)


def resolve_constraint_plane_origin(params, spec, object_center):
    """The origin of the plane the SOLVER constrains against: the table plane
    (:func:`resolve_table_origin`) raised by ``params.constraint_plane_height``
    along ``plane_normal``.

    The two are separate because they answer different questions. The table is a
    fixed physical landmark -- move it and the robot registration, the calibration
    grid and the URDF's workspace table all move with it -- whereas the constraint
    plane is a planning choice: where the support equality seats fingertips and
    where the avoidance half-space starts. Height 0 (the default) puts the
    constraint plane exactly on the table, which is the geometry every headless
    script has always solved, so nothing changes unless a caller asks for it.
    """
    n = np.asarray(params.plane_normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    # getattr: params-like objects predating the split keep the coincident planes.
    height = float(getattr(params, "constraint_plane_height", 0.0))
    return np.asarray(resolve_table_origin(params, spec, object_center),
                      float) + height * n


def default_half_space_axis(spec, rotation, plane_normal):
    """The opposition split axis (Eq 2.16-2.17's ``m_hat``) derived from the
    object's own geometry, for when ``HandSolveParams.half_space_axis`` is
    unset: perpendicular, within the support plane, to the object's longest
    in-plane axis (:func:`scene.object_principal_inplane_axis`), via
    :func:`config.opposition_axis_from_object`.

    This is what makes the split LINE run along the object's length (e.g.
    lengthwise along a pen, thumb on one side and fingers on the other across
    its diameter) instead of across it. The length is measured off the object's
    silhouette on the support plane, so it finds a direction that is not one of
    the object's own frame axes -- a YCB screwdriver lying at 27 degrees to its
    export frame gets 27 degrees, not the nearest axis. Falls back to world +Y
    (giving ``m_hat = -X``, thumb on the -X side) only when the object is
    in-plane isotropic (below the degeneracy ratio) --
    :func:`scene.object_principal_inplane_axis`'s own fallback.

    Returns the LINE, not the side assignment: the sign is inherited from the
    object's principal-axis direction, which is an arbitrary convention (which
    end of the sweep the widest direction landed on, or the +Y fallback). The
    fallback's sign is chosen to agree with the hand rather than fight it, but
    the measured one cannot be. Which half the THUMB is asked to
    stay on is a statement about the hand, not the object -- see
    :func:`orient_opposition_axis`, which every caller must apply before
    building the constraint."""
    e_long, _ratio = object_principal_inplane_axis(spec, rotation, plane_normal)
    return opposition_axis_from_object(plane_normal, e_long)


def orient_opposition_axis(axis, thumb_pt, finger_pts, flip=None):
    """``(oriented_axis, flipped)`` -- ``axis`` signed so that ``+m_hat`` points
    from the opposing fingers TOWARD the thumb at the posture given.

    :func:`config.opposition_directions` hands the thumb ``+m_hat`` and every
    other finger ``-m_hat``, so the sign of ``m_hat`` IS the side assignment.
    Deriving it from the object alone (:func:`default_half_space_axis`) picks
    that assignment by a coin flip, and when it lands the wrong way up the
    constraint asks the thumb and the fingers to TRADE sides -- a ~180 degree
    roll of the whole hand about the object. Measured on the phase-0 pen scene
    that is a 32 mm demand on the thumb and 70-75 mm on the fingers, from a
    start pose that already satisfies the constraint in the other orientation:
    the AL stalls at 3 outer iterations with a violation of 1.09e3 and the hand
    never moves. Orienting by the hand instead turns the same scene into a
    solve that runs to a 3e-7 violation.

    ``flip`` overrides the measurement: None (default) picks the nearer
    orientation as described, False keeps the derived sign, True inverts it --
    the way to ask for the opposition the hand is NOT already in, which is a
    genuine (large) repositioning move rather than a mislabeling.
    """
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    if flip is not None:
        return (-axis if flip else axis), bool(flip)
    pts = np.asarray(finger_pts, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return axis, False
    reach = float((np.asarray(thumb_pt, float).reshape(3) - pts.mean(axis=0))
                  @ axis)
    return (-axis, True) if reach < 0.0 else (axis, False)
