"""The support plane: where it sits, and how to draw it.

The slab is a real box with thickness, so the constraint plane is its TOP face --
a viewer that draws the slab centred on the plane origin puts the object half
inside the table.
"""

import numpy as np

from .constants import GRASP_SPHERE_CENTER

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
# Constant regardless of what object is on it: the plane's HEIGHT is seated
# from the object (see solvers.auto_table_origin), its size and its in-plane
# position (TABLE_ANCHOR below) never are.
TABLE_SPAN = 0.4          # m, edge of the square slab


# Where the slab sits WITHIN its own plane. The table is a fixed physical
# landmark -- the bench the robot, the URDF's workspace table and the calibration
# grid are all registered against -- so only its height follows the object, which
# is what makes an object rest ON it. Sliding an object across the table must
# leave the table where it is; seating the in-plane position from the object too
# (as this used to, by taking the whole object centre) drags the slab, its corner
# frame, its grid and the robot registration along with every x/y nudge of the
# object-pose sliders. Only the components perpendicular to the plane normal are
# read, so the height here is ignored. Anchored on the grasp scene's nominal
# object location so the square is centred under the objects posed on it.
TABLE_ANCHOR = np.array(GRASP_SPHERE_CENTER, dtype=float)


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
