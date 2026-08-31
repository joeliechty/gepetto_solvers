"""Drawing the grasp object: its analytic shell and its scanned mesh.

A mixin of :class:`~gepetto_solvers.core.plotting.viser_hand.scene.ViserHandScene`.
Split out of what was one 968-line class; the methods here use only ``self.scene``
and ``self._dynamic``, which the composed class owns.
"""

import numpy as np
import trimesh

from ._geometry import _object_trimesh, _wxyz_from_R
from .palette import (
    _OBJECT_EXCLUDED_OPACITY,
    _OBJECT_EXCLUDED_RGB,
    _OBJECT_MESH_RGB,
    _OBJECT_RGB,
)


class ObjectSceneMixin:
    # -- static scene: object + table --------------------------------------

    def set_object(self, spec, center, rotation=None, contact_subset=None):
        """(Re)build the grasp object mesh. Sphere/cube/ellipsoid/ellipsoid_set use
        native viser primitives (translucent); cylinder/capsule fall back to a
        trimesh.

        ``contact_subset`` (``scene.grasp_subset_indices``; None = all) marks
        which members of an ``ellipsoid_set`` the fingertips may be driven onto.
        The rest are still drawn -- greyed, not hidden -- because they are still
        collision geometry: a shell removed from the picture would say the hand
        can pass through it, when what it actually cannot do is grab it.

        ``rotation`` is the object's world orientation (3x3). It matters for any
        primitive that is not rotationally symmetric -- an ellipsoid drawn without
        it appears axis-aligned no matter how the solver has it posed, and for an
        ellipsoid SET it would scatter the members to the wrong places entirely.

        Every shape is drawn as a CHILD of ``/object``, never on ``/object``
        itself, so that node stays a pure frame at the identity. viser keeps a
        node's transform in its own persistent message, separate from the
        geometry: removing a node drops the geometry but the position/orientation
        messages survive in the broadcast buffer, and adding a child later
        resurrects the parent as an implicit frame that those stale messages then
        apply to. Drawing a single-shape object on ``/object`` therefore poisoned
        the frame for the next ellipsoid SET -- its members are positioned in
        world coordinates, so they all came back offset by the previous object's
        centre. Keeping ``/object`` transform-free is what makes it safe to reuse
        the name across object kinds.
        """
        center = np.asarray(center, float)
        R = np.eye(3) if rotation is None else np.asarray(rotation, float)
        wxyz = tuple(_wxyz_from_R(R))
        t = spec["type"]
        name = "/object"
        shell = f"{name}/shell"          # single-shape objects; see above
        self.clear_object()
        if t == "sphere":
            self.scene.add_icosphere(shell, radius=float(spec["radius"]),
                                     color=_OBJECT_RGB, opacity=0.35,
                                     position=tuple(center))
        elif t == "ellipsoid":
            a, b, c = (float(v) for v in spec["semi_axes"])
            self.scene.add_icosphere(shell, radius=1.0, scale=(a, b, c),
                                     color=_OBJECT_RGB, opacity=0.35,
                                     position=tuple(center), wxyz=wxyz)
        elif t == "ellipsoid_set":
            # One shell per member (Section 1.2). Each member's pose is constant in
            # the OBJECT frame, so the world placement is the object pose composed
            # with it -- exactly what EllipsoidSetCollisionGapFactor evaluates, so
            # what is drawn is the geometry the graph sees.
            targets = None if contact_subset is None else set(contact_subset)
            for index, member in enumerate(spec["members"]):
                a, b, c = (float(v) for v in member["semi_axes"])
                R_member = R @ np.asarray(member["rotation"], float)
                pos = center + R @ np.asarray(member["center"], float)
                excluded = targets is not None and index not in targets
                self.scene.add_icosphere(
                    f"{name}/e{index}", radius=1.0, scale=(a, b, c),
                    color=_OBJECT_EXCLUDED_RGB if excluded else _OBJECT_RGB,
                    opacity=_OBJECT_EXCLUDED_OPACITY if excluded else 0.35,
                    position=tuple(pos), wxyz=tuple(_wxyz_from_R(R_member)))
        elif t == "cube":
            hx, hy, hz = spec["half_extents"]
            self.scene.add_box(shell, color=_OBJECT_RGB,
                               dimensions=(2 * hx, 2 * hy, 2 * hz),
                               opacity=0.35, position=tuple(center))
        else:
            mesh = _object_trimesh(spec, center)
            if mesh is not None:
                self.scene.add_mesh_trimesh(shell, mesh)


    def clear_object(self):
        """Drop the object geometry.

        Removing ``/object`` takes its whole subtree with it, which is what makes
        switching object KINDS safe: a K=7 ellipsoid set followed by a K=4 one
        would otherwise leave the last three shells floating, and the single-shape
        ``/object/shell`` would survive underneath a set."""
        try:
            self.server.scene.remove_by_name("/object")
        except Exception:
            pass


    def set_object_mesh(self, mesh, center, rotation=None, *, opacity=0.55):
        """Overlay the object's real scanned mesh (a trimesh), posed like the
        analytic geometry.

        For a YCB object the shells are an APPROXIMATION of this, so showing both
        is how the approximation gets judged: where the hand stops is set by the
        shells, and the mesh says how much object is really there. Same role for
        the megaminx's dodecahedron inside its circumsphere (see
        :meth:`hull_mesh`). Pass ``None`` to clear.
        """
        self.clear_object_mesh()
        if mesh is None:
            return
        posed = mesh.copy()
        transform = np.eye(4)
        transform[:3, :3] = np.eye(3) if rotation is None else np.asarray(rotation, float)
        transform[:3, 3] = np.asarray(center, float)
        posed.apply_transform(transform)
        posed.visual = trimesh.visual.ColorVisuals(
            posed, vertex_colors=np.array(
                [*_OBJECT_MESH_RGB, int(255 * opacity)], dtype=np.uint8))
        self.scene.add_mesh_trimesh("/object_mesh", posed)


    @staticmethod
    def hull_mesh(vertices):
        """Convex hull of ``vertices`` (object-local, m) as a trimesh, for a spec
        that carries its real solid as a point set -- the megaminx's 20
        dodecahedron corners. Feed the result to :meth:`set_object_mesh`, which
        poses it exactly like the analytic shell it sits inside."""
        return trimesh.Trimesh(vertices=np.asarray(vertices, float)).convex_hull


    def clear_object_mesh(self):
        try:
            self.server.scene.remove_by_name("/object_mesh")
        except Exception:
            pass
