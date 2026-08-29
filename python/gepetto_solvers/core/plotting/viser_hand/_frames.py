"""Coordinate-frame triads: world, object, table, calibration, mount.

A mixin of :class:`~gepetto_solvers.core.plotting.viser_hand.scene.ViserHandScene`.
Split out of what was one 968-line class; the methods here use only ``self.scene``
and ``self._dynamic``, which the composed class owns.
"""

import numpy as np

from ._geometry import _wxyz_from_R
from .palette import (
    _MOUNT_RGB,
)


class FrameSceneMixin:
    def set_world_frame(self, show=True, *, axes_length=0.03):
        """Show or hide the world-origin triad.

        Toggled by re-adding the node with ``show_axes`` rather than removing
        it: ``/world`` is the scene's root frame and pins the up-direction set
        in ``__init__``, so it has to keep existing even when its axes are not
        drawn.
        """
        self.scene.add_frame("/world", show_axes=bool(show),
                             axes_length=axes_length, axes_radius=0.0008)


    def set_object_frame(self, center, rotation=None, *, axes_length=0.04):
        """Draw the object's own frame: the pose every contact and collision
        factor is written against.

        Worth seeing on its own because the object's ORIENTATION is otherwise
        invisible on a rotationally symmetric primitive, and for an ellipsoid
        set it is the frame the members are placed in -- so it is what the
        Object-pose rotation sliders actually drive. Lives at ``/object_frame``,
        a sibling of ``/object`` rather than a child, so :meth:`clear_object`
        (which prunes the whole ``/object`` subtree) does not take it down with it.
        """
        R = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.scene.add_frame("/object_frame", show_axes=True,
                             axes_length=axes_length,
                             axes_radius=axes_length * 0.03,
                             wxyz=_wxyz_from_R(R),
                             position=tuple(np.asarray(center, float).reshape(3)))


    def clear_object_frame(self):
        try:
            self.server.scene.remove_by_name("/object_frame")
        except Exception:
            pass


    def set_table_frame(self, corner, *, axes_length=0.06, label=None):
        """Draw the table's landmark frame: a PURE TRANSLATION of the world frame
        to ``corner`` -- same axis directions, no rotation, so a coordinate read
        off this frame differs from a world coordinate by an offset alone.

        That is the point of it. Setting up a real experiment means measuring the
        bench, and a table corner is the one feature this scene and the physical
        rig share; with the axes parallel to world, going between the two is a
        subtraction rather than a change of basis. Drawn longer than the object
        (0.04) and mount (0.05) frames because it is the scene-scale reference.

        ``label`` is optional billboard text (the app passes the slab's
        dimensions, so the numbers are legible from inside the 3D view).
        """
        self.scene.add_frame("/table_frame", show_axes=True,
                             axes_length=axes_length,
                             axes_radius=axes_length * 0.025,
                             wxyz=(1.0, 0.0, 0.0, 0.0),   # world-aligned
                             position=tuple(np.asarray(corner, float).reshape(3)))
        try:
            self.server.scene.remove_by_name("/table_frame_label")
        except Exception:
            pass
        if label:
            self.scene.add_label("/table_frame_label", label,
                                 position=tuple(np.asarray(corner, float).reshape(3)))


    def clear_table_frame(self):
        for name in ("/table_frame", "/table_frame_label"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass


    def set_calibration_frame(self, T, *, axes_length=0.05, label=None):
        """Draw the calibration target: where a chosen hand landmark is being
        asked to go.

        Unlike :meth:`set_table_frame` this carries a full ORIENTATION, because
        the thing it is aligned against is a disc's body frame and not just a
        point -- so the triad has to show which way the landmark will be facing
        when it lands, not only where.

        ``label`` is optional billboard text; the app passes the target's
        table-frame coordinates, which is what you compare against the grid
        printed on the bench.
        """
        T = np.asarray(T, float)
        self.scene.add_frame("/calibration_frame", show_axes=True,
                             axes_length=axes_length,
                             axes_radius=axes_length * 0.03,
                             wxyz=_wxyz_from_R(T[:3, :3]),
                             position=tuple(T[:3, 3]))
        try:
            self.server.scene.remove_by_name("/calibration_frame_label")
        except Exception:
            pass
        if label:
            self.scene.add_label("/calibration_frame_label", label,
                                 position=tuple(T[:3, 3]))


    def clear_calibration_frame(self):
        for name in ("/calibration_frame", "/calibration_frame_label"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass


    def set_mount_frames(self, T_world_wrist, T_flange_wrist, *, axes_length=0.05):
        """Draw the wrist frame and the robot flange frame it hangs off.

        ``T_flange_wrist`` is the measured mount (``mount.measured_mount_pose()``),
        so the flange sits at ``T_world_wrist @ inv(T_flange_wrist)``. Both frames
        are drawn with axes plus a line between them, which is what makes a wrong
        mount obvious: the flange should land where the metal bracket actually
        bolts on, with its axes matching the CAD assembly's origin triad.
        """
        T_world_wrist = np.asarray(T_world_wrist, float)
        T_world_flange = T_world_wrist @ np.linalg.inv(
            np.asarray(T_flange_wrist, float))
        for name, T, length in (("/mount/wrist", T_world_wrist, axes_length * 0.7),
                                ("/mount/flange", T_world_flange, axes_length)):
            self.scene.add_frame(name, show_axes=True, axes_length=length,
                                 axes_radius=length * 0.03,
                                 wxyz=_wxyz_from_R(T[:3, :3]),
                                 position=tuple(T[:3, 3]))
        self.scene.add_line_segments(
            "/mount/link",
            points=np.array([[T_world_flange[:3, 3], T_world_wrist[:3, 3]]]),
            colors=np.array([[_MOUNT_RGB, _MOUNT_RGB]], dtype=np.uint8),
            line_width=2.0)
        self.scene.add_label("/mount/flange_label", "flange (robot mount)",
                             position=tuple(T_world_flange[:3, 3]))


    def clear_mount_frames(self):
        for name in ("/mount/wrist", "/mount/flange", "/mount/link",
                     "/mount/flange_label"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass
