"""viser scene renderer for the tendon hand -- the web-viewer analogue of the
PyVista ``TendonHandPlotter``.

Pure rendering: given a viser server and one solved hand *frame* (the
``{finger_name: solution}`` shim the solvers produce), it draws each finger's
Cosserat backbone, its tendons, and -- optionally -- the routing discs, the
per-fingertip contact spheres, the disc-node collision spheres, the grasp
object, the support-plane "table", and the per-finger pinch planes. No solving
and no GUI live here; the interactive app (``tendon_hand/viz_interactive.py``)
owns those.

The world-frame geometry reproduces exactly what the PyVista mesh managers
compute (``_plotting/tendon_hand_plotter.py``):

* backbone  -- node translations ``rod.states[n].pose.mean[:3, 3]``.
* tendons   -- ``R @ hole + t`` for each active hole, with the disc pose
  ``rod.states[tendon_config.disc_pose_idx[disc]]``.
* discs / disc frames / collision spheres -- one per disc node
  (``disc_pose_idx``); the frame is the body frame its hole locations are in.

Nodes are addressed by stable scene-tree names; re-adding a name upserts it, and
handles for a frame's dynamic geometry are tracked so toggled-off / vanished
geometry is removed on the next update.
"""

import numpy as np
import trimesh


# Tendon colours (RGB 0-255), cycled per tendon; loosely matches the PyVista
# TENDON_COLORS palette.
_TENDON_RGB = [
    (220, 20, 60), (34, 139, 34), (65, 105, 225), (186, 85, 211),
    (218, 165, 32), (255, 20, 147),
]
_ROD_RGB = (40, 90, 200)
_OBJECT_RGB = (218, 165, 32)
# The real scanned mesh behind an ellipsoid-set approximation. Deliberately a
# different hue from _OBJECT_RGB: the two are drawn together so the fit can be
# judged, and in one colour the shells would be indistinguishable from the mesh
# they are approximating.
_OBJECT_MESH_RGB = (120, 150, 190)
_CONTACT_RGB = (80, 200, 120)
_COLLISION_RGB = (230, 120, 60)
_DISC_RGB = (100, 149, 237)
_TABLE_RGB = (150, 150, 160)
_HALF_SPACE_RGB = (255, 140, 0)
_CENTER_TARGET_RGB = (180, 60, 220)
_MOUNT_RGB = (240, 240, 240)
# Per-finger pinch-plane patches, in finger_names order (index, middle, ring,
# pinky, thumb). One colour per finger rather than one for the overlay: the five
# planes all pass through the same pinch point, so a single colour would draw
# five sheets fanned about one line with no way to tell whose is whose.
_FINGER_PLANE_RGB = [
    (231, 76, 60), (241, 196, 15), (46, 204, 113), (52, 152, 219),
    (155, 89, 182),
]

# Fingertip-to-object gap overlay: green within GAP_GREEN_MAX_M of the surface
# (including interpenetration, which is simply "not far"), red beyond it.
GAP_GREEN_MAX_M = 0.015
_GAP_NEAR_RGB = (0, 190, 60)
_GAP_FAR_RGB = (220, 40, 40)
# Opposition half-space margin overlay: green when the constraint is
# SATISFIED (margin >= 0, i.e. the finger is on its designated side), red when
# violated -- a sign-based rule rather than GAP_GREEN_MAX_M's magnitude-based
# one, since what matters here is which side of the plane, not how far.
_MARGIN_OK_RGB = _GAP_NEAR_RGB
_MARGIN_VIOLATED_RGB = _GAP_FAR_RGB
# Pre-grasp short-axis alignment overlay: green within ANGLE_GREEN_MAX_DEG of
# the target axis (either direction), red beyond it.
ANGLE_GREEN_MAX_DEG = 10.0


def _wxyz_from_R(R):
    """viser's (w, x, y, z) quaternion from a 3x3 rotation.

    Shepperd's method: pick the largest of the four diagonal combinations so the
    square root is never taken of something near zero. The naive w-first formula
    loses all precision at 180 deg rotations, and the mount transform is exactly
    that kind of pose.
    """
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def _recenter(mesh):
    """Translate a trimesh primitive so its bounding box is centered on the origin
    (trimesh's capsule/cylinder are not all origin-centered)."""
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def _object_trimesh(spec, center):
    """Best-effort trimesh for a grasp object, in final world orientation --
    mirrors ``ik_5f_contact._add_object_mesh`` (the spec's rotation is already
    baked into how each primitive is drawn, so it is not re-applied here)."""
    t = spec["type"]
    if t == "cylinder":
        mesh = trimesh.creation.cylinder(radius=spec["radius"], height=spec["height"])
    elif t == "capsule":
        mesh = trimesh.creation.capsule(height=spec["height"], radius=spec["radius"])
    else:
        return None
    _recenter(mesh)
    mesh.apply_translation(np.asarray(center, float))
    mesh.visual.vertex_colors = np.array([*_OBJECT_RGB, 120], dtype=np.uint8)
    return mesh


class ViserHandScene:
    """Renders/refreshes the tendon hand in a viser scene.

    Parameters mirror the PyVista plotter's spirit: construct once with the finger
    names, then call :meth:`set_object` on object change and :meth:`update` per
    frame. Display toggles (discs / contact spheres / collision spheres) are read
    on each :meth:`update` from the attributes set here.
    """

    def __init__(self, server, finger_names, *, backbone_width=4.0,
                 show_discs=False, show_disc_frames=False,
                 show_contact_spheres=True,
                 show_collision_spheres=True, show_gap_lines=True,
                 show_finger_planes=False, show_planar_gap=False):
        self.server = server
        self.scene = server.scene
        self.finger_names = list(finger_names)
        self.backbone_width = backbone_width
        self.show_discs = show_discs
        self.show_disc_frames = show_disc_frames
        self.show_contact_spheres = show_contact_spheres
        self.show_collision_spheres = show_collision_spheres
        self.show_gap_lines = show_gap_lines
        # Off by default: five translucent sheets through the middle of the
        # grasp hide the fingertips and the object surface behind them, so this
        # is something you switch on to answer a question, not scene furniture.
        self.show_finger_planes = show_finger_planes
        self.show_planar_gap = show_planar_gap

        # name -> handle, for dynamic (per-frame) geometry so we can prune it.
        self._dynamic = {}

        self.scene.add_frame("/world", show_axes=True, axes_length=0.03,
                             axes_radius=0.0008)
        # The hand is built +Z-up, fingers extending +Y and curling toward -X
        # (palmar). The PyVista demos view it with +Z up from azimuth 165 (the -X
        # side), so the curl comes toward the viewer and reads as a grasp. viser's
        # default is +Y-up looking from the opposite side, where that same correct
        # curl looks like the fingers bending *backwards* -- so pin the world up.
        self.scene.set_up_direction("+z")

    @staticmethod
    def grasp_camera(focal, *, azimuth_deg=165.0, elevation_deg=20.0,
                     distance=0.28):
        """(position, look_at) for a camera orbiting ``focal`` on the -X (palmar)
        side, using the same azimuth/elevation/+Z-up spherical convention as the
        PyVista TendonHandPlotter. Viewing from here the finger curl reads as a
        grasp closing toward you rather than hyperextending away."""
        f = np.asarray(focal, float).reshape(3)
        az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        offset = distance * np.array([np.cos(el) * np.cos(az),
                                      np.cos(el) * np.sin(az),
                                      np.sin(el)])
        return f + offset, f

    # -- static scene: object + table --------------------------------------

    def set_object(self, spec, center, rotation=None):
        """(Re)build the grasp object mesh. Sphere/cube/ellipsoid/ellipsoid_set use
        native viser primitives (translucent); cylinder/capsule fall back to a
        trimesh.

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
            for index, member in enumerate(spec["members"]):
                a, b, c = (float(v) for v in member["semi_axes"])
                R_member = R @ np.asarray(member["rotation"], float)
                pos = center + R @ np.asarray(member["center"], float)
                self.scene.add_icosphere(
                    f"{name}/e{index}", radius=1.0, scale=(a, b, c),
                    color=_OBJECT_RGB, opacity=0.35,
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

    def set_table(self, origin, normal, *, span, thickness):
        """Draw the support-plane slab (visual aid; the solver uses the analytic
        half-space). Thin along the dominant normal axis, and hung half a
        thickness BELOW ``origin`` so its top face is the plane itself -- matching
        ``scene.table_plot_spec`` / ``scene.table_slab_center``.

        ``span``/``thickness`` are required rather than defaulted: the slab is a
        measured landmark for real-robot setup, and a second set of dimensions
        living here would be free to drift from ``scene.TABLE_SPAN``, which is
        the one definition callers are supposed to pass.
        """
        origin = np.asarray(origin, float).reshape(3)
        n = np.asarray(normal, float).reshape(3)
        axis = int(np.argmax(np.abs(n)))
        extents = [span, span, span]
        extents[axis] = thickness
        center = origin.copy()
        center[axis] -= np.sign(n[axis]) * thickness / 2.0
        self.scene.add_box("/table", color=_TABLE_RGB, dimensions=tuple(extents),
                           opacity=0.4, position=tuple(center))

    def clear_table(self):
        try:
            self.server.scene.remove_by_name("/table")
        except Exception:
            pass

    def set_half_space_plane(self, split_point, axis, *, margin=0.0, span=0.25,
                             thickness=0.003):
        """Draw the Eq 2.16-2.17 opposition split plane -- a thin translucent
        slab through ``split_point``, thin along ``axis`` (the in-plane
        direction separating the thumb's half from the other fingers'; NOT the
        table normal -- this plane stands roughly vertical, cutting across the
        table). Visual aid only, mirroring :meth:`set_table`; the solver uses
        the analytic half-space directly.

        ``margin`` is the minimum standoff (m) the constraint now demands of
        each side (``solvers.HandSolveParams.half_space_margin``). Nonzero, the
        split itself is no longer the boundary anyone is held to, so the two
        planes that ARE -- ``split_point +- margin * axis``, the thumb's and the
        opposing fingers' -- are drawn alongside it, fainter. The corridor
        between them is the region the constraint now keeps empty."""
        origin = np.asarray(split_point, float).reshape(3)
        a = np.asarray(axis, float).reshape(3)
        a = a / (np.linalg.norm(a) or 1.0)
        ax = int(np.argmax(np.abs(a)))
        extents = [span, span, span]
        extents[ax] = thickness
        self.scene.add_box("/half_space_plane", color=_HALF_SPACE_RGB,
                           dimensions=tuple(extents), opacity=0.25,
                           position=tuple(origin))
        for name, side in (("/half_space_margin_pos", +1.0),
                           ("/half_space_margin_neg", -1.0)):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass
            if margin > 0.0:
                self.scene.add_box(name, color=_HALF_SPACE_RGB,
                                   dimensions=tuple(extents), opacity=0.15,
                                   position=tuple(origin + side * margin * a))

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

    def clear_half_space_plane(self):
        for name in ("/half_space_plane", "/half_space_margin_pos",
                     "/half_space_margin_neg"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass

    # -- per-frame hand ----------------------------------------------------

    def update(self, frame, *, tip_radii=None, collision_radius=0.003,
               collision=False, gaps=None, table_gaps=None,
               half_space_gaps=None, center_gap=None, axis_align=None,
               centroid_gap=None, finger_planes=None, planar_gaps=None):
        """Refresh the hand geometry for one frame. ``frame`` maps finger name to
        an object exposing ``.marginals`` (a ``TendonFingerMarginals``).

        ``gaps`` is the optional fingertip-to-object overlay: a
        ``{finger: (sphere_pt, surface_pt, gap_m)}`` map as returned by
        ``HandResult.contact_witness``. ``table_gaps`` is the same shape measured
        against the support plane (``solvers.plane_witness``) and drawn alongside,
        so a solve touching both surfaces shows both distances.

        ``half_space_gaps`` is the Eq 2.16-2.17 opposition overlay: a
        ``{finger: (sphere_pt, foot_pt, signed_margin_m)}`` map (as returned by
        ``solvers.half_space_witness``) drawn per finger like the gap overlays
        above, but colored by SIGN (green = on the correct side, red =
        violating) rather than by distance.

        ``center_gap`` is the Eq 2.18-2.19 pre-grasp centering overlay: a single
        ``(hand_centroid_pt, target_pt, gap_m)`` tuple (as returned by
        ``solvers.pregrasp_center_witness``) or None -- a HAND-level quantity,
        not per finger, so it is drawn once rather than per finger name.

        ``centroid_gap`` is the pre-grasp PINCH-CENTROID overlay: a single
        ``(pinch_pt, target_pt, gap_m)`` tuple (as returned by
        ``solvers.pregrasp_centroid_witness``) or None. Same shape and same
        HAND-level treatment as ``center_gap``, and drawn under its own scene
        path so both can be shown at once -- but ``pinch_pt`` is the measured
        hand-frame meeting point carried through the wrist pose, not the
        fingertips' achieved midpoint, so the two lines genuinely differ until
        the hand is closed.

        ``axis_align`` is the pre-grasp short-axis alignment overlay: a single
        ``(c_thumb, c_others_mean, angle_deg)`` tuple (as returned by
        ``solvers.pregrasp_axis_witness``) or None -- also HAND-level, drawn
        under its own scene path so it coexists with ``center_gap`` when both
        are on (the two lines connect the same two points but mean different
        things -- distance to a target vs. angle off an axis -- so they are
        never merged into one draw call).

        All the above are gated on ``self.show_gap_lines`` (the existing
        "contact distance" display toggle) -- one category of overlay, one
        switch.

        ``finger_planes`` is the per-finger PINCH-PLANE overlay: a
        ``{finger: (base_pt, tip_pt, pinch_pt)}`` map (as returned by
        ``solvers.finger_plane_witness``) or None. It gets its own switch
        (``self.show_finger_planes``) rather than riding on the gap toggle,
        because it is opaque geometry rather than a thin measurement line -- it
        occludes the very contact it is drawn around.

        ``planar_gaps`` is the IN-PLANE distance overlay that goes with it: a
        ``{finger: solvers.PlanarGap}`` map (from ``solvers.planar_gap_witness``)
        or None, drawn under ``self.show_planar_gap``. Separate switch again,
        and separate from the plane patches too -- the cross-sections are worth
        looking at without five translucent sheets over them.

        Rendering only -- nothing here feeds the solver."""
        keep = set()
        tip_radii = tip_radii or [None] * len(self.finger_names)
        plane_rgb = dict(zip(self.finger_names, _FINGER_PLANE_RGB * len(self.finger_names)))

        for name, radius in zip(self.finger_names, tip_radii):
            if name not in frame:
                continue
            fm = frame[name].marginals
            states = fm.rod.states
            poses = [np.asarray(st.pose.mean) for st in states]
            positions = np.array([T[:3, 3] for T in poses])

            # Backbone.
            n = f"/hand/{name}/rod"
            self._dynamic[n] = self.scene.add_spline_catmull_rom(
                n, positions, curve_type="catmullrom",
                line_width=self.backbone_width, color=_ROD_RGB)
            keep.add(n)

            # Tendons.
            keep |= self._update_tendons(name, fm, poses)

            # Contact sphere at the fingertip node.
            if self.show_contact_spheres and radius:
                cn = f"/hand/{name}/contact"
                self._dynamic[cn] = self.scene.add_icosphere(
                    cn, radius=float(radius), color=_CONTACT_RGB, opacity=0.3,
                    position=tuple(positions[-1]))
                keep.add(cn)

            # Fingertip -> object-surface gap: a coloured line with the distance
            # in mm labelled at its midpoint. viser labels carry no colour, so the
            # near/far cue lives on the line.
            if self.show_gap_lines and gaps and name in gaps:
                keep |= self._update_gap(name, *gaps[name])
            if self.show_gap_lines and table_gaps and name in table_gaps:
                keep |= self._update_gap(name, *table_gaps[name],
                                         kind="table_gap")
            if self.show_gap_lines and half_space_gaps and name in half_space_gaps:
                keep |= self._update_half_space(name, *half_space_gaps[name])

            # Collision spheres on the disc nodes.
            if collision and self.show_collision_spheres:
                for di, node_idx in enumerate(fm.tendon_config.disc_pose_idx):
                    kn = f"/hand/{name}/collision/{di}"
                    self._dynamic[kn] = self.scene.add_icosphere(
                        kn, radius=float(collision_radius), color=_COLLISION_RGB,
                        opacity=0.25, position=tuple(poses[node_idx][:3, 3]))
                    keep.add(kn)

            # Routing discs.
            if self.show_discs:
                keep |= self._update_discs(name, fm, poses)

            # The disc nodes' own frames -- what the hole locations, and every
            # other per-disc quantity, are expressed in.
            if self.show_disc_frames:
                keep |= self._update_disc_frames(name, fm, poses)

            # Pinch plane through base / tip / pinch centroid.
            if self.show_finger_planes and finger_planes and name in finger_planes:
                keep |= self._update_finger_plane(name, *finger_planes[name],
                                                  rgb=plane_rgb[name])

            # What the object looks like inside that plane, and how far away it is.
            if self.show_planar_gap and planar_gaps and name in planar_gaps:
                keep |= self._update_planar_gap(name, planar_gaps[name],
                                                rgb=plane_rgb[name])

        # Pre-grasp centering (Eq 2.18-2.19): a HAND-level overlay, drawn once
        # rather than per finger.
        if self.show_gap_lines and center_gap is not None:
            keep |= self._update_center(*center_gap)

        # Pre-grasp short-axis alignment: also HAND-level, drawn once.
        if self.show_gap_lines and axis_align is not None:
            keep |= self._update_axis_align(*axis_align)

        # Pre-grasp pinch-centroid centering: also HAND-level, drawn once.
        if self.show_gap_lines and centroid_gap is not None:
            keep |= self._update_centroid(*centroid_gap)

        # The point the finger planes fan about -- one marker for all of them.
        if self.show_finger_planes and finger_planes:
            keep |= self._update_pinch_point(next(iter(finger_planes.values()))[2])

        self._prune(keep)

    def _update_tendons(self, name, fm, poses):
        keep = set()
        tc = fm.tendon_config
        for ti in range(tc.num_tendons):
            pts = []
            for di in range(tc.num_discs):
                hole = tc.hole_locations[di][ti]
                if hole is None:
                    break
                T = poses[tc.disc_pose_idx[di]]
                pts.append(T[:3, :3] @ np.asarray(hole) + T[:3, 3])
            if len(pts) < 2:
                continue
            pts = np.asarray(pts)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (S, 2, 3)
            n = f"/hand/{name}/tendon/{ti}"
            self._dynamic[n] = self.scene.add_line_segments(
                n, segs, colors=_TENDON_RGB[ti % len(_TENDON_RGB)], line_width=3.0)
            keep.add(n)
        return keep

    def _update_gap(self, name, sphere_pt, surface_pt, gap, kind="gap"):
        """One fingertip-to-surface line + labelled distance. ``kind`` namespaces
        the scene handles, so an object gap and a table gap on the SAME finger are
        two overlays rather than one overwriting the other."""
        p0 = np.asarray(sphere_pt, float).reshape(3)
        p1 = np.asarray(surface_pt, float).reshape(3)
        rgb = _GAP_NEAR_RGB if gap < GAP_GREEN_MAX_M else _GAP_FAR_RGB

        ln = f"/hand/{name}/{kind}/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=3.0)

        lb = f"/hand/{name}/{kind}/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{gap * 1000.0:.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        return {ln, lb}

    def _update_half_space(self, name, sphere_pt, foot_pt, margin):
        """One fingertip-to-split-plane line + labelled signed margin (Eq
        2.16-2.17). Colored by SIGN, not distance: green while ``margin >= 0``
        (the finger is on its designated side of the opposition plane), red
        once it crosses to the wrong side. The label keeps the sign so a
        violation reads as a negative number rather than looking like a small
        satisfied gap."""
        p0 = np.asarray(sphere_pt, float).reshape(3)
        p1 = np.asarray(foot_pt, float).reshape(3)
        rgb = _MARGIN_OK_RGB if margin >= 0.0 else _MARGIN_VIOLATED_RGB

        ln = f"/hand/{name}/half_space/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=3.0)

        lb = f"/hand/{name}/half_space/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{margin * 1000.0:+.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        return {ln, lb}

    def _update_center(self, hand_pt, target_pt, gap):
        """The pre-grasp hand-centering overlay (Eq 2.18-2.19): a marker at the
        target point (object centroid + clearance), a marker at the achieved
        hand-centroid midpoint, and a labelled line between them. Colored by
        distance like the fingertip gap lines (near = converged)."""
        p0 = np.asarray(hand_pt, float).reshape(3)
        p1 = np.asarray(target_pt, float).reshape(3)
        rgb = _GAP_NEAR_RGB if gap < GAP_GREEN_MAX_M else _GAP_FAR_RGB
        keep = set()

        tgt = "/pregrasp_center/target"
        self._dynamic[tgt] = self.scene.add_icosphere(
            tgt, radius=0.004, color=_CENTER_TARGET_RGB, opacity=0.8,
            position=tuple(p1))
        keep.add(tgt)

        mid = "/pregrasp_center/midpoint"
        self._dynamic[mid] = self.scene.add_icosphere(
            mid, radius=0.004, color=rgb, opacity=0.8, position=tuple(p0))
        keep.add(mid)

        ln = "/pregrasp_center/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=3.0)
        keep.add(ln)

        lb = "/pregrasp_center/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{gap * 1000.0:.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        keep.add(lb)
        return keep

    def _update_centroid(self, pinch_pt, target_pt, gap):
        """The pre-grasp pinch-centroid overlay: a marker at the target (object
        centroid + clearance), a marker at where the checked digits WOULD meet
        given the current wrist pose, and a labelled line between them.

        Visually a twin of ``_update_center``, deliberately under its own
        ``/pregrasp_centroid/...`` path so both can be on at once. Reading them
        together is the point: the centering line ends at the fingertips'
        achieved midpoint, this one at the hand's measured pinch point, and the
        two converge only as the fingers close."""
        p0 = np.asarray(pinch_pt, float).reshape(3)
        p1 = np.asarray(target_pt, float).reshape(3)
        rgb = _GAP_NEAR_RGB if gap < GAP_GREEN_MAX_M else _GAP_FAR_RGB
        keep = set()

        tgt = "/pregrasp_centroid/target"
        self._dynamic[tgt] = self.scene.add_icosphere(
            tgt, radius=0.004, color=_CENTER_TARGET_RGB, opacity=0.8,
            position=tuple(p1))
        keep.add(tgt)

        pin = "/pregrasp_centroid/pinch"
        self._dynamic[pin] = self.scene.add_icosphere(
            pin, radius=0.005, color=rgb, opacity=0.55, position=tuple(p0))
        keep.add(pin)

        ln = "/pregrasp_centroid/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=3.0)
        keep.add(ln)

        lb = "/pregrasp_centroid/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{gap * 1000.0:.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        keep.add(lb)
        return keep

    def _update_axis_align(self, c_thumb, c_others, angle_deg):
        """The pre-grasp short-axis alignment overlay: a labelled line between
        the thumb's and the opposing fingers' contact centroids, colored by
        the angle off the target axis (near 0 deg = converged; the target
        itself is direction-agnostic, so unlike ``_update_center`` there is no
        single target POINT to mark, only the achieved angle). Own scene path
        (``/pregrasp_align/...``) so it coexists with the centering overlay,
        which can be drawn between the same two points but means something
        different (distance to a point, not an angle)."""
        p0 = np.asarray(c_thumb, float).reshape(3)
        p1 = np.asarray(c_others, float).reshape(3)
        rgb = _GAP_NEAR_RGB if angle_deg < ANGLE_GREEN_MAX_DEG else _GAP_FAR_RGB
        keep = set()

        ln = "/pregrasp_align/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=3.0)
        keep.add(ln)

        lb = "/pregrasp_align/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{angle_deg:.1f} deg", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        keep.add(lb)
        return keep

    def _update_finger_plane(self, name, base_pt, tip_pt, pinch_pt, *, rgb,
                             margin=0.25, min_pad=0.01):
        """One finger's pinch plane: the plane through its metacarpal base, its
        fingertip and the pinch centroid, drawn as a translucent quad with the
        defining triangle outlined on top of it and the finger's name at the
        triangle's middle.

        A plane is unbounded, so what is drawn is a choice: the smallest
        axis-aligned rectangle *in the plane's own basis* that contains all
        three defining points, padded by ``margin`` of its own size (at least
        ``min_pad`` metres, so a nearly-degenerate triangle still shows as a
        patch rather than a sliver). Sizing it off the points themselves keeps
        the sheet around the grasp instead of across the whole scene, and makes
        it grow with the finger as it reaches.

        The outline is what makes the patch readable -- the quad's own edges are
        arbitrary, the triangle's are the actual inputs, so the triangle is what
        you check when the plane looks wrong.

        Returns an EMPTY set when the three points are collinear (within 0.1 mm
        of the base-tip line): no plane exists there, and drawing the quad from
        a near-zero second basis vector would show one at an arbitrary
        orientation. Returning nothing lets ``_prune`` take last frame's patch
        down, so the plane disappears rather than lying."""
        p = np.stack([np.asarray(q, float).reshape(3)
                      for q in (base_pt, tip_pt, pinch_pt)])

        e1 = p[1] - p[0]
        len1 = np.linalg.norm(e1)
        if len1 < 1e-9:
            return set()
        e1 = e1 / len1
        w = p[2] - p[0]
        e2 = w - (w @ e1) * e1          # component of the centroid off the finger axis
        off = np.linalg.norm(e2)
        if off < 1e-4:
            return set()
        e2 = e2 / off

        # The three points in the plane's own (u, v) coordinates, origin at the base.
        uv = np.stack([(p - p[0]) @ e1, (p - p[0]) @ e2], axis=1)
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        pad = np.maximum(margin * (hi - lo), min_pad)
        lo, hi = lo - pad, hi + pad
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                            [hi[0], hi[1]], [lo[0], hi[1]]])
        verts = p[0] + corners[:, :1] * e1 + corners[:, 1:] * e2
        faces = np.array([[0, 1, 2], [0, 2, 3]])

        keep = set()
        pn = f"/hand/{name}/pinch_plane/patch"
        # Two-sided: a one-sided patch vanishes as the camera orbits past it,
        # which for a plane reads as the overlay having switched itself off.
        self._dynamic[pn] = self.scene.add_mesh_simple(
            pn, verts, faces, color=rgb, opacity=0.18, side="double",
            flat_shading=True, cast_shadow=False)
        keep.add(pn)

        ol = f"/hand/{name}/pinch_plane/outline"
        self._dynamic[ol] = self.scene.add_line_segments(
            ol, np.stack([p, np.roll(p, -1, axis=0)], axis=1), colors=rgb,
            line_width=2.0)
        keep.add(ol)

        lb = f"/hand/{name}/pinch_plane/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, name, position=tuple(p.mean(axis=0)), anchor="center-center")
        keep.add(lb)
        return keep

    def _update_planar_gap(self, name, gap, *, rgb):
        """One finger's in-plane distance (Eq 11 / Eq 13): the cross-section the
        pulling plane cuts out of the object, and the distance to it.

        Three pieces, each saying a different thing:

        * the cross-section outlines -- ``G_planar``, drawn in the finger's own
          plane colour so it is obvious which plane cut them. This is EXACT
          geometry;
        * the line from the contact sphere to the nearest point on that outline,
          coloured near/far like every other gap overlay;
        * the label, which is the FACTOR's number, not the line's length. The two
          differ by the Taubin approximation error, and seeing that difference is
          the point of drawing both. In fallback (the plane missed everything, or
          it is degenerate) the label says so, because then the number is the
          ordinary 3D distance and the plane is not involved at all.
        """
        keep = set()
        for index, curve in enumerate(gap.sections):
            pts = np.asarray(curve, float)
            sn = f"/hand/{name}/planar_gap/section/{index}"
            self._dynamic[sn] = self.scene.add_line_segments(
                sn, np.stack([pts[:-1], pts[1:]], axis=1), colors=rgb,
                line_width=2.0)
            keep.add(sn)

        p0 = np.asarray(gap.sphere_pt, float).reshape(3)
        p1 = np.asarray(gap.foot, float).reshape(3)
        rgb_line = _GAP_NEAR_RGB if gap.gap < GAP_GREEN_MAX_M else _GAP_FAR_RGB

        ln = f"/hand/{name}/planar_gap/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb_line, line_width=3.0)
        keep.add(ln)

        text = f"{gap.gap * 1000.0:.1f} mm"
        if gap.fallback:
            text += " (3D)"
        lb = f"/hand/{name}/planar_gap/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, text, position=tuple(0.5 * (p0 + p1)), anchor="center-center")
        keep.add(lb)
        return keep

    def _update_pinch_point(self, pinch_pt):
        """The pinch centroid every finger plane passes through, marked once.

        Hand-level, not per finger: it is one point from
        ``config.HAND_PINCH_POSES``, and drawing it per finger would stack five
        spheres on it. Deliberately NOT the same node as
        ``/pregrasp_centroid/pinch`` -- that one only exists while the
        pinch-centroid CONSTRAINT is on, and this overlay has to stand on its
        own."""
        n = "/finger_plane/pinch"
        self._dynamic[n] = self.scene.add_icosphere(
            n, radius=0.005, color=_CENTER_TARGET_RGB, opacity=0.6,
            position=tuple(np.asarray(pinch_pt, float).reshape(3)))
        return {n}

    def _update_discs(self, name, fm, poses):
        keep = set()
        tc = fm.tendon_config
        r = 1.3 * tc.routing_radius
        h = 0.3 * tc.routing_radius
        for di, node_idx in enumerate(tc.disc_pose_idx):
            if di == 0:
                continue
            mesh = _recenter(trimesh.creation.cylinder(radius=r, height=h))
            mesh.apply_transform(poses[node_idx])
            mesh.visual.vertex_colors = np.array([*_DISC_RGB, 90], dtype=np.uint8)
            n = f"/hand/{name}/disc/{di}"
            self._dynamic[n] = self.scene.add_mesh_trimesh(n, mesh)
            keep.add(n)
        return keep

    def _update_disc_frames(self, name, fm, poses):
        """A triad on every disc node, in that disc's own body frame.

        This is the frame the routing geometry is written in: a hole location
        ``tendon_config.hole_locations[disc][tendon]`` is a body-frame vector on
        this triad, and the world tendon point drawn by :meth:`_update_tendons`
        is ``R @ hole + t`` off exactly these axes. Seeing them is what turns a
        wrong-looking tendon path into a readable one -- a routing angle
        measured from the wrong axis, or a whole finger's discs rolled 180
        degrees, is invisible in the disc cylinders (rotationally symmetric) and
        obvious the moment the axes are drawn.

        Unlike :meth:`_update_discs` the BASE disc is included. It is skipped
        there because its cylinder sits inside the palm and only adds clutter,
        but its frame is the finger's mounting frame -- the one thing on the
        chain you can check against the CAD assembly -- so dropping it would
        hide the frame most worth seeing, and would silently shift the numbering
        of every triad on screen relative to ``disc_pose_idx``.
        """
        keep = set()
        tc = fm.tendon_config
        # Sized off the disc rather than fixed: the axes have to read against the
        # disc they sit on (drawn at 1.3 * routing_radius) at any finger scale,
        # and just past its rim is where they are legible without swamping the
        # neighbouring disc.
        length = 2.0 * tc.routing_radius
        for di, node_idx in enumerate(tc.disc_pose_idx):
            T = poses[node_idx]
            n = f"/hand/{name}/disc_frame/{di}"
            self._dynamic[n] = self.scene.add_frame(
                n, show_axes=True, axes_length=float(length),
                axes_radius=float(length) * 0.04,
                wxyz=_wxyz_from_R(T[:3, :3]),
                position=tuple(T[:3, 3]))
            keep.add(n)
        return keep

    def _prune(self, keep):
        """Remove dynamic handles that were not re-added this frame (e.g. a toggle
        turned off, or a tendon became inactive)."""
        for n in list(self._dynamic):
            if n not in keep:
                handle = self._dynamic.pop(n)
                try:
                    handle.remove()
                except Exception:
                    pass
