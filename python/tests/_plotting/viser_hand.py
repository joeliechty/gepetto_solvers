"""viser scene renderer for the tendon hand -- the web-viewer analogue of the
PyVista ``TendonHandPlotter``.

Pure rendering: given a viser server and one solved hand *frame* (the
``{finger_name: solution}`` shim the solvers produce), it draws each finger's
Cosserat backbone, its tendons, and -- optionally -- the routing discs, the
per-fingertip contact spheres, the disc-node collision spheres, the grasp
object, and the support-plane "table". No solving and no GUI live here; the
interactive app (``tendon_hand/viz_interactive.py``) owns those.

The world-frame geometry reproduces exactly what the PyVista mesh managers
compute (``_plotting/tendon_hand_plotter.py``):

* backbone  -- node translations ``rod.states[n].pose.mean[:3, 3]``.
* tendons   -- ``R @ hole + t`` for each active hole, with the disc pose
  ``rod.states[tendon_config.disc_pose_idx[disc]]``.
* discs / collision spheres -- one per disc node (``disc_pose_idx``).

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
_CONTACT_RGB = (80, 200, 120)
_COLLISION_RGB = (230, 120, 60)
_DISC_RGB = (100, 149, 237)
_TABLE_RGB = (150, 150, 160)

# Fingertip-to-object gap overlay: green within GAP_GREEN_MAX_M of the surface
# (including interpenetration, which is simply "not far"), red beyond it.
GAP_GREEN_MAX_M = 0.015
_GAP_NEAR_RGB = (0, 190, 60)
_GAP_FAR_RGB = (220, 40, 40)


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
                 show_discs=False, show_contact_spheres=True,
                 show_collision_spheres=True, show_gap_lines=True):
        self.server = server
        self.scene = server.scene
        self.finger_names = list(finger_names)
        self.backbone_width = backbone_width
        self.show_discs = show_discs
        self.show_contact_spheres = show_contact_spheres
        self.show_collision_spheres = show_collision_spheres
        self.show_gap_lines = show_gap_lines

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
        """(Re)build the grasp object mesh. Sphere/cube/ellipsoid use native viser
        primitives (translucent); cylinder/capsule fall back to a trimesh."""
        center = np.asarray(center, float)
        t = spec["type"]
        name = "/object"
        if t == "sphere":
            self.scene.add_icosphere(name, radius=float(spec["radius"]),
                                     color=_OBJECT_RGB, opacity=0.35,
                                     position=tuple(center))
        elif t == "ellipsoid":
            a, b, c = (float(v) for v in spec["semi_axes"])
            self.scene.add_icosphere(name, radius=1.0, scale=(a, b, c),
                                     color=_OBJECT_RGB, opacity=0.35,
                                     position=tuple(center))
        elif t == "cube":
            hx, hy, hz = spec["half_extents"]
            self.scene.add_box(name, color=_OBJECT_RGB,
                               dimensions=(2 * hx, 2 * hy, 2 * hz),
                               opacity=0.35, position=tuple(center))
        else:
            mesh = _object_trimesh(spec, center)
            if mesh is not None:
                self.scene.add_mesh_trimesh(name, mesh)

    def set_table(self, origin, normal, *, span=0.3, thickness=0.005):
        """Draw the support-plane slab (visual aid; the solver uses the analytic
        half-space). Thin along the dominant normal axis, matching
        ``scene.table_plot_spec``."""
        origin = np.asarray(origin, float).reshape(3)
        n = np.asarray(normal, float).reshape(3)
        axis = int(np.argmax(np.abs(n)))
        extents = [span, span, span]
        extents[axis] = thickness
        self.scene.add_box("/table", color=_TABLE_RGB, dimensions=tuple(extents),
                           opacity=0.4, position=tuple(origin))

    def clear_table(self):
        try:
            self.server.scene.remove_by_name("/table")
        except Exception:
            pass

    # -- per-frame hand ----------------------------------------------------

    def update(self, frame, *, tip_radii=None, collision_radius=0.003,
               collision=False, gaps=None):
        """Refresh the hand geometry for one frame. ``frame`` maps finger name to
        an object exposing ``.marginals`` (a ``TendonFingerMarginals``).

        ``gaps`` is the optional fingertip-to-object overlay: a
        ``{finger: (sphere_pt, surface_pt, gap_m)}`` map as returned by
        ``HandResult.contact_witness``. Rendering only -- nothing here feeds the
        solver."""
        keep = set()
        tip_radii = tip_radii or [None] * len(self.finger_names)

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

    def _update_gap(self, name, sphere_pt, surface_pt, gap):
        p0 = np.asarray(sphere_pt, float).reshape(3)
        p1 = np.asarray(surface_pt, float).reshape(3)
        rgb = _GAP_NEAR_RGB if gap < GAP_GREEN_MAX_M else _GAP_FAR_RGB

        ln = f"/hand/{name}/gap/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=3.0)

        lb = f"/hand/{name}/gap/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{gap * 1000.0:.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        return {ln, lb}

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
