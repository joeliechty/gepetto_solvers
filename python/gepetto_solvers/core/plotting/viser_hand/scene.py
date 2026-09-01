""":class:`ViserHandScene` -- the composed renderer.

Construction, the camera helper, the per-frame ``update`` that draws the
backbones and dispatches to the overlays, and the handle pruning that removes
geometry which vanished since the last frame.
"""

import numpy as np

from ._frames import FrameSceneMixin
from ._object import ObjectSceneMixin
from ._overlays import OverlayMixin
from ._support import SupportSceneMixin
from .palette import (
    _COLLISION_RGB,
    _CONTACT_RGB,
    _FINGER_PLANE_RGB,
    _ROD_RGB,
)


class ViserHandScene(
    ObjectSceneMixin,
    SupportSceneMixin,
    FrameSceneMixin,
    OverlayMixin,
):
    """Renders/refreshes the tendon hand in a viser scene.

    Parameters mirror the PyVista plotter's spirit: construct once with the finger
    names, then call :meth:`set_object` on object change and :meth:`update` per
    frame. Display toggles (discs / contact spheres / collision spheres) are read
    on each :meth:`update` from the attributes set here.

    The drawing methods live in the four mixins above -- object, support plane,
    coordinate frames, and per-frame overlays -- which share ``self.scene`` and
    the ``self._dynamic`` handle table this constructor sets up.
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
        PyVista HandPlotter. Viewing from here the finger curl reads as a
        grasp closing toward you rather than hyperextending away."""
        f = np.asarray(focal, float).reshape(3)
        az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
        offset = distance * np.array([np.cos(el) * np.cos(az),
                                      np.cos(el) * np.sin(az),
                                      np.sin(el)])
        return f + offset, f


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
            states = fm.sites
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
                for di, node_idx in enumerate(fm.extras.tendon_config.disc_pose_idx):
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
