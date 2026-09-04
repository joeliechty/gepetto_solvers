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
from .toggles import DistanceOverlays


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
                 show_link_frames=False,
                 show_contact_spheres=True,
                 show_collision_spheres=True, show_gap_lines=True,
                 show_link_meshes=True,
                 show_finger_planes=False, show_planar_gap=False,
                 distances=None):
        self.server = server
        self.scene = server.scene
        self.finger_names = list(finger_names)
        self.backbone_width = backbone_width
        self.show_discs = show_discs
        self.show_disc_frames = show_disc_frames
        # The joint-space counterpart of show_disc_frames: a triad on every
        # rigid link. A hand has one or the other -- routing discs and rigid
        # links are two different things to have frames ON -- so the workbench
        # only ever builds one of the two checkboxes.
        self.show_link_frames = show_link_frames
        self.show_contact_spheres = show_contact_spheres
        self.show_collision_spheres = show_collision_spheres
        # On by default: a hand that HAS meshes looks like itself with
        # them, and the overlays are drawn over the top.
        self.show_link_meshes = show_link_meshes
        # The MASTER switch for every measurement overlay: one control that
        # takes the whole category off the scene, so a picture can be cleared for
        # a screenshot without losing which families were selected.
        self.show_gap_lines = show_gap_lines
        # ...and which families those are. Independent of the constraints
        # themselves -- see toggles.DistanceOverlays for why that separation is
        # the point rather than a convenience.
        self.distances = distances if distances is not None else DistanceOverlays()
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
               centroid_gap=None, finger_planes=None, planar_gaps=None,
               object_clearances=None, table_clearances=None, pair_gaps=None,
               ellipsoid_metrics=None, grasp_wrench=None, object_center=None,
               link_meshes=None, wrist_pose=None):
        """Refresh the hand geometry for one frame. ``frame`` maps finger name to
        an object exposing ``.marginals`` (a ``DigitState``).

        ``link_meshes`` is the hand's optional visual geometry, as
        ``[(attach, path, T_local)]`` from ``Hand.visual_meshes`` -- ``attach``
        is None for a mesh riding on the wrist (``wrist_pose``) or
        ``(digit, site)`` for one riding on a site, and ``T_local`` takes the
        mesh out of its own coordinates into that frame (glTF is Y-up where
        URDF is Z-up, and the file itself does not say so). PURELY COSMETIC: collision in this repository is
        the sphere set the solve carries, never a mesh, so these are drawn and
        nothing else. Omit them and the hand is drawn as a skeleton, which is a
        complete drawing in its own right.

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

        ``object_clearances`` and ``table_clearances`` are the COLLISION
        counterparts of the two gap maps: ``{"finger/node": (sphere_pt,
        surface_pt, signed_gap)}`` over every sphere the contact equalities do
        not own (``solvers.object_collision_witness`` /
        ``solvers.free_sphere_plane_witness``). Same 3-tuple, drawn by sign
        rather than by magnitude, because ``h_pen`` is an inequality and 1 mm of
        clearance is a different verdict from 1 mm of penetration.

        ``pair_gaps`` is the finger-finger counterpart:
        ``{"a/i/b/j": (point_on_a, point_on_b, signed_gap)}`` from
        ``solvers.self_collision_witness``, already narrowed to the pairs near
        touching.

        ``ellipsoid_metrics`` is ``{finger: solvers.EllipsoidMetric}`` -- the
        exact and Taubin ellipsoid distances side by side, for reading the
        approximation error the ``ellipsoid_taubin`` flag would put into the
        residuals. None on an object with no ellipsoid form, where the flag is
        inert and there is nothing to compare.

        ``grasp_wrench`` is the h_grasp readout (``solvers.GraspWrench``), drawn
        with ``object_center`` as the origin the moment arms and the net residual
        are measured about -- the same ``t_obj`` the constraint uses. Both are
        needed together; either alone draws nothing.

        Every one of the above is gated on ``self.show_gap_lines`` (the master
        "distance overlays" toggle) AND on its own family flag in
        ``self.distances``. The family flags are deliberately NOT tied to whether
        the matching constraint is in the graph: see ``toggles.DistanceOverlays``.

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

            # The digit itself. A continuum rod really is a smooth curve
            # through its nodes, so it gets a spline; a rigid linkage is
            # straight segments between joint frames, and drawing a spline
            # through those would show a bend where the hardware has none.
            #
            # Told apart by whether the digit carries continuum state, which is
            # the honest question -- not by the hand's name.
            n = f"/hand/{name}/rod"
            if getattr(fm, "extras", None) is not None:
                self._dynamic[n] = self.scene.add_spline_catmull_rom(
                    n, positions, curve_type="catmullrom",
                    line_width=self.backbone_width, color=_ROD_RGB)
                keep.add(n)
            else:
                # Degenerate segments are dropped. A link's frame sits at its
                # own joint's origin, so a digit's FIRST link is coincident with
                # that digit's mount and the bar between them has zero length.
                # It is a real frame -- it is what the base joint rotates -- so
                # it keeps its site; it just has nothing to draw.
                segs = np.stack([positions[:-1], positions[1:]], axis=1)
                segs = segs[np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1) > 1e-9]
                if len(segs):
                    self._dynamic[n] = self.scene.add_line_segments(
                        n, segs, colors=_ROD_RGB,
                        line_width=self.backbone_width)
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
            if self._shown("object_contact") and gaps and name in gaps:
                keep |= self._update_gap(name, *gaps[name])
            if (self._shown("table_contact") and table_gaps
                    and name in table_gaps):
                keep |= self._update_gap(name, *table_gaps[name],
                                         kind="table_gap")
            if (self._shown("half_space") and half_space_gaps
                    and name in half_space_gaps):
                keep |= self._update_half_space(name, *half_space_gaps[name])

            # The exact-vs-Taubin comparison rides beside the object gap line it
            # is measured along, so it is per finger like that one.
            if (self._shown("ellipsoid_metric") and ellipsoid_metrics
                    and name in ellipsoid_metrics):
                keep |= self._update_ellipsoid_metric(
                    name, ellipsoid_metrics[name])

            # Collision spheres on the disc nodes.
            if collision and self.show_collision_spheres:
                # Off the STATE's own site list, so this marks exactly the
                # spheres the solve carried -- on any hand.
                for di, node_idx in enumerate(fm.collision_sites):
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

            # The rigid links' own frames, for a hand whose digits are a
            # linkage rather than a rod.
            if self.show_link_frames:
                keep |= self._update_link_frames(name, fm, poses)

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
        if self._shown("pregrasp") and center_gap is not None:
            keep |= self._update_center(*center_gap)

        # Pre-grasp short-axis alignment: also HAND-level, drawn once.
        if self._shown("pregrasp") and axis_align is not None:
            keep |= self._update_axis_align(*axis_align)

        # Pre-grasp pinch-centroid centering: also HAND-level, drawn once.
        if self._shown("pregrasp") and centroid_gap is not None:
            keep |= self._update_centroid(*centroid_gap)

        # The collision inequalities' own distances. Keyed per SPHERE rather
        # than per finger, so they are drawn from the flat witness maps here
        # instead of inside the per-finger loop above.
        if self._shown("object_collision") and object_clearances:
            for key, measure in object_clearances.items():
                keep |= self._update_clearance(key, *measure, kind="object")
        if self._shown("table_collision") and table_clearances:
            for key, measure in table_clearances.items():
                keep |= self._update_clearance(key, *measure, kind="table")
        if self._shown("self_collision") and pair_gaps:
            for key, measure in pair_gaps.items():
                keep |= self._update_pair_gap(key, *measure)

        # h_grasp: hand-level, and measured ABOUT the object, so it needs the
        # object's origin as well as the residual.
        if (self._shown("grasp_wrench") and grasp_wrench is not None
                and object_center is not None):
            keep |= self._update_grasp_wrench(grasp_wrench, object_center)

        # The point the finger planes fan about -- one marker for all of them.
        if self.show_finger_planes and finger_planes:
            keep |= self._update_pinch_point(next(iter(finger_planes.values()))[2])

        # The palm link's frame: hand-level, so drawn once rather than per digit.
        if self.show_link_frames:
            keep |= self._update_palm_frame(wrist_pose)

        if self.show_link_meshes:
            keep |= self._update_link_meshes(link_meshes, frame, wrist_pose)

        self._prune(keep)


    def _shown(self, family):
        """Whether one distance-overlay family should be drawn: the master
        switch AND that family's own flag.

        A method rather than the condition written out nine times, because the
        two-level rule is the thing that has to stay consistent -- a family that
        forgot the master would keep drawing after the category was switched
        off, which is exactly the failure the master exists to prevent."""
        return self.show_gap_lines and getattr(self.distances, family)


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
