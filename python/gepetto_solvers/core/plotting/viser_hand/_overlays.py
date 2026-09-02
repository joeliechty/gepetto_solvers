"""Per-frame overlays: tendons, gaps, discs, and the constraint witnesses.

A mixin of :class:`~gepetto_solvers.core.plotting.viser_hand.scene.ViserHandScene`.
Split out of what was one 968-line class; the methods here use only ``self.scene``
and ``self._dynamic``, which the composed class owns.
"""

import numpy as np
import trimesh

from ._geometry import _recenter, _wxyz_from_R
from .palette import (
    _CENTER_TARGET_RGB,
    _DISC_RGB,
    _GAP_FAR_RGB,
    _GAP_NEAR_RGB,
    _MARGIN_OK_RGB,
    _MARGIN_VIOLATED_RGB,
    _TENDON_RGB,
    ANGLE_GREEN_MAX_DEG,
    GAP_GREEN_MAX_M,
)


class OverlayMixin:
    def _update_tendons(self, name, fm, poses):
        """Tendon routes, for a digit that has any.

        A hand with no tendons carries no routing to draw, and says so by
        leaving ``extras`` unset -- so this returns nothing rather than the
        caller having to know which kind of hand it is holding."""
        keep = set()
        extras = getattr(fm, "extras", None)
        if extras is None:
            return keep
        tc = extras.tendon_config
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
        extras = getattr(fm, "extras", None)
        if extras is None:
            return keep   # no routing discs on this hand
        tc = extras.tendon_config
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
        extras = getattr(fm, "extras", None)
        if extras is None:
            return set()   # no routing discs on this hand
        keep = set()
        tc = extras.tendon_config
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
