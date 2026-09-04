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
    _CLEAR_RGB,
    _DISC_RGB,
    _GAP_FAR_RGB,
    _GAP_NEAR_RGB,
    _MARGIN_OK_RGB,
    _MARGIN_VIOLATED_RGB,
    _PENETRATING_RGB,
    _SELF_PAIR_RGB,
    _TAUBIN_RGB,
    _TENDON_RGB,
    _WRENCH_ARM_RGB,
    _WRENCH_FORCE_RGB,
    ANGLE_GREEN_MAX_DEG,
    GAP_GREEN_MAX_M,
    GRASP_WRENCH_GREEN_MAX,
)

#: Link-mesh shading. Light and translucent on purpose: the meshes are scenery,
#: and the contact spheres, gap lines and frames drawn over them are the things
#: a reader is actually measuring by.
_LINK_MESH_RGB = (170, 178, 189)
_LINK_MESH_ALPHA = 140


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


    def _update_link_frames(self, name, fm, poses):
        """A triad on every site of a digit -- the RIGID LINK frames.

        The joint-space counterpart of :meth:`_update_disc_frames`. A rigid
        digit's sites are its link frames: each one is the frame its own joint
        rotates, and it is what the URDF's visual origin, the collision sphere
        centre and the contact site are all written against. Drawing them is how
        a mesh that rides on the wrong axis, or a joint whose zero is rotated
        from the manufacturer's, becomes visible -- the link geometry alone
        cannot show it.

        Every site is drawn, the digit's first included. That one is coincident
        with the digit's mount and so has no bar to draw (see the segment filter
        in ``ViserHandScene.update``), but it is a real frame and skipping it
        would shift every triad's index relative to the state's site list.
        """
        keep = set()
        # Fixed length rather than sized off the geometry: unlike a routing disc
        # (whose radius the disc triads scale with) a link has no single
        # characteristic radius to read against. 15 mm reads on an Allegro
        # phalanx without swamping the neighbouring joint.
        length = 0.015
        for si in range(len(poses)):
            T = poses[si]
            n = f"/hand/{name}/link_frame/{si}"
            self._dynamic[n] = self.scene.add_frame(
                n, show_axes=True, axes_length=length,
                axes_radius=length * 0.04,
                wxyz=_wxyz_from_R(T[:3, :3]),
                position=tuple(T[:3, 3]))
            keep.add(n)
        return keep


    def _update_palm_frame(self, wrist_pose):
        """The palm's own link frame, drawn with the digits' under the same
        switch.

        Hangs off the wrist rather than off a digit because that IS the palm
        link: the hand's base frame, the one every digit mount is written
        relative to. Drawn a little larger than the digit triads for the same
        reason the table frame is drawn larger than the object's -- it is the
        reference the others are read against.

        Distinct from the 'mount frames' overlay, which draws this same wrist
        frame next to the ROBOT FLANGE to check a mount measurement. Here it is
        the first link of the kinematic chain, and it belongs with the rest of
        the chain.
        """
        if wrist_pose is None:
            return set()
        T = np.asarray(wrist_pose, float)
        length = 0.025
        n = "/hand/link_frame/palm"
        self._dynamic[n] = self.scene.add_frame(
            n, show_axes=True, axes_length=length,
            axes_radius=length * 0.04,
            wxyz=_wxyz_from_R(T[:3, :3]),
            position=tuple(T[:3, 3]))
        return {n}


    # -- link meshes -------------------------------------------------------
    #
    # VISUAL ONLY. Collision in this repository is the sphere set the solve
    # carries (DigitState.collision_sites), and the graph never sees a mesh --
    # so nothing here can affect a solve, and a checkout without the mesh files
    # loses the skin and nothing else.

    def _link_mesh(self, path):
        """One loaded mesh, cached for the life of the scene.

        Loading is the expensive part and the geometry never changes -- only the
        transform it is drawn at does -- so the file is read once and reused
        every frame. A file that fails to load is cached as None so a broken
        mesh is not re-read once per frame forever.
        """
        cache = self.__dict__.setdefault("_mesh_cache", {})
        key = str(path)
        if key not in cache:
            try:
                import trimesh

                cache[key] = trimesh.load(path, force="mesh")
            except Exception:
                cache[key] = None
        return cache[key]

    def _update_link_meshes(self, link_meshes, frame, wrist_pose):
        """Draw each link mesh at the pose the solve already put its site at.

        No second kinematic pass: every visual origin on the hands that carry
        meshes is the identity, so a mesh rides directly on its site's frame.
        """
        keep = set()
        if not link_meshes:
            return keep

        for i, entry in enumerate(link_meshes):
            attach, path, T_local = entry
            mesh = self._link_mesh(path)
            if mesh is None:
                continue
            if attach is None:
                if wrist_pose is None:
                    continue
                T = np.asarray(wrist_pose, float)
            else:
                digit, site = attach
                if digit >= len(self.finger_names):
                    continue
                fs = frame.get(self.finger_names[digit])
                if fs is None or site >= len(fs.marginals.sites):
                    continue
                T = np.asarray(fs.marginals.sites[site].pose.mean, float)

            # T_local first: it takes the mesh out of its own coordinates and
            # into the frame it attaches to (for glTF, the Y-up to Z-up
            # correction). Then the site pose puts that frame in the world.
            placed = mesh.copy()
            placed.apply_transform(np.asarray(T) @ np.asarray(T_local, float))
            placed.visual.vertex_colors = np.array(
                [*_LINK_MESH_RGB, _LINK_MESH_ALPHA], dtype=np.uint8)
            n = f"/hand/mesh/{i}"
            self._dynamic[n] = self.scene.add_mesh_trimesh(n, placed)
            keep.add(n)
        return keep


    # -- constraint distances ---------------------------------------------
    #
    # One method per constraint FAMILY, each drawn under its own scene path and
    # switched on its own flag (see toggles.DistanceOverlays). None of them
    # consult whether the constraint is in the factor graph: the number a
    # collision inequality WOULD see is exactly what you want while collision is
    # off, and gating the picture on the solve makes that impossible to look at.

    def _update_clearance(self, key, sphere_pt, surface_pt, gap, kind):
        """One collision sphere's clearance from a surface: a line to the nearest
        point on it and the signed gap in mm.

        Coloured by SIGN, like the half-space overlay and unlike the contact gap
        lines: ``h_pen`` is satisfied exactly while ``gap >= 0``, so 1 mm of
        clearance and 1 mm of penetration are opposite verdicts. Colouring them
        both "near", as ``GAP_GREEN_MAX_M`` would, hides the only thing this
        overlay exists to show.

        ``key`` is ``"{finger}/{node}"`` from the witness, which nests straight
        into the scene path so a sphere's line lives under its own finger.
        ``kind`` namespaces the surface, so the same sphere's object clearance
        and table clearance are two overlays rather than one overwriting the
        other.
        """
        p0 = np.asarray(sphere_pt, float).reshape(3)
        p1 = np.asarray(surface_pt, float).reshape(3)
        rgb = _CLEAR_RGB if gap >= 0.0 else _PENETRATING_RGB

        ln = f"/clearance/{kind}/{key}/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=2.0)

        lb = f"/clearance/{kind}/{key}/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{gap * 1000.0:+.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        return {ln, lb}


    def _update_pair_gap(self, key, point_a, point_b, gap):
        """One finger-finger sphere pair: the segment between the two spheres'
        facing surface points, and the signed gap.

        Surface points rather than centres on purpose -- a centre-to-centre line
        passes through both spheres and reads as penetration in every posture,
        which is precisely the state this overlay has to be able to distinguish.

        Hand-level path (``/self_collision/...``) because the measurement belongs
        to neither finger: filing it under one of the two would leave it hidden
        whenever that finger's overlays were the ones switched off.
        """
        p0 = np.asarray(point_a, float).reshape(3)
        p1 = np.asarray(point_b, float).reshape(3)
        rgb = _SELF_PAIR_RGB if gap >= 0.0 else _PENETRATING_RGB

        ln = f"/self_collision/{key}/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=rgb, line_width=2.0)

        lb = f"/self_collision/{key}/label"
        self._dynamic[lb] = self.scene.add_label(
            lb, f"{gap * 1000.0:+.1f} mm", position=tuple(0.5 * (p0 + p1)),
            anchor="center-center")
        return {ln, lb}


    def _update_ellipsoid_metric(self, name, metric):
        """One fingertip's exact and Taubin distances to an ellipsoid, together.

        Draws the line the TAUBIN number would measure along -- the same segment
        the exact witness found, since the approximation has no witness point of
        its own -- and labels both numbers with the difference between them, so
        the approximation error is a figure on screen rather than something to
        infer from two overlays.

        The metric the SOLVE was built with is marked. Without it the pair is
        ambiguous in the worst possible way: the two agree near the surface, so
        a converged contact looks identical under either setting, and the label
        would silently stop describing the residual the moment the flag moved.

        Deliberately its own scene path rather than extra text on the contact gap
        label: that label is the constraint's number, and hanging a second metric
        off it would make a debugging aid look like part of the readout the solve
        is judged by.
        """
        p0 = np.asarray(metric.sphere_pt, float).reshape(3)
        p1 = np.asarray(metric.surface_pt, float).reshape(3)
        keep = set()

        ln = f"/hand/{name}/ellipsoid_metric/line"
        self._dynamic[ln] = self.scene.add_line_segments(
            ln, np.stack([p0, p1])[None], colors=_TAUBIN_RGB, line_width=2.0)
        keep.add(ln)

        mark = {"exact": ("*", ""), "taubin": ("", "*")}[metric.in_use]
        lb = f"/hand/{name}/ellipsoid_metric/label"
        self._dynamic[lb] = self.scene.add_label(
            lb,
            f"exact {metric.exact * 1000.0:+.1f}{mark[0]} / "
            f"taubin {metric.taubin * 1000.0:+.1f}{mark[1]} mm "
            f"(d {abs(metric.exact - metric.taubin) * 1000.0:.1f})",
            # Offset toward the sphere rather than the midpoint: the contact gap
            # line runs along the same segment and labels its own middle, so two
            # labels at one point would overprint.
            position=tuple(p0 + 0.25 * (p1 - p0)), anchor="center-center")
        keep.add(lb)
        return keep


    def _update_grasp_wrench(self, wrench, object_center, *, arrow=0.03):
        """The ``h_grasp`` equality, drawn as the arrangement that produces it.

        The residual is a 6-vector summed over the contacts, and a number alone
        cannot say WHY it is nonzero -- a balanced grasp and a grasp nobody
        measured both report zero. So what is drawn is the sum's terms:

        * one arrow per contact, from the contact point ``p_i`` along the INWARD
          normal ``-n_i`` -- the unit virtual force the constraint is written
          over. All the same length, because they are unit vectors; the grasp is
          balanced when they point at each other.
        * a faint moment arm from the object origin ``t_obj`` out to each
          ``p_i``. This is the half of the torque term that the force arrows do
          not show: two contacts whose forces cancel perfectly still spin the
          object when their arms are offset, and that offset is only visible as
          a drawn arm.
        * the NET force ``sum(-n_i)`` from the object origin, at ``arrow`` per
          unit -- so its length against one contact arrow reads directly as
          "this many contacts' worth of push is left over".
        * the NET torque, scaled by the longest moment arm so that a unit force
          acting tangentially at that radius draws one arrow length. Torque is in
          force-metres and cannot share the force's scale honestly; tying it to
          the object's own size is what makes the two arrows comparable at a
          glance instead of one of them always being invisible.

        The two residual arrows are green together when the whole 6-vector is
        under ``GRASP_WRENCH_GREEN_MAX`` and red otherwise -- one verdict, since
        the constraint is one equality over both halves.
        """
        keep = set()
        t_obj = np.asarray(object_center, float).reshape(3)
        points = [np.asarray(p, float).reshape(3) for p in wrench.points]
        if not points:
            return keep

        # Per-contact virtual forces: p_i -> p_i - arrow * n_i (inward).
        shafts = np.stack(
            [np.stack([p, p - arrow * np.asarray(n, float).reshape(3)])
             for p, n in zip(points, wrench.normals)])
        cn = "/grasp_wrench/contacts"
        self._dynamic[cn] = self.scene.add_arrows(
            cn, shafts, colors=_WRENCH_FORCE_RGB,
            shaft_radius=arrow * 0.035, head_radius=arrow * 0.10,
            head_length=arrow * 0.25)
        keep.add(cn)

        arms = np.stack([np.stack([t_obj, p]) for p in points])
        an = "/grasp_wrench/arms"
        self._dynamic[an] = self.scene.add_line_segments(
            an, arms, colors=_WRENCH_ARM_RGB, line_width=1.5)
        keep.add(an)

        on = "/grasp_wrench/origin"
        self._dynamic[on] = self.scene.add_icosphere(
            on, radius=arrow * 0.12, color=_WRENCH_ARM_RGB, opacity=0.9,
            position=tuple(t_obj))
        keep.add(on)

        rgb = (_GAP_NEAR_RGB if wrench.norm <= GRASP_WRENCH_GREEN_MAX
               else _GAP_FAR_RGB)
        # The reference radius for the torque scale: the longest arm actually in
        # the sum. Falls back to `arrow` for contacts sitting on the origin,
        # where no radius can be read off the grasp.
        radius = max((float(np.linalg.norm(p - t_obj)) for p in points),
                     default=0.0) or arrow

        # Either half can be exactly zero while the other is not -- a radially
        # symmetric object has every contact normal through its centre, so its
        # torque term vanishes identically -- so each is drawn and named on its
        # own. An arrow of zero length renders as a stray cone at the origin,
        # which reads as an unbalanced contact rather than as none.
        residuals, tags = [], []
        for tag, vector, scale in (("force", wrench.force, arrow),
                                   ("torque", wrench.torque, arrow / radius)):
            v = np.asarray(vector, float).reshape(3) * scale
            if np.linalg.norm(v) > 1e-6:
                residuals.append(np.stack([t_obj, t_obj + v]))
                tags.append(tag)
        if residuals:
            rn = "/grasp_wrench/residual"
            self._dynamic[rn] = self.scene.add_arrows(
                rn, np.stack(residuals),
                colors=np.array([rgb] * len(residuals), dtype=np.uint8),
                shaft_radius=arrow * 0.06, head_radius=arrow * 0.16,
                head_length=arrow * 0.30)
            keep.add(rn)
            # The two share a colour -- they are halves of one verdict -- and an
            # origin, so the only thing telling them apart is the tip label.
            for tag, segment in zip(tags, residuals):
                tl = f"/grasp_wrench/{tag}_label"
                self._dynamic[tl] = self.scene.add_label(
                    tl, tag, position=tuple(segment[1]), anchor="center-center")
                keep.add(tl)

        lb = "/grasp_wrench/label"
        self._dynamic[lb] = self.scene.add_label(
            lb,
            f"|h_grasp| {wrench.norm:.3f}  "
            f"(f {np.linalg.norm(wrench.force):.3f}, "
            f"t {np.linalg.norm(wrench.torque) * 1000.0:.1f} mm) "
            f"over {len(points)}",
            position=tuple(t_obj + np.array([0.0, 0.0, 1.4 * arrow])),
            anchor="center-center")
        keep.add(lb)
        return keep
