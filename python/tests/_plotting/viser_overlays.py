"""viser overlays for the Section 1.8 phase goals -- the debug layer over
``viser_hand.ViserHandScene``.

``ViserHandScene`` draws what the robot IS. This draws what each phase is TRYING
TO DO: the pre-grasp target frame and the axes being aligned to it, the
support-plane distances, the opposition split, the object proxy, the witness
points and their contact frames. ``phase_violations()`` reports one scalar per
constraint family, which tells you a phase is unhappy but not where; these show
where.

Pure rendering, exactly like ``ViserHandScene``: construct with a viser server,
call :meth:`update` with the dict from ``solvers.goal_geometry`` per frame, and
toggle items through :attr:`show`. Nothing here computes geometry (that lives
next to the constraints it mirrors, in ``tendon_hand/solvers.py``) and nothing
here feeds a solver.

Everything is named ``/goals/<overlay key>/...`` so it never collides with the
hand (``/hand``), the object or the table, and dynamic handles are tracked and
pruned the same way -- toggling an item off removes it on the next update.

Two naming rules the drawing helpers keep, because viser's scene tree is a real
tree: no drawn node may be an ancestor path of another (removing a parent
cascades to its children, and the prune pass would then try to remove an
already-removed handle), and every node sits at least one level below its
overlay key so the key stays a clean prefix to filter by. Hence the ``_lbl`` /
``_head`` SUFFIXES on a segment's label and an arrow's head -- they are siblings
of the line, not children of it.

:data:`OVERLAYS` is the single source of truth for what exists: this module
dispatches on it and the visualizer builds its checkboxes from it, so adding an
overlay is one entry plus one ``_draw_*`` method and the GUI follows.
"""

from collections import namedtuple

import numpy as np
import trimesh

from .viser_hand import GAP_GREEN_MAX_M, _GAP_FAR_RGB, _GAP_NEAR_RGB


# key      -- stable id, also the scene-node segment and the _draw_<key> method
# label    -- checkbox text
# phase    -- which §1.8 phase this is a goal of (grouping only; every overlay can
#             be drawn in any phase, which is the point -- watching phase 1's
#             plane distances during phase 2 is how you see the slide hold contact)
# default  -- on at startup
# hint     -- checkbox tooltip
OverlaySpec = namedtuple("OverlaySpec", "key label phase default hint")

OVERLAYS = (
    # -- Phase 0: pre-grasp positioning (Eq 1.92-1.98) --
    OverlaySpec("pre_target", "T_base,pre target frame", 0, True,
                "Eq 1.93: the pre-grasp base pose phase 0 servos toward."),
    OverlaySpec("base_frame", "current base frame", 0, True,
                "The achieved T_base. Together with the target frame these are "
                "the two frames phase 0 is aligning."),
    OverlaySpec("waypoint", "slewed waypoint frame", 0, False,
                "What is actually commanded THIS tick: the target slewed by "
                "pregrasp_slew_pos / _rot. Phase 0 never commands T_pre "
                "outright -- a stiff prior that far away stalls the inner LM."),
    OverlaySpec("align_axes", "alignment axes (a/g/s vs -n, m)", 0, False,
                "The hand's measured triad in world, drawn with the -n_hat and "
                "m_hat it is matched to. a_hat should end on -n_hat (palm down) "
                "and s_hat on m_hat (thumb primed for opposition)."),
    OverlaySpec("hover_point", "hover point + h_clear", 0, True,
                "p_obj + h_clear*n_hat, and the clearance height from the "
                "object centroid up to it."),
    OverlaySpec("centroid", "contact centroid p_bar", 0, True,
                "The live centroid of the contact spheres and its segment to "
                "the hover point -- that segment IS the Eq 1.93 position error."),

    # -- Phase 1: support contact (Eq 1.99-1.107) --
    OverlaySpec("plane_lines", "plane distance lines", 1, True,
                "Eq 1.104 c_support per contact sphere, labelled in mm. Zero is "
                "the goal in phases 1-2; in phase 3 it relaxes to a clearance."),
    OverlaySpec("free_plane_lines", "free-sphere plane lines", 1, False,
                "The Eq 1.106 avoidance inequality on every other sphere. Off "
                "by default (one line per sphere), but one sphere driven "
                "through the table is what silently stalls a phase."),
    OverlaySpec("split", "p_split + m_hat arrows", 1, True,
                "Eq 1.99 opposition: the splitting point and each contact "
                "finger's own half-space normal, labelled with its c_half."),
    OverlaySpec("split_plane", "half-space split plane", 1, True,
                "The splitting plane as a slab. A visual aid -- the constraint "
                "is the analytic half-space, not this box."),

    # -- Phase 2: object approach (Eq 1.108-1.113) --
    OverlaySpec("proxy", "proxy ellipsoid", 2, False,
                "The hyper-ellipsoid phase 2 drives onto (Eq 1.108). For a "
                "cube / cylinder / capsule this is NOT the drawn object mesh, "
                "and the difference is why a phase-2 contact can look wrong."),

    # -- Phase 3: on-object servoing (Eq 1.114-1.118) --
    OverlaySpec("witness", "witness points p_c,obj", 3, True,
                "The Eq 1.114 c_R segment from each contact sphere to its "
                "witness. Uses the solver's own Symbol('Y',i) when the binding "
                "exposes it, else the analytic surface projection."),
    OverlaySpec("witness_frames", "witness C-frames", 3, False,
                "The contact frame (N_obj, t1, t2) the Eq 1.116/1.117 tangent "
                "slip residuals are written in."),
    OverlaySpec("witness_targets", "witness targets", 3, False,
                "Eq 1.118 target regions and the geodesic pull toward them, "
                "when params.witness_targets is set."),
)

OVERLAY_KEYS = tuple(s.key for s in OVERLAYS)


# Goal geometry reads against the hand's own palette, so nothing here reuses a
# colour ViserHandScene already means something by (rod blue, contact green,
# collision orange, object gold).
_TARGET_RGB = (255, 120, 0)      # where phase 0 is going
_CURRENT_RGB = (60, 130, 250)    # where the base is now
_WAYPOINT_RGB = (150, 150, 150)  # what is commanded this tick
_SPLIT_RGB = (230, 160, 60)
_CENTROID_RGB = (255, 60, 200)
_WITNESS_RGB = (120, 220, 255)
_PROXY_RGB = (190, 110, 220)
_HOVER_RGB = (255, 210, 40)

# Alignment triad: the hand's measured axes vs the world directions they are
# matched to. Same hue per pair, so a matched pair reads as one colour.
_AXIS_RGB = {"a": (220, 60, 60), "g": (60, 200, 90), "s": (90, 120, 255)}

_AXES_LEN = 0.035          # frame triad arm length (m); the hand is ~0.1 m
_AXES_RADIUS = 0.0012
_MARKER_R = 0.0035         # point marker radius (m)
_ARROW_LEN = 0.035


def _wxyz(R):
    """viser's (w, x, y, z) quaternion for a 3x3 rotation.

    ``trimesh.transformations.quaternion_from_matrix`` already returns wxyz order
    and trimesh is a hard dependency of the renderer next door, so this needs no
    extra package."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    return tuple(float(v) for v in trimesh.transformations.quaternion_from_matrix(T))


def _gap_rgb(gap):
    """The hand renderer's near/far cue, reused so a table clearance and an
    object gap are read the same way (green within GAP_GREEN_MAX_M, including
    interpenetration, which is simply 'not far')."""
    return _GAP_NEAR_RGB if gap < GAP_GREEN_MAX_M else _GAP_FAR_RGB


class ConstraintOverlays:
    """Draws the §1.8 phase goals into a viser scene, one togglable item at a time.

    :attr:`show` maps every :data:`OVERLAYS` key to a bool and is read on each
    :meth:`update`; the visualizer wires its checkboxes straight into it.
    """

    def __init__(self, server, *, root="/goals"):
        self.server = server
        self.scene = server.scene
        self.root = root
        self.show = {s.key: s.default for s in OVERLAYS}
        self._dynamic = {}      # name -> handle, pruned per update

    # -- entry point -------------------------------------------------------

    def update(self, geom):
        """Redraw the enabled overlays from a ``solvers.goal_geometry`` dict.

        ``geom`` may be ``None`` (nothing solved yet), which clears everything.
        Individual items skip themselves when the geometry they need is absent,
        so an overlay stays checked across a phase that has no such quantity and
        comes back when it does.
        """
        if geom is None:
            self._prune(set())
            return
        keep = set()
        for spec in OVERLAYS:
            if not self.show.get(spec.key):
                continue
            keep |= getattr(self, f"_draw_{spec.key}")(geom) or set()
        self._prune(keep)

    def clear(self):
        self._prune(set())

    # -- phase 0 -----------------------------------------------------------

    def _draw_pre_target(self, geom):
        return self._frame("pre_target/T", geom.get("T_pre"), _TARGET_RGB,
                           "T_base,pre")

    def _draw_base_frame(self, geom):
        return self._frame("base_frame/T", geom.get("T_base"), _CURRENT_RGB,
                           "T_base")

    def _draw_waypoint(self, geom):
        return self._frame("waypoint/T", geom.get("waypoint"), _WAYPOINT_RGB,
                           "waypoint")

    def _draw_align_axes(self, geom):
        """The hand's measured triad against the world directions phase 0 matches
        it to: a_hat onto -n_hat and s_hat onto m_hat. Both members of a pair
        share a hue, so alignment is 'the two same-coloured arrows agree'."""
        T = geom.get("T_base")
        if T is None or geom.get("a_hat") is None:
            return set()
        T = np.asarray(T, float)
        origin = T[:3, 3]
        R = T[:3, :3]

        keep = set()
        # The hand's own axes, rotated into world by the achieved base pose.
        for tag in ("a", "g", "s"):
            v = geom.get(f"{tag}_hat")
            if v is None:
                continue
            keep |= self._arrow(f"align_axes/{tag}", origin,
                                R @ np.asarray(v, float), _AXIS_RGB[tag],
                                label=f"{tag}_hat")
        # The world directions they are aimed at. g_hat has no target: the two
        # constraints consume all three rotational DOF, so it is a consequence.
        keep |= self._arrow("align_axes/target_a", origin,
                            -np.asarray(geom["n_hat"], float),
                            _AXIS_RGB["a"], label="-n_hat", scale=0.7)
        keep |= self._arrow("align_axes/target_s", origin,
                            np.asarray(geom["m_hat"], float),
                            _AXIS_RGB["s"], label="m_hat", scale=0.7)
        return keep

    def _draw_hover_point(self, geom):
        hover, center = geom.get("hover_point"), geom.get("object_center")
        if hover is None or center is None:
            return set()
        keep = self._point("hover_point/p", hover, _HOVER_RGB)
        keep |= self._segment("hover_point/h_clear", center, hover, _HOVER_RGB,
                              label=f"h_clear {geom['h_clear'] * 1000.0:.0f} mm")
        return keep

    def _draw_centroid(self, geom):
        """The live contact-sphere centroid, and its offset from the hover point
        -- that segment is the Eq 1.93 position error phase 0 is driving out."""
        c = geom.get("centroid")
        if c is None:
            return set()
        keep = self._point("centroid/p", c, _CENTROID_RGB)
        hover = geom.get("hover_point")
        if hover is not None:
            err = float(np.linalg.norm(np.asarray(hover, float) - c))
            keep |= self._segment("centroid/err", c, hover, _CENTROID_RGB,
                                  label=f"{err * 1000.0:.1f} mm")
        return keep

    # -- phase 1 -----------------------------------------------------------

    def _draw_plane_lines(self, geom):
        return self._measure_set("plane_lines", geom.get("plane_gaps"),
                                 labels=True)

    def _draw_free_plane_lines(self, geom):
        # No labels: one per sphere is unreadable, and the colour already says
        # which ones are near or through the plane.
        return self._measure_set("free_plane_lines", geom.get("free_plane_gaps"),
                                 labels=False)

    def _draw_split(self, geom):
        keep = self._point("split/p_split", geom["p_split"], _SPLIT_RGB)
        for name, (c, m_i, c_half) in (geom.get("half_gaps") or {}).items():
            # c_half <= 0 is satisfied; colour it like a gap so "green is fine"
            # keeps meaning the same thing across every overlay.
            rgb = _GAP_NEAR_RGB if c_half <= 0.0 else _GAP_FAR_RGB
            keep |= self._arrow(f"split/{name}", c, m_i, rgb,
                                label=f"{c_half * 1000.0:+.1f} mm")
        return keep

    def _draw_split_plane(self, geom):
        """The splitting plane through ``p_split``, perpendicular to ``m_hat``.
        Purely a visual aid -- the constraint is the analytic half-space."""
        m = np.asarray(geom["m_hat"], float)
        m = m / (np.linalg.norm(m) or 1.0)
        extents = [0.25, 0.25, 0.25]
        extents[int(np.argmax(np.abs(m)))] = 0.002    # thin along m_hat
        n = f"{self.root}/split_plane/slab"
        self._dynamic[n] = self.scene.add_box(
            n, color=_SPLIT_RGB, dimensions=tuple(extents), opacity=0.35,
            position=tuple(float(v) for v in geom["p_split"]))
        return {n}

    # -- phase 2 -----------------------------------------------------------

    def _draw_proxy(self, geom):
        a, b, c = (float(v) for v in geom["proxy_semi_axes"])
        n = f"{self.root}/proxy/surface"
        self._dynamic[n] = self.scene.add_icosphere(
            n, radius=1.0, scale=(a, b, c), color=_PROXY_RGB, opacity=0.25,
            wireframe=True, wxyz=_wxyz(geom["object_rotation"]),
            position=tuple(float(v) for v in geom["object_center"]))
        return {n}

    # -- phase 3 -----------------------------------------------------------

    def _witness_points(self, geom):
        """``{name: (sphere_centre, witness_pt)}``, preferring the solver's own
        Symbol('Y', i) over the analytic surface projection.

        The solved witness exists only in phase 3; everywhere else (and on a
        binding without the accessor) this falls back to the analytic point,
        which tracks the surface but is not the variable being optimised.
        """
        analytic = geom.get("object_witness") or {}
        solved = geom.get("witness_points")
        out = {}
        for i, name in enumerate(geom["contact_names"]):
            if name not in analytic:
                continue
            sphere_pt, surface_pt, _ = analytic[name]
            p = None
            if solved is not None and i < len(solved):
                p = solved[i]
            out[name] = (np.asarray(sphere_pt, float),
                         np.asarray(surface_pt if p is None else p, float),
                         p is not None)
        return out

    def _draw_witness(self, geom):
        keep = set()
        for name, (sphere_pt, witness, exact) in self._witness_points(geom).items():
            keep |= self._point(f"witness/{name}/p", witness, _WITNESS_RGB)
            # c_R (Eq 1.114) is |p - c| - r; the analytic sphere_pt is already
            # the point on the sphere, so the drawn segment IS the residual.
            d = float(np.linalg.norm(witness - sphere_pt))
            keep |= self._segment(
                f"witness/{name}/c_R", sphere_pt, witness, _gap_rgb(d),
                label=f"{d * 1000.0:.1f} mm{'' if exact else ' ~'}")
        return keep

    def _draw_witness_frames(self, geom):
        keep = set()
        for name, (p, n_hat, t1, t2) in (geom.get("contact_frames") or {}).items():
            for tag, v, rgb in (("N", n_hat, _AXIS_RGB["a"]),
                                ("t1", t1, _AXIS_RGB["g"]),
                                ("t2", t2, _AXIS_RGB["s"])):
                keep |= self._arrow(f"witness_frames/{name}/{tag}", p, v, rgb,
                                    scale=0.5)
        return keep

    def _draw_witness_targets(self, geom):
        targets = geom.get("witness_targets")
        if not targets:
            return set()
        keep = set()
        current = self._witness_points(geom)
        for i, name in enumerate(geom["contact_names"]):
            if i >= len(targets) or targets[i] is None:
                continue
            t = np.asarray(targets[i], float).reshape(3)
            keep |= self._point(f"witness_targets/{name}/p", t, _TARGET_RGB)
            if name in current:
                # The Eq 1.118 prior acts as a geodesic pull along the surface.
                keep |= self._segment(f"witness_targets/{name}/pull",
                                      current[name][1], t, _TARGET_RGB)
        return keep

    # -- primitives --------------------------------------------------------

    def _frame(self, key, T, rgb, label):
        """An axis triad with a coloured origin ball. The triad's own RGB arms are
        fixed by viser, so the ball and the label are what tell two frames
        apart."""
        if T is None:
            return set()
        T = np.asarray(T, float)
        n = f"{self.root}/{key}"
        self._dynamic[n] = self.scene.add_frame(
            n, show_axes=True, axes_length=_AXES_LEN, axes_radius=_AXES_RADIUS,
            origin_radius=_MARKER_R, origin_color=rgb, wxyz=_wxyz(T[:3, :3]),
            position=tuple(float(v) for v in T[:3, 3]))
        keep = {n}
        keep |= self._label(f"{key}_lbl", label, T[:3, 3])
        return keep

    def _point(self, key, p, rgb, radius=_MARKER_R):
        n = f"{self.root}/{key}"
        self._dynamic[n] = self.scene.add_icosphere(
            n, radius=float(radius), color=rgb, opacity=0.9,
            position=tuple(float(v) for v in np.asarray(p, float).reshape(3)))
        return {n}

    def _segment(self, key, p0, p1, rgb, *, label=None, width=3.0):
        p0 = np.asarray(p0, float).reshape(3)
        p1 = np.asarray(p1, float).reshape(3)
        n = f"{self.root}/{key}"
        self._dynamic[n] = self.scene.add_line_segments(
            n, np.stack([p0, p1])[None], colors=rgb, line_width=width)
        keep = {n}
        if label is not None:
            keep |= self._label(f"{key}_lbl", label, 0.5 * (p0 + p1))
        return keep

    def _arrow(self, key, origin, direction, rgb, *, label=None, scale=1.0):
        """A segment with a small head, since viser has no arrow primitive.
        ``direction`` need not be normalized -- only its direction is used."""
        origin = np.asarray(origin, float).reshape(3)
        d = np.asarray(direction, float).reshape(3)
        d = d / (np.linalg.norm(d) or 1.0)
        tip = origin + _ARROW_LEN * scale * d
        keep = self._segment(key, origin, tip, rgb, label=label)
        keep |= self._point(f"{key}_head", tip, rgb, radius=0.0015)
        return keep

    def _measure_set(self, key, gaps, *, labels):
        """A batch of signed-distance measurements drawn as one line-segments node
        (one draw call for all of them) plus, optionally, a label per entry."""
        if not gaps:
            return set()
        segs, colors, keep = [], [], set()
        for name, (p0, p1, gap) in gaps.items():
            p0 = np.asarray(p0, float).reshape(3)
            p1 = np.asarray(p1, float).reshape(3)
            segs.append([p0, p1])
            colors.append([_gap_rgb(gap)] * 2)
            if labels:
                keep |= self._label(f"{key}/{name}_lbl",
                                    f"{gap * 1000.0:+.1f} mm", 0.5 * (p0 + p1))
        n = f"{self.root}/{key}/lines"
        self._dynamic[n] = self.scene.add_line_segments(
            n, np.asarray(segs, float),
            colors=np.asarray(colors, dtype=np.uint8), line_width=3.0)
        keep.add(n)
        return keep

    def _label(self, key, text, position):
        n = f"{self.root}/{key}"
        self._dynamic[n] = self.scene.add_label(
            n, text, position=tuple(float(v) for v in
                                    np.asarray(position, float).reshape(3)),
            anchor="center-center")
        return {n}

    def _prune(self, keep):
        """Drop handles not re-added this update -- an overlay toggled off, or a
        quantity that does not exist in the current phase."""
        for n in list(self._dynamic):
            if n not in keep:
                handle = self._dynamic.pop(n)
                try:
                    handle.remove()
                except Exception:
                    pass
