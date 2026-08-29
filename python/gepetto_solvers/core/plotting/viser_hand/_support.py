"""The support plane, its grid, and the opposition half-space.

A mixin of :class:`~gepetto_solvers.core.plotting.viser_hand.scene.ViserHandScene`.
Split out of what was one 968-line class; the methods here use only ``self.scene``
and ``self._dynamic``, which the composed class owns.
"""

import numpy as np

from .palette import (
    _CONTACT_RGB,
    _HALF_SPACE_RGB,
    _TABLE_GRID_RGB,
    _TABLE_RGB,
)


class SupportSceneMixin:
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


    def set_constraint_plane(self, origin, normal, *, span, thickness=0.001):
        """Draw the plane the SOLVER constrains against, when it has been raised
        off the table surface :meth:`set_table` draws.

        A separate node from ``/table`` because the two are separate things: the
        slab is the physical bench the robot is registered to and never moves for
        a planning reason, while this is where the support equality seats
        fingertips and where the avoidance half-space begins. Drawn thinner, more
        transparent and in the contact colour so it reads as a constraint rather
        than as a second table -- and so a viewer can tell at a glance which
        surface the fingers are actually stopping on.

        The caller decides when to draw it: coincident with the table it is
        nothing but z-fighting, so the app clears it at zero height.
        """
        origin = np.asarray(origin, float).reshape(3)
        n = np.asarray(normal, float).reshape(3)
        axis = int(np.argmax(np.abs(n)))
        extents = [span, span, span]
        extents[axis] = thickness
        self.scene.add_box("/constraint_plane", color=_CONTACT_RGB,
                           dimensions=tuple(extents), opacity=0.2,
                           position=tuple(origin))


    def clear_constraint_plane(self):
        try:
            self.server.scene.remove_by_name("/constraint_plane")
        except Exception:
            pass


    def set_table_grid(self, origin, normal, *, span, spacing):
        """Rule the table's top face into a grid, matching the one drawn on the
        physical bench.

        The point is comparison by eye: the real table carries a grid at this
        spacing, so a landmark commanded to an intersection here can be read
        against the same intersection there without measuring anything. Lines run
        corner to corner of the same square :meth:`set_table` draws -- the
        in-plane axes come from the same dominant-normal rule -- and are lifted a
        hair ABOVE the top face, because coplanar geometry z-fights the
        translucent slab and the grid flickers in and out as the camera moves.

        ``origin`` is the PLANE origin -- the same argument :meth:`set_table`
        takes, which sits at the middle of the square, not at its corner. Taking
        it in that form rather than pre-cornered is deliberate: the two methods
        are called with the same value from the same place, so the grid cannot end
        up describing a different square from the slab it is ruled on.

        ``span``/``spacing`` are required for the reason :meth:`set_table`'s
        dimensions are: a second copy of the bench's numbers living here would be
        free to drift from the caller's.
        """
        origin = np.asarray(origin, float).reshape(3)
        n = np.asarray(normal, float).reshape(3)
        axis = int(np.argmax(np.abs(n)))
        u, v = [i for i in range(3) if i != axis]
        # The minimum corner, by the same rule ``scene.table_corner`` uses, lifted
        # clear of the top face.
        base = origin.copy()
        base[u] -= span / 2.0
        base[v] -= span / 2.0
        base[axis] += np.sign(n[axis]) * 5e-4

        # Inclusive of both edges, so the square's own border is a grid line and
        # the count reads as span/spacing + 1 the way a ruled sheet does.
        steps = int(round(span / spacing))
        segments = []
        for i in range(steps + 1):
            offset = i * spacing
            for along, across in ((u, v), (v, u)):
                a, b = base.copy(), base.copy()
                a[along] += offset
                b[along] += offset
                b[across] += span
                segments.append([a, b])
        colors = np.tile(np.array(_TABLE_GRID_RGB, dtype=np.uint8),
                         (len(segments), 2, 1))
        self.scene.add_line_segments(
            "/table_grid", points=np.array(segments, dtype=np.float32),
            colors=colors, line_width=1.5)


    def clear_table_grid(self):
        try:
            self.server.scene.remove_by_name("/table_grid")
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


    def clear_half_space_plane(self):
        for name in ("/half_space_plane", "/half_space_margin_pos",
                     "/half_space_margin_neg"):
            try:
                self.server.scene.remove_by_name(name)
            except Exception:
                pass
