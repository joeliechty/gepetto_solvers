"""A docked viser panel that plots the CONTROL TRAJECTORY of a hand solve.

Eleven stacked subplots -- the five actuated (flexor) TENDON LENGTHS and the
solved wrist pose broken into x/y/z/roll/pitch/yaw -- against the iteration the
solve is on. Sample 0 is where the run started (the FK pose on screen, i.e. the
current kinematics), and every subsequent sample is one Augmented Lagrangian
outer iteration, so the panel fills in live as ``Auto solve`` runs and, when the
solve ends, holds the whole path the hand took to get there.

LENGTHS, NOT TENSIONS. The tension is what the solve is *asked* for; the length
is what the hand actually took in, and it is the half of the state the hardware
is commanded on -- ``robot_plan`` builds every waypoint out of
``open_lengths[name] - length``. So this is the trajectory in the units the
robot moves in, and it can be read straight against the *Tensions* folder's
per-finger length readout, which prints the same number for the frame on screen.

WHY ELEVEN AND NOT SIX. The robot is commanded with six things -- five tendons
and one wrist pose -- but a pose is not plottable as a scalar, so the pose is
split into its xyzrpy components here. That split is a VIEWING choice
and is not what the trajectory is interpolated in downstream: ``robot_plan``
paces the wrist through ``se3_log``/``se3_exp``, so between two knots the real
path is a screw motion and the straight line this panel draws between two
roll samples is the projection of that, not the thing itself. It is the right
picture for "did the wrist wander", the wrong one for "what exactly does the arm
do between iterate 7 and 8".

The lines ARE the linear interpolation between states: uPlot joins consecutive
samples with a straight segment, and the knots are drawn as points on top, so
the two are distinguishable at a glance -- a long flat run with two dots at its
ends is one AL iteration that moved nothing, not a dense stretch of samples.

UNITS FOLLOW WHATEVER ALREADY PRINTS THE SAME QUANTITY. Tendon lengths in mm,
matching the *Tensions* folder's length table (``_report_tendon_lengths``);
wrist position in m and angles in rad, matching the *Wrist start pose* sliders,
so a pose number picked off a plot can be typed straight into the slider that
commands it. Neither is converted on the way in, so nothing here can disagree
with the readout beside it.

Colours match the 3D scene where there is a match to make: the five tendon
traces reuse ``viser_hand._FINGER_PLANE_RGB`` in ``finger_names`` order, and
x/y/z reuse the red/green/blue of the triads the scene draws, with roll/pitch/
yaw dashed in the same hue as the axis each turns about.

The panel is a pure sink: it is handed an ``(N, 11)`` array of numbers and knows
nothing about solvers, results or iterates. Extracting that array from a
``HandResult`` is the caller's job -- see ``viz_interactive._traj_samples``.
"""

import numpy as np

from .viser_hand import _FINGER_PLANE_RGB


def _css(rgb):
    """An (r, g, b) 0-255 tuple as the CSS string uPlot wants."""
    return "rgb(%d, %d, %d)" % tuple(int(c) for c in rgb)


# The triad colours, so "x" on a plot is the same red as the x arrow in the
# scene. Rotations are the axis they turn about, which is why roll/pitch/yaw
# repeat red/green/blue -- they are told apart by the dashed stroke, not the hue.
_AXIS_RGB = ((214, 62, 62), (60, 176, 92), (62, 118, 214))

# One entry per subplot, in plot order: (title, dashed?). The five tension rows
# are built per finger at construction, since their labels come from the caller.
_POSE_CHANNELS = (("wrist x (m)", 0, False), ("wrist y (m)", 1, False),
                  ("wrist z (m)", 2, False), ("wrist roll (rad)", 0, True),
                  ("wrist pitch (rad)", 1, True), ("wrist yaw (rad)", 2, True))

# Height of one subplot in px. Eleven of them do not fit a laptop viewport, and
# that is fine -- the panel scrolls. Shrinking them to fit would cost the
# vertical resolution that makes a 2 mm wrist drift visible, which is the whole
# reason for looking.
_PLOT_HEIGHT = 112


class TrajectoryPanel:
    """The docked plot window. One instance per app; built once, updated often.

    ``update`` is called from the solve worker thread as well as from viser
    callback threads (the same way ``ViserHandScene.update`` is), which is safe
    because assigning to a uPlot handle's ``data`` only queues a message.
    """

    #: Number of columns the ``values`` array handed to :meth:`update` must have:
    #: five tendon lengths (mm) then x, y, z (m), roll, pitch, yaw (rad).
    N_CHANNELS = 11

    def __init__(self, server, finger_labels, *, width=340, visible=True):
        self.server = server
        self.finger_labels = list(finger_labels)
        self._n = 0
        self.panel = server.gui.add_panel(visible=visible)
        # The icon as its literal string rather than via ``viser.Icon``: this
        # module, like ``viser_hand``, is handed a server and never imports
        # viser itself, and ``viser.Icon.CHART_LINE`` IS "chart-line".
        with self.panel.add_tab("Trajectory", "chart-line"):
            # One readout above the stack rather than a live legend on each of
            # eleven charts: the legends would cost more vertical space than the
            # plots they annotate, and the numbers are wanted together anyway --
            # a control vector is read across, not one component at a time.
            # (Vertical space is the binding constraint here: eleven subplots
            # already scroll, and the plot area is what carries the resolution
            # that makes a sub-millimetre drift visible.)
            self.header = server.gui.add_markdown(self._IDLE)
            self.plots = []
            for label, rgb in zip(self.finger_labels, _FINGER_PLANE_RGB):
                self.plots.append(self._add_plot(f"{label} tendon (mm)",
                                                 _css(rgb), dashed=False))
            for title, axis, dashed in _POSE_CHANNELS:
                self.plots.append(self._add_plot(title, _css(_AXIS_RGB[axis]),
                                                 dashed=dashed))
        # Left, as asked, and narrow: this is a companion to the 3D view, not a
        # replacement for it, so it must not eat the viewport the hand is in.
        self.panel.dock_left()
        self.panel.set_width(width)

    _IDLE = ("*press **FK** for the current state, then **Step** / **Auto "
             "solve** to trace the trajectory*")

    def _add_plot(self, title, stroke, dashed):
        """One subplot: the trajectory series plus a marker series for the
        iterate the convergence scrubber is parked on.

        The marker is a second series that is NaN everywhere except at the
        scrubbed index, which is how uPlot draws a single isolated point -- there
        is no annotation/vline API to do it directly, and a whole second chart
        overlaid would not share the first's autoscaled y range."""
        series = (
            {"label": "iterate"},
            {"label": title, "stroke": stroke, "width": 1.6,
             "dash": (4.0, 3.0) if dashed else (),
             # spanGaps off so a NaN reads as a hole rather than being bridged:
             # a channel that could not be recovered from an iterate is a fact
             # worth seeing, not one to interpolate over.
             "spanGaps": False,
             "points": {"show": True, "size": 5.0, "stroke": stroke,
                        "fill": stroke}},
            # Drawn as points only (width 0): it is a cursor, not a trace.
            #
            # ``auto: False`` is NOT cosmetic and must not be dropped. This
            # series is NaN at every sample except the marked one, and a series
            # holding NaN at its first sample makes uPlot's y auto-range come
            # out null -- which silently kills the WHOLE subplot, trace
            # included, because both series share the "y" scale. Measured in a
            # headless browser: 420 painted pixels (the axes alone) against 1902
            # with this flag set. `auto` takes the series out of the scale's
            # range calculation while still drawing it against that scale, which
            # is exactly right here -- the marker must line up with the trace,
            # and a lone dot should never be what sets the y limits anyway.
            {"label": "shown", "stroke": "rgb(255, 255, 255)", "width": 0.0,
             "auto": False,
             "points": {"show": True, "size": 9.0,
                        "stroke": "rgb(255, 255, 255)", "fill": stroke}},
        )
        return self.server.gui.add_uplot(
            data=(np.zeros(1), np.full(1, np.nan), np.full(1, np.nan)),
            series=series, title=title, height=_PLOT_HEIGHT,
            # Integers only on the x axis: the samples ARE iteration indices, and
            # a "3.5th iterate" does not exist.
            scales={"x": {"time": False}},
            axes=({"scale": "x", "incrs": (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)},
                  {"scale": "y"}),
            legend={"show": False},
            # Drag-to-zoom on x is worth keeping -- a 60-iterate solve where the
            # interesting part is iterates 40-45 is the normal case.
            cursor={"drag": {"x": True, "y": False}},
            padding=(0, 10, 0, 0))

    def update(self, values, cursor=None, note=""):
        """Redraw every subplot from ``values``, an ``(N, N_CHANNELS)`` array.

        ``cursor`` is the sample index the 3D view is currently showing (the
        convergence scrubber's position), marked on each plot with a dot; None
        leaves it unmarked. ``note`` is appended to the header line.
        """
        values = np.asarray(values, float)
        if values.ndim != 2 or values.shape[1] != self.N_CHANNELS:
            raise ValueError(
                f"expected an (N, {self.N_CHANNELS}) array, got {values.shape}")
        n = len(values)
        self._n = n
        # uPlot needs a length to draw against; one NaN sample is an empty chart
        # that still has axes, where a zero-length array is a frontend error.
        if n == 0:
            x, values = np.zeros(1), np.full((1, self.N_CHANNELS), np.nan)
            n = 1
            cursor = None
        else:
            x = np.arange(n, dtype=float)
        mark = np.full(n, np.nan)
        if cursor is not None and 0 <= int(cursor) < n:
            mark[int(cursor)] = 1.0
        for i, plot in enumerate(self.plots):
            col = values[:, i]
            plot.data = (x, col, col * mark)
        self.header.content = self._header(values, cursor, note)

    def _header(self, values, cursor, note):
        """The value of all eleven channels at the marked sample, as one block.

        Reads the SHOWN sample rather than the last one so it agrees with the 3D
        view: scrubbing back to iterate 3 must not leave a readout describing
        iterate 30."""
        n = self._n
        if n == 0:
            return self._IDLE
        i = int(cursor) if cursor is not None and 0 <= int(cursor) < n else n - 1
        row = values[i]
        lines = [f"**sample {i} / {n - 1}**" + (f" &nbsp; {note}" if note else "")]
        # Code spans so the columns survive markdown's whitespace collapsing --
        # the same trick _report_tendon_lengths uses for its length table.
        for label, q in zip(self.finger_labels, row[:5]):
            lines.append(f"`{label:<6} {q:7.2f} mm`")
        for label, q in zip(("x", "y", "z"), row[5:8]):
            lines.append(f"`{label:<6} {q:+7.4f} m`")
        for label, q in zip(("roll", "pitch", "yaw"), row[8:11]):
            lines.append(f"`{label:<6} {q:+7.4f} rad`")
        return "  \n".join(lines)

    def clear(self):
        """Blank every subplot -- nothing solved, nothing to show."""
        self.update(np.zeros((0, self.N_CHANNELS)))

    def error(self, exc):
        """Report a failed extraction in the header, leaving the plots alone.

        This panel is a READOUT hanging off the render path, and the render path
        runs inside the auto-solve loop. A raise here would abort a solve over a
        plotting bug, so the caller catches instead -- but a caught exception
        that goes nowhere is the worse failure, because the panel then just
        quietly stops updating. So it lands here, on screen, next to the stale
        plots it explains."""
        self.header.content = (f"**trajectory unavailable**  \n`{exc}`")

    def set_visible(self, visible):
        self.panel.visible = bool(visible)

    def remove(self):
        self.panel.remove()
