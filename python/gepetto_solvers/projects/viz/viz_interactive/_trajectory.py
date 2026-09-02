"""The trajectory panel and the robot trace it overlays.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""


import numpy as np

from gepetto_solvers.core.geometry.scene import table_corner
from gepetto_solvers.core.solvers import R_to_euler


class TrajectoryMixin:
    # -- control-trajectory panel (left-docked plot window) --
    #
    # The six things this robot is commanded with -- five actuated tendons and
    # one wrist pose -- plotted against the iteration the solve is on, in the
    # window traj_panel.TrajectoryPanel owns. Everything below is
    # EXTRACTION: the panel is handed plain numbers and knows nothing about
    # results or iterates, which is what keeps the solver vocabulary on this side
    # of the line and the plotting on that one.

    def _traj_row(self, res):
        """The eleven control numbers of ONE solved state, in panel order:
        five actuated tendon lengths in mm, then the wrist as xyz (m) + rpy (rad).

        The LENGTH, not the tension that produced it. The tension is what the
        solve was asked for; the length is what the hand took in, it is the L
        half of the state a Section 1.8 control tick anchors on, and it is what
        the hardware is actually commanded on -- `robot_plan.build_plan` turns
        each waypoint into ``open_lengths[name] - length``. Same array
        `_report_tendon_lengths` prints under the tension sliders, and in the
        same mm, so the plot and that readout cannot disagree.

        Everything is re-read from the RESULT rather than from the sliders that
        commanded it, for the reason `_report_tendon_lengths` gives: past the
        first iterate neither the wrist nor the tendon is the slider's any more.
        The wrist is a variable with a soft prior, so a contact solve ends a long
        way from the commanded pose -- which is precisely the drift this panel
        exists to make visible. Reading the sliders would draw flat lines.

        The wrist also has to be RECOVERED rather than read: nothing in a result
        reports it directly, so the state bundle carries the solved wrist
        offset out of its node-0 pose. Split into xyzrpy here because a 4x4 is
        not plottable, using the same ZYX convention (and the same radians) the
        Wrist start pose sliders use, so a number read off a plot goes straight
        back into the slider it came from.
        """
        T = np.asarray(res.wrist_pose(0),
                       float)
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        lengths = [float(np.asarray(length, float)[self._drive_index()]) * 1e3
                   for length in res.displacements(0)]
        return lengths + [T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw]


    def _robot_traj_row(self, state):
        """One MEASURED robot state as the panel's eleven channels.

        The exact inverse of what `_traj_row` reads off a solve, in the same
        units and the same order, because the whole point is to draw the two on
        one axis: five actuated tendon LENGTHS in mm, then the wrist as xyz (m)
        and rpy (rad) in the viser world frame.

        The hardware reports DISPLACEMENT from the hand-open pose and
        `robot_plan.build_plan` commands it as ``open_lengths[name] - length``,
        so recovering a length is ``open_lengths[name] - displacement`` -- the
        same identity read backwards. Doing it here rather than plotting the
        displacement directly is what makes the measured line comparable to the
        commanded one instead of being a differently-zeroed cousin of it.

        A finger the hardware could not report is left NaN rather than zeroed: a
        motor whose position read failed is a hole in the measurement, and
        `spanGaps: False` on the series draws it as one. Zero would draw as a
        fully open finger, which is a claim about the hand rather than an
        admission that nothing was heard.
        """
        T = np.asarray(state.wrist_pose, float)
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        open_lengths = self._open_lengths()
        # The ORDER has to be the result's own, because that is the order
        # `_traj_row` reads `tendon_lengths(0)` in and therefore the order the
        # panel's first five channels are in. Falling back to self.digit_names only
        # covers the case where nothing is solved, where there is no plot to
        # align with anyway.
        names = (list(self.result.finger_names)
                 if self.result is not None else list(self.digit_names))
        lengths = []
        for name in names:
            disp = state.tendon_disp.get(name)
            lengths.append(np.nan if disp is None or name not in open_lengths
                           else (open_lengths[name] - float(disp)) * 1e3)
        # Fixed to five, so a result carrying a different number of digits can
        # never slide the wrist channels along and draw them on a tendon plot.
        lengths = (lengths + [np.nan] * 5)[:5]
        return lengths + [T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw]


    def _sample_robot_trace(self, feedback):
        """Record where the robot IS, against the waypoint the plan is heading to.

        Called about ten times a second off the playback feedback, on the action
        client's callback thread. Cheap on purpose -- two TF lookups and a cached
        tendon reading -- because it runs while the robot is moving and must not
        become a reason the feedback path falls behind.

        Keyed by WAYPOINT rather than by time, and the last sample for a waypoint
        wins. That makes entry ``k`` "where the machine was as the reference
        finished waypoint k", which is exactly the quantity the plot is being
        asked for: the gap between the two lines at ``k`` is the tracking error
        at that waypoint. Time would have to be resampled onto the iterate grid
        to be plotted at all, and would answer a question nobody asked.

        The feedback's waypoint index is already the CLIENT's -- the executor
        takes its own prepended approach waypoint back off -- so it indexes the
        iterates directly, with no offset to get wrong here.
        """
        trace = self._robot_trace
        if trace is None:
            return          # not collecting: not a history playback
        try:
            state = self.bridge.read_state(self._corner_viz())
            trace[int(feedback.waypoint)] = self._robot_traj_row(state)
        except Exception:
            # Diagnostics on the feedback path of a moving robot. A failed TF
            # lookup mid-run is a missing sample, not a reason to raise into the
            # action client's callback.
            pass


    def _robot_trace_array(self, n):
        """The recorded trace as an ``(n, 11)`` array, NaN where nothing landed.

        Returns None when there is nothing to draw, which the panel takes as "no
        measured line" -- distinct from an all-NaN array, which would mean a
        playback happened and recorded nothing.
        """
        trace = getattr(self, "_robot_trace", None)
        if not trace:
            return None
        out = np.full((n, self.traj.N_CHANNELS), np.nan)
        hit = False
        for index, row in trace.items():
            if 0 <= index < n:
                out[index] = row
                hit = True
        return out if hit else None


    def _traj_samples(self):
        """The whole trajectory on screen as an ``(N, 11)`` array.

        The recorded AL iterates when there are any -- sample 0 is where the run
        started, which under the warm-start latch IS the FK pose that was on
        screen when Step was first pressed, i.e. the current kinematics -- and
        the single solved state when there are none, which is what an FK pose
        is. So the panel shows a lone dot after FK and grows a line from it as
        the solve steps, with no special-casing at the boundary.

        Recomputed in full on every render rather than appended to. Measured at
        1.1 ms over a 26-iterate solve -- 1.6 ms including the eleven uplot
        pushes, against a `_render_frame` that costs 52 ms -- so it is 3% of a
        redraw, and nowhere near worth a cache that would have to know about cold
        restarts, Close/Lift overwriting the history, and the scrubber: three
        separate ways to serve a stale plot.

        Close and Lift record their ramp substeps as iterates too, so those get
        plotted by exactly the same path: a phase-4 close draws as five tendon
        lengths ramping together, which is the claim in its name made visible."""
        res = self.result
        if res is None or getattr(self, "fk_solver", None) is None:
            return np.zeros((0, self.traj.N_CHANNELS))
        n = res.num_iterates()
        views = [res.at_iterate(i) for i in range(n)] if n else [res]
        return np.array([self._traj_row(v) for v in views], float)


    def _traj_cursor(self, n, live):
        """Which sample the 3D view is showing, for the marker dot.

        Follows `_iter_view` exactly, because agreeing with it is the point: the
        dot claims "this plotted sample is the hand you are looking at", and the
        one case where that is easy to get wrong is mid-run, where the scrubber
        still describes the PREVIOUS solve and the render is drawing the newest
        state regardless."""
        if n == 0:
            return None
        if live or getattr(self, "iter_slider", None) is None:
            return n - 1
        return min(self._current_iterate(), n - 1)


    def _update_traj(self, live=False):
        """Redraw the trajectory panel for whatever is on screen.

        Called from `_render_frame`, so it follows the live re-solve, every AL
        step (from the auto-solve worker thread, which is safe -- assigning to a
        uPlot handle only queues a message) and the convergence scrubber alike.

        Exceptions are caught and shown IN the panel rather than raised: this is
        a readout on the render path, and that path runs inside the auto-solve
        loop, so a raise here would end a solve over a plotting bug."""
        panel = getattr(self, "traj", None)
        if panel is None:
            return          # rendered before __init__ built it
        try:
            values = self._traj_samples()
            n = len(values)
            panel.update(
                values, cursor=self._traj_cursor(n, live),
                actual=self._robot_trace_array(n),
                # Worth saying only in the case that looks like a broken panel:
                # one dot and no line is a correct picture of an FK pose.
                note=("current kinematics (FK pose)"
                      if n == 1 and self.result.num_iterates() == 0 else ""))
        except Exception as exc:
            panel.error(exc)


    def _corner_viz(self):
        """The table square's minimum corner in viser world coordinates -- this
        app's half of the registration against the physical bench.

        Read live rather than cached: the support plane is seated UNDER the
        object (see ``auto_table_origin``), so switching objects moves it, and a
        stale corner would silently offset every pose sent to the robot by
        however far the table had moved."""
        return table_corner(self._table_origin(),
                            np.asarray(self.params.plane_normal, float))
