"""Phases 4 and 5: closing the hand, and lifting it.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import threading

import numpy as np

from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.solvers import FLEXOR_IDX, lift_wrist, synchronized_close

from .constants import (
    FINGER_LABELS,
)
from .estop import Refused


class MotionMixin:
    # -- phase 4: the synchronized close --

    def _close_hand(self, _=None):
        """Phase 4: shut every ticked contact finger together, recording the ramp.

        Threaded and gated exactly like Auto solve, and for the same two reasons:
        the gate is what stops a close and a solve overlapping (they share the FK
        solver, its warm start and ``self.result``), and a viser callback thread
        that blocks for the length of a close -- a few seconds of FK solves --
        cannot service the E-STOP click.

        The stop INTERRUPTS this one rather than merely refusing it, like
        auto-solve and unlike Step: the walk polls between poses, so a trip lands
        within one FK solve (~100 ms, not the AL loop's ~1.7 s) and keeps every
        pose recorded so far. See :func:`~.solvers.synchronized_close`.
        """
        try:
            gate = self.estop.admit("synchronized close")
        except Refused:
            return
        try:
            self._set_solving(True)
            self._sync_params()
            self._refresh_object()
        except Exception as exc:
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise

        def worker():
            try:
                self._close_admitted()
            except Exception as exc:
                self._error_status(exc)
            finally:
                # Gate first, so the refreshes below read an idle latch and can
                # hand the controls back -- the auto-solve worker's ordering.
                gate.release()
                self._set_solving()
                self._report_estop()

        self._close_thread = threading.Thread(target=worker, daemon=True)
        try:
            self._close_thread.start()
        except Exception as exc:
            # A gate never released refuses every solve for the rest of the
            # session; the same failure _ik_auto guards against.
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise


    def _close_admitted(self):
        """The close itself, for a caller that ALREADY HOLDS the gate."""
        fingers = [label for label, box in zip(FINGER_LABELS, self.g_contacts)
                   if box.value]
        if not fingers:
            self._set_status(
                "**nothing to close** -- tick at least one digit under "
                "*Constraints / fingers* first. Phase 4 closes the GRASPING "
                "set, and leaves every other finger where it is.")
            return
        # The motors' travel, not the model's: a close planned against what the
        # rod can bend would spend its last waypoints past the hardware stop,
        # which the plan export clamps and the servo node saturates. None (no
        # gepetto_core) is passed straight through -- synchronized_close falls
        # back to the model's reach and says in its notes that it did.
        limits = robot_plan.hardware_travel_limits()
        travel = None if limits is None else {name: hi
                                              for name, (_lo, hi) in limits.items()}
        # Resolved BEFORE the carry below, deliberately: the open lengths are
        # measured by FK solves at the CALIBRATED OPEN tensions, which leave the
        # solver's retained values on an open hand and would eat the posture the
        # carry commits. Cached, so this ordering costs nothing after the first
        # close.
        open_lengths = self._open_lengths()
        carried = self._carry_solve_into_fk()
        result, notes = synchronized_close(
            self.fk_solver, open_lengths, fingers, travel,
            fraction=self.g_close_frac.value,
            # The tension ceiling is the flexor slider's own top, so a close can
            # never command a pull the panel could not have been dragged to.
            tension_ceiling=float(self.g_flexors[0].max),
            on_progress=self._set_status,
            should_stop=self.estop.is_tripped)
        if result is None:
            self._set_status("  \n".join(notes))
            return

        # Not "FK": _live_fk re-solves on every tension-slider drag while the
        # mode is FK, and the first such drag would throw the recorded ramp away
        # -- along with the scrubber and everything the Robot folder plays. The
        # close is a recorded path, like a stepped solve, so it is held the way
        # one is: press FK (or Close again) to move on from it.
        self.mode = "Close"
        self.result = result
        # A close re-poses the hand outside the AL loop, so any multipliers a
        # stepper is holding describe a hand that no longer exists.
        self._invalidate_stepper()
        self._rebuild_iter_slider()
        # Move the sliders onto the tensions the close ended at, for the reason
        # _adopt_solved_tensions exists: after it they no longer describe where
        # the hand is, and the next FK solve would haul the fingers back open.
        # After the scrubber rebuild, so it reads the LAST pose rather than
        # whichever index the previous solve's slider was parked on.
        self._adopt_solved_tensions()
        self._render_frame()
        # Where to go from here. Both halves are non-obvious: the mode change
        # above quietly turns the live re-solve off, which is what keeps the
        # recorded ramp alive for the Robot folder to play. The scrubber parking
        # on the last pose no longer matters to playback -- it decides what is
        # drawn, not what is sent (see robot_plan.build_plan).
        self._set_status(
            f"**Phase 4 close** &nbsp; {self._fingers_label(fingers)}  \n"
            + "  \n".join(([carried] if carried else []) + notes)
            + "  \nthe tension sliders now hold the close, and the live "
              "re-solve is off so the recorded ramp survives -- press **FK** to "
              "go back to posing by hand. *Play solve on robot* sends the whole "
              "ramp, every step of it, wherever the **Solve steps** scrubber "
              "happens to be parked.")


    # -- phase 5: the lift --

    def _lift_hand(self, _=None):
        """Phase 5: raise the wrist straight up, and record the whole ramp.

        Threaded and gated exactly like the close, for the close's reasons: the
        gate is what stops a lift and a solve overlapping (they share the FK
        solver, its warm start and ``self.result``), and a viser callback thread
        blocked for the length of a lift cannot service the E-STOP click. The
        stop INTERRUPTS this one rather than merely refusing it -- the walk polls
        between poses -- and keeps every pose recorded so far.
        See :func:`~.solvers.lift_wrist`.
        """
        try:
            gate = self.estop.admit("wrist lift")
        except Refused:
            return
        try:
            self._set_solving(True)
            self._sync_params()
            self._refresh_object()
        except Exception as exc:
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise

        def worker():
            try:
                self._lift_admitted()
            except Exception as exc:
                self._error_status(exc)
            finally:
                # Gate first, so the refreshes below read an idle latch and can
                # hand the controls back -- the close worker's ordering.
                gate.release()
                self._set_solving()
                self._report_estop()

        self._lift_thread = threading.Thread(target=worker, daemon=True)
        try:
            self._lift_thread.start()
        except Exception as exc:
            # A gate never released refuses every solve for the rest of the
            # session; the same failure _ik_auto and _close_hand guard against.
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise


    def _lift_admitted(self):
        """The lift itself, for a caller that ALREADY HOLDS the gate."""
        # Same hand-over as the close, for the same reason: a lift pressed
        # straight off an IK phase (skipping the close, which is allowed --
        # nothing sequences these buttons) would otherwise start by dropping the
        # hand back onto the wrist and tensions the sliders still command.
        carried = self._carry_solve_into_fk()
        result, notes = lift_wrist(
            self.fk_solver, height=self.g_lift_height.value,
            on_progress=self._set_status,
            should_stop=self.estop.is_tripped)

        # "Lift", not "FK", for the close's reason: _live_fk re-solves on every
        # slider drag while the mode is FK, and the first such drag would throw
        # the recorded ramp away along with the scrubber.
        self.mode = "Lift"
        self.result = result
        # A lift re-poses the hand outside the AL loop, so any multipliers a
        # stepper is holding describe a hand that no longer exists.
        self._invalidate_stepper()
        self._rebuild_iter_slider()
        # The wrist half of what _adopt_solved_tensions does after a close, and
        # just as necessary: lift_wrist moved params.wrist_pose, but the six
        # Wrist-start-pose sliders still read the pose from BEFORE the lift, and
        # _sync_params rebuilds params.wrist_pose straight off them -- so the
        # next FK solve (or any slider drag) would drop the hand back down.
        # _write_wrist_sliders also grows the +-0.1 m demo range, which a lift
        # always needs: the default hover is 0.075 m and 150 mm of it lands well
        # outside.
        widened = self._write_wrist_sliders(self.fk_solver.params.wrist_pose)
        self._render_frame()
        lines = ["**Phase 5 lift**  \n"
                 + "  \n".join(([carried] if carried else []) + notes)]
        if widened:
            lines.append(
                f"_wrist slider range widened ({', '.join(widened)}) to hold "
                f"the raised pose._")
        # Said every time, because the picture invites the opposite reading: the
        # hand rises and the object does not follow it up.
        lines.append(
            "the object stayed where it is -- an FK lift enforces nothing, so "
            "no contact carries it, and whether this grasp would actually hold "
            "it is not a question this phase asks.")
        lines.append(
            "the wrist sliders now hold the raised pose and the live re-solve "
            "is off so the recorded ramp survives -- press **FK** to go back to "
            "posing by hand. *Play solve on robot* sends the whole lift, every "
            "step of it, wherever the **Solve steps** scrubber is parked.")
        self._set_status("  \n".join(lines))


    # -- robot (ROS mode only) --
    #
    # Everything below is dead code with ros_mode off: the Robot folder is not
    # built, so nothing calls it. The division of labour is that this side knows
    # about SOLVES and viser, the bridge knows about ROS and frames, and
    # robot_plan.py -- which neither imports -- is the pure conversion between
    # them.

    def _open_lengths(self):
        """The hand-open tendon lengths every commanded displacement is measured
        from, built once and cached.

        Costs two FK solves (the open pose, plus the probe that proves flexion
        SHORTENS the actuated tendon), so it is not done at startup: an app opened
        to look at a solve should not pay for hardware it may never talk to. The
        cache is never invalidated because it cannot go stale -- the open pose is
        the hand's morphology posed at the calibrated open-hand tensions
        (``HandConfig.zero_bend_flexor_tensions``), both loaded from gepetto_core
        and neither of them a control on this page. In particular it does NOT
        follow the tension sliders: those say how hard THIS solve is pulling, and
        the zero every displacement is measured from has to stay put while they
        move.

        Raises if the sign check fails: every displacement in a plan would have
        the wrong sign, which on real hardware means driving the fingers into the
        extension stop.
        """
        if self._open_lengths_cache is None:
            lengths = robot_plan.open_tendon_lengths(self.params, self.fk_solver)
            notes, ok = robot_plan.check_open_lengths(lengths, self.params)
            if not ok:
                self._open_notes = notes
                raise RuntimeError("  \n".join(notes))
            self._open_lengths_cache, self._open_notes = lengths, notes
        return self._open_lengths_cache


    def _report_tendon_lengths(self, res):
        """The ACTUATED (flexor) tendon length the solve arrived at, per finger,
        printed under the tension sliders that commanded it.

        The sliders say what the hand is being pulled with; this says what came
        back -- the L half of the state a control tick anchors on
        (``HandResult.tendon_lengths``), which until now was only visible after
        an export to the robot. ``res`` is the frame being drawn, so scrubbing
        the convergence slider walks the lengths through the solve too.

        The tension is re-read from the RESULT rather than from the slider
        because IK treats the flexor tension as a variable with a soft prior:
        past the first iterate the tension that produced this length is the
        solver's, not the slider's, and printing the slider's next to a solved
        length would be a pairing that never happened.

        Displacement from the open hand -- the quantity actually commanded to
        the hardware -- is appended only once :meth:`_open_lengths` has been
        computed for some other reason (ROS mode). Deriving it here would cost
        an FK solve on the shared warm solver from whichever thread happens to
        be rendering, and this is a readout, not a reason to solve.
        """
        handle = getattr(self, "g_tendon_lengths", None)
        if handle is None:
            return  # rendered before the GUI was built
        if res is None:
            handle.content = self.TENDON_IDLE
            return
        lengths = dict(zip(res.finger_names, res.tendon_lengths(0)))
        open_lengths = self._open_lengths_cache
        # Whole lines as code spans: markdown collapses runs of spaces
        # everywhere else, and the columns are the point of the readout.
        lines = [f"**actuated tendon ({self.mode})**"]
        for name in res.finger_names:
            tension = float(np.asarray(
                res.frames[0][name].marginals.tensions.mean,
                float)[FLEXOR_IDX])
            length = float(lengths[name][FLEXOR_IDX])
            row = f"{name:<6} {tension:5.2f} N   {length * 1e3:7.2f} mm"
            if open_lengths is not None and name in open_lengths:
                # Signed the way robot_plan commands it: POSITIVE is tendon
                # pulled in from the open hand, negative is paid back out.
                # Snapped to zero inside half a displayed digit, so the open
                # hand reads a column of +0.00 rather than a mix of +0.00 and
                # the "-0.00" a solver residual of -1e-9 would otherwise print.
                disp = (open_lengths[name] - length) * 1e3
                if abs(disp) < 0.005:
                    disp = 0.0
                row += f"   {disp:+6.2f} mm vs open"
            lines.append(f"`{row}`")
        handle.content = "  \n".join(lines)
