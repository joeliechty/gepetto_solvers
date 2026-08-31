"""Talking to the hardware: playback, state readback, the e-stop wiring.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import math
import threading
import traceback

import numpy as np

from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.solvers import FLEXOR_IDX, R_to_euler

from .constants import (
    MAX_TENDON_SPEED,
    PLAY_FINAL,
    PLAY_HISTORY,
    SERVO_SCALE_LINEAR,
    SERVO_SCALE_ROTATIONAL,
)
from .estop import Refused


class RobotMixin:
    # How far past a measured wrist value to grow the slider when the value falls
    # outside its range, as a fraction of how far outside it fell -- headroom
    # enough that the widened slider is still draggable either side of where the
    # robot actually is, rather than pinned against its own new end stop.
    _WRIST_RANGE_MARGIN = 0.25

    # Bisection budget for the tension recovery below. 14 halvings of the 0-3 N
    # slider range resolve tension to 0.2 mN, far finer than the ~0.1 mm of
    # displacement the hardware can distinguish, so the tolerance is what
    # actually ends it.
    _TENSION_BISECT_STEPS = 14
    _TENSION_BISECT_TOL_M = 5e-4

    def _robot_speeds(self):
        """The playback-speed slider resolved into real units, per channel.

        ONE fraction scaling all three ceilings together, which is exactly the
        time-scaling factor: a segment lasts
        ``(1 / fraction) * max(t_linear, t_angular, t_tendon)``, so the slider is
        a pure duration multiplier and the geometric path is untouched by it.
        Scaling the channels independently could only change WHICH one is the
        slowest -- it can never make them arrive at different times, because they
        share a duration by construction -- so two sliders were two numbers for
        one question.

        Fractions rather than absolute speeds so they keep meaning something if
        MoveIt Servo's scales or HandConfig's tendon speed are ever retuned.
        """
        fraction = float(self.g_speed.value)
        return dict(max_linear=fraction * SERVO_SCALE_LINEAR,
                    max_angular=fraction * SERVO_SCALE_ROTATIONAL,
                    max_tendon=fraction * MAX_TENDON_SPEED)


    def _speed_note(self):
        speeds = self._robot_speeds()
        return (f"speed {float(self.g_speed.value):.2f} &nbsp; "
                f"(wrist {speeds['max_linear']:.2f} m/s / "
                f"{speeds['max_angular']:.2f} rad/s, "
                f"tendon {speeds['max_tendon'] * 1e3:.1f} mm/s)")


    def _set_robot_status(self, text):
        self.g_robot_status.content = text


    def _refresh_robot_status(self, extra=None, standing=True):
        """The Robot folder's own readout: what the bridge can see, and what the
        buttons would do right now. Separate from the solver status line because
        the two answer different questions and overwrite each other otherwise.

        ``standing=False`` reuses the cached lower half instead of re-polling the
        bridge. Playback feedback arrives about ten times a second and every poll
        is a pair of TF lookups for three lines that cannot have changed since the
        run started -- worth skipping even now that the robot is not waiting on
        this thread.
        """
        # The transient message LEADS: it is the answer to whatever the operator
        # just pressed (playing, refused, failed), and the standing readout below
        # it is the same three lines every time. Buried under them it reads as
        # part of the furniture.
        lines = [extra] if extra else []
        if standing or self._standing_status is None:
            standing_lines = []
            try:
                standing_lines.append(self.bridge.status())
            except Exception as exc:
                standing_lines.append(f"**bridge unavailable:** `{exc}`")
            standing_lines.append(
                ("**ARMED** -- the next *Play* moves the arm and the hand"
                 if self.g_armed.value else
                 "not armed -- tick *move the real robot* to allow one playback")
                + f" &nbsp; {self._speed_note()}")
            standing_lines.extend(self._open_notes)
            self._standing_status = standing_lines
        lines.extend(self._standing_status)
        self._set_robot_status("  \n".join(lines))


    def _set_robot_busy(self, busy=None):
        """Grey the robot buttons in step with the rest of the panel.

        Same rule as :meth:`_set_solving`, which is what actually calls this: a
        robot action is admitted through the very same gate a solve is, so
        "something is running" and "the latch is engaged" both have to disable
        these two buttons as well."""
        if not self.ros_mode:
            return
        if busy is None:
            busy = self.estop.busy is not None
        blocked = busy or self.estop.is_tripped()
        self.g_play.disabled = blocked
        self.g_get_state.disabled = blocked


    def _on_estop_change(self, tripped, reason):
        """The latch moved -- tell the bridge, so the publishers stop.

        Registered on ``EStop`` rather than hung off the button handler because
        the latch has several sources (the button, an abort from the playback
        watchdog, a trip arriving over ROS) and every one of them has to reach
        the hardware. Called on whatever thread tripped it, with no locks held.

        Both directions go through one call: engaging halts the publishers and
        latches the cell-wide e-stop topic, releasing clears it. A rearm that did
        not clear the topic would leave every other node in the cell refusing to
        move while this app looked live again.
        """
        self.bridge.set_estop(tripped, reason)
        # A trip disarms. Releasing the latch deliberately does NOT re-arm: coming
        # back from an E-STOP should take the same explicit gesture as arming did
        # the first time, not hand back a live panel because someone cleared a
        # fault. `getattr` because the latch exists outside ROS mode, where the
        # Robot folder -- and this checkbox -- was never built.
        if tripped:
            armed = getattr(self, "g_armed", None)
            if armed is not None:
                armed.value = False
        self._set_robot_busy()


    def _play_on_robot(self, _=None):
        """Send the solve on screen to the robot.

        Claims the SAME gate a solve does, which is the point: a playback and a
        solve can never overlap, the latch already refuses both, and the panel
        greys out for one exactly as it does for the other. The work runs on a
        worker thread for the same reason auto-solve does -- a viser callback
        thread that blocks for the length of a trajectory cannot service the
        E-STOP click.
        """
        if not self.ros_mode:
            return
        if not self.g_armed.value:
            self._refresh_robot_status(
                "**not armed** -- tick *move the real robot* first. It arms one "
                "press and clears itself afterwards.")
            return
        try:
            gate = self.estop.admit("robot playback")
        except Refused as exc:
            self._refresh_robot_status(f"**refused:** {exc}")
            return

        # Collect the measured trace only for a recorded-path playback: it is
        # keyed by waypoint index, and only for `history` does a waypoint index
        # mean an iterate index -- which is what the plot's x axis is. A `final
        # state only` run has one waypoint and nothing to line up against.
        self._robot_trace = ({} if self.g_play_source.value == PLAY_HISTORY
                             else None)

        def worker():
            try:
                self._set_solving(True)
                plan = self._build_robot_plan()
                self.bridge.play(
                    plan,
                    # Both channels, always: the wrist and the fingers are one
                    # time-scaled trajectory and playing half of it would move the
                    # arm to the grasp with the hand wherever it happened to be.
                    enable_arm=True,
                    enable_hand=True,
                    speeds=self._robot_speeds(),
                    on_progress=lambda text: self._refresh_robot_status(
                        text, standing=False),
                    on_sample=self._sample_robot_trace,
                    should_stop=self.estop.is_tripped)
            except Exception as exc:
                traceback.print_exc()
                self._refresh_robot_status(f"**playback failed:** `{exc}`")
            finally:
                # DISARM FIRST, before anything that can fail or block. This runs
                # on every exit -- finished, aborted, stopped, threw -- because
                # the box arms one press and a press has now been spent. Ahead of
                # the gate release so there is no window in which the panel is
                # live again while still armed.
                self.g_armed.value = False
                # Gate next, so the refreshes below see an idle latch and can
                # hand the controls back -- the auto-solve worker's ordering.
                gate.release()
                self._set_solving()
                self._report_estop()
                # The measured trace is complete now, including whatever the
                # terminal hold added. Redraw so the dashed line reflects where
                # the robot actually finished rather than the last feedback
                # tick before it stopped moving.
                self._update_traj()

        self._play_thread = threading.Thread(target=worker, daemon=True)
        try:
            self._play_thread.start()
        except Exception:
            # A gate never released refuses every solve for the rest of the
            # session; the same failure _ik_auto guards against. Disarm here too:
            # the worker's `finally` is what normally does it and it never ran.
            self.g_armed.value = False
            gate.release()
            self._set_solving()
            raise


    def _build_robot_plan(self):
        """The plan for whatever is on screen, clamped to the hand's real travel."""
        source = "final" if self.g_play_source.value == PLAY_FINAL else "history"
        if self.result is None:
            raise RuntimeError("nothing solved yet -- press FK, Step or Auto solve")
        # The WHOLE recorded path, whatever the scrubber is showing. See
        # build_plan on why there is no "play from here": the scrubber opens on
        # the last iterate, so honouring it turned every playback into a single
        # hop to the final pose.
        plan = robot_plan.build_plan(
            self.result, self.fk_solver.configs, self._corner_viz(),
            self._open_lengths(), source=source)
        plan, notes = robot_plan.clamp_to_travel(plan)
        if notes:
            self._refresh_robot_status("  \n".join(notes))
        return plan


    def _get_robot_state(self, _=None):
        """Adopt the robot's measured state as the visualizer's state.

        The wrist is a direct write (the bridge hands it over already in viser
        coordinates). The tendons are not: this app's kinematic input is TENSION,
        and the hardware reports LENGTH, so the tensions that produce the measured
        lengths have to be recovered -- see :meth:`_tensions_for_displacement`.

        Runs on a worker thread and through the solve gate, because the recovery
        is a series of FK solves.
        """
        if not self.ros_mode:
            return
        try:
            gate = self.estop.admit("read robot state")
        except Refused as exc:
            self._refresh_robot_status(f"**refused:** {exc}")
            return

        def worker():
            try:
                self._set_solving(True)
                state = self.bridge.read_state(self._corner_viz())
                notes = self._adopt_robot_state(state)
                self._refresh_robot_status("  \n".join(notes))
            except Exception as exc:
                traceback.print_exc()
                self._refresh_robot_status(f"**read failed:** `{exc}`")
            finally:
                gate.release()
                self._set_solving()
                self._report_estop()

        threading.Thread(target=worker, daemon=True).start()


    def _fit_wrist_range(self, handle, value):
        """Grow ``handle``'s range until it contains ``value``. Returns the new
        ``(min, max)`` if the range moved, else None.

        Named for the caller it was written for, but it is a plain "make this
        handle able to hold this number": the Calibration folder uses it for the
        same reason, on both the wrist sliders it commands and its own x/y/z when
        a captured landmark falls outside the table square.

        The wrist sliders open on a DEMO range (+-0.1 m) that says nothing about
        where the robot is: a read comes back in the scene frame, whose origin is
        the table corner, so a wrist a third of a metre above the table is an
        ordinary measurement rather than an error. Clamping it into the range drew
        the hand somewhere the robot is not -- the one thing this readout must
        never do -- so the range yields to the measurement instead. The new bound
        is snapped OUT to a whole step, since a bound off the step grid leaves the
        end of the track unreachable by dragging.
        """
        lo, hi = float(handle.min), float(handle.max)
        if lo <= value <= hi:
            return None
        step = float(handle.step or 1e-3)
        if value < lo:
            lo = math.floor(
                (value - max((lo - value) * self._WRIST_RANGE_MARGIN, step)) / step
            ) * step
        else:
            hi = math.ceil(
                (value + max((value - hi) * self._WRIST_RANGE_MARGIN, step)) / step
            ) * step
        handle.min, handle.max = lo, hi
        return lo, hi


    def _adopt_robot_state(self, state):
        """Write one :class:`RobotState` onto the controls. Returns status lines."""
        notes = []
        T = np.asarray(state.wrist_pose, float)
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        widened = []
        self._restoring = True   # our writes; no live-FK re-solve per slider
        try:
            for handle, value, label in zip(
                    (self.g_tx, self.g_ty, self.g_tz,
                     self.g_roll, self.g_pitch, self.g_yaw),
                    (*T[:3, 3], roll, pitch, yaw),
                    ("x", "y", "z", "roll", "pitch", "yaw")):
                value = float(value)
                grown = self._fit_wrist_range(handle, value)
                if grown is not None:
                    widened.append(
                        f"{label} to [{grown[0]:+.3f}, {grown[1]:+.3f}]")
                # The MEASURED number, not a rounded or bounded one: viser does
                # not snap a programmatic write to the step grid, and _sync_wrist
                # rebuilds params.wrist_pose straight off these six handles, so
                # what is written here is exactly the pose that gets solved and
                # drawn. See _adopt_solved_wrist, which writes the same way.
                handle.value = value
        finally:
            self._restoring = False
        if widened:
            notes.append(
                f"_wrist slider range widened ({', '.join(widened)}) so it holds "
                f"the measured pose -- the robot is outside the volume this "
                f"scene's demo sliders cover. The hand on screen IS where the "
                f"robot is; if that looks wrong, check the table registration._")
        notes.append(f"wrist read at ({T[0, 3]:+.6f}, {T[1, 3]:+.6f}, "
                     f"{T[2, 3]:+.6f}) m in the scene frame")

        if state.tendon_disp:
            notes.extend(self._tensions_for_displacement(state.tendon_disp))
        else:
            notes.append("_no tendon state on "
                         "`/finger_servo_node/measured_state` -- tensions left "
                         "as they were._")
        if state.age is not None and state.age > 1.0:
            notes.append(f"**tendon state is {state.age:.1f} s old** -- is "
                         "finger_servo_node still running?")
        # Already admitted (the whole read holds the gate), so this must not try
        # to claim it again -- see _fk_solve_admitted.
        self._fk_solve_admitted()
        return notes


    def _tensions_for_displacement(self, measured):
        """Recover the flexor tensions that reproduce the MEASURED tendon
        displacements, and put them on the sliders.

        The visualizer poses the hand from tension; the hardware measures length.
        There is no inverse in the solver -- the FK graph runs tension to length
        -- so this inverts it numerically. Bisection rather than anything cleverer
        because the map is monotone (more flexor tension pulls more tendon in,
        which ``robot_plan.check_open_lengths`` proves rather than assumes) and
        because a bisection cannot diverge on a finger whose measurement is
        outside what the model can reach: it simply converges onto the nearest
        end of the slider range and the residual reported below says so.

        All five digits are bisected TOGETHER: one FK solve poses the whole hand,
        so five independent searches would cost five times the solves for the same
        answer. Each solve is warm-started off the last (the cached FK solver), so
        the whole recovery is ~14 cheap solves.
        """
        open_lengths = self._open_lengths()
        names = [n for n in self.fk_solver.finger_names if n in measured]
        if not names:
            return ["_no measured finger matched the model's digits._"]

        lo = {n: float(self.g_flexors[0].min) for n in names}
        hi = {n: float(self.g_flexors[0].max) for n in names}
        # Preserve whatever the sliders hold for digits the robot did not report,
        # so a partial readback does not silently zero the rest of the hand.
        tensions = list(self.params.flexor_tensions)
        index_of = {n: i for i, n in enumerate(self.fk_solver.finger_names)}
        residual = {}

        for _ in range(self._TENSION_BISECT_STEPS):
            for name in names:
                tensions[index_of[name]] = 0.5 * (lo[name] + hi[name])
            self.params.flexor_tensions = list(tensions)
            result = self.fk_solver.solve()
            lengths = dict(zip(result.finger_names, result.tendon_lengths(0)))
            for name in names:
                got = open_lengths[name] - float(lengths[name][FLEXOR_IDX])
                residual[name] = got - measured[name]
                # Monotone increasing: too little displacement means not enough
                # tension, so the answer is in the upper half.
                if got < measured[name]:
                    lo[name] = 0.5 * (lo[name] + hi[name])
                else:
                    hi[name] = 0.5 * (lo[name] + hi[name])
            if max(abs(v) for v in residual.values()) <= self._TENSION_BISECT_TOL_M:
                break

        self._restoring = True
        try:
            for handle, value in zip(self.g_flexors, tensions):
                handle.value = float(min(max(value, handle.min), handle.max))
        finally:
            self._restoring = False
        self.params.flexor_tensions = list(tensions)

        worst = max(residual, key=lambda n: abs(residual[n]))
        line = (f"tendons matched to "
                f"{abs(residual[worst]) * 1e3:.2f} mm (worst: {worst}); "
                f"measured " + ", ".join(f"{n} {measured[n] * 1e3:.1f}"
                                         for n in names) + " mm")
        if abs(residual[worst]) > 10 * self._TENSION_BISECT_TOL_M:
            line = (f"**{line}** -- {worst} is outside what the model reaches "
                    f"within the 0-{self.g_flexors[0].max:g} N slider range")
        return [line]


    # -- e-stop --

    def _estop(self, _=None):
        """Engage the e-stop. Runs on a viser callback thread and must never
        block: it flips the latch and repaints, and does NOT wait for the
        running solve to notice. A running auto-solve breaks out at its next
        iteration boundary; anything else is simply refused from here on."""
        self.estop.trip("E-STOP pressed")
        self._set_solving()
        self._report_estop()


    def _rearm(self, _=None):
        """Release the e-stop and hand the controls back.

        Refused while a stopped solve is still winding down -- rearming around
        a solve that has not yet returned would re-enable the panel while the
        hand is still moving, which is the one thing the button exists to
        prevent."""
        if not self.estop.rearm():
            self._set_status(
                "**E-STOP still engaged** -- the running solve has not returned "
                "yet (it stops at the end of the current AL iteration). Try "
                "again in a moment.")
            return
        self._set_solving()
        # Put the real readout back: whatever the solve was showing when it was
        # stopped is still on screen and still true.
        if self.mode == "IK" and self.stepper is not None:
            self._report_step_status(self.stepper.status())
        elif self.result is not None:
            self._report()
        else:
            self._set_status("**Rearmed.**")


    def _report_estop(self):
        """Write the e-stop banner over the status line, if engaged.

        Called after every solve path returns so the last thing written is the
        latch's state rather than a step readout that has been overtaken."""
        if not self.estop.is_tripped():
            return
        note = ("" if self.caps["gil_release"] else
                "  \n*this binding does not release the GIL during a solve, so "
                "this click landed only when the iteration ended -- rebuild "
                "with `pip install .` from the crest-sparse root*")
        self._set_status(
            f"# &#9888; E-STOP ENGAGED  \n{self.estop.reason} -- every solve is "
            f"refused until you press **Rearm**. Nothing was lost: the solve is "
            f"paused where it stopped and resumes from there." + note)


    def _build_robot_folder(self, gui):
        """The Robot folder: play a solve on the hardware, read the hardware back.

        Built only in ROS mode. It opens DISARMED -- everything else on this page
        is a picture, and this is not, so moving the robot takes a deliberate
        gesture that does not survive the run that used it.
        """
        with gui.add_folder("Robot"):
            gui.add_markdown(
                "Commands the **real robot**. The scene's table square is "
                "registered against `lbr_workspace_table_link`, so the hand on "
                "screen and the hand on the arm are the same hand.")
            self.g_armed = gui.add_checkbox(
                "move the real robot", False,
                hint="Safety interlock, and nothing else: *Play solve on robot* "
                     "refuses while this is clear. It ARMS ONE PRESS -- it clears "
                     "itself when the run ends, however it ends, and an E-STOP "
                     "clears it too. So it can never be left on and forgotten, "
                     "and Reset cannot set it. Playback always drives the arm and "
                     "the hand together; they are one coordinated trajectory and "
                     "running half of it is not a thing this panel offers. For "
                     "channel-at-a-time bring-up use `play_client.py --hand-only` "
                     "(with `finger_servo_node dry_run:=true`).")
            self.g_play_source = gui.add_dropdown(
                "waypoints", [PLAY_HISTORY, PLAY_FINAL],
                initial_value=PLAY_HISTORY,
                hint="What to play. *recorded path* walks the WHOLE recorded "
                     "path, one waypoint per recorded state, from the first -- "
                     "the Solve steps scrubber decides what is drawn, not what is "
                     "played. After Step/Auto solve those states are OPTIMIZER "
                     "iterations, so early ones can move oddly before the solve "
                     "settles; after a phase-4 Close they are the ramp itself, "
                     "which is a real planned path. *final state only* makes one "
                     "interpolated move to the end state and ignores the path.")
            self.g_speed = gui.add_slider(
                "playback speed (fraction)", 0.05, 1.0, 0.05, 0.50,
                hint=f"How fast to play the trajectory, as a fraction of every "
                     f"channel's own maximum ({SERVO_SCALE_LINEAR} m/s linear, "
                     f"{SERVO_SCALE_ROTATIONAL} rad/s rotational, "
                     f"{MAX_TENDON_SPEED} m/s tendon). ONE number because the "
                     f"wrist and the fingers are one trajectory: each segment "
                     f"takes as long as its slowest channel needs, so scaling "
                     f"them apart only changes which one waits. A ceiling rather "
                     f"than a setpoint -- halving it doubles the duration and "
                     f"leaves the path itself identical. The executor caps the "
                     f"wrist below the servo's full scale so its tracking "
                     f"correction always has room to work.")
            self.g_play = gui.add_button(
                "Play solve on robot", icon=self.viser.Icon.ROBOT,
                hint="Export the solve on screen as waypoints, interpolate them "
                     "at the speed above, and servo the robot along them. Goes "
                     "through the same admission gate as a solve, so it cannot "
                     "run alongside one and the E-STOP refuses it outright.")
            self.g_get_state = gui.add_button(
                "Get robot state", icon=self.viser.Icon.DOWNLOAD,
                hint="Read the wrist pose (TF) and the measured tendon lengths "
                     "(finger_servo_node) and make them the state on screen. The "
                     "tendon lengths are inverted back into flexor tensions -- "
                     "this app poses from tension -- so the sliders move too.")
            self.g_robot_status = gui.add_markdown("")

        self.g_play.on_click(self._play_on_robot)
        self.g_get_state.on_click(self._get_robot_state)
        for handle in (self.g_armed, self.g_speed):
            handle.on_update(lambda _: self._refresh_robot_status())
