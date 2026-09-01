"""Driving the solve: FK, the AL stepper, and the warm-start carry.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import threading

import numpy as np

from gepetto_solvers.core.solvers import (
    HandIKStepper,
    R_to_euler,
    solved_wrist_pose,
)

from .constants import (
    binding_path,
)
from .estop import Refused


class StepperMixin:
    # -- solve --

    def _set_solving(self, solving=None):
        """Put the control panel into the state the e-stop says it should be in.

        Two things grey the panel out: a solve running (it is busy) and the
        e-stop being engaged (it is refusing). Both are read off the latch, so
        this can be called with no argument from anywhere -- the trip and rearm
        handlers do exactly that -- and cannot disagree with what admit() will
        actually allow. ``solving`` is accepted only to spare the callers that
        already know the answer.
        """
        if solving is None:
            solving = self.estop.busy is not None
        tripped = self.estop.is_tripped()
        # While the latch is engaged NOTHING may start, so every control that
        # begins work greys out, not just the ones a running solve blocks.
        blocked = solving or tripped
        self.g_fk.disabled = blocked
        # Names the state, not the button that caused it: labelling this one
        # "E-STOP" would read as a second stop button sitting next to the real
        # one.
        self.g_fk.label = ("FK (stopped)" if tripped
                           else "Solving..." if solving else "FK")
        for btn in (getattr(self, "g_ik_step", None), getattr(self, "g_ik_auto", None)):
            if btn is not None:
                btn.disabled = blocked or not self.caps["ik_stepping"]
        # Close follows the same rule but NOT the ik_stepping capability: it is
        # a run of FK solves, so it works on a binding that cannot step an AL
        # solve at all.
        if getattr(self, "g_close", None) is not None:
            self.g_close.disabled = blocked
        # Lift is the same kind of thing as Close -- a run of FK solves -- so it
        # follows the same rule, ik_stepping capability included (not needed).
        if getattr(self, "g_lift", None) is not None:
            self.g_lift.disabled = blocked
        # Reset would pull the params out from under a running step, and flipping
        # the warm start mid-run cannot affect the loop already built, so both
        # wait it out rather than looking like they did something.
        if getattr(self, "g_warm", None) is not None:
            self.g_warm.disabled = blocked or not self.caps["solver_seed"]
        if getattr(self, "g_reset", None) is not None:
            self.g_reset.disabled = blocked
        # Rearm is live only when there is something to rearm FROM, and only
        # once the stopped solve has actually returned -- rearming around a
        # solve still winding down would hand the controls back while the hand
        # is still moving. EStop.rearm() enforces the same rule; this just makes
        # the button agree with it.
        if getattr(self, "g_rearm", None) is not None:
            self.g_rearm.disabled = not tripped or solving
        # The two buttons that move a physical robot follow exactly the same
        # rule -- they are admitted through the same gate, so they must grey out
        # with everything else.
        self._set_robot_busy(solving)


    def _fk_solve(self, _=None):
        """Re-pose the hand from the current sliders with the FK solver.

        Also the "start over" action for the IK loop: it drops any partially
        stepped solve. What the next Step then starts from is the warm-start
        latch's business -- off, the cold guess; on, this FK pose.

        Refused outright while the e-stop is engaged: an FK solve re-poses the
        hand, which is exactly what a stopped app must not do."""
        try:
            gate = self.estop.admit("FK solve")
        except Refused:
            return
        try:
            with gate:
                self._set_solving(True)
                self._fk_solve_admitted()
        finally:
            # Outside the gate, so it reads a released latch and can re-enable
            # the controls (or leave them grey, if the e-stop tripped meanwhile).
            self._set_solving()
            self._report_estop()


    def _fk_solve_admitted(self):
        """The FK solve itself, for a caller that ALREADY HOLDS the gate.

        Split out because the gate is not re-entrant, by design -- it is what
        stops two solves running at once. A caller that has already been admitted
        for a longer operation (reading the robot's state, which is a whole
        series of FK solves) therefore cannot go through :meth:`_fk_solve`: its
        admit() would be refused as "already running", and refusal there is
        silent, so the hand would simply never be re-posed.
        """
        try:
            self._sync_params()
            self._refresh_object()
            self._invalidate_stepper()
            self.mode = "FK"
            self._set_status("Solving (FK)...")
            # Reuse the cached FK solver (shares self.params) so this warm-starts.
            self.result = self.fk_solver.solve()
            self._rebuild_iter_slider()
            self._render_frame()
            self._report()
        except Exception as exc:  # surface it in the GUI, keep serving
            self._error_status(exc)
            raise


    # -- IK stepping: drive the AL outer loop one iteration at a time --

    def _invalidate_stepper(self):
        """Drop the cached stepper so the next Step cold-starts.

        Only for changes to the CONSTRAINT SET (object, contact mask, collision,
        table): the stepper carries Lagrange multipliers and a penalty weight
        that describe the constraints it has been working on, and those mean
        nothing against a different set. Tensions and the wrist pose are passed
        fresh every step, so they deliberately do NOT invalidate -- nudging the
        flexor mid-solve and continuing is the point."""
        self.stepper = None


    def _ensure_stepper(self):
        """The cached stepper, built on demand.

        This is the ONE place the warm start is applied, and it is applied at
        BUILD time: a rebuilt solver is exactly the thing that would otherwise
        cold-start, and reading the on-screen state here means the posture handed
        over is whatever is showing at that instant -- no capture to go stale, no
        ordering to get right."""
        if self.stepper is None:
            if self.warm_start:
                self.params.initial_state = self._seed_state()
                # The other half: the multipliers holding each constraint. The
                # posture alone restarts the penalty schedule at mu = al_mu with
                # every multiplier at zero, so the solve lets go of everything it
                # had already satisfied and hauls it back over the next few
                # iterations -- the fingers lifting off the table. Matched onto
                # the new constraint set by identity, so the shared constraints
                # keep their multipliers and the new ones start at zero.
                self.params.initial_duals = (
                    self._seed_duals() if self.g_carry_duals.value else None)
                # Carrying the posture is only half of it. The wrist and the
                # flexor tensions are VARIABLES with soft priors commanded from
                # the sliders, and a contact solve moves both a long way from
                # what is commanded; rebuilding those priors at the old slider
                # values hauls the hand straight back. Adopt where the solve
                # actually ended up -- and show it on the sliders.
                self._adopt_solved_wrist()
                self._adopt_solved_tensions()
            else:
                self.params.initial_state = None
                self.params.initial_duals = None
            self.stepper = HandIKStepper(self.params)
        return self.stepper


    def _seed_duals(self):
        """The AL multipliers of the solve on screen, or None.

        Taken from ``self.result`` rather than the scrubbed iterate view: the
        multipliers belong to the solve as a whole (the AL outer loop's running
        state), not to the snapshot the scrubber happens to be showing."""
        return None if self.result is None else self.result.duals


    def _adopt_solved_wrist(self):
        """Re-aim the wrist prior at the pose the last solve reached, and move
        the sliders to match.

        The sliders are the prior's only input -- every step re-commands it from
        ``params.wrist_pose``, which ``_sync_params`` reads straight off them --
        so leaving them on the old numbers would undo this on the very next step.
        Moving them is also the honest thing: after a solve they no longer
        describe where the hand is."""
        res = self._iter_view()
        if res is None:
            return
        T = solved_wrist_pose(self.fk_solver.configs, res.frames[0])
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        self._restoring = True   # these are OUR writes; no live-FK re-solve
        try:
            for handle, value in zip(
                    (self.g_tx, self.g_ty, self.g_tz,
                     self.g_roll, self.g_pitch, self.g_yaw),
                    (*T[:3, 3], roll, pitch, yaw)):
                handle.value = float(value)
        finally:
            self._restoring = False
        # Exact, not the round trip through the sliders (they hold the same
        # numbers; viser does not snap a programmatic write to the step grid).
        self.params.wrist_pose = np.asarray(T, float)


    def _adopt_solved_tensions(self):
        """Move the flexor sliders to the tensions the last solve reached.

        Same failure as the wrist, one variable over: the flexor prior is soft by
        design (``_IK_TENSION_COV`` gives it variance 1e-1 so contact can drive
        it), so a grasp ends with the flexor far from what the slider commands --
        1.28 N against a commanded 0.6 N here. Rebuild the prior at the slider
        value and it pulls the fingers back open even without a settling step.

        Only the flexors: the five passives are pinned at 1e-6 and come back
        within 7e-6 N of what was commanded, so the one shared passive slider is
        already telling the truth and moving it would be noise."""
        res = self._iter_view()
        if res is None:
            return
        lo, hi = self.g_flexors[0].min, self.g_flexors[0].max
        solved, clamped = [], False
        for name in res.finger_names:
            q = float(np.asarray(
                res.frames[0][name].marginals.tensions.mean, float)[self._drive_index()])
            clamped = clamped or not (lo <= q <= hi)
            solved.append(min(max(q, lo), hi))
        self._restoring = True
        try:
            for handle, q in zip(self.g_flexors, solved):
                handle.value = q
        finally:
            self._restoring = False
        self.params.flexor_tensions = solved
        if clamped:
            # A clamped value is a prior commanded somewhere the hand is not, so
            # say so rather than let it look like the warm start misbehaving.
            self._set_status(
                f"**warm start:** a solved flexor tension fell outside the "
                f"slider range [{lo}, {hi}] N and was clamped -- the hand will "
                f"move toward the clamped value.")


    def _carry_solve_into_fk(self):
        """Hand the solve on screen over to the FK solver, for the two phases
        that are commanded ramps rather than solves (4's close, 5's lift).
        Returns a markdown line saying which way it went, or None when there was
        nothing to carry.

        The IK phases carry themselves: ``_ensure_stepper`` adopts the solved
        wrist and tensions and seeds the posture every time a changed constraint
        set forces a rebuild, which is what makes phase 0 -> 1 -> 2 one
        continuous move. Nothing did that across the boundary INTO FK. The
        sliders still hold the pose and the tensions the last phase was
        *commanded* with -- deliberately, since a contact solve ends far from
        both -- and the FK solver holds whatever its own last solve left, which
        can be a phase and an object ago. So pressing Close after a phase-2
        approach re-posed the hand at the wrist phase 1 was aimed at and let the
        fingers spring back to the slider tensions, all before the ramp took its
        first step.

        What crosses is what an FK ramp can be told: the wrist pose, the flexor
        tensions, and the posture to start looking from. The CONTACT does not,
        and cannot -- phase 4 enforces nothing, so the fingers settle where
        those tensions put them. That gap is the phase, not a shortfall of the
        carry.

        Gated on the warm-start latch, this app's one switch for "continue from
        the state on screen", so a close can still be run from what the panel
        commands by turning it off."""
        if self.result is None:
            return None
        if not self.warm_start:
            return ("*warm start is off, so this ramp starts from the wrist and "
                    "tensions the sliders command rather than from the solve on "
                    "screen.*")
        # Both adopt from the SCRUBBED iterate, like every other warm start
        # here, so a ramp branches from whichever step of the solve is showing.
        self._adopt_solved_wrist()
        self._adopt_solved_tensions()
        self.fk_solver.seed_posture(self._seed_state())
        return ("*carried the solve on screen into the ramp: its wrist pose and "
                "flexor tensions are now on the sliders and the FK solver starts "
                "from its posture. Contact does not cross -- nothing is enforced "
                "from here.*")


    def _show_step(self, result, status):
        """Render one stepped state and update both status readouts. Called from
        the auto-run thread as well as the Step button."""
        self.result = result
        self._render_frame(live=True)
        # Deliberately not _report(): during stepping the AL numbers below are
        # the interesting readout, and writing both just overwrites one with the
        # other every frame.
        self._report_step_status(status)


    def _report_step_status(self, status):
        # "stalled" is the solver's own stopping rule (a step that changed
        # nothing), not a dead end: mu still grows, so pressing Auto solve again
        # resumes and often unwedges. Say so, or it reads as a failure.
        verdict = {"running": "stepping",
                   "converged": "**converged**",
                   "stalled": "**stalled** -- last step changed nothing; "
                              "Auto solve again to continue, FK to restart",
                   }[status.state]
        if self.estop.is_tripped() and status.state == "running":
            verdict = "**E-STOP** -- press Rearm to resume from here"
        gaps = [f"object {self.result.worst_gap(0):+.5f} m"] \
            if self.params.object_contact else []
        if self.result.table_contact_names():
            gaps.append(f"table {self.result.worst_table_gap(self.params, 0):+.5f} m")
        # What the rebuild inherited. Worth a line of its own: 0 matched against
        # a non-empty carry is a tag drift (a constraint built without one),
        # which otherwise just looks like the warm start not working.
        rep = self.result.dual_transfer
        carried = ("" if rep is None else
                   f"  \nduals carried: {rep.matched}/{rep.total} constraints")
        pinch = (self._half_space_note() + self._wrist_gauge_note()
                 + self._pinch_note())
        self._set_status(
            f"**IK step {status.steps}** &nbsp; {verdict}  \n"
            f"violation={status.violation:.3e} &nbsp; cost={status.cost:.4g} "
            f"&nbsp; mu={status.mu:.3g}  \n"
            f"worst gap: {' &nbsp; '.join(gaps) or 'n/a'}{carried}"
            + ("  \n" + "  \n".join(pinch) if pinch else ""))


    def _ik_step(self, _=None):
        """One Augmented Lagrangian outer iteration, continuing the last one.

        Refused while the e-stop is engaged. The iteration itself is not
        interruptible once started -- one step is one uninterruptible call into
        C++ -- so the latch's job here is purely to refuse to begin."""
        if not self.caps["ik_stepping"]:
            return
        try:
            gate = self.estop.admit("IK step")
        except Refused:
            return
        try:
            with gate:
                self._set_solving(True)
                try:
                    self._sync_params()
                    self._refresh_object()
                    self.mode = "IK"
                    stepper = self._ensure_stepper()
                    self._show_step(stepper.step(), stepper.status())
                    self._rebuild_iter_slider()
                except Exception as exc:
                    self._error_status(exc)
                    raise
        finally:
            self._set_solving()
            self._report_estop()


    def _ik_auto(self, _=None):
        """Step to convergence on a worker thread, redrawing after each iteration.

        The loop has to leave viser's callback thread free: run inline and the
        E-STOP click would sit behind the whole solve, which is precisely when
        it stops being useful.

        This is the one path the e-stop can interrupt rather than merely refuse.
        ``run()`` polls ``should_stop`` between iterations, so a trip breaks the
        loop at the next boundary -- bounded by one AL iteration (~1.7 s worst
        case measured), because that is a single call into C++ with no interrupt
        hook. Everything the stepper holds (multipliers, mu, the full history)
        survives, so Rearm + Auto solve resumes rather than restarts."""
        if not self.caps["ik_stepping"]:
            return
        try:
            gate = self.estop.admit("IK auto solve")
        except Refused:
            return
        try:
            self._set_solving(True)
            self._sync_params()
            self._refresh_object()
            self.mode = "IK"
            stepper = self._ensure_stepper()
        except Exception as exc:
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise

        def worker():
            try:
                status = stepper.run(max_steps=self.g_ik_max.value,
                                     on_step=self._show_step,
                                     should_stop=self.estop.is_tripped)
                self._report_step_status(status)
                # Only now: rebuilding the slider once per frame mid-animation
                # would tear a GUI handle down and re-add it every iteration.
                self._rebuild_iter_slider()
            except Exception as exc:
                self._error_status(exc)
            finally:
                # Release the gate FIRST, so the refresh below sees an idle
                # latch and can hand the controls back (or offer Rearm).
                gate.release()
                self._set_solving()
                self._report_estop()

        self._auto_thread = threading.Thread(target=worker, daemon=True)
        try:
            self._auto_thread.start()
        except Exception as exc:
            # Nothing else will ever release the gate if the worker never runs,
            # and a gate held forever means the app refuses every solve for the
            # rest of the session. Rare, but the failure mode is permanent.
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise


    # -- warm start and reset --

    def _seed_state(self):
        """The posture on screen right now, as a solver seed (None if nothing has
        been solved). Follows the *iterate* scrubber, so rewinding to an earlier
        iteration and branching from there works."""
        res = self._iter_view()
        return None if res is None else res.state(0)


    def _toggle_warm_start(self, _=None):
        """Flip the warm-start latch.

        A LATCH, not a one-shot capture: the state is read at the moment the
        stepper is (re)built, so it cannot go stale and the order you press
        things in does not matter. Capturing on the button press instead meant
        that anything re-posing the hand in between -- pressing FK, or just
        dragging a tension slider, which re-solves FK live -- silently threw the
        captured state away and the next Step cold-started."""
        if not self.caps["solver_seed"]:
            return
        self.warm_start = not self.warm_start
        self._refresh_warm_start()


    def _refresh_warm_start(self):
        """Put the latch's state on the button and in its readout."""
        on = self.warm_start
        self.g_warm.label = f"Warm start: {'ON' if on else 'off'}"
        self.g_warm.color = "blue" if on else None
        if not self.caps["solver_seed"]:
            # Name the .so that is actually loaded: the usual cause is not a
            # missing rebuild but the STALE IN-TREE copy at
            # python/gepetto_solvers/_gepetto_solvers*.so shadowing the installed one,
            # which happens whenever the app is run from the python/ directory.
            self.g_warm_status.content = (
                "**unavailable** -- this binding has no "
                "`HandSolverConfig.initial_state`  \n"
                f"loaded from `{binding_path()}`  \n"
                "*(run from the crest-sparse root -- `python -m "
                "python.tests.tendon_hand.viz_interactive` -- so the installed "
                "build is used, and rebuild with `pip install .`)*")
        elif on:
            self.g_warm_status.content = (
                "the next **Step**, **Close** or **Lift** starts from the state "
                "on screen"
                + (", carrying the AL multipliers"
                   if self.caps["dual_transfer"] and self.g_carry_duals.value
                   else ""))
        else:
            self.g_warm_status.content = (
                "the next **Step** cold-starts (straight hand, Q = 0); "
                "**Close**/**Lift** start from what the sliders command")


    # -- solve-iteration scrubber (IK) --

    def _current_iterate(self):
        """The scrubbed iterate index, clamped to the CURRENT result.

        The slider is rebuilt after the render that follows a step, so between
        the two it can still hold an index from a longer history -- a cold
        restart drops from ~30 snapshots to 2 and would otherwise index off the
        end."""
        if getattr(self, "iter_slider", None) is None or self.result is None:
            return 0
        return int(np.clip(int(self.iter_slider.value), 0,
                           max(self.result.num_iterates() - 1, 0)))


    def _rebuild_iter_slider(self):
        """Rebuild the convergence scrubber for the steps taken so far.

        Also drops any measured robot trace, and this is the right hook for it:
        the trace is indexed by iterate, every path that records new iterates
        comes through here, and a trace left over from the PREVIOUS solve would
        draw against waypoints it was never measured at. Silently wrong is the
        one thing an overlay claiming to be "what the robot did" must not be.

        Torn down and re-created rather than resized because the iteration count
        is whatever the AL outer loop has run to date -- and an FK pose (or a
        freshly restarted loop) must leave no slider at all."""
        self._robot_trace = None
        for name in ("iter_slider", "g_iter_status"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.remove()
            setattr(self, name, None)
        n = self.result.num_iterates() if self.result else 0
        if n <= 1:
            # Say why the folder is empty rather than leaving a bare header the
            # user has to guess at.
            with self.iter_folder:
                self.g_iter_status = self.server.gui.add_markdown(
                    "press **Step** or **Auto solve** to record IK iterations, "
                    "or **Close** to record a phase-4 close")
        else:
            with self.iter_folder:
                self.iter_slider = self.server.gui.add_slider(
                    "iterate", min=0, max=n - 1, step=1, initial_value=n - 1,
                    hint="0 is where the run started, the last is the most "
                         "recent step; in between, one Augmented Lagrangian "
                         "outer iteration each after Step/Auto solve, or one "
                         "substep of the ramp after a phase-4 Close.")
                self.g_iter_status = self.server.gui.add_markdown("")
            self.iter_slider.on_update(lambda _: self._render_frame())


    def _iter_view(self, live=False):
        """The result to render: the selected snapshot when the scrubber is up,
        otherwise the solve's own final state.

        ``live`` is the animation talking. While a solve is actually stepping,
        the newest state is the thing to draw -- not whatever index the scrubber
        is parked on. The scrubber CANNOT follow along mid-run: a viser slider's
        max is fixed at construction (there is no setter), which is why
        _rebuild_iter_slider tears it down and re-creates it, and why that is
        deliberately not done per frame. So during a run the slider still
        describes the PREVIOUS run. Honouring it here pinned every frame of the
        new solve to that stale index and the hand sat frozen until the run
        ended and the slider was rebuilt.

        The default stays scrubber-first, because _seed_state reads this to warm
        start from a rewound iterate -- that is the rewind-and-branch feature,
        and it must keep seeing the scrubbed state even though it is called from
        inside a running solve.
        """
        if self.result is None or live:
            return self.result
        if getattr(self, "iter_slider", None) is None:
            return self.result
        return self.result.at_iterate(self._current_iterate())


    def _report_iterate(self, live=False):
        """The AL convergence numbers for the iteration on screen, as the stepper
        labelled them when it took the step.

        ``live`` follows _iter_view: mid-run the newest snapshot is being drawn,
        so report THAT one. Reading the slider instead would label the picture
        with the index of a frame it is not showing."""
        # Gated on the SLIDER, not the markdown: with no slider up the markdown
        # is carrying the "nothing recorded" note, which must survive re-renders.
        if getattr(self, "iter_slider", None) is None:
            return
        n = self.result.num_iterates()
        i = max(n - 1, 0) if live else self._current_iterate()
        notes = self.result.iterate_notes
        body = notes[i] if notes is not None else ""
        self.g_iter_status.content = f"iterate {i} / {n - 1}  \n{body}"
