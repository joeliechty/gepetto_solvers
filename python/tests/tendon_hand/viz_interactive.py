"""Interactive viser visualizer for the tendon-hand FK solver and the stepped IK
solve.

Exposes the solver knobs as live web GUI controls -- object picker, wrist start
pose, per-finger flexor tensions, per-finger contact toggles, collision / table
options, AL settings. *FK* re-poses the hand from the current sliders (and the
pose / tension sliders re-solve it live as they move); *Step* advances the IK
solve by exactly one Augmented Lagrangian outer iteration, *Auto solve* keeps
stepping until it converges or stalls, and *Stop* breaks out of a running
auto-solve. Every step is kept, so the *Solve steps* scrubber replays the
convergence one iteration at a time (initial guess -> each outer iteration).

Changing the constraint set (object, contacts, collision, table) restarts the IK
loop, because the Augmented Lagrangian duals it carries describe the constraints
it has been working on. *Seed from current* is the way to change one anyway and
keep the posture: it commits the state on screen as the starting values of the
next solve, so the restart begins where the last solve got to instead of at a
straight hand with Q = 0. The same button starts an IK solve from an FK pose you
dialled in by hand. *Reset defaults* puts every control back and cold-starts.

Object contact, table contact, object collision and table collision are four
independent switches (*Contact targets*, *Collision*, *Table*), each acting on
the shared per-finger mask in *Contact fingers*. That is what makes a stalled
grasp bisectable: solve for the object alone, the table alone, or both, with or
without either avoidance, and see which constraint family is the one refusing to
close.

The solvers are the reusable ``HandFKSolver`` / ``HandIKStepper`` classes in
``tendon_hand/solvers.py``; the 3D scene is drawn by
``_plotting/viser_hand.ViserHandScene``. The trajectory planner and the Section
1.8 controller are not part of this app -- see the ``traj_*`` scripts and
``viz_controller.py``.

Run (from the ``python/`` directory):
    python -m tests.tendon_hand.viz_interactive
then open the printed http://localhost:8080 URL.

Optional headless self-check of the solver classes (no browser):
    python -m tests.tendon_hand.viz_interactive --smoke
"""

import argparse
import sys
import threading

import numpy as np

from .scene import get_primitive_specs, GRASP_FLEXOR_TENSION, TABLE_NORMAL
from .solvers import (
    HandSolveParams, HandFKSolver, HandIKStepper,
    resolve_scene, resolve_table_origin, capabilities,
    euler_to_R, plane_witness, DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY)


FINGER_LABELS = ["index", "middle", "ring", "pinky", "thumb"]

# Display-only suffix for the baked-SDF spheres in the object dropdown, so they
# read apart from the analytic ``*_sphere_ellipsoid`` look-alikes. The spec keys
# (and the demo scripts' argparse choices) keep the un-suffixed names; only the
# label the user picks from carries "_sdf".
SDF_DROPDOWN_LABELS = {"sphere": "sphere_sdf", "big_sphere": "big_sphere_sdf"}


# The wrist sliders and the solvers must agree on what "pitch" means, so the
# convention lives with the params rather than here.
_euler_to_R = euler_to_R


# ---------------------------------------------------------------------------
# Headless smoke test -- validates the solver classes independently of viser.
# ---------------------------------------------------------------------------

def _smoke():
    print(f"Smoke-testing the hand solver classes "
          f"({HandSolveParams().primitive}, defaults)...")
    ok = True
    caps = capabilities()

    # FK: one frame, no contact.
    res = HandFKSolver(HandSolveParams()).solve()
    status = "ok" if len(res.frames) == 1 else "BAD"
    ok = ok and status == "ok"
    print(f"  [{'FK':>8}] frames={len(res.frames)} (expect 1) [{status}] | "
          f"iters={res.meta.iterations} err={res.meta.error:.3g}")

    # The stepper, in the three contact configurations the split toggles exist
    # to bisect. A handful of steps is enough to prove the loop runs and carries;
    # convergence is what the GUI's Auto solve is for.
    if not caps["ik_stepping"]:
        print("  [IK-step] skipped -- binding has no TendonHandSolver.reset_al_duals")
    else:
        cases = [("IK", False, False)]
        if caps["table"]:
            cases += [("IK-table", True, False), ("IK-both", True, True)]
        else:
            print("  [IK-table] skipped -- binding has no support-plane env fields")
        for label, table, obj in cases:
            params = HandSolveParams()
            if table:
                params.table = True
                params.table_contact = True
                params.object_contact = obj
            # The last stepped result, captured the way the GUI does -- run()
            # returns only the status, and the gaps live on the result.
            last = {}
            st = HandIKStepper(params).run(
                max_steps=5, on_step=lambda r, s: last.update(res=r))
            res = last.get("res")
            # One snapshot per step plus the initial guess.
            n = res.num_iterates() if res is not None else 0
            status = "ok" if st.steps > 0 and n == st.steps + 1 else "BAD"
            if status == "BAD":
                ok = False
            extra = ""
            if res is not None:
                if params.object_contact:
                    extra += f" | worst object gap {res.worst_gap(0):+.5f} m"
                if res.table_contact_names():
                    extra += (f" | worst table gap "
                              f"{res.worst_table_gap(params, 0):+.5f} m")
            print(f"  [{label:>8}] steps={st.steps} state={st.state} "
                  f"snapshots={n} [{status}] | violation={st.violation:.3e} "
                  f"cost={st.cost:.4g}{extra}")
    print("Smoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Interactive app.
# ---------------------------------------------------------------------------

class HandVizApp:

    def __init__(self, server):
        import viser  # local import so --smoke needs no viser
        self.viser = viser
        self.server = server
        self.params = HandSolveParams()
        # Which solver produced what is on screen: "FK" for a posed hand, "IK"
        # once the stepper has been driven. Gates the live FK re-solve and labels
        # the status readout; there is no mode picker.
        self.mode = "FK"
        self.result = None
        self._solving = False
        # Cached IK stepper: it owns the AL outer loop being advanced one
        # iteration per Step, so it has to outlive a single step.
        self.stepper = None
        self._auto_stop = threading.Event()
        self._auto_thread = None
        # True while Reset is writing the controls back to their defaults, so the
        # per-handle callbacks (live FK, object rebuild) sit out the restore and
        # the one solve at the end of it is the only one that runs.
        self._restoring = False
        # What this installed binding supports, so we can gate controls a stale
        # .so would crash on (ellipsoid objects, the table, cull margin).
        self.caps = capabilities()

        from .._plotting.viser_hand import ViserHandScene
        self.scene = ViserHandScene(server, FINGER_LABELS)

        # Park every (current and future) client's camera on the -X/palmar side so
        # the finger curl reads as a grasp instead of bending backwards. Without
        # this viser opens from the opposite side and the correct solve looks wrong.
        server.on_client_connect(lambda client: self._aim_camera(client))

        self._build_gui()
        # A cached FK solver so wrist/tension tweaks warm-start (rebuilt on object
        # change only).
        self._rebuild_fk()
        self._refresh_object()
        self._fk_solve()

    def _aim_camera(self, client):
        """Point one client's camera at the current object from the demo viewpoint."""
        _spec, center, _rot, _pose = resolve_scene(self.params)
        pos, look = self.scene.grasp_camera(center)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(float(v) for v in pos)
        client.camera.look_at = tuple(float(v) for v in look)

    def _aim_all_cameras(self):
        for client in self.server.get_clients().values():
            self._aim_camera(client)

    # -- solver plumbing --

    def _rebuild_fk(self):
        self.fk_solver = HandFKSolver(self.params)
        # The FK solver is rebuilt whenever the object changes, and the object is
        # part of the stepper's constraint set too.
        self._invalidate_stepper()

    def _sync_wrist(self):
        R = _euler_to_R(self.g_roll.value, self.g_pitch.value, self.g_yaw.value)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.g_tx.value, self.g_ty.value, self.g_tz.value]
        self.params.wrist_pose = T

    def _sync_params(self):
        p = self.params
        p.primitive = self._label_to_key[self.g_object.value]
        # let center/rotation re-derive from the primitive
        p.object_center = None
        p.object_rotation = None
        self._sync_wrist()
        p.passive_tension = self.g_passive.value
        p.flexor_tensions = [s.value for s in self.g_flexors]
        p.contact_fingers = [c.value for c in self.g_contacts]
        p.object_contact = self.g_obj_contact.value
        p.table_contact = self.g_tbl_contact.value
        p.sigma_wrist_pos = 10.0 ** self.g_sig_pos.value
        p.sigma_wrist_rot = 10.0 ** self.g_sig_rot.value
        # AL
        p.al_mu = self.g_al_mu.value
        p.al_rate = self.g_al_rate.value
        p.al_iters = self.g_al_iters.value
        p.ik_settle_steps = int(self.g_ik_settle.value)
        # collision
        p.collision = self.g_collision.value
        p.collision_radius = self.g_coll_radius.value
        p.collision_sigma = 10.0 ** self.g_coll_sigma.value
        p.cull_margin = (None if not self.caps["collision_cull"] or self.g_cull.value <= 0
                         else self.g_cull.value)
        # table
        p.table = self.g_table.value and self.caps["table"]
        p.plane_normal = np.array(TABLE_NORMAL, float)
        p.plane_avoidance = self.g_plane_avoid.value
        p.plane_origin = None  # auto (under object); offset applied below
        # display toggles
        self.scene.show_discs = self.g_show_discs.value
        self.scene.show_contact_spheres = self.g_show_contact.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self.scene.show_gap_lines = self.g_show_gaps.value

    def _table_origin(self):
        spec, center, _rot, _pose = resolve_scene(self.params)
        origin = resolve_table_origin(self.params, spec, center)
        origin = np.asarray(origin, float) + self.g_plane_offset.value * np.array(
            TABLE_NORMAL, float)
        return origin

    # -- rendering --

    def _refresh_object(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        self.scene.set_object(spec, center, rotation)
        if self.params.table:
            self.scene.set_table(self._table_origin(), self.params.plane_normal)
        else:
            self.scene.clear_table()

    def _render_frame(self):
        if self.result is None:
            return
        # Render whichever solve snapshot the convergence scrubber selects; with
        # no scrubber up this is the result itself, so the gap readouts below
        # describe the intermediate state without knowing about iterates at all.
        # Every result here is a single state, so there is only ever frame 0.
        res = self._iter_view()
        # Only the fingers this solve drove onto a surface get a gap line for it;
        # a distance readout on a finger nothing asked to touch is just noise.
        # The two sets are independent, so a finger can carry both lines, one, or
        # neither.
        gaps = res.contact_witness(0)
        names = set(res.contact_names() if self.params.object_contact else [])
        gaps = {name: v for name, v in gaps.items() if name in names}
        table_names = res.table_contact_names()
        table_gaps = (plane_witness(self.params, res, 0, names=table_names)
                      if table_names else None)
        self._report_iterate()
        self.scene.update(res.frames[0],
                          tip_radii=res.tip_radii,
                          collision_radius=self.params.collision_radius,
                          # The spheres are drawn whenever EITHER consumer is
                          # using them, matching what the solve actually built.
                          collision=(self.params.collision
                                     or (self.params.table
                                         and self.params.plane_avoidance)),
                          gaps=gaps,
                          table_gaps=table_gaps)

    def _set_status(self, text):
        self.g_status.content = text

    # -- solve --

    def _set_solving(self, solving):
        """Grey out the solve buttons while a solve runs so it's clear the app is
        busy; restore them when it finishes."""
        self.g_fk.disabled = solving
        self.g_fk.label = "Solving..." if solving else "FK"
        # Step / Auto go grey too, so an auto-run cannot be re-entered; Stop is
        # driven separately and stays live for exactly that run.
        for btn in (getattr(self, "g_ik_step", None), getattr(self, "g_ik_auto", None)):
            if btn is not None:
                btn.disabled = solving or not self.caps["ik_stepping"]
        # Seeding mid-solve would capture a half-drawn state and Reset would pull
        # the params out from under a running step, so both wait it out.
        if getattr(self, "g_seed", None) is not None:
            self.g_seed.disabled = solving or not self.caps["solver_seed"]
        if getattr(self, "g_reset", None) is not None:
            self.g_reset.disabled = solving

    def _fk_solve(self, _=None):
        """Re-pose the hand from the current sliders with the FK solver.

        Also the explicit "start over" action for the IK loop: it drops any
        partially stepped solve AND the committed seed, so the next Step is a
        true cold start. (Seed again afterwards to hand the fresh FK pose to the
        solver instead.)"""
        if self._solving:
            return
        self._solving = True
        self._set_solving(True)
        try:
            self._sync_params()
            self._refresh_object()
            self._invalidate_stepper()
            self._clear_seed()
            self.mode = "FK"
            self._set_status("Solving (FK)...")
            # Reuse the cached FK solver (shares self.params) so this warm-starts.
            self.result = self.fk_solver.solve()
            self._rebuild_iter_slider()
            self._render_frame()
            self._report()
        except Exception as exc:  # surface solver errors in the GUI, keep serving
            self._set_status(f"**Error:** {exc}")
            raise
        finally:
            self._set_solving(False)
            self._solving = False

    def _report(self):
        m = self.result.meta
        lines = [f"**{self.mode}** &nbsp; iters={m.iterations} &nbsp; "
                 f"err={m.error:.3g} &nbsp; {m.total_time_ms:.0f} ms"]
        if self.mode != "FK":
            lines.extend(self._contact_lines(-1))
        self._set_status("  \n".join(lines))

    def _fingers_label(self, names):
        return ("none" if not names
                else ", ".join(names)
                if len(names) < len(self.result.finger_names) else "all")

    def _contact_lines(self, k):
        """The per-surface contact readout: which fingers were driven onto each
        surface and how far they ended up from it. Reported per surface because
        the whole point of splitting the toggles is telling the two apart -- one
        combined number cannot say which family is the one refusing to close."""
        lines = []
        if self.params.object_contact:
            names = self.result.contact_names()
            lines.append(f"object contact: {self._fingers_label(names)}")
            lines.append(f"terminal worst object gap: "
                         f"{self.result.worst_gap(k):+.5f} m")
        table_names = self.result.table_contact_names()
        if table_names:
            lines.append(f"table contact: {self._fingers_label(table_names)}")
            lines.append(f"terminal worst table gap: "
                         f"{self.result.worst_table_gap(self.params, k):+.5f} m")
        if not lines:
            lines.append("contact: none (no surface targeted)")
        return lines

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
        if self.stepper is None:
            self.stepper = HandIKStepper(self.params)
        return self.stepper

    def _show_step(self, result, status):
        """Render one stepped state and update both status readouts. Called from
        the auto-run thread as well as the Step button."""
        self.result = result
        self._render_frame()
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
        if self._auto_stop.is_set() and status.state == "running":
            verdict = "**stopped**"
        gaps = [f"object {self.result.worst_gap(0):+.5f} m"] \
            if self.params.object_contact else []
        if self.result.table_contact_names():
            gaps.append(f"table {self.result.worst_table_gap(self.params, 0):+.5f} m")
        self._set_status(
            f"**IK step {status.steps}** &nbsp; {verdict}  \n"
            f"violation={status.violation:.3e} &nbsp; cost={status.cost:.4g} "
            f"&nbsp; mu={status.mu:.3g}  \n"
            f"worst gap: {' &nbsp; '.join(gaps) or 'n/a'}")

    def _ik_step(self, _=None):
        """One Augmented Lagrangian outer iteration, continuing the last one."""
        if self._solving or not self.caps["ik_stepping"]:
            return
        self._solving = True
        self._set_solving(True)
        try:
            self._sync_params()
            self._refresh_object()
            self.mode = "IK"
            self._auto_stop.clear()
            stepper = self._ensure_stepper()
            self._show_step(stepper.step(), stepper.status())
            self._rebuild_iter_slider()
        except Exception as exc:
            self._set_status(f"**Error:** {exc}")
            raise
        finally:
            self._set_solving(False)
            self._solving = False

    def _ik_auto(self, _=None):
        """Step to convergence on a worker thread, redrawing after each iteration.

        The loop has to leave viser's callback thread free: run inline and the
        Stop click would sit in the queue until the whole solve finished, which
        is precisely when it stops being useful."""
        if self._solving or not self.caps["ik_stepping"]:
            return
        self._sync_params()
        self._refresh_object()
        self.mode = "IK"
        self._auto_stop.clear()
        self._solving = True
        self._set_solving(True)
        self.g_ik_stop.disabled = False
        stepper = self._ensure_stepper()

        def worker():
            try:
                status = stepper.run(max_steps=self.g_ik_max.value,
                                     on_step=self._show_step,
                                     should_stop=self._auto_stop.is_set)
                self._report_step_status(status)
                # Only now: rebuilding the slider once per frame mid-animation
                # would tear a GUI handle down and re-add it every iteration.
                self._rebuild_iter_slider()
            except Exception as exc:
                self._set_status(f"**Error:** {exc}")
            finally:
                self.g_ik_stop.disabled = True
                self._set_solving(False)
                self._solving = False

        self._auto_thread = threading.Thread(target=worker, daemon=True)
        self._auto_thread.start()

    def _ik_stop(self, _=None):
        """Ask a running auto-solve to stop; it breaks before the next step."""
        self._auto_stop.set()

    # -- seeding and reset --

    def _set_seed_status(self, text):
        self.g_seed_status.content = text

    def _clear_seed(self):
        """Forget the committed warm start: the next Step cold-starts."""
        self.params.initial_state = None
        self._set_seed_status("seed: none (Step cold-starts)")

    def _seed_from_current(self, _=None):
        """Commit the state on screen as the starting posture of the next solve.

        The stepper cannot absorb a change to the CONSTRAINT SET -- its duals
        describe the old one -- so any such change rebuilds it, and a rebuilt
        solver cold-starts from a straight hand with Q = 0. That is what this
        works around: the committed posture is handed to the new solver as its
        initial values, so "change a setting and carry on from where the solve
        got to" becomes possible. It is equally the way to start an IK solve from
        an FK pose you dialled in by hand instead of from the cold guess.

        Whatever the *iterate* scrubber is showing is what gets committed, so you
        can also rewind to an earlier iteration and branch from there."""
        if self._solving or not self.caps["solver_seed"]:
            return
        res = self._iter_view()
        state = None if res is None else res.state(0)
        if state is None:
            self._set_seed_status("**nothing to seed from** -- solve first")
            return
        self.params.initial_state = state
        # The committed posture only reaches the solver through a construction,
        # so the loop in progress has to go; keeping it would silently ignore
        # the seed until the next constraint-set change.
        self._invalidate_stepper()
        where = (f"IK step {self._current_iterate()}" if self.mode == "IK"
                 else "the FK pose")
        self._set_seed_status(f"seed: **{where}** -- next Step starts there")

    def _reset_defaults(self, _=None):
        """Put every control back to the value it was built with and cold-start.

        A full reset rather than a re-solve: fresh params (so the seed and any
        derived scene state go too), fresh FK solver, no stepper, camera back on
        the default object."""
        if self._solving:
            return
        self._restoring = True
        try:
            for handle, value in self._gui_defaults:
                handle.value = value
        finally:
            self._restoring = False
        self.params = HandSolveParams()
        self._sync_params()
        self._rebuild_fk()          # also drops the stepper
        self._refresh_object()
        self._aim_all_cameras()
        self._fk_solve()            # clears the seed and re-renders
        self._set_status("**reset** to defaults  \n" + self.g_status.content)

    def _live_fk(self, _=None):
        """FK is fast and warm-starts, so re-solve live as sliders move.

        Only while the hand is FK-posed: once the stepper is running, the same
        sliders are read live by every step, and re-solving FK here would throw
        that loop away mid-solve."""
        if self.mode == "FK" and not self._solving and not self._restoring:
            self._fk_solve()

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

        Torn down and re-created rather than resized because the iteration count
        is whatever the AL outer loop has run to date -- and an FK pose (or a
        freshly restarted loop) must leave no slider at all."""
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
                    "press **Step** or **Auto solve** to record IK iterations")
        else:
            with self.iter_folder:
                self.iter_slider = self.server.gui.add_slider(
                    "iterate", min=0, max=n - 1, step=1, initial_value=n - 1,
                    hint="0 is the initial guess, the last is the most recent "
                         "step; in between, one Augmented Lagrangian outer "
                         "iteration each.")
                self.g_iter_status = self.server.gui.add_markdown("")
            self.iter_slider.on_update(lambda _: self._render_frame())

    def _iter_view(self):
        """The result to render: the selected snapshot when the scrubber is up,
        otherwise the solve's own final state."""
        if self.result is None or getattr(self, "iter_slider", None) is None:
            return self.result
        return self.result.at_iterate(self._current_iterate())

    def _report_iterate(self):
        """The AL convergence numbers for the scrubbed iteration, as the stepper
        labelled them when it took the step."""
        # Gated on the SLIDER, not the markdown: with no slider up the markdown
        # is carrying the "nothing recorded" note, which must survive re-renders.
        if getattr(self, "iter_slider", None) is None:
            return
        i, n = self._current_iterate(), self.result.num_iterates()
        notes = self.result.iterate_notes
        body = notes[i] if notes is not None else ""
        self.g_iter_status.content = f"iterate {i} / {n - 1}  \n{body}"

    # -- GUI construction --

    def _input_handles(self):
        """Every value-carrying control, in build order. Buttons and markdown are
        deliberately absent -- Reset restores values, not widgets."""
        return ([self.g_object, self.g_ik_max, self.g_ik_settle,
                 self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw,
                 self.g_sig_pos, self.g_sig_rot, self.g_passive]
                + self.g_flexors
                + [self.g_obj_contact, self.g_tbl_contact]
                + self.g_contacts
                + [self.g_collision, self.g_coll_radius, self.g_coll_sigma,
                   self.g_cull,
                   self.g_table, self.g_plane_offset, self.g_plane_avoid,
                   self.g_al_mu, self.g_al_rate, self.g_al_iters,
                   self.g_show_contact, self.g_show_collision,
                   self.g_show_discs, self.g_show_gaps])

    def _build_gui(self):
        gui = self.server.gui
        # Ellipsoid objects need the analytic-surface env fields; hide them on a
        # binding that lacks them.
        keys = [k for k, v in get_primitive_specs().items()
                if v["type"] != "ellipsoid" or self.caps["ellipsoid"]]
        # Map the displayed dropdown label back to the real spec key (identity
        # except for the "_sdf"-suffixed baked spheres).
        self._label_to_key = {SDF_DROPDOWN_LABELS.get(k, k): k for k in keys}
        labels = list(self._label_to_key)

        step_hint = (None if self.caps["ik_stepping"]
                     else "requires a rebuilt _crest_sparse with "
                          "TendonHandSolver.reset_al_duals")

        with gui.add_folder("Solver"):
            # Opens on HandSolveParams' own default (the mid analytic sphere),
            # so the GUI and a headless HandSolveParams() describe the same
            # scene. Falls back to the first entry if that primitive is hidden
            # -- an ellipsoid needs the analytic-surface env fields, which a
            # stale binding may not have.
            default_label = SDF_DROPDOWN_LABELS.get(self.params.primitive,
                                                    self.params.primitive)
            if default_label not in self._label_to_key:
                default_label = labels[0]
            self.g_object = gui.add_dropdown(
                "object", labels, initial_value=default_label)
            self.g_fk = gui.add_button(
                "FK", icon=self.viser.Icon.PLAYER_PLAY,
                hint="Re-pose the hand from the current wrist / tension sliders "
                     "with the FK solver -- no contact, no collision. Also the "
                     "restart button for the IK loop: it drops any stepped solve "
                     "in progress.")
            self.g_ik_step = gui.add_button(
                "Step", icon=self.viser.Icon.PLAYER_TRACK_NEXT,
                disabled=not self.caps["ik_stepping"],
                hint=step_hint or (
                    "Advance the IK solve by exactly one Augmented Lagrangian "
                    "outer iteration, continuing the previous one. Tensions and "
                    "the wrist pose are re-read every step, so you can nudge "
                    "them mid-solve; changing the object, contacts, collision "
                    "or table starts over."))
            self.g_ik_auto = gui.add_button(
                "Auto solve", icon=self.viser.Icon.PLAYER_PLAY,
                disabled=not self.caps["ik_stepping"],
                hint=step_hint or (
                    "Keep stepping until the solve converges, stalls or hits "
                    "the cap, redrawing after every iteration."))
            self.g_ik_stop = gui.add_button(
                "Stop", icon=self.viser.Icon.PLAYER_STOP, disabled=True,
                hint="Break out of a running auto-solve before the next step; "
                     "the steps already taken are kept.")
            self.g_ik_max = gui.add_slider(
                "max steps", 5, 300, 5, 200,
                disabled=not self.caps["ik_stepping"],
                hint="Backstop for Auto solve. A converging grasp takes ~30 "
                     "iterations, so hitting this means it is not converging.")
            self.g_ik_settle = gui.add_slider(
                "settle steps", 0, 5, 1, HandSolveParams().ik_settle_steps,
                disabled=not self.caps["ik_stepping"],
                hint="Leading steps that pin the flexor prior as tightly as the "
                     "passives, to settle the cold start. The solver seeds every "
                     "tension at zero on an already-curled rod, and the flexor -- "
                     "by far the softest variable -- absorbs that inconsistency by "
                     "swinging NEGATIVE, i.e. the fingers hyperextend and then "
                     "spend ~13 steps crawling back to the FK pose. One settling "
                     "step lands step 1 on the FK pose instead, and reaches the "
                     "same grasp. Set 0 to watch the old behaviour.")
            self.g_status = gui.add_markdown("")

        # The scrubber over the steps taken. The slider and its readout are built
        # per step by _rebuild_iter_slider().
        self.iter_folder = gui.add_folder("Solve steps")

        with gui.add_folder("Seed / reset"):
            self.g_seed = gui.add_button(
                "Seed from current", icon=self.viser.Icon.PIN,
                disabled=not self.caps["solver_seed"],
                hint=("Commit the state on screen (the FK pose, or whichever "
                      "iteration the scrubber is showing) as the starting "
                      "posture of the next solve. Changing the object, "
                      "contacts, collision or table restarts the IK loop from a "
                      "straight hand; seed first and it restarts from here "
                      "instead, so you can change a setting and carry on. "
                      "Pressing FK drops the seed."
                      if self.caps["solver_seed"]
                      else "requires a rebuilt _crest_sparse with "
                           "TendonHandSolverConfig.initial_state"))
            self.g_reset = gui.add_button(
                "Reset defaults", icon=self.viser.Icon.ROTATE,
                hint="Put every control back to the value it opened with, drop "
                     "the seed and the stepped solve, and re-pose with FK.")
            self.g_seed_status = gui.add_markdown("")

        with gui.add_folder("Wrist start pose"):
            # Seeded from the shared default (solvers.DEFAULT_WRIST_*) so the
            # pose the GUI opens on IS HandSolveParams' default -- a headless
            # repro of what is on screen needs no numbers copied across.
            x0, y0, z0 = DEFAULT_WRIST_XYZ
            r0, p0, yw0 = DEFAULT_WRIST_RPY
            self.g_tx = gui.add_slider("x (m)", -0.1, 0.1, 0.001, x0)
            self.g_ty = gui.add_slider("y (m)", -0.1, 0.1, 0.001, y0)
            self.g_tz = gui.add_slider("z (m)", -0.1, 0.1, 0.001, z0)
            self.g_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, r0)
            self.g_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, p0)
            self.g_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, yw0)
            self.g_sig_pos = gui.add_slider("log10 sigma_pos", -6, 2, 0.5, -4)
            self.g_sig_rot = gui.add_slider("log10 sigma_rot", -6, 2, 0.5, -3)

        with gui.add_folder("Tensions (N)"):
            self.g_passive = gui.add_slider("passive", 0.0, 3.0, 0.05, 0.5)
            self.g_flexors = [
                gui.add_slider(lbl, 0.0, 3.0, 0.05, GRASP_FLEXOR_TENSION)
                for lbl in FINGER_LABELS]

        with gui.add_folder("Contact targets"):
            self.g_obj_contact = gui.add_checkbox(
                "object contact", True,
                hint="Drive the checked fingertips onto the OBJECT surface. "
                     "Turn off to leave the object as pure collision geometry "
                     "-- the way to see what the table constraints do on their "
                     "own. Applies on the next solve.")
            self.g_tbl_contact = gui.add_checkbox(
                "table contact", False,
                hint="Drive the checked fingertips onto the SUPPORT PLANE (one "
                     "equality per finger on the distance from its contact "
                     "sphere to the plane). Needs the Table folder's *enabled*; "
                     "combine with object contact to solve for both at once.")

        with gui.add_folder("Contact fingers"):
            self.g_contacts = [
                gui.add_checkbox(
                    lbl, True,
                    hint="Which fingers the contact targets above apply to (IK "
                         "only; FK never uses contact). Unchecked fingers keep "
                         "collision avoidance, so they stay out of the object "
                         "and off the table without being driven onto either. "
                         "Applies on the next solve.")
                for lbl in FINGER_LABELS]

        with gui.add_folder("Collision"):
            self.g_collision = gui.add_checkbox(
                "object collision", False,
                hint="Keep every non-contact sphere out of the OBJECT. The "
                     "table's own avoidance is separate (Table folder); "
                     "finger-finger avoidance comes on with either.")
            self.g_coll_radius = gui.add_slider("sphere radius (m)", 0.001, 0.01, 0.0005, 0.003)
            self.g_coll_sigma = gui.add_slider("log10 sigma", -6, 0, 0.5, -4)
            self.g_cull = gui.add_slider("cull margin (m, 0 off)", 0.0, 0.1, 0.005, 0.0)

        with gui.add_folder("Table"):
            self.g_table = gui.add_checkbox(
                "enabled", False, disabled=not self.caps["table"],
                hint=None if self.caps["table"]
                else "requires a newer _crest_sparse build (plane env fields)")
            # Offset from the scene's own seating, which now half-buries the
            # object (HandSolveParams.table_burial = 0.5) rather than resting it
            # on the plane. Dial in -half_extent to recover the old geometry.
            self.g_plane_offset = gui.add_slider("height offset (m)", -0.1, 0.1, 0.002, 0.0)
            self.g_plane_avoid = gui.add_checkbox(
                "table collision", True,
                hint="Keep every non-contact sphere out of the half-space. "
                     "Independent of object collision -- the collision spheres "
                     "are attached for whichever of the two is on.")

        with gui.add_folder("Augmented Lagrangian"):
            self.g_al_mu = gui.add_slider("mu", 0.1, 10.0, 0.1, 1.0)
            self.g_al_rate = gui.add_slider("rate", 1.1, 5.0, 0.1, 2.0)
            self.g_al_iters = gui.add_slider("max iters", 5, 100, 5, 40)

        with gui.add_folder("Display"):
            self.g_show_contact = gui.add_checkbox("contact spheres", True)
            self.g_show_collision = gui.add_checkbox("collision spheres", True)
            self.g_show_discs = gui.add_checkbox("routing discs", False)
            self.g_show_gaps = gui.add_checkbox(
                "contact distance", True,
                hint="Fingertip-to-object gap in mm; green under 15 mm, red over.")

        # Every value-carrying control, captured as built: this IS the definition
        # of "defaults" that Reset restores, so the two cannot drift.
        self._gui_defaults = [(h, h.value) for h in self._input_handles()]
        self._clear_seed()

        # -- callbacks --
        self.g_fk.on_click(self._fk_solve)
        self.g_ik_step.on_click(self._ik_step)
        self.g_ik_auto.on_click(self._ik_auto)
        self.g_ik_stop.on_click(self._ik_stop)
        self.g_seed.on_click(self._seed_from_current)
        self.g_reset.on_click(self._reset_defaults)

        @self.g_object.on_update
        def _(_):
            if self._restoring:
                return
            self.params.primitive = self._label_to_key[self.g_object.value]
            self.params.object_center = None
            self.params.object_rotation = None
            self._rebuild_fk()      # FK solver carries the object for its result/spec
            self._refresh_object()
            self._aim_all_cameras()  # re-center on the new object's location
            self._fk_solve()

        # Live FK re-solve on the pose / tension sliders (fast, warm-started).
        for h in ([self.g_tx, self.g_ty, self.g_tz, self.g_roll, self.g_pitch,
                   self.g_yaw, self.g_passive] + self.g_flexors):
            h.on_update(self._live_fk)

        # Display toggles re-render the current frame without re-solving.
        for h in (self.g_show_contact, self.g_show_collision, self.g_show_discs,
                  self.g_show_gaps):
            h.on_update(lambda _: (self._sync_params(), self._render_frame()))
        # The contact checkboxes ride along only to keep self.params in sync;
        # like every other solver knob they take effect on the next solve, and
        # the gap lines keep describing the solve that is actually on screen
        # until then. They do change the stepper's constraint set, though -- which
        # surface a finger is driven onto IS that set, so a carried dual is
        # meaningless and the loop has to restart.
        for h in self.g_contacts + [self.g_obj_contact, self.g_tbl_contact]:
            h.on_update(lambda _: (self._sync_params(),
                                   self._invalidate_stepper(),
                                   self._render_frame()))
        # Table toggle / height updates the static slab immediately.
        for h in (self.g_table, self.g_plane_offset):
            h.on_update(lambda _: (self._sync_params(), self._refresh_object(),
                                   self._invalidate_stepper()))
        # Collision knobs are part of the constraint set too.
        for h in (self.g_collision, self.g_coll_radius, self.g_coll_sigma,
                  self.g_cull, self.g_plane_avoid):
            h.on_update(lambda _: self._invalidate_stepper())
        # AL knobs are baked into the stepper's config at construction, unlike
        # the tensions it re-reads every step, so they need a rebuild too.
        # "settle steps" is read live, but it counts off steps ALREADY taken, so
        # changing it part-way through a run would do nothing without a restart.
        for h in (self.g_al_mu, self.g_al_rate, self.g_sig_pos, self.g_sig_rot,
                  self.g_ik_settle):
            h.on_update(lambda _: self._invalidate_stepper())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Headless self-check of the solver classes (no viser).")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.smoke:
        sys.exit(_smoke())

    import viser
    server = viser.ViserServer(port=args.port)
    HandVizApp(server)
    print(f"viser hand visualizer running -- open http://localhost:{args.port}")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
