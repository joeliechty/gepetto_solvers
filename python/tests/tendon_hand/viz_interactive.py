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
it has been working on. *Warm start* is the way to change one anyway and keep the
posture: while the latch is on, every restart begins from the state on screen
instead of from a straight hand with Q = 0 -- so it also starts an IK solve from
an FK pose you dialled in by hand. Only the posture carries; the penalty schedule
restarts regardless. *Reset defaults* puts every control back and cold-starts.

Run (from the ``crest-sparse`` root, so ``crest_sparse`` resolves to the
INSTALLED build -- launching from ``python/`` picks up the stale in-tree
``python/crest_sparse/_crest_sparse*.so`` instead, and every capability-gated
control silently goes dead):
    python -m python.tests.tendon_hand.viz_interactive

Every constraint from the paper's Chapter 2 (Eq 2.8-2.19) is an independent
switch in the *Constraints* folder -- object/table contact, object/self/table
collision, drop-normal-row SDF contact, opposition half-space, pre-grasp
centering -- each acting on the shared per-finger mask in its nested *fingers*
sub-folder. A box is the whole story: checked means that constraint family is in
the graph, with no second toggle it silently waits on. That is what makes a
stalled grasp bisectable: solve for the object alone, the table alone, or both,
with or without any of the three avoidances, and see which constraint family is
the one refusing to close. (Their tuning sliders --
collision radius/sigma/cull margin, table height offset -- stay in the
*Collision*/*Table* folders alongside the object/primitive picker.)

The solvers are the reusable ``HandFKSolver`` / ``HandIKStepper`` classes in
``tendon_hand/solvers.py``; the 3D scene is drawn by
``_plotting/viser_hand.ViserHandScene``. The trajectory planner is not part of
this app -- see the ``traj_*`` scripts.

then open the printed http://localhost:8080 URL. The startup line names the
binding that was actually loaded and lists any capability missing from it.

Optional headless self-check of the solver classes (no browser):
    python -m python.tests.tendon_hand.viz_interactive --smoke
"""

import argparse
import math
import sys
import threading

import numpy as np

from .scene import get_primitive_specs, GRASP_FLEXOR_TENSION, TABLE_NORMAL
from .solvers import (
    HandSolveParams, HandFKSolver, HandIKStepper,
    resolve_scene, resolve_table_origin, capabilities,
    euler_to_R, R_to_euler, solved_wrist_pose, plane_witness,
    half_space_witness, pregrasp_center_witness, pregrasp_axis_witness,
    pregrasp_centroid_witness, default_half_space_axis, PHASE_PRESETS,
    FLEXOR_IDX, DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY)
from .config import pinch_pose


FINGER_LABELS = ["index", "middle", "ring", "pinky", "thumb"]

# This app's own startup object -- see HandVizApp.__init__ for why it's set
# there rather than just changed on the dropdown widget.
DEFAULT_OBJECT_PRIMITIVE = "pen"

# Display-only suffix for the baked-SDF spheres in the object dropdown, so they
# read apart from the analytic ``*_sphere_ellipsoid`` look-alikes. The spec keys
# (and the demo scripts' argparse choices) keep the un-suffixed names; only the
# label the user picks from carries "_sdf".
SDF_DROPDOWN_LABELS = {"sphere": "sphere_sdf", "big_sphere": "big_sphere_sdf"}

# Which half of the split the THUMB is sent to -- the opposition axis's sign,
# which the object's own geometry cannot answer (see
# solvers.orient_opposition_axis). Label -> HandSolveParams.half_space_flip.
OPPOSITION_SIDES = {
    "auto (match the hand)": None,
    "as derived": False,
    "flipped": True,
}


# The wrist sliders and the solvers must agree on what "pitch" means, so the
# convention lives with the params rather than here.
_euler_to_R = euler_to_R


def binding_path():
    """Where the loaded ``crest_sparse`` came from.

    Worth reporting because there are two of them: the installed build in
    site-packages, and a stale in-tree ``python/crest_sparse/_crest_sparse*.so``
    that shadows it whenever the app is launched from the ``python/`` directory.
    A control gated on ``capabilities()`` then goes quietly dead against a build
    the source has long since moved past, which looks like the feature failing
    rather than the import resolving somewhere unexpected."""
    import crest_sparse
    return crest_sparse.__file__


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
        # This app's own startup default -- not HandSolveParams' own default,
        # which stays whatever headless callers/other scripts expect. Set
        # here (not just on the dropdown widget below) because _rebuild_fk()/
        # _refresh_object() run before the first _sync_params() and read
        # self.params.primitive directly; the dropdown's own default_label
        # computation already reads it too, so it follows automatically.
        self.params.primitive = DEFAULT_OBJECT_PRIMITIVE
        # Which solver produced what is on screen: "FK" for a posed hand, "IK"
        # once the stepper has been driven. Gates the live FK re-solve and labels
        # the status readout; there is no mode picker.
        self.mode = "FK"
        self.result = None
        self._solving = False
        # Cached IK stepper: it owns the AL outer loop being advanced one
        # iteration per Step, so it has to outlive a single step.
        self.stepper = None
        # Warm-start latch: while on, every (re)build of the stepper starts from
        # the state on screen rather than the cold guess. See _ensure_stepper.
        self.warm_start = False
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
        p.flexor_tension_sigma = 10.0 ** self.g_flexor_sigma.value
        p.passive_tension_sigma = 10.0 ** self.g_passive_sigma.value
        p.contact_fingers = [c.value for c in self.g_contacts]
        p.object_contact = self.g_obj_contact.value
        p.table_contact = self.g_tbl_contact.value
        p.contact_drop_normal_row = self.g_drop_normal_row.value
        p.half_space = self.g_half_space.value
        p.half_space_split = None   # derive from object_center
        # Cleared every sync so the SIGN is re-resolved against the posture on
        # screen: _attach_opposition writes the oriented axis back here, and a
        # stale one would keep sending the thumb to the side it was on two
        # solves ago.
        p.half_space_axis = None
        p.half_space_flip = OPPOSITION_SIDES[self.g_half_sides.value]
        p.half_space_margin = self.g_half_margin.value
        p.pregrasp_center = self.g_pregrasp_center.value
        p.h_clear = self.g_h_clear.value
        p.pregrasp_axis_align = self.g_axis_align.value
        p.pregrasp_centroid = self.g_pregrasp_centroid.value
        p.sigma_wrist_pos = 10.0 ** self.g_sig_pos.value
        p.sigma_wrist_rot = 10.0 ** self.g_sig_rot.value
        # AL
        p.al_mu = self.g_al_mu.value
        p.al_rate = self.g_al_rate.value
        p.al_iters = self.g_al_iters.value
        p.ik_settle_steps = int(self.g_ik_settle.value)
        # collision
        p.collision = self.g_collision.value
        p.self_collision = self.g_self_collision.value
        p.collision_radius = self.g_coll_radius.value
        p.collision_sigma = 10.0 ** self.g_coll_sigma.value
        p.cull_margin = (None if not self.caps["collision_cull"] or self.g_cull.value <= 0
                         else self.g_cull.value)
        # table
        p.table = self.g_table.value and self.caps["table"]
        p.plane_normal = np.array(TABLE_NORMAL, float)
        p.plane_avoidance = self.g_plane_avoid.value
        # Resolve the auto (under-object) seating once, then bake the "height
        # offset" slider into an EXPLICIT plane_origin -- shared by the factor
        # graph (via this params object) and the rendered slab (_table_origin
        # just reads it back). Previously this stayed None and the offset was
        # applied only in _table_origin() for the visual scene, so dragging the
        # slider moved the drawn table without moving the solver's plane.
        p.plane_origin = None
        spec, center, _rot, _pose = resolve_scene(p)
        p.plane_origin = np.asarray(
            resolve_table_origin(p, spec, center), float) + (
                self.g_plane_offset.value * np.array(TABLE_NORMAL, float))
        # display toggles
        self.scene.show_discs = self.g_show_discs.value
        self.scene.show_contact_spheres = self.g_show_contact.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self.scene.show_gap_lines = self.g_show_gaps.value

    def _table_origin(self):
        """The rendered slab's origin -- reads straight off ``params.plane_origin``,
        which ``_sync_params`` bakes the height-offset slider into, so the drawn
        table and the factor graph's plane always agree."""
        spec, center, _rot, _pose = resolve_scene(self.params)
        return resolve_table_origin(self.params, spec, center)

    # -- rendering --

    def _refresh_object(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        self.scene.set_object(spec, center, rotation)
        if self.params.table:
            self.scene.set_table(self._table_origin(), self.params.plane_normal)
        else:
            self.scene.clear_table()
        if self.params.half_space:
            axis = (self.params.half_space_axis if self.params.half_space_axis is not None
                   else default_half_space_axis(spec, rotation, self.params.plane_normal))
            split = (self.params.half_space_split if self.params.half_space_split is not None
                    else center)
            self.scene.set_half_space_plane(
                split, axis, margin=self.params.half_space_margin)
        else:
            self.scene.clear_half_space_plane()

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
        half_gaps = (half_space_witness(self.params, res, 0)
                    if self.params.half_space else None)
        center_gap = (pregrasp_center_witness(self.params, res, 0)
                     if self.params.pregrasp_center else None)
        axis_align = (pregrasp_axis_witness(self.params, res, 0)
                     if self.params.pregrasp_axis_align else None)
        centroid_gap = (pregrasp_centroid_witness(self.params, res, 0)
                       if self.params.pregrasp_centroid else None)
        self._report_iterate()
        self.scene.update(res.frames[0],
                          tip_radii=res.tip_radii,
                          collision_radius=self.params.collision_radius,
                          # The spheres are drawn whenever ANY of the three
                          # consumers is using them, matching what the solve
                          # actually built.
                          collision=(self.params.collision
                                     or self.params.self_collision
                                     or (self.params.table
                                         and self.params.plane_avoidance)),
                          gaps=gaps,
                          table_gaps=table_gaps,
                          half_space_gaps=half_gaps,
                          center_gap=center_gap,
                          axis_align=axis_align,
                          centroid_gap=centroid_gap)

    def _set_status(self, text):
        self.g_status.content = text

    def _error_status(self, exc):
        """Show a solver exception with the reason attached where we can name it.

        GTSAM's IndeterminantLinearSystem text explains what an ill-posed system
        IS, not which of the controls on this page caused one. In a pre-grasp
        solve there is essentially one answer -- the wrist prior stopped fixing
        the wrist -- so say that instead of leaving a Doxygen link on screen."""
        note = []
        if "Indeterminant" in str(exc):
            note = self._wrist_gauge_note() or [
                "*Indeterminant means some variable has no information left. In "
                "a pre-grasp solve the usual culprit is the wrist prior: "
                "inequalities (collision, opposition) contribute nothing while "
                "satisfied, so the prior is what fixes the wrist.*"]
        self._set_status("  \n".join([f"**Error:** {exc}"] + note))

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
        # Reset would pull the params out from under a running step, and flipping
        # the warm start mid-run cannot affect the loop already built, so both
        # wait it out rather than looking like they did something.
        if getattr(self, "g_warm", None) is not None:
            self.g_warm.disabled = solving or not self.caps["solver_seed"]
        if getattr(self, "g_reset", None) is not None:
            self.g_reset.disabled = solving

    def _fk_solve(self, _=None):
        """Re-pose the hand from the current sliders with the FK solver.

        Also the "start over" action for the IK loop: it drops any partially
        stepped solve. What the next Step then starts from is the warm-start
        latch's business -- off, the cold guess; on, this FK pose."""
        if self._solving:
            return
        self._solving = True
        self._set_solving(True)
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
        except Exception as exc:  # surface solver errors in the GUI, keep serving
            self._error_status(exc)
            raise
        finally:
            self._set_solving(False)
            self._solving = False

    def _pinch_note(self):
        """Warn when pinch-centroid centering is checked but the selected
        digits have no measured pinch pose.

        Only combinations INCLUDING THE THUMB were measured, so a selection
        like index+middle silently attaches nothing -- the C++ layer skips an
        unconfigured constraint without complaint, which is the same trap the
        other pre-grasp toggles set (they need the thumb plus one other finger
        or they vanish too). Say it out loud instead."""
        if not self.params.pregrasp_centroid:
            return []
        names = [n for n, c in zip(FINGER_LABELS, self.g_contacts) if c.value]
        pose = pinch_pose(names)
        if pose is None:
            return [f"**pinch-centroid: INACTIVE** -- no measured pinch pose "
                    f"for ({', '.join(names) or 'no fingers'}); only "
                    f"combinations including the thumb were measured"]
        if not pose.touches():
            # A real measurement, but of a closest approach rather than a
            # contact -- the centroid is still well-defined, the digits just
            # never actually meet there.
            return [f"pinch-centroid: these digits close to "
                    f"{pose.gap * 1000:.1f} mm apart (they never touch)"]
        return []

    def _half_space_note(self):
        """The opposition half-space's own status line: inert when no finger is
        checked for it, and the standoff it is holding when there is one.

        The constraint is built per finger off its own ``half_space_node``, so
        the only way to check the box and get nothing is to check no fingers --
        the finger mask is the one dependency it has left."""
        if not self.params.half_space:
            return []
        names = [n for n, c in zip(FINGER_LABELS, self.g_contacts) if c.value]
        if not names:
            return ["**opposition half-space: INACTIVE** -- no fingers checked "
                    "in the *fingers* folder, so there is nothing to oppose"]
        if not self.caps["half_space_margin"]:
            return ["opposition half-space: this binding has no "
                    "`EnvironmentConfig.half_space_margin`, so the standoff "
                    "slider is inert (rebuild with `pip install .`)"]
        lines = []
        if self.params.half_space_margin > 0.0:
            lines.append(f"opposition standoff: each side held "
                         f"{self.params.half_space_margin * 1000:.0f} mm off "
                         f"the split "
                         f"({self.params.half_space_margin * 2000:.0f} mm "
                         f"corridor)")
        lines.extend(self._opposition_side_note())
        lines.extend(self._rotation_driver_note())
        return lines

    # Wrist priors looser than this (sigma, m and rad) stop fixing the wrist's
    # gauge in any solve whose other constraints are all INEQUALITIES. Measured
    # on the pen pre-grasp scene (pinch-centroid + collision + opposition): every
    # cell at sigma >= 10 on EITHER prior throws IndeterminantLinearSystem near
    # W0, every cell at <= 1 solves, and adding the other pre-grasp constraints
    # does not move the boundary.
    WRIST_PRIOR_GAUGE_LIMIT = 10.0

    def _wrist_gauge_note(self):
        """Warn when the wrist prior is the only thing fixing the wrist, and is
        too loose to do it.

        An inequality that is SATISFIED contributes no rows to the linearized
        system -- collision, table avoidance and the opposition half-space are
        all inequalities, and in a pre-grasp scene they sit slack. Contact
        equalities are what would otherwise pin the hand, and a pre-grasp solve
        has none by definition. So the 6 dof of the wrist are held by: the
        pre-grasp constraints that touch it (pinch-centroid, 3 rows on a point;
        centering, 3 rows on a fingertip midpoint; short-axis alignment, ONE
        scalar row) and the prior. Loosen the prior far enough and what is left
        is rank-deficient -- which surfaces as GTSAM's IndeterminantLinearSystem
        near W0, not as anything that names the prior."""
        pos, rot = self.params.sigma_wrist_pos, self.params.sigma_wrist_rot
        loose = [n for n, v in (("position", pos), ("rotation", rot))
                 if v >= self.WRIST_PRIOR_GAUGE_LIMIT]
        if not loose:
            return []
        equality_backed = (self.params.object_contact or self.params.table_contact)
        if equality_backed:
            return []
        return [f"**wrist prior ({' and '.join(loose)}) is very loose "
                f"(sigma {pos:g} m / {rot:g} rad)** -- with no contact "
                f"equalities on, nothing else fixes the wrist: collision and "
                f"the opposition half-space are inequalities and contribute "
                f"nothing while satisfied. Expect "
                f"*IndeterminantLinearSystem near W0*; keep sigma at or below "
                f"1 (log10 0)."]

    def _rotation_driver_note(self):
        """Say when nothing in the constraint set can rotate the hand.

        The half-space is a one-sided inequality on POSITIONS: once each digit
        is on its own side it goes slack and contributes no gradient at all, so
        it cannot turn the wrist however large the standoff. Pinch-centroid is
        three rows on a single point -- satisfiable by translation alone.
        Short-axis alignment is the only constraint here that says anything
        about ORIENTATION. Measured on this scene: 1.5 degrees of wrist rotation
        without it, 45 with."""
        if not (self.params.half_space or self.params.pregrasp_centroid):
            return []
        if self.params.pregrasp_axis_align or self.params.object_contact:
            return []
        return ["*nothing here rotates the hand*: the half-space is a "
                "one-sided inequality on positions (slack => no gradient) and "
                "pinch-centroid is satisfiable by translation alone. Tick "
                "**short-axis alignment** for the orientation constraint."]

    def _opposition_side_note(self):
        """How far the current posture is from the side assignment being asked
        for -- the one number that says whether this constraint is a nudge or a
        demand that the hand turn itself inside out.

        Worth a line of its own because the failure is silent otherwise: with
        the sides the wrong way up the solve stalls on iteration 3 with the hand
        visibly untouched, which reads as the solver giving up rather than as
        the constraint asking for a 180 degree roll."""
        res = self._iter_view()
        if res is None or "thumb" not in res.finger_names:
            return []
        gaps = half_space_witness(self.params, res, 0)
        if not gaps or "thumb" not in gaps:
            return []
        worst = min(v[2] for v in gaps.values())
        mode = self.g_half_sides.value
        if worst >= 0.0:
            return [f"opposition sides ({mode}): satisfied, worst digit "
                    f"{worst * 1000:+.0f} mm inside its half"]
        return [f"**opposition sides ({mode}): the hand is on the WRONG side "
                f"by up to {-worst * 1000:.0f} mm** -- it has to trade thumb "
                f"and fingers over to satisfy this, which normally stalls the "
                f"solve. Try *auto* (or *flipped*) in the sides dropdown."]

    def _report(self):
        m = self.result.meta
        lines = [f"**{self.mode}** &nbsp; iters={m.iterations} &nbsp; "
                 f"err={m.error:.3g} &nbsp; {m.total_time_ms:.0f} ms"]
        if self.mode != "FK":
            lines.extend(self._contact_lines(-1))
        lines.extend(self._half_space_note())
        lines.extend(self._wrist_gauge_note())
        lines.extend(self._pinch_note())
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
                res.frames[0][name].marginals.tensions.mean, float)[FLEXOR_IDX])
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
            self._error_status(exc)
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
                self._error_status(exc)
            finally:
                self.g_ik_stop.disabled = True
                self._set_solving(False)
                self._solving = False

        self._auto_thread = threading.Thread(target=worker, daemon=True)
        self._auto_thread.start()

    def _ik_stop(self, _=None):
        """Ask a running auto-solve to stop; it breaks before the next step."""
        self._auto_stop.set()

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
            # python/crest_sparse/_crest_sparse*.so shadowing the installed one,
            # which happens whenever the app is run from the python/ directory.
            self.g_warm_status.content = (
                "**unavailable** -- this binding has no "
                "`TendonHandSolverConfig.initial_state`  \n"
                f"loaded from `{binding_path()}`  \n"
                "*(run from the crest-sparse root -- `python -m "
                "python.tests.tendon_hand.viz_interactive` -- so the installed "
                "build is used, and rebuild with `pip install .`)*")
        elif on:
            self.g_warm_status.content = (
                "the next **Step** starts from the state on screen"
                + (", carrying the AL multipliers"
                   if self.caps["dual_transfer"] and self.g_carry_duals.value
                   else ""))
        else:
            self.g_warm_status.content = (
                "the next **Step** cold-starts (straight hand, Q = 0)")

    def _reset_defaults(self, _=None):
        """Put every control back to the value it was built with and cold-start.

        A full reset rather than a re-solve: fresh params (so the warm-start
        posture and any derived scene state go too), fresh FK solver, no
        stepper, camera back on the default object."""
        if self._solving:
            return
        self._restoring = True
        try:
            for handle, value in self._gui_defaults:
                handle.value = value
        finally:
            self._restoring = False
        self.params = HandSolveParams()
        self.warm_start = False     # a button, so not in _gui_defaults
        self._refresh_warm_start()
        self._sync_params()
        self._rebuild_fk()          # also drops the stepper
        self._refresh_object()
        self._aim_all_cameras()
        self._fk_solve()
        self._set_status("**reset** to defaults  \n" + self.g_status.content)

    # -- phase presets --

    def _preset_widget(self, field):
        """The GUI handle a ``PHASE_PRESETS`` override field writes onto, for
        the plain 1:1 cases (everything except ``contact_fingers``,
        ``sigma_wrist_pos``/``sigma_wrist_rot`` and ``flexor_tension_sigma``,
        which :meth:`_apply_phase_preset` special-cases itself)."""
        return {
            "object_contact": self.g_obj_contact,
            "table_contact": self.g_tbl_contact,
            "collision": self.g_collision,
            "self_collision": self.g_self_collision,
            "table": self.g_table,
            "plane_avoidance": self.g_plane_avoid,
            "half_space": self.g_half_space,
            "half_space_margin": self.g_half_margin,
            "pregrasp_center": self.g_pregrasp_center,
            "pregrasp_axis_align": self.g_axis_align,
            "pregrasp_centroid": self.g_pregrasp_centroid,
            "h_clear": self.g_h_clear,
            "contact_drop_normal_row": self.g_drop_normal_row,
        }[field]

    def _apply_phase_preset(self, name):
        """Write ``PHASE_PRESETS[name]``'s overrides directly onto the
        corresponding GUI widgets, so checking the preset box is a single
        visible action: every affected checkbox/slider jumps to the preset's
        value on screen. One solve-ready sync/invalidate happens at the end --
        Auto solve is a separate, manual next step, not triggered here."""
        overrides = PHASE_PRESETS[name].overrides
        self._restoring = True   # batch write; no live-FK/other side effects
        try:
            for field, value in overrides.items():
                if field == "contact_fingers":
                    for handle, v in zip(self.g_contacts, value):
                        handle.value = bool(v)
                elif field == "sigma_wrist_pos":
                    self.g_sig_pos.value = math.log10(value)
                elif field == "sigma_wrist_rot":
                    self.g_sig_rot.value = math.log10(value)
                elif field == "flexor_tension_sigma":
                    self.g_flexor_sigma.value = math.log10(value)
                else:
                    self._preset_widget(field).value = value
        finally:
            self._restoring = False
        self._sync_params()
        self._invalidate_stepper()
        self._refresh_object()
        self._render_frame()

    def _phase_checkboxes(self):
        """Every phase-preset checkbox, name -> handle. Small and built on
        demand rather than cached, so a future phase3 checkbox only needs
        adding here (and to ``_build_gui``/``_input_handles``)."""
        return {"phase0": self.g_phase0, "phase1": self.g_phase1,
                "phase2": self.g_phase2}

    def _on_phase_toggle(self, name, _=None):
        """Checking a phase preset applies it and unchecks every OTHER phase
        checkbox first -- they are mutually exclusive stages of the same
        pipeline, and leaving two checked at once would show a state whose
        settings actually contradict each other (e.g. phase 0's half_space=
        True vs. phase 1's False). Unchecking is a no-op -- the controls a
        preset wrote stay exactly where it left them, freely editable
        afterward; there is nothing to "undo" back to."""
        if self._restoring:
            return
        checkbox = self._phase_checkboxes()[name]
        if not checkbox.value:
            return
        self._restoring = True
        try:
            for other_name, other_box in self._phase_checkboxes().items():
                if other_name != name:
                    other_box.value = False
        finally:
            self._restoring = False
        self._apply_phase_preset(name)

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
        return ([self.g_object, self.g_ik_max, self.g_ik_settle, self.g_carry_duals,
                 self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw,
                 self.g_sig_pos, self.g_sig_rot, self.g_passive]
                + self.g_flexors
                + [self.g_flexor_sigma, self.g_passive_sigma,
                   self.g_phase0, self.g_phase1, self.g_phase2]
                + [self.g_obj_contact, self.g_tbl_contact, self.g_drop_normal_row,
                   self.g_half_space, self.g_half_sides, self.g_half_margin,
                   self.g_pregrasp_center, self.g_h_clear,
                   self.g_pregrasp_centroid, self.g_axis_align]
                + self.g_contacts
                + [self.g_collision, self.g_self_collision,
                   self.g_coll_radius, self.g_coll_sigma, self.g_cull,
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
                     "same grasp. Set 0 to watch the old behaviour. Ignored "
                     "entirely on a warm start -- a seeded posture is already "
                     "consistent, and pinning its tendons back to the commanded "
                     "means is exactly what would undo it.")
            self.g_status = gui.add_markdown("")

        # The scrubber over the steps taken. The slider and its readout are built
        # per step by _rebuild_iter_slider().
        self.iter_folder = gui.add_folder("Solve steps")

        with gui.add_folder("Warm start / reset"):
            self.g_warm = gui.add_button(
                "Warm start: off", icon=self.viser.Icon.PIN,
                disabled=not self.caps["solver_seed"],
                hint=("Latch: while it is ON, every restart of the IK loop "
                      "begins from the state on screen instead of from a "
                      "straight hand with Q = 0. Changing the object, contacts, "
                      "collision, table or an AL knob restarts the loop (the "
                      "carried multipliers describe the old constraint set), so "
                      "this is what lets you change a setting and carry on from "
                      "the posture you had. It also starts an IK solve from an "
                      "FK pose you dialled in, and it follows the iterate "
                      "scrubber, so you can rewind and branch. Only the POSTURE "
                      "carries -- the penalty schedule restarts either way."
                      if self.caps["solver_seed"]
                      else "requires a rebuilt _crest_sparse with "
                           "TendonHandSolverConfig.initial_state"))
            self.g_carry_duals = gui.add_checkbox(
                "carry AL duals", True, disabled=not self.caps["dual_transfer"],
                hint=("Also carry the Augmented Lagrangian multipliers, matched "
                      "to the new constraint set by identity. This is what stops "
                      "the hand letting go of constraints it had already "
                      "satisfied: without it the rebuilt solve restarts the "
                      "penalty schedule at mu = al_mu with every multiplier at "
                      "zero. Untick to see the difference."
                      if self.caps["dual_transfer"]
                      else "requires a rebuilt _crest_sparse with "
                           "TendonHandSolver.set_initial_duals"))
            self.g_reset = gui.add_button(
                "Reset defaults", icon=self.viser.Icon.ROTATE,
                hint="Put every control back to the value it opened with, turn "
                     "the warm start off, drop the stepped solve, and re-pose "
                     "with FK.")
            self.g_warm_status = gui.add_markdown("")

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
            self.g_sig_pos = gui.add_slider("log10 sigma_pos", -6, 2, 0.5, -2)
            self.g_sig_rot = gui.add_slider("log10 sigma_rot", -6, 2, 0.5, -2)

        with gui.add_folder("Tensions (N)"):
            self.g_passive = gui.add_slider("passive", 0.0, 3.0, 0.05, 0.5)
            self.g_flexors = [
                gui.add_slider(lbl, 0.0, 3.0, 0.05, GRASP_FLEXOR_TENSION)
                for lbl in FINGER_LABELS]
            self.g_flexor_sigma = gui.add_slider(
                "log10 flexor tension sigma", -3.0, 5.0, 0.1,
                math.log10(HandSolveParams().flexor_tension_sigma),
                hint="How loose the ACTUATED (flexor) tendon's tension prior "
                     "is once contact is expected to move it away from its "
                     "commanded value above. Read live every IK step, so a "
                     "drag takes effect on the next Step/Auto solve with no "
                     "rebuild. Has no effect on FK.")
            self.g_passive_sigma = gui.add_slider(
                "log10 passive tension sigma", -6.0, 1.0, 0.1,
                math.log10(HandSolveParams().passive_tension_sigma),
                hint="How loose the five PASSIVE tendons' tension prior is -- "
                     "normally left tight (their physics is a spring holding "
                     "roughly constant tension), unlike the actuated flexor "
                     "above. Below ~1e-3 (variance ~1e-6) risks an "
                     "IndeterminantLinearSystem against the flexor's much "
                     "looser scale. Read live every IK step, so a drag takes "
                     "effect on the next Step/Auto solve with no rebuild. "
                     "Has no effect on FK.")

        # One-click constraint-set presets, backed by solvers.PHASE_PRESETS so
        # the same data is usable headlessly. Checking a box writes its whole
        # preset onto the Constraints controls below in one go; press Auto
        # solve afterward to run it. Mutually exclusive -- checking one
        # unchecks the other, see _on_phase_toggle.
        with gui.add_folder("Presets"):
            self.g_phase0 = gui.add_checkbox(
                PHASE_PRESETS["phase0"].label, False,
                hint="Apply the phase-0 preset: no object/table contact yet, "
                     "collision avoidance on, all three pre-grasp constraints "
                     "(opposition half-space, centering, short-axis "
                     "alignment) on, a loose wrist prior (this is a big "
                     "repositioning move), and a 3-finger pinch "
                     "(index/middle/thumb). Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve. Unchecking is a no-op.")
            self.g_phase1 = gui.add_checkbox(
                PHASE_PRESETS["phase1"].label, False,
                hint="Apply the phase-1 preset: table contact ON (object "
                     "contact stays off), the three pre-grasp constraints "
                     "(opposition half-space, centering, short-axis "
                     "alignment) turned back OFF now that they've done their "
                     "job, and a tighter wrist prior than phase 0 (held "
                     "closer to where it ended up, not free to roam). Same "
                     "3-finger pinch. Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve. Unchecking is a no-op.")
            self.g_phase2 = gui.add_checkbox(
                PHASE_PRESETS["phase2"].label, False,
                hint="Apply the phase-2 preset: object contact turned back "
                     "ON alongside table contact (approaching the object "
                     "while still sliding on the table), pre-grasp "
                     "constraints still off, and the wrist prior loosened "
                     "back to phase 0's level -- object approach is another "
                     "significant motion, not phase 1's small settle. Same "
                     "3-finger pinch. Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve. Unchecking is a no-op.")

        # Every constraint on/off toggle lives here (Chapter 2, Eq 2.8-2.19),
        # grouped by the paper's structure. Numeric tuning sliders that go with
        # a toggle (collision radius/sigma/cull margin, table height offset)
        # stay behind in Collision/Table below -- only the booleans move.
        with gui.add_folder("Constraints"):
            with gui.add_folder("Collision (Eq 2.8-2.9)"):
                self.g_collision = gui.add_checkbox(
                    "object collision", True,
                    hint="Keep every non-contact sphere out of the OBJECT. "
                         "Independent of the other two collision families "
                         "below -- the three share one set of collision "
                         "spheres but each is its own switch. Sliders in the "
                         "Collision folder below.")
                self.g_self_collision = gui.add_checkbox(
                    "self collision", True,
                    disabled=not self.caps["self_collision"],
                    hint="Keep the FINGERS out of each other: one inequality "
                         "per cross-finger sphere pair, minus the pairs the "
                         "cull margin drops. On by default -- a hand passing "
                         "through itself is never wanted -- and needs neither "
                         "the object nor the table. The most expensive of the "
                         "three families by factor count, so this is the one "
                         "to untick when bisecting a slow solve."
                         if self.caps["self_collision"]
                         else "requires a rebuilt _crest_sparse with "
                              "EnvironmentConfig.self_collision")
                self.g_plane_avoid = gui.add_checkbox(
                    "table collision", True,
                    hint="Keep every non-contact sphere out of the "
                         "half-space. Independent of the other two collision "
                         "families. Needs the table enabled below -- with no "
                         "plane there is no half-space to stay out of.")

            with gui.add_folder("Contact (Eq 2.11-2.15)"):
                self.g_table = gui.add_checkbox(
                    "table enabled", True, disabled=not self.caps["table"],
                    hint=None if self.caps["table"]
                    else "requires a newer _crest_sparse build (plane env fields)")
                self.g_obj_contact = gui.add_checkbox(
                    "object contact", True,
                    hint="Drive the checked fingertips onto the OBJECT "
                         "surface. Turn off to leave the object as pure "
                         "collision geometry -- the way to see what the "
                         "table constraints do on their own.")
                self.g_tbl_contact = gui.add_checkbox(
                    "table contact", False,
                    hint="Drive the checked fingertips onto the SUPPORT "
                         "PLANE (one equality per finger on the distance "
                         "from its contact sphere to the plane). Needs "
                         "*table enabled*; combine with object contact to "
                         "solve for both at once.")
                self.g_drop_normal_row = gui.add_checkbox(
                    "drop contact normal row", False,
                    disabled=not self.caps["drop_normal_row"],
                    hint="Eq 2.12-2.15: use the 4-row [c_R, c_O, c_T1, "
                         "c_T2] SDF witness contact form (c_N dropped) "
                         "instead of the default 5-row form. Only affects "
                         "non-ellipsoid (SDF) object contact.")

            with gui.add_folder("Pre-grasp (Eq 2.16-2.19)"):
                self.g_half_space = gui.add_checkbox(
                    "opposition half-space", False,
                    disabled=not self.caps["opposition"],
                    hint="Eq 2.16-2.17: keep each checked finger on its "
                         "designated side of the splitting plane, thumb "
                         "opposite the rest. Independent of table contact and "
                         "of the table itself -- it constrains a fingertip "
                         "against a line, not against the plane.")
                self.g_half_sides = gui.add_dropdown(
                    "opposition sides", list(OPPOSITION_SIDES),
                    initial_value="auto (match the hand)",
                    hint="Which half of the split the THUMB is sent to. The "
                         "object fixes the split LINE; its sign -- the side "
                         "assignment -- is an arbitrary object-frame "
                         "convention, and the wrong one asks the thumb and the "
                         "fingers to TRADE sides, i.e. to roll the hand ~180 "
                         "degrees about the object. That is infeasible from any "
                         "normal start pose and the solve stalls immediately, "
                         "at any standoff and any wrist sigma. *auto* orients "
                         "it by where the digits already are (the nearer "
                         "opposition); *flipped* deliberately asks for the "
                         "other one; *as derived* is the old object-only "
                         "behaviour, kept for comparison.")
                self.g_half_margin = gui.add_slider(
                    "half-space standoff (m)", 0.0, 0.05, 0.001, 0.0,
                    disabled=not self.caps["half_space_margin"],
                    hint="Minimum distance each contact finger must keep from "
                         "the splitting plane, along its own side's direction: "
                         "HalfSpaceGapFactor's d_min, making the constraint "
                         "-(c - p_split).m_hat + standoff <= 0 rather than "
                         "<= 0 alone. At 0 a fingertip sitting exactly ON the "
                         "split is already legal, so opposition alone does not "
                         "stop the thumb and the fingers closing onto each "
                         "other; a positive standoff holds them 2x this far "
                         "apart, which is what makes it a pre-grasp opening. "
                         "Drawn as the two fainter planes either side of the "
                         "split."
                         if self.caps["half_space_margin"]
                         else "requires a rebuilt _crest_sparse with "
                              "EnvironmentConfig.half_space_margin")
                self.g_pregrasp_center = gui.add_checkbox(
                    "pre-grasp centering", False,
                    disabled=not self.caps["pregrasp_center"],
                    hint="Eq 2.18-2.19: center the midpoint of the thumb + "
                         "opposing fingers' contact points over the object, "
                         "raised by the clearance slider along the table "
                         "normal. Needs the thumb AND at least one other "
                         "finger checked below.")
                self.g_h_clear = gui.add_slider(
                    "clearance (m)", 0.0, 0.08, 0.002, 0.07,
                    hint="Pre-grasp centering's height above the object "
                         "centroid, along the table normal.")
                self.g_pregrasp_centroid = gui.add_checkbox(
                    "pinch-centroid centering", False,
                    disabled=not self.caps["pregrasp_centroid"],
                    hint="Drive the point where the CHECKED digits are "
                         "measured to meet -- a constant in the hand frame, "
                         "looked up from config.HAND_PINCH_POSES -- onto the "
                         "object centroid, raised by the clearance slider "
                         "above. Unlike pre-grasp centering (which averages "
                         "where the fingertips actually are, so it only says "
                         "something once they are nearly closed) this "
                         "constrains the WRIST alone: it positions the hand "
                         "so that closing those digits would close them ON "
                         "the object, whatever the fingers are doing now. "
                         "Only combinations INCLUDING THE THUMB were "
                         "measured; any other selection leaves it inert and "
                         "says so in the status line."
                         if self.caps["pregrasp_centroid"]
                         else "requires a rebuilt _crest_sparse with "
                              "EnvironmentConfig.pregrasp_centroid_point")
                self.g_axis_align = gui.add_checkbox(
                    "short-axis alignment", False,
                    disabled=not self.caps["pregrasp_axis_align"],
                    hint="Align the thumb-vs-opposing-fingers connecting "
                         "vector with the perpendicular to the opposition "
                         "split plane (the same axis opposition half-space "
                         "uses), direction-agnostic -- it doesn't matter "
                         "which way it points, just that it's colinear. "
                         "Needs the thumb AND at least one other finger "
                         "checked below. Independent of opposition "
                         "half-space and pre-grasp centering -- computes its "
                         "own copy of the axis either way.")

            with gui.add_folder("fingers"):
                # Default to a 3-finger pinch (thumb, index, middle) rather
                # than the whole-hand grasp; ring/pinky keep collision
                # avoidance but are not driven onto a surface.
                _pinch_default = {"index", "middle", "thumb"}
                self.g_contacts = [
                    gui.add_checkbox(
                        lbl, lbl in _pinch_default,
                        hint="Which fingers every constraint above applies "
                             "to (IK only; FK never uses contact). "
                             "Unchecked fingers keep collision avoidance, so "
                             "they stay out of the object and off the table "
                             "without being driven onto either, opposed "
                             "against, or centered on.")
                    for lbl in FINGER_LABELS]

        with gui.add_folder("Collision"):
            self.g_coll_radius = gui.add_slider("sphere radius (m)", 0.001, 0.01, 0.0005, 0.003)
            self.g_coll_sigma = gui.add_slider("log10 sigma", -6, 0, 0.5, -4)
            self.g_cull = gui.add_slider("cull margin (m, 0 off)", 0.0, 0.1, 0.005, 0.0)

        with gui.add_folder("Table"):
            # Offset from the scene's own seating, which now half-buries the
            # object (HandSolveParams.table_burial = 0.5) rather than resting it
            # on the plane. Dial in -half_extent to recover the old geometry.
            # -0.006 m is this app's own default seating for the pen.
            self.g_plane_offset = gui.add_slider("height offset (m)", -0.1, 0.1, 0.002, -0.006)

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
                hint="Fingertip-to-surface gap/margin overlays in mm: object "
                     "and table contact (green under 15 mm, red over), "
                     "opposition half-space (green = correct side, red = "
                     "violating), and pre-grasp centering (green under 15 mm "
                     "to target).")

        # Every value-carrying control, captured as built: this IS the definition
        # of "defaults" that Reset restores, so the two cannot drift.
        self._gui_defaults = [(h, h.value) for h in self._input_handles()]
        self._refresh_warm_start()

        # -- callbacks --
        self.g_fk.on_click(self._fk_solve)
        self.g_ik_step.on_click(self._ik_step)
        self.g_ik_auto.on_click(self._ik_auto)
        self.g_ik_stop.on_click(self._ik_stop)
        self.g_warm.on_click(self._toggle_warm_start)
        self.g_reset.on_click(self._reset_defaults)
        self.g_phase0.on_update(lambda _: self._on_phase_toggle("phase0"))
        self.g_phase1.on_update(lambda _: self._on_phase_toggle("phase1"))
        self.g_phase2.on_update(lambda _: self._on_phase_toggle("phase2"))

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
        # Table toggle / height updates the static slab immediately; opposition
        # half-space rides along since it draws its own static split-plane slab
        # (set_half_space_plane) the same way -- as does its standoff slider,
        # which draws the two boundary planes either side of that split.
        for h in (self.g_table, self.g_plane_offset, self.g_half_space,
                  self.g_half_margin, self.g_half_sides):
            h.on_update(lambda _: (self._sync_params(), self._refresh_object(),
                                   self._invalidate_stepper()))
        # Collision knobs are part of the constraint set too. The sphere-drawing
        # flag follows the toggles, so these re-render as well as invalidate.
        for h in (self.g_collision, self.g_self_collision, self.g_coll_radius,
                  self.g_coll_sigma, self.g_cull, self.g_plane_avoid):
            h.on_update(lambda _: (self._sync_params(),
                                   self._invalidate_stepper(),
                                   self._render_frame()))
        # drop-normal-row / pre-grasp centering / clearance: no static geometry
        # of their own (the centering line only appears once a solve has run,
        # via _render_frame), so just invalidate the stepper -- self.params is
        # refreshed from every handle at the start of the next solve regardless
        # of which widget triggered it.
        for h in (self.g_drop_normal_row, self.g_pregrasp_center, self.g_h_clear,
                  self.g_axis_align, self.g_pregrasp_centroid):
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
    app = HandVizApp(server)
    # Which binding got loaded, and what it can do. Printed unconditionally: a
    # capability-gated control that is silently disabled is indistinguishable
    # from one that does not work, and the usual cause is the in-tree .so
    # shadowing the installed build (see binding_path()).
    print(f"crest_sparse: {binding_path()}")
    missing = [k for k, v in app.caps.items() if not v]
    if missing:
        print(f"  capabilities MISSING from this build: {', '.join(missing)}")
    print(f"viser hand visualizer running -- open http://localhost:{args.port}")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
