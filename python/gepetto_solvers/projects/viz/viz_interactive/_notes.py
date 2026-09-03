"""The status line and the per-constraint readouts under each control.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""


import numpy as np

from gepetto_solvers.core.geometry.scene import (
    INPLANE_DEGENERACY_RATIO,
    ellipsoid_members,
    get_primitive_specs,
    object_principal_inplane_axis,
)
from gepetto_solvers.core.hands.tendon_5f import (
    pinch_pose,
)
from gepetto_solvers.core.objects import has_exact_form, names_exact_form
from gepetto_solvers.core.solvers import (
    half_space_witness,
    phase_presets,
    resolve_scene,
)


class NotesMixin:
    # Wrist priors looser than this (sigma, m and rad) stop fixing the wrist's
    # gauge in any solve whose other constraints are all INEQUALITIES. Measured
    # on the pen pre-grasp scene (pinch-centroid + collision + opposition): every
    # cell at sigma >= 10 on EITHER prior throws IndeterminantLinearSystem near
    # W0, every cell at <= 1 solves, and adding the other pre-grasp constraints
    # does not move the boundary.
    WRIST_PRIOR_GAUGE_LIMIT = 10.0

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
        names = [n for n, c in zip(self.digit_names, self.g_contacts) if c.value]
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


    def _finger_plane_note(self):
        """Say when the pinch-plane overlay is switched on but has no plane to
        draw -- the same measured-pinch-pose dependency :meth:`_pinch_note`
        covers, hit from the display side.

        Worth its own line because here there is no constraint to blame: the
        checkbox is ticked, the hand is on screen, and nothing appears. The
        planes are anchored on the centroid of the digits the SOLVE designated,
        so a thumbless selection leaves them undefined.

        Reads the result's digits, not the checkboxes -- unlike
        :meth:`_pinch_note`, which describes the constraint the NEXT solve will
        attach. This describes an overlay on the posture already drawn, and
        that overlay is keyed off the same ``contact_names`` the result carries,
        so a contact box unticked since the last solve must not change this line
        while the hand it describes is still on screen."""
        if self.g_show_finger_planes is None:
            return []          # no pinch table on this hand: no overlay to explain
        if not self.g_show_finger_planes.value or self.result is None:
            return []
        names = self.result.contact_names()
        if pinch_pose(names) is None:
            return [f"**finger pinch planes: NOTHING DRAWN** -- no measured "
                    f"pinch pose for ({', '.join(names) or 'no fingers'}); the "
                    f"planes pass through that centroid, and only combinations "
                    f"including the thumb were measured"]
        return []


    def _object_contact_boxes(self):
        """Every object-contact FORM box this hand has, 3D first.

        The list, not a pair: there are three forms now (3D and in-plane against
        the ellipsoid, witness against the baked SDF) and a hand may have two or
        three of them, so every rule about "the other one" has to be written
        against whatever is actually present."""
        return [h for h in (self.g_obj_contact, self.g_obj_contact_plane,
                            self.g_obj_contact_exact) if h is not None]

    def _enforce_object_contact(self, source):
        """Keep the object-contact FORMS mutually exclusive.

        3D, in-plane and exact-SDF are three ways of stating ONE constraint --
        one factor per contact finger whichever is chosen -- so "two at once" is
        not a state the graph has. Rather than silently preferring one at build
        time, the box you just touched wins and the rest clear, which is the same
        rule stated in the hints and visible the moment you click.

        The first two differ only in the METRIC and both measure against the
        ellipsoid; the third changes the SURFACE to the baked grid. That
        distinction matters everywhere else -- it is why ``object_contact_exact``
        does not ride on ``object_contact`` in the params -- but not here: from
        the panel's point of view they are three radio buttons.

        ``source`` is the handle that changed. Re-entrant by construction (it
        writes the OTHER handles, whose own callbacks land right back here), so
        it is latched; it also sits out :attr:`_restoring`, since Reset and the
        phase presets write every box as one batch and settle it themselves at
        the end.
        """
        if self._restoring or self._contact_guard:
            return
        if not source.value:
            return             # clearing a box never conflicts with anything
        others = [h for h in self._object_contact_boxes() if h is not source]
        if not any(h.value for h in others):
            return
        self._contact_guard = True
        try:
            for handle in others:
                handle.value = False
        finally:
            self._contact_guard = False


    def _planar_contact_available(self):
        """``(ok, reason)`` for whether Eq 13 in-plane contact can be built for
        the scene AS SET UP IN THE GUI -- checked before a solve rather than
        after, so an impossible request never reaches the solver.

        Mirrors the three refusals in :func:`config.attach_contact` exactly. The
        two live here as well because the GUI knows the answer while the box is
        still being offered, and greying a control is a better way to say "not
        for this object" than an exception after Auto solve."""
        if self.g_obj_contact_plane is None:
            return False, ("this hand has no measured pinch table, so Eq 11 has "
                           "no centroid to span a pulling plane with")
        if not self.caps["planar_contact"]:
            return False, ("this binding cannot build it (no "
                           "EnvironmentConfig.object_contact_in_plane)")
        # Through resolve_scene, so the answer is read off the SAME spec the next
        # solve will build from rather than a second lookup that could disagree.
        spec = resolve_scene(self.params)[0]
        if ellipsoid_members(spec) is None:
            return False, (f"a `{spec['type']}` object has no ellipsoid "
                           f"cross-section for the pulling plane to cut")
        names = [n for n, c in zip(self.digit_names, self.g_contacts) if c.value]
        if pinch_pose(names) is None:
            return False, (f"no measured pinch pose for "
                           f"({', '.join(names) or 'no fingers'}), so Eq 11 has "
                           f"no centroid to span the plane with")
        return True, ""


    def _refresh_planar_contact_gate(self):
        """Grey the in-plane contact box -- and clear it if it was on -- whenever
        the current object or digit set cannot support it.

        Clearing rather than leaving it checked-but-disabled is deliberate: a
        ticked box that the next solve would refuse is a lie about what is in the
        graph. The status line says why (see :meth:`_planar_contact_note`)."""
        if self.g_obj_contact_plane is None:
            return             # the box was never built; nothing to grey
        ok, _reason = self._planar_contact_available()
        self.g_obj_contact_plane.disabled = not ok
        if not ok and self.g_obj_contact_plane.value:
            # Only this box is cleared -- the 3D box is deliberately NOT ticked
            # in compensation. Substituting the other metric for the one that was
            # asked for is exactly the silent fallback attach_contact refuses to
            # make; object contact simply goes off, and the status line says why.
            self._contact_guard = True
            try:
                self.g_obj_contact_plane.value = False
            finally:
                self._contact_guard = False


    def _exact_contact_available(self):
        """``(ok, reason)`` for whether the exact-SDF contact form can be built
        for the object currently selected.

        The sibling of :meth:`_planar_contact_available`, and the same argument
        for existing: the GUI knows the answer while the box is still being
        offered, and greying a control says "not for this object" better than an
        exception after Auto solve does.

        The distinction the reason has to carry is between an object that has no
        exact form and a MACHINE that has not baked one. Every object has an
        exact form; almost no fresh checkout has any of them on disk, because the
        grids are build output. Reporting the second as the first would tell the
        user their object cannot be grasped precisely when what it actually needs
        is one command."""
        if not self.caps["contact_exact"]:
            return False, ("this binding cannot build it (no "
                           "EnvironmentConfig.object_contact_exact)")
        spec = resolve_scene(self.params)[0]
        if not names_exact_form(spec):
            return False, (f"a `{spec['type']}` object carries no exact "
                           f"geometry to contact")
        if not has_exact_form(spec):
            return False, (f"`{self.params.primitive}` has not been baked on "
                           f"this machine -- run "
                           f"`python scripts/objects/setup_objects.py`")
        return True, ""


    def _refresh_exact_contact_gate(self):
        """Grey the exact-contact box, the grasp-alignment box and the phases
        that need them -- clearing any that were on -- whenever the selected
        object has no baked grid.

        Grasp alignment goes with it because it is built on the witness points
        the exact form creates; a phase box goes with it because checking one
        would write a constraint set the next solve refuses. Clearing rather than
        leaving them ticked-but-disabled is the rule
        :meth:`_refresh_planar_contact_gate` already follows: a ticked box the
        solve would reject is a lie about what is in the graph."""
        if self.g_obj_contact_exact is None:
            return
        ok, _reason = self._exact_contact_available()
        boxes = [self.g_obj_contact_exact, self.g_grasp_align]
        # The phases whose preset asks for the exact form -- read off the
        # presets rather than by phase NUMBER, since the number means different
        # things on different hands.
        for name, box in self._phase_checkboxes().items():
            preset = phase_presets(self.hand.name)[name]
            if preset.overrides.get("object_contact_exact"):
                boxes.append(box)
        self._contact_guard = True
        try:
            for box in boxes:
                if box is None:
                    continue
                box.disabled = not ok
                if not ok and box.value:
                    box.value = False
        finally:
            self._contact_guard = False


    def _grasp_subset_note(self):
        """The loaded object's shell counts as ``(n_subset, n_members)``, or None
        when there is no choice to describe.

        None for three different reasons, all of which mean "leave the control
        greyed": the binding cannot narrow a set, the object is not a set, or the
        object's fit names no proper subset. Only the first is a shortcoming."""
        if not self.caps.get("grasp_subset", False):
            return None
        spec = get_primitive_specs().get(self.params.primitive, {})
        subset = spec.get("grasp_subset")
        if not subset:
            return None
        return len(subset), len(spec["members"])


    def _refresh_grasp_subset_gate(self):
        """Grey the contact-shells dropdown, and say what it would do, for the
        object now loaded.

        Unlike :meth:`_refresh_planar_contact_gate` this does NOT reset the value
        when it greys out. There is nothing to reset: an object with no authored
        subset is contacted on every shell whichever mode is selected, so the
        setting is inert rather than a lie, and preserving it means it still
        applies when the user returns to an object that does have one."""
        counts = self._grasp_subset_note()
        self.g_contact_shells.disabled = counts is None
        if counts is None:
            reason = ("needs a rebuilt _gepetto_solvers with EnvironmentConfig."
                      "contact_ellipsoid_subset"
                      if not self.caps.get("grasp_subset", False)
                      else "this object's fit names no grasp subset, so every "
                           "shell is a target")
            self.g_contact_shells.hint = (
                f"Which shells of the object the fingertips may be driven onto "
                f"-- inert here: {reason}.")
            return
        n_subset, n_members = counts
        self.g_contact_shells.hint = (
            f"Which shells of the object the fingertips may be driven onto. "
            f"{n_subset} of this object's {n_members} shells are authored grasp "
            f"targets; the other {n_members - n_subset} bound its shape rather "
            f"than offering a handle. Either way ALL {n_members} still collide, "
            f"so 'grasp subset' narrows what the hand reaches FOR, never what it "
            f"can reach THROUGH.")


    def _planar_contact_note(self):
        """Say why the in-plane contact box is greyed, when it is."""
        if self.g_obj_contact_plane is None:
            return []          # no box, so nothing on screen to account for
        ok, reason = self._planar_contact_available()
        if ok:
            return []
        return [f"*in-plane object contact unavailable: {reason}*"]


    def _planar_gap_note(self):
        """Say when the in-plane overlay is on but has nothing to measure.

        Three ways to get an empty overlay, none of them a failure: a binding
        that cannot evaluate the factor, an object with no ellipsoid form
        (cube/cylinder/capsule -- the factor takes an ellipsoid set), or solved
        digits with no measured pinch pose, which leaves Eq 11 without its
        centroid and so without a plane."""
        if self.g_show_planar_gap is None:
            return []          # no pinch table on this hand: no overlay to explain
        if not self.g_show_planar_gap.value or self.result is None:
            return []
        if not self.caps["planar_gap"]:
            return ["**in-plane distance: UNAVAILABLE** -- this binding has no "
                    "`ellipsoid_set_planar_gap`; rebuild it (`pip install .` "
                    "from the crest-sparse root)"]
        if ellipsoid_members(self.result.spec) is None:
            return [f"**in-plane distance: NOTHING DRAWN** -- a "
                    f"`{self.result.spec['type']}` object has no ellipsoid "
                    f"cross-section; use a sphere, an ellipsoid or a ycb: object"]
        if pinch_pose(self.result.contact_names()) is None:
            return ["**in-plane distance: NOTHING DRAWN** -- no measured pinch "
                    "pose for the solved digits, so Eq 11 has no plane to cut "
                    "the object with"]
        return []


    def _half_space_note(self):
        """The opposition half-space's own status line: inert when no finger is
        checked for it, and the standoff it is holding when there is one.

        The constraint is built per finger off its own ``half_space_node``, so
        the only way to check the box and get nothing is to check no fingers --
        the finger mask is the one dependency it has left."""
        if not self.params.half_space:
            return []
        names = [n for n, c in zip(self.digit_names, self.g_contacts) if c.value]
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
        lines.extend(self._split_axis_note())
        lines.extend(self._opposition_side_note())
        lines.extend(self._rotation_driver_note())
        return lines


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


    def _split_axis_note(self):
        """Which way the split LINE is pointing, and whether the object chose it.

        The line is derived silently from the object's silhouette on the support
        plane, and a derivation with no readout is one nobody can check: the
        degenerate case (every ball, can and bowl) hands back a fixed world
        direction that looks identical on screen to a measured one, so a wrong
        axis and a defaulted axis are indistinguishable without the ratio."""
        spec, _center, rotation, _pose = resolve_scene(self.params)
        try:
            e_long, ratio = object_principal_inplane_axis(
                spec, rotation, self.params.plane_normal)
        except ValueError:
            return []
        # Reported as the LINE (mod 180 deg), since the sign shown to the user is
        # the side assignment, and _opposition_side_note already covers that.
        deg = np.degrees(np.arctan2(e_long[1], e_long[0])) % 180.0
        if ratio < INPLANE_DEGENERACY_RATIO:
            return [f"split line: object is in-plane isotropic ({ratio:.2f}x), "
                    f"so the default +Y split is used — thumb on the -X side"]
        return [f"split line: {deg:.0f}° from +X in the table plane "
                f"(object is {ratio:.1f}x longer that way)"]


    def _opposition_side_note(self):
        """How far the current posture is from the side assignment being asked
        for -- the one number that says whether this constraint is a nudge or a
        demand that the hand turn itself inside out.

        Worth a line of its own because the failure is silent otherwise: with
        the sides the wrong way up the solve stalls on iteration 3 with the hand
        visibly untouched, which reads as the solver giving up rather than as
        the constraint asking for a 180 degree roll."""
        if self.g_half_sides is None:
            return []          # no pre-grasp panel: the constraint cannot be on
        res = self._iter_view()
        opposing = self.hand.opposing_digit
        if res is None or opposing is None or opposing not in res.finger_names:
            return []
        gaps = half_space_witness(self.params, res, 0)
        if not gaps or opposing not in gaps:
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
        # FK, a phase-4 close and a phase-5 lift all enforce NOTHING, so a
        # contact/table gap line under them would be reporting a distance nobody
        # asked to close.
        if self.mode not in ("FK", "Close", "Lift"):
            lines.extend(self._contact_lines(-1))
        lines.extend(self._half_space_note())
        lines.extend(self._wrist_gauge_note())
        lines.extend(self._pinch_note())
        lines.extend(self._finger_plane_note())
        lines.extend(self._planar_contact_note())
        lines.extend(self._planar_gap_note())
        lines.extend(self._object_size_note())
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
