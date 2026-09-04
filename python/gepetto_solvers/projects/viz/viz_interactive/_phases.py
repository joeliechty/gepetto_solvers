"""The staged pipeline: applying and toggling phase presets.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import math

from gepetto_solvers.core.solvers import phase_presets

from .constants import (
    DEFAULT_PHASE,
)


class PhaseMixin:
    # -- phase presets --

    @staticmethod
    def _set(handle, value):
        """Assign to a GUI handle if it exists. See _apply_phase_preset."""
        if handle is not None:
            handle.value = value

    def _preset_widget(self, field):
        """The GUI handle a ``PHASE_PRESETS`` override field writes onto, for
        the plain 1:1 cases (everything except the two object-contact form
        boxes, ``sigma_wrist_pos``/``sigma_wrist_rot``,
        ``flexor_tension_sigma`` and ``passive_tension_sigma``, which
        :meth:`_apply_phase_preset` special-cases itself, and
        ``contact_fingers``, which it deliberately ignores)."""
        return {
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
            # Registered so a future preset CAN name it, though none should:
            # planarity is a property of the hand, not of a phase.
            "planar_bending": self.g_planar_bend,
            # The exact-SDF contact form and h_grasp. Plain 1:1 cases: unlike
            # the two ellipsoid forms below, `object_contact_exact` is its own
            # box and needs no exclusivity dance here -- _apply_phase_preset
            # writes every form field a preset names, and the presets never name
            # two at once.
            "object_contact_exact": self.g_obj_contact_exact,
            "grasp_alignment": self.g_grasp_align,
            # Attaching BOTH object representations is not a control the panel
            # offers: it is implied by the contact form (see _sync_params), so a
            # preset naming it writes nothing and _sync_params derives it.
            "object_proxy_and_exact": None,
        }[field]


    def _standing_preset_fields(self):
        """Preset fields the PANEL refuses to write, beyond ``contact_fingers``.

        The Allegro's phases retune the wrist prior (loose for phase 0's big
        repositioning move, an order of magnitude tighter for the four contact
        phases after it), and that is right for a headless run of one phase from
        cold. It is wrong for the workbench, where the phases are STEPPED on a
        single scene and the wrist prior is a control the user tunes and then
        watches across the whole pipeline: having it jump under the cursor on
        every phase toggle means a hand-picked value survives exactly one click.

        So on this hand the wrist sigmas stay where the sliders were built (-1,
        see GuiMixin) or dragged. The joint sigma needs no entry: no Allegro
        preset names a joint or tension sigma, so that slider is already the
        standing control this makes the wrist ones.
        """
        if self.hand.name == "allegro":
            return ("sigma_wrist_pos", "sigma_wrist_rot")
        return ()


    def _apply_phase_preset(self, name):
        """Write ``PHASE_PRESETS[name]``'s overrides directly onto the
        corresponding GUI widgets, so checking the preset box is a single
        visible action: every affected checkbox/slider jumps to the preset's
        value on screen. One solve-ready sync/invalidate happens at the end --
        Auto solve is a separate, manual next step, not triggered here.

        The one field no preset writes here is ``contact_fingers``: the finger
        mask carries across phases untouched, see the branch below."""
        overrides = phase_presets(self.hand.name)[name].overrides
        self._restoring = True   # batch write; no live-FK/other side effects
        try:
            for field, value in overrides.items():
                if field == "contact_fingers":
                    # NOT written. Which digits are grasping is the user's
                    # standing choice, not part of what a phase IS: the panel
                    # is stepped phase0 -> phase1 -> phase2 on one scene, and a
                    # preset that re-imposed its own mask would silently
                    # un-pick the hand between stages -- tick all five for the
                    # pre-grasp, check phase 1, and three of them quietly go
                    # away. Every preset that names the field names the SAME
                    # three-finger pinch anyway (phase 5 deliberately names
                    # none), so honouring it here only ever overwrote a
                    # deliberate selection with the value it started at. The
                    # boxes are seeded at build time and put back by Reset,
                    # neither of which goes through a preset, so the opening
                    # pinch set is unaffected. Headless callers still get the
                    # mask -- solvers.apply_phase_preset writes the field, and
                    # a script has no standing selection to protect.
                    continue
                if field in self._standing_preset_fields():
                    # Same treatment, for the same reason: a hand whose phases
                    # are meant to be stepped on ONE wrist prior (see that
                    # method) keeps whatever the sliders say, and the preset's
                    # own value is ignored HERE only -- solvers.apply_phase_
                    # preset still writes it for headless callers.
                    continue
                if field in ("object_contact", "object_contact_in_plane") \
                        and self.g_obj_contact_plane is None:
                    # No in-plane box on this hand, so there is no FORM to
                    # choose and no pair to keep exclusive: object_contact is
                    # the whole statement, and a preset naming only the form is
                    # a statement about a metric this hand does not offer.
                    if field == "object_contact":
                        self.g_obj_contact.value = bool(value)
                elif field == "object_contact":
                    # A preset that names object_contact ALONE says only WHETHER
                    # the object is contacted, with no opinion on which metric,
                    # so the form the user picked survives it. Off clears both
                    # boxes (otherwise a checked in-plane form would keep contact
                    # alive through a phase that asked for none); on writes the
                    # 3D box only if no form is selected yet. A preset that also
                    # names the FORM is handled by the branch below, which writes
                    # both boxes -- so this one stands aside for it rather than
                    # racing it on dict order.
                    if not value:
                        self.g_obj_contact.value = False
                        self.g_obj_contact_plane.value = False
                    elif "object_contact_in_plane" in overrides:
                        pass
                    elif not self.g_obj_contact_plane.value:
                        self.g_obj_contact.value = True
                elif field == "object_contact_in_plane":
                    # An explicit choice of metric, so it sets the mutually
                    # exclusive pair itself: the callback that normally enforces
                    # that (_enforce_object_contact) is suppressed during this
                    # batch. Gated on the preset's own object_contact, so a form
                    # cannot switch contact back on for a phase that asked for
                    # none.
                    on = bool(value) and bool(overrides.get("object_contact", True))
                    self.g_obj_contact_plane.value = on
                    self.g_obj_contact.value = (
                        not on and bool(overrides.get("object_contact", False)))
                elif field == "passive_tension_sigma":
                    self._set(self.g_passive_sigma, math.log10(value))
                elif field == "sigma_wrist_pos":
                    self.g_sig_pos.value = math.log10(value)
                elif field == "sigma_wrist_rot":
                    self.g_sig_rot.value = math.log10(value)
                elif field == "flexor_tension_sigma":
                    self._set(self.g_flexor_sigma, math.log10(value))
                else:
                    # A preset may name a control this hand does not have -- the
                    # phase presets were written for the tendon hand, and a
                    # joint-space one has no tension sigmas or rod bending. Skip
                    # rather than fail: the rest of the phase still applies, and
                    # the missing control is missing BECAUSE it would not mean
                    # anything here.
                    self._set(self._preset_widget(field), value)
        finally:
            self._restoring = False
        # The batch ran with every per-handle callback suppressed, so the
        # in-plane form's gate -- which is keyed off the object AND the finger
        # mask (it needs a measured thumb pinch pose) -- has to be re-run by
        # hand against whichever object and digits are currently selected.
        self._refresh_planar_contact_gate()
        self._refresh_exact_contact_gate()
        self._refresh_normal_row_gate()
        # ...and that gate may have just cleared an in-plane box the preset asked
        # for (SDF object, or a digit set with no measured pinch pose). Falling
        # back to the 3D metric is right HERE, unlike when the user ticks the box
        # by hand: a phase preset's claim is that the object IS contacted during
        # this phase, and dropping the contact entirely would break the phase
        # rather than substitute a metric. The 3D box ticks visibly, so the panel
        # still says exactly what is in the graph.
        if (overrides.get("object_contact")
                and not (self.g_obj_contact.value
                         or (self.g_obj_contact_plane is not None
                             and self.g_obj_contact_plane.value))):
            self.g_obj_contact.value = True
        self._sync_params()
        self._invalidate_stepper()
        self._refresh_object()
        self._render_frame()


    def _apply_default_phase(self):
        """Write :data:`DEFAULT_PHASE`'s preset onto the panel, for the two
        moments the app declares a starting stage: opening, and Reset.

        Needed because both of those get the box TICKED by a mechanism that does
        not fire its callback -- the build-time value, and Reset's ``_restoring``
        batch -- so without this the checkbox would claim a phase the constraint
        controls below it are not actually in."""
        if DEFAULT_PHASE is not None:
            self._apply_phase_preset(DEFAULT_PHASE)


    def _phase_checkboxes(self):
        """Every phase-preset checkbox, name -> handle.

        Just the dict ``_build_gui`` filled, which is already exactly the phases
        this hand HAS a box for -- a phase the hand's own preset set does not
        name, or one whose runner it cannot walk, never got one. Returned rather
        than rebuilt so the wiring cannot drift from the build; every caller here
        either reads ``.value`` off a handle or writes it, so a None would be a
        crash rather than a skip."""
        return dict(self.g_phases)


    def _on_phase_toggle(self, name, _=None):
        """Checking a phase preset applies it and unchecks every OTHER phase
        checkbox first -- they are mutually exclusive stages of the same
        pipeline, and leaving two checked at once would show a state whose
        settings actually contradict each other (e.g. phase 0's
        pregrasp_centroid=True vs. phase 1's False). Unchecking is a no-op --
        the controls a
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
