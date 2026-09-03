"""Reading the GUI controls into HandSolveParams, and back again.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""


import numpy as np

from gepetto_solvers.core.geometry.scene import TABLE_NORMAL
from gepetto_solvers.core.solvers import HandFKSolver, HandSolveParams

from .constants import (
    CONTACT_SHELL_MODES,
    OPPOSITION_SIDES,
    _euler_to_R,
)


class ParamsSyncMixin:
    def _fresh_params(self):
        """A cold ``HandSolveParams`` plus this app's OWN scene defaults.

        The single source of "defaults" for the fields no GUI control owns, so
        startup and *Reset defaults* cannot drift apart. Reset used to build a
        bare ``HandSolveParams()``, which restored the headless
        ``table_burial = 0.5`` under sliders that still read 0 and re-seated the
        table -- moving the object and its ellipsoids with it.

        Only the not-GUI-backed fields belong here; everything a widget drives
        is written by :meth:`_sync_params` from the (already restored) handles.
        """
        params = HandSolveParams()
        # Not HandSolveParams' own default, which stays whatever headless
        # callers/other scripts expect. Set on the params (not just on the
        # dropdown widget) because _rebuild_fk()/_refresh_object() run before
        # the first _sync_params() and read params.primitive directly; the
        # dropdown's own default_label computation reads it too, so the widget
        # follows automatically.
        params.primitive = self._resolve_default_primitive()
        # Seat the object ON the table rather than half-buried in it. Another of
        # this app's own defaults (like the object above): HandSolveParams keeps
        # 0.5 for the §1.8 low-profile-object case its docstring argues for, but
        # for browsing arbitrary objects -- YCB scans included -- an object sunk
        # halfway through the table reads as a bug in the scene. The seating rule
        # derives the height from the object's own along-normal extent, so this
        # is correct per object with nothing to re-tune per pick.
        params.table_burial = 0.0
        # The hand's own start pose. HandSolveParams' wrist default is the
        # TENDON hand's measured hover, which aims a hand whose fingers extend
        # differently at nothing -- so it is the hand that says where it starts,
        # not the params dataclass.
        wrist, means = self.hand.default_pose()
        params.wrist_pose = wrist
        if self.has("single_drive"):
            params.flexor_tensions = [
                float(self.hand.actuation.drive_value(m)) for m in means]
        else:
            params.joint_targets = [list(map(float, m)) for m in means]
        return params


    # -- solver plumbing --

    def _rebuild_fk(self):
        # `hand` is not optional here: without it the solver builds the
        # DEFAULT hand, and the app would pose one robot while every panel
        # described another -- which looks like a working workbench.
        self.fk_solver = HandFKSolver(self.params, self.hand)
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
        p.object_center, p.object_rotation = self._object_pose_from_sliders()
        self._sync_wrist()
        # The actuation panel, in whichever shape this hand got. Only one of
        # the two exists -- see GuiMixin._build_joint_sliders.
        if self.g_passive is not None:
            p.passive_tension = self.g_passive.value
            p.flexor_tensions = [s.value for s in self.g_flexors]
            p.flexor_tension_sigma = 10.0 ** self.g_flexor_sigma.value
            p.passive_tension_sigma = 10.0 ** self.g_passive_sigma.value
        if self.g_joints:
            # q_S, the mean of p(q). One vector per digit.
            p.joint_targets = [[j.value for j in row] for row in self.g_joints]
            p.flexor_tension_sigma = 10.0 ** self.g_joint_sigma.value
            p.passive_tension_sigma = 10.0 ** self.g_joint_sigma.value
        p.ellipsoid_set_beta = float(self.g_set_beta.value)
        p.use_grasp_subset = CONTACT_SHELL_MODES[self.g_contact_shells.value]
        p.contact_fingers = [c.value for c in self.g_contacts]
        # Which FORM, once object contact is on at all. The two boxes are kept
        # mutually exclusive by _enforce_object_contact, so this cannot be read
        # as "both": object_contact says whether there is a contact,
        # object_contact_in_plane says which distance it is measured with. A
        # hand with no pinch table has only the 3D box, and only the 3D form.
        in_plane = (self.g_obj_contact_plane is not None
                    and self.g_obj_contact_plane.value)
        # ...and the third form: the witness contact against the object's baked
        # SDF. Unlike the in-plane box this one does not ride on object_contact
        # -- it names a different SURFACE, not a different metric -- so it turns
        # the other two off rather than qualifying them.
        exact = (self.g_obj_contact_exact is not None
                 and self.g_obj_contact_exact.value)
        p.object_contact = (self.g_obj_contact.value or in_plane) and not exact
        p.object_contact_in_plane = in_plane and not exact
        p.object_contact_exact = exact
        # Both representations on every env whenever the exact form is in play,
        # so the collision inequalities keep reading E_obj while the contact
        # reads the grid. Also written when the exact form is OFF but the
        # object has a grid, so switching the box mid-session does not need a
        # scene rebuild -- attaching an unused surface costs nothing, since
        # which one each family reads is decided per family.
        p.object_proxy_and_exact = exact
        p.grasp_alignment = (self.g_grasp_align is not None
                             and self.g_grasp_align.value)
        if self.g_grasp_sigma_force is not None:
            p.sigma_grasp_force = 10.0 ** self.g_grasp_sigma_force.value
            p.sigma_grasp_torque = 10.0 ** self.g_grasp_sigma_torque.value
        p.table_contact = self.g_tbl_contact.value
        # No box means no CHOICE, not a default: the 4-row [c_R, c_O, c_T1,
        # c_T2] witness form is what the formulation defines for sphere-only
        # collision geometry, since the tangential rows already force the
        # radius vector collinear with the surface normal. See the note beside
        # the checkbox in _build_gui.
        p.contact_drop_normal_row = (True if self.g_drop_normal_row is None
                                     else self.g_drop_normal_row.value)
        # The pre-grasp family, all four of them off for a hand that declares
        # no pre-grasp -- the panel is absent, so there is nothing to read and
        # nothing to attach.
        p.half_space = self.g_half_space is not None and self.g_half_space.value
        p.half_space_split = None   # derive from object_center
        # Cleared every sync so the SIGN is re-resolved against the posture on
        # screen: _attach_opposition writes the oriented axis back here, and a
        # stale one would keep sending the thumb to the side it was on two
        # solves ago.
        p.half_space_axis = None
        if self.g_half_sides is not None:
            p.half_space_flip = OPPOSITION_SIDES[self.g_half_sides.value]
            p.half_space_margin = self.g_half_margin.value
        p.pregrasp_center = (self.g_pregrasp_center is not None
                             and self.g_pregrasp_center.value)
        if self.g_h_clear is not None:
            p.h_clear = self.g_h_clear.value
        p.pregrasp_axis_align = (self.g_axis_align is not None
                                 and self.g_axis_align.value)
        p.pregrasp_centroid = (self.g_pregrasp_centroid is not None
                               and self.g_pregrasp_centroid.value)
        p.sigma_wrist_pos = 10.0 ** self.g_sig_pos.value
        p.sigma_wrist_rot = 10.0 ** self.g_sig_rot.value
        # AL
        p.al_mu = self.g_al_mu.value
        p.al_rate = self.g_al_rate.value
        p.al_iters = self.g_al_iters.value
        p.ik_settle_steps = int(self.g_ik_settle.value)
        # rod physics -- a rigid-linkage hand has no rod to keep planar, so the
        # folder is absent and the approximation stays off. The sigmas keep
        # their dataclass defaults; nothing reads them with the factor off.
        p.planar_bending = (self.g_planar_bend is not None
                            and self.g_planar_bend.value
                            and self.caps["planar_bending"])
        if self.g_planar_bend is not None:
            p.sigma_planar_bend = 10.0 ** self.g_planar_bend_sigma.value
            p.sigma_planar_twist = 10.0 ** self.g_planar_twist_sigma.value
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
        # The TABLE stays where the scene seats it: left as None, so
        # resolve_table_origin keeps answering with the seating rule and the slab,
        # its corner frame, its grid, the calibration target and the robot
        # registration (_corner_viz) all stay put. The slider moves the CONSTRAINT
        # plane instead -- a height above that surface, resolved by the solver on
        # demand rather than baked into an explicit origin here, so the two planes
        # cannot drift apart and the offset cannot compound across syncs.
        p.plane_origin = None
        p.constraint_plane_height = self.g_constraint_height.value
        # display toggles. Each is read only where its checkbox was built; an
        # overlay whose hand cannot draw it stays off for the life of the app.
        def _shown(handle):
            return handle is not None and handle.value

        self.scene.show_discs = _shown(self.g_show_discs)
        self.scene.show_disc_frames = _shown(self.g_show_disc_frames)
        self.scene.show_link_frames = _shown(self.g_show_link_frames)
        self.scene.show_contact_spheres = self.g_show_contact.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self.scene.show_gap_lines = self.g_show_gaps.value
        self.scene.show_finger_planes = _shown(self.g_show_finger_planes)
        self.scene.show_planar_gap = _shown(self.g_show_planar_gap)


    def _reset_defaults(self, _=None):
        """Put every control back to the value it was built with and cold-start.

        A full reset rather than a re-solve: fresh params (so the warm-start
        posture and any derived scene state go too), fresh FK solver, no
        stepper, camera back on the default object.

        Refused while a solve is running, and while the e-stop is engaged: a
        reset cold-starts, which would throw away exactly the state the stop was
        protecting. Rearm first, deliberately."""
        if self.estop.busy is not None or self.estop.is_tripped():
            return
        self._restoring = True
        try:
            for handle, value in self._gui_defaults:
                handle.value = value
        finally:
            self._restoring = False
        # _fresh_params, not a bare HandSolveParams(): the app's own scene
        # defaults (object seating on the table) are not GUI-backed, so a bare
        # one would put the table/object/ellipsoids somewhere the restored
        # sliders do not describe.
        self.params = self._fresh_params()
        # A button, so not in _gui_defaults -- restored by hand, to the same
        # default the app opens with rather than to off.
        self.warm_start = self.caps["solver_seed"]
        self._refresh_warm_start()
        # The restore above re-ticked DEFAULT_PHASE's box but, running under
        # _restoring, could not fire the callback that gives the tick meaning.
        self._apply_default_phase()
        self._refresh_planar_contact_gate()
        self._refresh_exact_contact_gate()   # restored object may not support it
        self._sync_params()
        self._rebuild_fk()          # also drops the stepper
        self._refresh_object()
        self._aim_all_cameras()
        self._fk_solve()
        self._set_status("**reset** to defaults  \n" + self.g_status.content)


    def _live_fk(self, _=None):
        """FK is fast and warm-starts, so re-solve live as sliders move.

        Only while the hand is FK-posed: once the stepper is running, the same
        sliders are read live by every step, and re-solving FK here would throw
        that loop away mid-solve.

        The busy/e-stop test is _fk_solve's gate anyway, so this is belt and
        braces -- but it is the reason the latch has to LATCH: this fires on
        every slider drag, and a momentary stop would let the next twitch of a
        tension slider re-pose the hand straight after the button was hit."""
        if self.mode == "FK" and not self._restoring:
            self._fk_solve()
