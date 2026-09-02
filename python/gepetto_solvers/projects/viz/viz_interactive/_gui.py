"""Building the viser control panel.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import math

import numpy as np

from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.geometry.scene import (
    ELLIPSOID_SET_BETA,
    GRASP_FLEXOR_TENSION,
    TABLE_SPAN,
)
from gepetto_solvers.core.solvers import (
    PHASE_PRESETS,
    HandSolveParams,
    R_to_euler,
)

from .constants import (
    CAL_GRID_SPACING,
    CONTACT_SHELL_MODES,
    DEFAULT_CONTACT_SHELL_MODE,
    DEFAULT_PHASE,
    OPPOSITION_SIDES,
    SDF_DROPDOWN_LABELS,
)


class GuiMixin:
    def _build_joint_sliders(self, gui):
        """The commanded posture for a JOINT-SPACE hand: one slider per joint.

        A tendon hand is commanded with one number per digit -- the pull on its
        actuated tendon -- so a single slider per digit says everything. A hand
        that drives every joint has no such summary, so it gets a slider each,
        grouped by digit, and their limits come from the URDF rather than from a
        guess: driving a joint past its stop produces a posture the real hand
        cannot reach, and nothing downstream would flag it (see the note on
        RigidHandKinematics about limits not being enforced in the solve).

        Writes ``params.joint_targets``, which is q_S -- the mean of p(q).
        """
        wrist, means = self.hand.default_pose()
        limits = self._joint_limits()
        self.g_joints = []
        for d, name in enumerate(self.digit_names):
            with gui.add_folder(name, expand_by_default=(d == 0)):
                row = []
                for j, label in enumerate(self.hand.actuation.names):
                    lo, hi = limits[d][j]
                    row.append(gui.add_slider(
                        label, float(lo), float(hi), 0.01,
                        float(np.clip(means[d][j], lo, hi)),
                        hint=f"{name} {label}, radians. Limits are the URDF's. "
                             f"This is q_S: the joint prior pulls the solve "
                             f"toward it, and the kinematics seeds there so a "
                             f"solve starts at zero FK residual."))
                self.g_joints.append(row)
        self.g_joint_sigma = gui.add_slider(
            "log10 joint sigma", -6.0, 1.0, 0.1, -2.0,
            hint="How loose p(q) is -- how far contact may pull the joints away "
                 "from the commanded posture above. Read live, so a drag takes "
                 "effect on the next solve with no rebuild.")
        self.g_actuation_report = gui.add_markdown(
            "_solve to see the joint states_")

    def _joint_limits(self):
        """Per-digit, per-joint ``(lo, hi)``, from the hand.

        The hand knows: the mapping from a digit's joints to its model's
        configuration indices is not arithmetic (see AllegroHand.joint_limits),
        so deriving it here would be re-deriving something already got right.
        Falls back to a generous symmetric range for a hand that cannot say, so
        the sliders still work rather than not existing."""
        limits = getattr(self.hand, "joint_limits", None)
        if limits is None:
            n = self.hand.actuation.n
            return [[(-np.pi, np.pi)] * n for _ in self.digit_names]
        return limits()

    def _input_handles(self):
        """Every value-carrying control, in build order. Buttons and markdown are
        deliberately absent -- Reset restores values, not widgets.

        Nones are stripped at the end: a control for something this hand does
        not have was never built, and Reset should skip it rather than crash."""
        handles = ([self.g_object, self.g_contact_shells,
                 self.g_ik_max, self.g_ik_settle, self.g_carry_duals,
                 self.g_obj_dx, self.g_obj_dy, self.g_obj_dz,
                 self.g_obj_roll, self.g_obj_pitch, self.g_obj_yaw,
                 self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw,
                 self.g_sig_pos, self.g_sig_rot, self.g_passive]
                + self.g_flexors
                + [self.g_flexor_sigma, self.g_passive_sigma,
                   self.g_phase0, self.g_phase1, self.g_phase2, self.g_phase4,
                   self.g_phase5, self.g_close_frac, self.g_lift_height]
                + [self.g_obj_contact, self.g_obj_contact_plane,
                   self.g_tbl_contact, self.g_drop_normal_row,
                   self.g_half_space, self.g_half_sides, self.g_half_margin,
                   self.g_pregrasp_center, self.g_h_clear,
                   self.g_pregrasp_centroid, self.g_axis_align]
                + self.g_contacts
                + [self.g_planar_bend, self.g_planar_bend_sigma,
                   self.g_planar_twist_sigma,
                   self.g_collision, self.g_self_collision,
                   self.g_coll_radius, self.g_coll_sigma, self.g_cull,
                   self.g_set_beta,
                   self.g_table, self.g_constraint_height,
                   self.g_show_constraint_plane, self.g_plane_avoid,
                   self.g_cal_finger, self.g_cal_disc,
                   self.g_cal_x, self.g_cal_y, self.g_cal_z,
                   self.g_cal_roll, self.g_cal_pitch, self.g_cal_yaw,
                   self.g_cal_show,
                   self.g_al_mu, self.g_al_rate, self.g_al_iters,
                   self.g_show_meshes, self.g_show_true_mesh,
                   self.g_show_contact, self.g_show_collision,
                   self.g_show_discs, self.g_show_disc_frames,
                   self.g_show_world, self.g_show_obj_frame,
                   self.g_show_table_frame, self.g_show_grid,
                   self.g_show_gaps, self.g_show_mount,
                   self.g_show_finger_planes, self.g_show_planar_gap,
                   self.g_show_traj]
                + [h for row in self.g_joints for h in row])
        return [h for h in handles if h is not None]


    def _build_gui(self):
        gui = self.server.gui
        # Map the displayed dropdown label back to the real spec key (identity
        # except for the "_sdf"-suffixed baked spheres, and the "ycb:"-prefixed
        # ellipsoid sets, which keep their prefix as the label so they group
        # together and read as "not one of the hand-authored primitives").
        labels, self._label_to_key = self._object_dropdown_labels()

        step_hint = (None if self.caps["ik_stepping"]
                     else "requires a rebuilt _gepetto_solvers with "
                          "HandSolver.reset_al_duals")

        with gui.add_folder("Solver"):
            # Opens on whatever __init__ resolved (see _resolve_default_primitive,
            # which already handled the object being unavailable), so the widget
            # and self.params cannot disagree about which scene is loaded.
            default_label = SDF_DROPDOWN_LABELS.get(self.params.primitive,
                                                    self.params.primitive)
            if default_label not in self._label_to_key:
                default_label = labels[0]
            self.g_object = gui.add_dropdown(
                "object", labels, initial_value=default_label)
            # Which SHELLS of the object the fingertips may be sent to, next to
            # the object it qualifies. Its enabled state and hint depend on the
            # loaded object, so both are (re)set by _refresh_grasp_subset_gate --
            # here and on every object change.
            self.g_contact_shells = gui.add_dropdown(
                "contact shells", list(CONTACT_SHELL_MODES),
                initial_value=DEFAULT_CONTACT_SHELL_MODE)
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
            # Phases 4 and 5 are open-loop RAMPS along measured travel, so
            # they exist only for a hand that has those measurements. A hand
            # without them gets no buttons rather than buttons that cannot run.
            self.g_close = self.g_close_frac = None
            self.g_lift = self.g_lift_height = None
            if self.has("close_ramp"):
                # Phase 4's runner. Sits with Step/Auto rather than in the Presets
                # folder because what it IS is a third way of moving the hand -- and
                # because it must sit above the E-STOP that stops it.
                self.g_close = gui.add_button(
                    "Close", icon=self.viser.Icon.HAND_GRAB,
                    hint="Phase 4: shut every ticked contact finger TOGETHER, and "
                         "record the whole ramp. Starts from the solve on screen "
                         "while Warm start is on -- its wrist pose and flexor "
                         "tensions are adopted onto the sliders and its posture "
                         "seeds the FK solver -- so a close follows a phase-2 "
                         "approach instead of jumping back to whatever the sliders "
                         "were last commanded to. Not a solve -- no constraint is "
                         "enforced and nothing converges, so the CONTACT does not "
                         "carry over: the fingers settle wherever their tensions "
                         "put them before the ramp starts. Each finger is commanded "
                         "along the same fraction of its own remaining tendon "
                         "travel, so they start together, arrive together, and none "
                         "races ahead or stalls on its stop; the status line reports "
                         "the worst gap between them that the poses actually came "
                         "back with. Every pose is kept, so the Solve steps scrubber "
                         "replays the close and the Robot folder plays it as "
                         "waypoints.")
                self.g_close_frac = gui.add_slider(
                    "close depth (fraction)", 0.1, 1.0, 0.05, self.hand.motion.close_fraction,
                    hint="How far into the tendon travel each finger HAS LEFT the "
                         "close goes. 1.0 drives every digit onto its hardware stop, "
                         "where finger_servo_node saturates and the last waypoints "
                         "command a position the motors cannot reach; the default "
                         "stops short of it. Fractions of REMAINING travel, so a "
                         "finger already half shut moves half as far as an open one "
                         "and both still finish at the same moment.")
                # Phase 5's runner, next to phase 4's for the same two reasons, and
                # in the order the two phases run.
                self.g_lift = gui.add_button(
                    "Lift", icon=self.viser.Icon.ARROW_UP,
                    hint="Phase 5: raise the wrist straight up in the WORLD frame "
                         "from wherever it is, and record the whole ramp. The mirror "
                         "of Close -- that one moves the tendons and leaves the "
                         "wrist alone, this one moves the wrist and leaves the "
                         "tendons alone, so the hand goes up holding exactly the "
                         "grasp it closed on. Carries the state on screen the way "
                         "Close does, so it can also be pressed straight off a "
                         "solve. Not a solve: no constraint is "
                         "enforced, and NOTHING IN THE MODEL HOLDS THE OBJECT, so "
                         "the hand rises and the object stays on the table. Every "
                         "pose is kept, so the Solve steps scrubber replays the lift "
                         "and the Robot folder plays it as waypoints.")
                self.g_lift_height = gui.add_slider(
                    "lift height (m)", 0.0, 0.3, 0.01, self.hand.motion.lift_height_m,
                    hint=f"How far up the wrist goes, along world +Z. Split into "
                         f"{self.hand.motion.lift_steps} equal steps whatever the height, so the "
                         f"taller the lift the bigger each step -- past ~50 mm a "
                         f"step the FK warm start stops carrying the hand and every "
                         f"pose rebuilds from cold (slower, still correct; the "
                         f"status line says when it happens).")

            # NEVER disabled -- not while solving, not while idle, not on a
            # binding missing every other capability. A stop button that can be
            # greyed out is not a stop button; this one is always available and
            # always the highest-priority thing on the page.
            self.g_ik_stop = gui.add_button(
                "E-STOP", icon=self.viser.Icon.ALERT_TRIANGLE, color="red",
                hint="Software e-stop. Latches: it halts a running auto-solve "
                     "and then REFUSES every solve -- FK, Step, Auto, and the "
                     "live re-solve on the pose/tension sliders -- until you "
                     "press Rearm, so nothing can restart the hand by accident. "
                     "Nothing is lost: the solve pauses where it stopped, "
                     "keeping its multipliers, penalty weight and full step "
                     "history, and Rearm resumes from there rather than "
                     "restarting. A running auto-solve stops at the end of the "
                     "current AL outer iteration (~1.7 s worst case) -- one "
                     "iteration is a single call into the C++ solver with no "
                     "interrupt hook, so that is a floor, not a setting."
                     + ("" if self.caps["gil_release"] else
                        " WARNING: this binding does not release the GIL during "
                        "a solve, so this click cannot even be received until "
                        "the current iteration ends -- rebuild with `pip "
                        "install .` from the crest-sparse root."))
            self.g_rearm = gui.add_button(
                "Rearm", icon=self.viser.Icon.LOCK_OPEN, disabled=True,
                hint="Release the e-stop and hand the controls back. Live only "
                     "while the latch is engaged AND the stopped solve has "
                     "actually returned -- rearming around a solve still "
                     "winding down would re-enable the panel while the hand is "
                     "still moving.")
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

        # Directly under Solver, so the buttons that move a physical robot sit
        # next to the E-STOP that stops them rather than pages away.
        if self.ros_mode:
            self._build_robot_folder(gui)

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
                      "scrubber, so you can rewind and branch. It governs the "
                      "phase-4 Close and phase-5 Lift the same way: on, they "
                      "adopt the solved wrist and tensions and start from the "
                      "posture on screen; off, they start from what the sliders "
                      "command. Only the POSTURE "
                      "carries -- the penalty schedule restarts either way."
                      if self.caps["solver_seed"]
                      else "requires a rebuilt _gepetto_solvers with "
                           "HandSolverConfig.initial_state"))
            self.g_carry_duals = gui.add_checkbox(
                "carry AL duals", False, disabled=not self.caps["dual_transfer"],
                hint=("Also carry the Augmented Lagrangian multipliers, matched "
                      "to the new constraint set by identity. This is what stops "
                      "the hand letting go of constraints it had already "
                      "satisfied: without it the rebuilt solve restarts the "
                      "penalty schedule at mu = al_mu with every multiplier at "
                      "zero. Off by default even so -- carried multipliers stiffen "
                      "a constraint that the NEW stage may need to move, so the "
                      "posture is carried on its own unless you ask for both. "
                      "Tick to see the difference."
                      if self.caps["dual_transfer"]
                      else "requires a rebuilt _gepetto_solvers with "
                           "HandSolver.set_initial_duals"))
            self.g_reset = gui.add_button(
                "Reset defaults", icon=self.viser.Icon.ROTATE,
                hint="Put every control back to the value it opened with -- "
                     "including the warm start and the opening phase preset, "
                     "which are restored to their startup state rather than to "
                     "off -- drop the stepped solve, and re-pose with FK.")
            self.g_warm_status = gui.add_markdown("")

        with gui.add_folder("Object pose", expand_by_default=False):
            # OFFSETS from whatever the primitive resolves to on its own, not
            # absolute coordinates. Two reasons: every object keeps its own
            # sensible default placement (the grasp locus for the graspable ones,
            # OBJECT_CENTER for the small ones), and switching objects therefore
            # cannot fling the new one somewhere arbitrary just because the
            # previous one had been dragged. All-zero == the derived pose, so the
            # scene opens exactly as it did before these existed.
            self.g_obj_dx = gui.add_slider("dx (m)", -0.15, 0.15, 0.001, 0.0)
            self.g_obj_dy = gui.add_slider("dy (m)", -0.15, 0.15, 0.001, 0.0)
            self.g_obj_dz = gui.add_slider("dz (m)", -0.15, 0.15, 0.001, 0.0)
            self.g_obj_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_obj_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_obj_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, 0.0)
            gui.add_markdown(
                "_Rotation is applied about the object's own centre, on top of "
                "the primitive's base orientation. Moving the object changes the "
                "constraint set, so it restarts the IK loop._")

        with gui.add_folder("Wrist start pose", expand_by_default=False):
            # Seeded from the HAND's own start pose (via _fresh_params, so the
            # sliders and params cannot disagree). Not the shared
            # solvers.DEFAULT_WRIST_* constants: those are the tendon hand's
            # measured hover and point a differently-built hand nowhere near the
            # object. A headless repro of what is on screen is
            # `hand.default_pose()`.
            x0, y0, z0 = self.params.wrist_pose[:3, 3]
            r0, p0, yw0 = R_to_euler(self.params.wrist_pose[:3, :3])
            # The range FOLLOWS the poses this hand actually needs to reach:
            # its opening hover AND its measured robot mount. A fixed +-100 mm
            # broke both ways -- viser refuses an out-of-range initial value, so
            # a hand whose default hover fell outside would not open the
            # workbench at all; and "Pose at measured robot mount" drives these
            # same sliders, so the tendon hand's mount at z = 134.7 mm was
            # unreachable on a +-100 mm slider and the button raised.
            # Rounded out to the next 50 mm, and shared by all three axes so
            # they stay comparable.
            mount = getattr(self.hand, "mount_pose", None)
            reach = max(abs(x0), abs(y0), abs(z0))
            if mount is not None:
                reach = max(reach, float(np.abs(mount()[:3, 3]).max()))
            span = max(0.1, np.ceil(reach / 0.05) * 0.05)
            self.g_tx = gui.add_slider("x (m)", -span, span, 0.001, x0)
            self.g_ty = gui.add_slider("y (m)", -span, span, 0.001, y0)
            self.g_tz = gui.add_slider("z (m)", -span, span, 0.001, z0)
            self.g_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, r0)
            self.g_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, p0)
            self.g_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, yw0)
            self.g_sig_pos = gui.add_slider("log10 sigma_pos", -6, 2, 0.5, -2)
            self.g_sig_rot = gui.add_slider("log10 sigma_rot", -6, 2, 0.5, -2)
            # The sliders above are a demo pose. This is the measured one: put
            # the wrist here and the viser world origin becomes the robot
            # flange, so the hand hangs exactly as it does on the arm.
            #
            # Read off the HAND (see _mount_pose), not from one shared constant:
            # the tendon hand's mount is a fit against its Onshape assembly, the
            # Allegro hand's is a tf2_echo off the running robot, and they are
            # different transforms. A hand with no measured mount gets the
            # button greyed out rather than a neighbour's numbers.
            has_mount = getattr(self.hand, "mount_pose", None) is not None
            self.g_mount = gui.add_button(
                "Pose at measured robot mount",
                disabled=not has_mount,
                hint="Set the six sliders to this hand's measured "
                     "T_flange<-wrist. The world origin then IS the robot "
                     "flange, and 'mount frames' below draws both frames so you "
                     "can check the hand sits on the arm the way it really does."
                     if has_mount else
                     f"{self.hand.name} has no measured mount transform, so "
                     f"there is nowhere to pose it. A hand supplies one as "
                     f"mount_pose().")

        # What the hand is COMMANDED with. Two shapes, because the two hands
        # are commanded differently: one scalar pull per digit on a tendon hand,
        # a full joint vector per digit on a joint-space one. Both write the
        # per-digit means that become q_S / the tension prior.
        self.g_passive = None
        self.g_flexors = []
        self.g_joints = []
        self.g_tendon_lengths = None
        self.g_flexor_sigma = self.g_passive_sigma = None
        # The joint-space counterparts. Both shapes are declared here whichever
        # branch runs, so every `self.g_*` a mixin references resolves on either
        # hand -- which tests/projects/test_mixin_surface.py checks.
        self.g_joint_sigma = None
        self.g_actuation_report = None
        if self.has("single_drive"):
            with gui.add_folder("Tensions (N)"):
                # Opens on the CALIBRATED OPEN HAND -- HandConfig's
                # zero_bend_passive_tension / zero_bend_flexor_tensions, read through
                # robot_plan.open_pose_tensions so the numbers live in exactly one
                # place and the GUI cannot drift from the calibration. That pose is
                # the zero robot_plan.open_tendon_lengths measures every commanded
                # displacement from, so the app starts at zero displacement and the
                # length readout below opens on +0.00 mm rather than on an offset
                # nobody asked for. (Same trick as the wrist sliders, one level out:
                # the headless repro of what is on screen is open_pose_tensions(),
                # not a HandSolveParams default -- ITS flexor default is still
                # GRASP_FLEXOR_TENSION, which is scene geometry, the tension the big
                # grasp sphere was sized at, and not a statement about this hand's
                # open pose.) The step is 0.01 N because the calibrated pull is
                # per-finger and does not land on a 0.05 grid.
                open_passive, open_flexors = robot_plan.open_pose_tensions()
                self.g_passive = gui.add_slider("passive", 0.0, 3.0, 0.05,
                                                open_passive)
                self.g_flexors = [
                    gui.add_slider(lbl, 0.0, 3.0, 0.01,
                                   open_flexors.get(lbl, GRASP_FLEXOR_TENSION))
                    for lbl in self.digit_names]
                # What the solve gives BACK for the tensions above: the sliders
                # command a pull, this says how much actuated tendon that pull
                # actually took in. Rewritten on every render, so it follows the
                # live re-solve and the convergence scrubber both -- see
                # _report_tendon_lengths.
                self.g_tendon_lengths = gui.add_markdown(self.TENDON_IDLE)
                self.g_flexor_sigma = gui.add_slider(
                    "log10 flexor tension sigma", -6.0, 6.0, 0.1,
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
        else:
            self._build_joint_sliders(gui)

        # One-click constraint-set presets, backed by solvers.PHASE_PRESETS so
        # the same data is usable headlessly. Checking a box writes its whole
        # preset onto the Constraints controls below in one go -- except the
        # per-finger mask, which is the user's and carries across phases (see
        # _apply_phase_preset); press Auto solve afterward to run it. Mutually
        # exclusive -- checking one unchecks the other, see _on_phase_toggle.
        #
        # DEFAULT_PHASE's box opens TICKED. The tick alone would be a claim the
        # panel does not back (the callback only fires on a change), so
        # _apply_default_phase writes the preset for real once the GUI exists --
        # and again after Reset, which restores this same tick.
        with gui.add_folder("Presets"):
            self.g_phase0 = gui.add_checkbox(
                PHASE_PRESETS["phase0"].label, DEFAULT_PHASE == "phase0",
                hint="Apply the phase-0 preset: no object/table contact yet, "
                     "collision avoidance on, pinch-centroid centering + "
                     "short-axis alignment on (the opposition half-space and "
                     "fingertip-midpoint centering stay OFF -- the pinch "
                     "centroid already positions the hand and the other two "
                     "fight it), and a loose wrist prior (this is a big "
                     "repositioning move). Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve. Your finger selection is left alone, as it "
                     "is by every preset. Unchecking is a no-op.")
            self.g_phase1 = gui.add_checkbox(
                PHASE_PRESETS["phase1"].label, DEFAULT_PHASE == "phase1",
                hint="Apply the phase-1 preset: table contact ON (object "
                     "contact stays off), table COLLISION avoidance OFF -- a "
                     "deliberate departure from the paper, since this phase "
                     "drives the fingers onto the plane the avoidance "
                     "half-space would push them off (the table itself stays "
                     "on; object/self collision are untouched) -- the three "
                     "pre-grasp constraints (opposition half-space, "
                     "centering, short-axis alignment) turned back OFF now "
                     "that they've done their job, and a tighter wrist prior "
                     "than phase 0 (held closer to where it ended up, not "
                     "free to roam). Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve; whichever fingers phase 0 was solved with "
                     "carry over untouched. Unchecking is a no-op.")
            self.g_phase2 = gui.add_checkbox(
                PHASE_PRESETS["phase2"].label, DEFAULT_PHASE == "phase2",
                hint="Apply the phase-2 preset: object contact turned back ON "
                     "and table contact turned OFF -- the fingers are handed "
                     "off from the plane they settled on to the object "
                     "itself, in the Eq 13 IN-PLANE form (measured inside "
                     "each finger's pulling plane, so the solve is not asked "
                     "for torsion the tendons cannot produce; falls back to "
                     "the 3D form on a scene that cannot build it). Table "
                     "collision avoidance still OFF as in phase 1 (the "
                     "fingers arrive still lying on the plane, so the "
                     "half-space would be violated from the first step; "
                     "object and self collision stay on), pre-grasp "
                     "constraints still off, and the wrist prior kept TIGHT "
                     "at phase 1's level -- with nothing else holding the "
                     "hand, a loose wrist rides the whole hand onto the "
                     "object instead of closing the fingers around it. Tendon "
                     "sigmas set to the standard "
                     "loose-flexor/tight-passive pair. Writes straight onto "
                     "the Constraints/Wrist/Tensions controls -- check this, "
                     "then press Auto solve; the finger selection carries "
                     "over from phase 1. Unchecking is a no-op.")
            self.g_phase4 = gui.add_checkbox(
                PHASE_PRESETS["phase4"].label, DEFAULT_PHASE == "phase4",
                hint="Apply the phase-4 preset: every constraint OFF -- object "
                     "and table contact, collision avoidance, the opposition "
                     "half-space and all three pre-grasp terms -- because this "
                     "phase does not SOLVE for anything. It shuts the grasping "
                     "fingers on a commanded schedule and whatever they meet on "
                     "the way, they meet. The runner is **Close**, up in the "
                     "Solver folder, NOT Auto solve: check this, then press "
                     "Close. The fingers it shuts are the ones checked below "
                     "-- the same set phases 0-2 positioned, since no preset "
                     "touches that mask -- and the wrist prior is left tight "
                     "(the close does not move the wrist at all). Unchecking "
                     "is a no-op.")
            self.g_phase5 = gui.add_checkbox(
                PHASE_PRESETS["phase5"].label, DEFAULT_PHASE == "phase5",
                hint="Apply the phase-5 preset: every constraint OFF, for "
                     "phase 4's reason -- this phase does not solve for "
                     "anything either. It raises the wrist on a commanded ramp "
                     "and the hand goes up holding whatever the close left it "
                     "holding; nothing in the model holds the OBJECT, so the "
                     "object stays where it is. The runner is **Lift**, up in "
                     "the Solver folder, NOT Auto solve: check this, then press "
                     "Lift. The finger checkboxes are left alone, as by every "
                     "preset -- a lift follows a close, and the grasping set is "
                     "whatever that close shut. Unchecking is a no-op.")

        # Every constraint on/off toggle lives here (Chapter 2, Eq 2.8-2.19),
        # grouped by the paper's structure. Numeric tuning sliders that go with
        # a toggle (collision radius/sigma/cull margin, table height offset)
        # stay behind in Collision/Table below -- only the booleans move.
        with gui.add_folder("Constraints"):
            with gui.add_folder("Rod (planar bending)", expand_by_default=False):
                pb_hint = (
                    "Keep each finger in its own flexion plane: one factor per "
                    "rod segment penalising the out-of-plane and torsional "
                    "components of Log(R_i^T R_j), leaving flexion free. The "
                    "discs are keyed to the backbone, so the real hand cannot "
                    "splay or twist -- the Cosserat rod can, and spends those "
                    "DOFs on contact, collision and the passive tendons routed "
                    "at +/-90 deg. On by default: it is hardware, not a tuning "
                    "choice."
                    if self.caps["planar_bending"]
                    else "requires a rebuilt _gepetto_solvers with "
                         "TendonFingerSolverConfig.planar_bending")
                self.g_planar_bend = gui.add_checkbox(
                    "planar bending", True,
                    disabled=not self.caps["planar_bending"], hint=pb_hint)
                # Curvatures (rad/m), so these read against sigma_twist_rot
                # (1e-2). Defaults are asymmetric on purpose: twist is the cause,
                # bend is the symptom -- see HandSolveParams.
                self.g_planar_bend_sigma = gui.add_slider(
                    "log10 sigma bend", -6, 0, 0.5, -2,
                    disabled=not self.caps["planar_bending"],
                    hint="Out-of-plane bending stiffness, as a curvature sigma "
                         "in rad/m. Lower is stiffer. Left SOFT by default: "
                         "this row fights reach without improving planarity, "
                         "because it constrains the accumulated splay rather "
                         "than what causes it. Kept only so a direct lateral "
                         "load still meets resistance.")
                self.g_planar_twist_sigma = gui.add_slider(
                    "log10 sigma twist", -6, 0, 0.5, -4,
                    disabled=not self.caps["planar_bending"],
                    hint="Torsion stiffness, same units. This is the load-"
                         "bearing one: the spiral-routed lateral tendons inject "
                         "twist, twist rotates the material frame, and the next "
                         "segment's flexion then lands out of plane. Tightening "
                         "THIS collapses the splay at no cost in reach.")

            with gui.add_folder("Collision (Eq 2.8-2.9)", expand_by_default=False):
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
                         else "requires a rebuilt _gepetto_solvers with "
                              "EnvironmentConfig.self_collision")
                self.g_plane_avoid = gui.add_checkbox(
                    "table collision", True,
                    hint="Keep every non-contact sphere out of the "
                         "half-space. Independent of the other two collision "
                         "families. Needs the table enabled below -- with no "
                         "plane there is no half-space to stay out of.")

            with gui.add_folder("Contact (Eq 2.11-2.15)", expand_by_default=False):
                self.g_table = gui.add_checkbox(
                    "table enabled", True, disabled=not self.caps["table"],
                    hint="Put the support plane in the factor graph. Affects the "
                         "SOLVER only -- the table square is always drawn, since "
                         "it doubles as the scene's landmark (see 'table frame' "
                         "under Display)."
                    if self.caps["table"]
                    else "requires a newer _gepetto_solvers build (plane env fields)")
                self.g_obj_contact = gui.add_checkbox(
                    "object contact (3D)", True,
                    hint="Drive the checked fingertips onto the OBJECT "
                         "surface, measuring the full 3D distance to it. Turn "
                         "off to leave the object as pure collision geometry -- "
                         "the way to see what the table constraints do on their "
                         "own. Mutually exclusive with the in-plane form below: "
                         "they are two metrics for the SAME constraint, so "
                         "checking one clears the other.")
                self.g_obj_contact_plane = gui.add_checkbox(
                    "object contact (in-plane)", False,
                    disabled=not self.caps["planar_contact"],
                    hint="Eq 13: the same fingertip-onto-object equality, but "
                         "with the distance measured inside each finger's "
                         "pulling plane (Eq 11: metacarpal base, fingertip, "
                         "pinch centroid) -- the plane a tendon can actually "
                         "pull along, so the solve is not asked for "
                         "out-of-plane torsion the hand cannot produce. Same "
                         "factor count and the same zero set (distance = tip "
                         "radius); only the metric differs. Needs an ellipsoid "
                         "or ycb: object and a digit set INCLUDING THE THUMB, "
                         "and greys out when the scene cannot support it. "
                         "Watch it with 'in-plane distance' under Display.")
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

            with gui.add_folder("Pre-grasp (Eq 2.16-2.19)", expand_by_default=False):
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
                         else "requires a rebuilt _gepetto_solvers with "
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
                         else "requires a rebuilt _gepetto_solvers with "
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
                # avoidance but are not driven onto a surface. Whatever is
                # ticked here survives every phase preset (see
                # _apply_phase_preset) -- only Reset puts this back.
                _pinch_default = set(self.hand.default_contact_digits)
                self.g_contacts = [
                    gui.add_checkbox(
                        lbl, lbl in _pinch_default,
                        hint="Which fingers every constraint above applies "
                             "to (IK only; FK never uses contact), and the set "
                             "a phase-4 Close shuts. Carries across the phase "
                             "presets -- pick the digits once and they hold "
                             "from pre-grasp through the lift. "
                             "Unchecked fingers keep collision avoidance, so "
                             "they stay out of the object and off the table "
                             "without being driven onto either, opposed "
                             "against, or centered on.")
                    for lbl in self.digit_names]

        with gui.add_folder("Collision", expand_by_default=False):
            self.g_coll_radius = gui.add_slider("sphere radius (m)", 0.001, 0.01, 0.0005, 0.003)
            self.g_coll_sigma = gui.add_slider("log10 sigma", -6, 0, 0.5, -4)
            self.g_cull = gui.add_slider("cull margin (m, 0 off)", 0.0, 0.1, 0.005, 0.0)
            self.g_set_beta = gui.add_slider(
                "ellipsoid-set beta", 100.0, 4000.0, 100.0, ELLIPSOID_SET_BETA,
                disabled=not self.caps.get("ellipsoid_set", False),
                hint="LogSumExp sharpness for a ycb: object (Eq 1.11). The smooth "
                     "min understates by up to ln(K)/beta, so the constraint "
                     "surface sits that far OUTSIDE the object -- 1.4 mm at K=4, "
                     "beta=1000. Raising it tightens that standoff but sharpens "
                     "the gradient at the seams between members, which is the "
                     "smoothness the set formulation exists to buy. Inert for "
                     "every non-ycb object.")

        self._build_ycb_folder(gui)

        with gui.add_folder("Table", expand_by_default=False):
            # Height of the SOLVER's plane above the table surface. The table
            # itself does not move: it is seated from the scene (table_burial = 0,
            # see _fresh_params) and carries the bench registration, the corner
            # frame, the grid and the calibration target with it. Zero default, so
            # the two planes open coincident -- the geometry every headless script
            # solves.
            self.g_constraint_height = gui.add_slider(
                "constraint plane height (m)", -0.1, 0.1, 0.002, -0.005,
                hint="Raise or lower the plane the SOLVER constrains against -- "
                     "where the support equality seats fingertips and where the "
                     "avoidance half-space starts -- relative to the table's top "
                     "face. Moves nothing else: the drawn square, its corner "
                     "frame, the grid, the calibration target and the robot "
                     "registration all stay on the physical bench. Positive "
                     "lifts the constraint off the table; negative sinks it "
                     "under.")
            self.g_show_constraint_plane = gui.add_checkbox(
                "draw constraint plane", True,
                hint="Show the solver's plane as a thin green sheet. Only "
                     "appears at a nonzero height -- sitting on the table it "
                     "would just z-fight the slab.")
            # Filled by _refresh_table_readout on every re-place of the slab.
            self.g_table_status = gui.add_markdown("")

        self._build_calibration_folder(gui)

        with gui.add_folder("Augmented Lagrangian", expand_by_default=False):
            self.g_al_mu = gui.add_slider("mu", 0.1, 10.0, 0.1, 1.0)
            self.g_al_rate = gui.add_slider("rate", 1.1, 5.0, 0.1, 2.0)
            self.g_al_iters = gui.add_slider("max iters", 5, 100, 5, 40)

        with gui.add_folder("Display"):
            # Only offered for a hand that HAS link meshes; a checkbox that
            # could never draw anything is worse than no checkbox.
            self.g_show_meshes = gui.add_checkbox(
                "hand meshes", True,
                disabled=not self._link_meshes(),
                hint="Draw the hand's own link geometry from its URDF, at the "
                     "poses the solve put each link at. VISUAL ONLY -- collision "
                     "is the sphere set the solve carries, never a mesh, so this "
                     "changes the picture and nothing else. Turn it off to see "
                     "the bare skeleton and the collision spheres, which is what "
                     "the graph actually reasons about.")
            self.g_show_true_mesh = gui.add_checkbox(
                "true object mesh", True,
                hint="Overlay the object's real geometry behind the analytic "
                     "surface: the scanned mesh inside a ycb: object's ellipsoid "
                     "shells, or the dodecahedron inside the megaminx's "
                     "circumsphere. The analytic surface is what the solver sees, "
                     "so showing both is how the approximation gets judged -- "
                     "where the fingers stop is set by the surface, and the mesh "
                     "says how much object is actually there.")
            self.g_show_contact = gui.add_checkbox("contact spheres", True)
            self.g_show_collision = gui.add_checkbox("collision spheres", True)
            self.g_show_discs = gui.add_checkbox("routing discs", False)
            self.g_show_disc_frames = gui.add_checkbox(
                "disc frames", False,
                hint="A triad on every disc node of every finger, including the "
                     "base disc -- the body frame the routing hole locations are "
                     "expressed in, so the tendon path is those axes plus "
                     "`R @ hole + t`. Off by default: five fingers' worth of "
                     "triads is dense, so this is for checking routing "
                     "orientation (a hole angle measured off the wrong axis is "
                     "invisible on the rotationally symmetric disc cylinders).")
            # The three frame toggles all default ON: which frame anything is
            # expressed in is the first question asked of this scene, and a
            # triad you have to go and switch on answers it too late.
            self.g_show_world = gui.add_checkbox(
                "world frame", True,
                hint="Triad at the world origin. Every position readout in this "
                     "app is in this frame, and it is the frame the wrist pose "
                     "sliders command -- so with the hand at the measured mount "
                     "it doubles as the robot flange frame.")
            self.g_show_obj_frame = gui.add_checkbox(
                "object frame", True,
                hint="Triad at the object's center, oriented by its rotation -- "
                     "the pose the contact and collision factors are written "
                     "against, and what the Object-pose sliders drive. The only "
                     "way to see the orientation of a symmetric primitive, and "
                     "for a ycb: set it is the frame its members sit in.")
            self.g_show_table_frame = gui.add_checkbox(
                "table frame", True,
                hint=f"Triad on the table square's -X/-Y corner, on the top "
                     f"face -- a pure translation of the world frame, so a "
                     f"coordinate read off it is a world coordinate minus the "
                     f"corner. The landmark to measure a real bench against. "
                     f"The square is {TABLE_SPAN:g} x {TABLE_SPAN:g} m "
                     f"whatever object is on it, but the plane is seated UNDER "
                     f"the object, so the frame moves when you switch objects "
                     f"-- the Table folder reports where it is.")
            self.g_show_grid = gui.add_checkbox(
                "table grid", True,
                hint=f"Rule the table square into "
                     f"{CAL_GRID_SPACING * 100:.0f} cm cells, matching the grid "
                     f"drawn on the physical bench. On by default because it is "
                     f"what the Calibration folder's x/y sliders are coordinates "
                     f"ON -- with it drawn you can check a commanded landmark "
                     f"against the same intersection in the room.")
            self.g_show_mount = gui.add_checkbox(
                "mount frames", True,
                hint="Draw the wrist frame and, offset from it by this hand's "
                     "measured mount transform, the robot flange frame it bolts "
                     "to. Use with 'Pose at measured robot mount' to check the "
                     "measurement by eye. Draws nothing for a hand that has no "
                     "measured mount.")
            self.g_show_gaps = gui.add_checkbox(
                "contact distance", True,
                hint="Fingertip-to-surface gap/margin overlays in mm: object "
                     "and table contact (green under 15 mm, red over), "
                     "opposition half-space (green = correct side, red = "
                     "violating), and pre-grasp centering (green under 15 mm "
                     "to target).")
            self.g_show_finger_planes = gui.add_checkbox(
                "finger pinch planes", False,
                hint="Per finger, the plane through its metacarpal base, its "
                     "fingertip and the pinch centroid the checked digits "
                     "close on (config.HAND_PINCH_POSES, carried through the "
                     "solved wrist). One colour per finger; the outlined "
                     "triangle is the three defining points. Off by default -- "
                     "five translucent sheets sit right where the grasp is. "
                     "Nothing enforces these planes; they describe the posture "
                     "on screen. Needs a measured pinch pose, so only digit "
                     "sets INCLUDING THE THUMB draw anything.")
            self.g_show_planar_gap = gui.add_checkbox(
                # On by default, but only where the binding can actually
                # measure it: a hard True would tick a DISABLED box on an
                # older .so, and the readout would then permanently report
                # "in-plane distance: UNAVAILABLE" with no way to turn it off.
                "in-plane distance", self.caps["planar_gap"],
                disabled=not self.caps["planar_gap"],
                hint="Eq 11/13: the distance from each fingertip to the object "
                     "measured INSIDE that finger's pulling plane, which is what "
                     "a tendon can actually pull along. Draws the cross-section "
                     "the plane cuts out of the object (exact) plus a line to the "
                     "nearest point on it; the mm label is the C++ factor's own "
                     "first-order value, so a visible mismatch between line and "
                     "label IS the approximation error. '(3D)' means the plane "
                     "missed the object entirely and the number has fallen back "
                     "to the ordinary 3D distance. Spheres, ellipsoids and ycb: "
                     "sets only -- cube/cylinder/capsule have no ellipsoid "
                     "cross-section. On by default wherever the binding "
                     "supports it. Measurement only: no solve uses this yet.")
            self.g_show_traj = gui.add_checkbox(
                "trajectory plots", True,
                hint="The window docked to the LEFT of the 3D view: the six "
                     "controls this robot is commanded with -- the five "
                     "actuated tendon lengths in mm (what the hand took in, "
                     "the same numbers the length table above prints, NOT the "
                     "tension that was commanded) and the wrist pose, split "
                     "into x/y/z/roll/pitch/yaw -- one subplot each, against the "
                     "iteration the solve is on. Sample 0 is where the run "
                     "started (the FK pose on screen, so with Warm start on it "
                     "is the current kinematics) and each later sample is one "
                     "AL outer iteration, joined by straight segments, so the "
                     "window fills in live as Auto solve runs and holds the "
                     "whole path afterwards. A Close or a Lift plots its ramp "
                     "substeps the same way. The white dot marks the sample the "
                     "3D view is showing, so it follows the Solve steps "
                     "scrubber. Angles are the wrist sliders' own radians and "
                     "positions their metres, so a pose value read off a plot "
                     "can be typed back into the slider that commands it -- but "
                     "note "
                     "the straight line between two rpy samples is NOT the path "
                     "the arm flies between them, which robot_plan interpolates "
                     "as a screw motion (se3_log/se3_exp).")

        # Every value-carrying control, captured as built: this IS the definition
        # of "defaults" that Reset restores, so the two cannot drift.
        self._gui_defaults = [(h, h.value) for h in self._input_handles()]
        self._refresh_warm_start()

        # -- callbacks --
        self.g_fk.on_click(self._fk_solve)
        self.g_ik_step.on_click(self._ik_step)
        self.g_ik_auto.on_click(self._ik_auto)
        self.g_ik_stop.on_click(self._estop)
        self.g_rearm.on_click(self._rearm)
        self.g_warm.on_click(self._toggle_warm_start)
        self.g_reset.on_click(self._reset_defaults)
        self.g_phase0.on_update(lambda _: self._on_phase_toggle("phase0"))
        self.g_phase1.on_update(lambda _: self._on_phase_toggle("phase1"))
        self.g_phase2.on_update(lambda _: self._on_phase_toggle("phase2"))
        self.g_phase4.on_update(lambda _: self._on_phase_toggle("phase4"))
        self.g_phase5.on_update(lambda _: self._on_phase_toggle("phase5"))
        if self.g_close is not None:
            self.g_close.on_click(self._close_hand)
        if self.g_lift is not None:
            self.g_lift.on_click(self._lift_hand)

        self.g_object.on_update(self._on_object_selected)

        # Live FK re-solve on the pose and actuation sliders (fast,
        # warm-started). Whichever actuation panel this hand got -- the tension
        # sliders or the per-joint ones -- drives the same re-solve.
        for h in ([self.g_tx, self.g_ty, self.g_tz, self.g_roll, self.g_pitch,
                   self.g_yaw, self.g_passive]
                  + self.g_flexors
                  + [j for row in self.g_joints for j in row]):
            if h is not None:
                h.on_update(self._live_fk)

        # Object pose: re-places the object and restarts the IK loop (see
        # _object_pose_changed for why it cannot just re-render).
        for h in (self.g_obj_dx, self.g_obj_dy, self.g_obj_dz,
                  self.g_obj_roll, self.g_obj_pitch, self.g_obj_yaw):
            h.on_update(self._object_pose_changed)

        self.g_mount.on_click(self._pose_at_mount)

        # The scan-mesh overlay and the world/object triads are static scene
        # geometry, not per-frame, so they go through _refresh_object rather
        # than the _render_frame path below. (The mount frames are the exception
        # among the frame toggles: they hang off the SOLVED wrist, which moves
        # every iterate, so they render with the hand.)
        for h in (self.g_show_true_mesh, self.g_show_world, self.g_show_obj_frame,
                  self.g_show_table_frame):
            @h.on_update
            def _(_):
                if self._restoring:
                    return
                self._refresh_object()

        # Display toggles re-render the current frame without re-solving.
        for h in (self.g_show_contact, self.g_show_collision, self.g_show_discs,
                  self.g_show_disc_frames, self.g_show_meshes,
                  self.g_show_gaps, self.g_show_mount, self.g_show_finger_planes,
                  self.g_show_planar_gap):
            h.on_update(lambda _: (self._sync_params(), self._render_frame()))
        # The contact checkboxes ride along only to keep self.params in sync;
        # like every other solver knob they take effect on the next solve, and
        # the gap lines keep describing the solve that is actually on screen
        # until then. They do change the stepper's constraint set, though -- which
        # surface a finger is driven onto IS that set, so a carried dual is
        # meaningless and the loop has to restart.
        # The two object-contact FORMS additionally settle each other first (they
        # are mutually exclusive), and the per-finger boxes re-check whether the
        # in-plane form is still buildable -- its plane is keyed off the pinch
        # centroid of exactly these digits, so unchecking the thumb can take it
        # away.
        # Purely a visibility switch -- the panel keeps its plotted data while
        # hidden, so unlike the toggles above this needs no re-render to come
        # back with the trajectory still on it.
        self.g_show_traj.on_update(
            lambda _: self.traj.set_visible(self.g_show_traj.value))

        for h in (self.g_obj_contact, self.g_obj_contact_plane):
            h.on_update(lambda _, src=h: self._enforce_object_contact(src))
        for h in self.g_contacts:
            h.on_update(lambda _: self._refresh_planar_contact_gate())
        for h in (self.g_contacts
                  + [self.g_obj_contact, self.g_obj_contact_plane,
                     self.g_tbl_contact]):
            h.on_update(lambda _: (self._sync_params(),
                                   self._invalidate_stepper(),
                                   self._render_frame()))
        # Which SHELLS may be touched is the same kind of change as which FINGERS
        # touch -- the contact equality is written against a different surface, so
        # a carried dual is meaningless -- but it also restyles the object, since
        # the excluded shells are drawn greyed. Hence _refresh_object as well.
        self.g_contact_shells.on_update(
            lambda _: (self._sync_params(), self._refresh_object(),
                       self._invalidate_stepper(), self._render_frame()))
        # Table toggle / constraint-plane height updates the static slabs
        # immediately; opposition half-space rides along since it draws its own
        # static split-plane slab (set_half_space_plane) the same way -- as does
        # its standoff slider, which draws the two boundary planes either side of
        # that split. The height slider invalidates the stepper too: it moves a
        # plane the factor graph is written against, even though the drawn table
        # stays put.
        for h in (self.g_table, self.g_constraint_height,
                  self.g_show_constraint_plane, self.g_half_space,
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
        # Planar bending changes the GRAPH, not the scene, so it invalidates the
        # stepper but draws nothing of its own.
        for h in (self.g_planar_bend, self.g_planar_bend_sigma,
                  self.g_planar_twist_sigma):
            h.on_update(lambda _: (self._sync_params(),
                                   self._invalidate_stepper()))
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
