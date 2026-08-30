"""Placing a physical landmark and aligning the wrist to it.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import traceback

import numpy as np

from gepetto_solvers.core.geometry.scene import TABLE_SPAN
from gepetto_solvers.core.hand.config import proximal_disc_flags
from gepetto_solvers.core.solvers import (
    R_to_euler,
    disc_frame_error,
    disc_pose,
    wrist_pose_for_disc_target,
)

from .constants import (
    CAL_DEFAULT_DISC,
    CAL_DISCS,
    CAL_GRID_SPACING,
    CAL_REFINE_PASSES,
    FINGER_LABELS,
    _euler_to_R,
)
from .estop import Refused


class CalibrationMixin:
    # -- table-grid calibration --
    #
    # See the CAL_* constants at module scope for what this is FOR. The short
    # version: the hand's geometry is from CAD and the table's placement is from
    # a ruler, so the way to test the ruler is to send a CAD-known landmark to a
    # grid intersection and look at where it physically ends up.

    def _cal_finger(self):
        return self.g_cal_finger.value


    def _cal_disc(self):
        """The selected disc's INDEX, back out of its dropdown label."""
        return self._cal_disc_labels[self.g_cal_disc.value]


    def _cal_config(self):
        """The selected finger's ``TendonFingerConfig``.

        ``fk_solver.configs`` is a list of ``(name, cfg)`` pairs -- the shape
        ``solved_wrist_pose`` unpacks -- so a named lookup has to walk it.
        """
        for name, cfg in self.fk_solver.configs:
            if name == self._cal_finger():
                return cfg
        raise KeyError(f"no finger named {self._cal_finger()!r}")


    def _cal_check_rigid(self):
        """The reason this alignment is closed-form, checked rather than assumed.

        Returns a complaint string if the selected disc is NOT rigidly attached to
        the palm, else None. ``CAL_DISCS`` only offers discs that are, so this
        cannot normally fire -- it exists because the thing it guards is a silent
        wrong answer rather than an exception: a disc past the MCP joint still has
        a ``T_wrist<-disc``, it just describes the one posture it was measured in,
        and the hand would land somewhere plausible-looking and wrong.
        """
        disc = self._cal_disc()
        flags = proximal_disc_flags(self._cal_config())
        if disc < len(flags) and flags[disc]:
            return None
        return (f"disc {disc} is not rigidly attached to the palm, so there is no "
                f"constant wrist-to-disc transform to invert. Pick one of: "
                f"{', '.join(str(d) for d in sorted(CAL_DISCS))}.")


    def _calibration_target(self):
        """The target frame in VISER WORLD coordinates.

        The sliders are in the TABLE frame, which is the useful one: the
        registration against the robot is a pure translation (see
        ``gepetto_control.frames``), so the viser world and
        ``lbr_workspace_table_link`` share axis directions and a table-frame
        coordinate is just a world coordinate minus the corner. That makes the x/y
        sliders read directly as positions on the grid drawn on the bench, with
        [0, TABLE_SPAN] spanning the square.
        """
        T = np.eye(4)
        T[:3, :3] = _euler_to_R(self.g_cal_roll.value, self.g_cal_pitch.value,
                                self.g_cal_yaw.value)
        T[:3, 3] = self._corner_viz() + np.array(
            [self.g_cal_x.value, self.g_cal_y.value, self.g_cal_z.value], float)
        return T


    def _refresh_calibration_frame(self, _=None):
        """Draw (or clear) the target triad. NO SOLVE -- this is the feedback while
        you pick where the frame goes, so it has to be free enough to fire on
        every slider drag.

        Also called from :meth:`_refresh_object`, because the support plane is
        seated under the object: changing the object moves the table corner the
        sliders are measured from, and a target left where it was would silently
        stop meaning what the sliders say.
        """
        if not self.g_cal_show.value:
            self.scene.clear_calibration_frame()
            return
        self.scene.set_calibration_frame(
            self._calibration_target(),
            label=f"calibration  ({self.g_cal_x.value:.3f}, "
                  f"{self.g_cal_y.value:.3f}, {self.g_cal_z.value:.3f}) m "
                  f"in table frame")


    def _cal_landmark_pose(self):
        """World pose of the selected landmark on the hand right now, or None
        before anything has been solved.

        Read off the scrubbed iterate rather than ``self.result`` so it describes
        the hand ON SCREEN -- the same rule ``_adopt_solved_wrist`` follows.
        """
        res = self._iter_view()
        if res is None:
            return None
        return disc_pose(res.frames[0], self._cal_finger(), self._cal_disc())


    def _write_wrist_sliders(self, T):
        """Put a wrist pose on the six Wrist-start-pose sliders.

        The idiom is ``_adopt_robot_state``'s, for its reasons: ``_restoring``
        latched so the per-slider live-FK hook does not fire six times on the way
        through, ``_fit_wrist_range`` to grow the +-0.1 m demo range (essential
        here -- the scene origin is the object, and the table corner is 200 mm
        away from it before the target offset is added), and the raw float
        written, since viser does not snap a programmatic write to the step grid
        and ``_sync_wrist`` rebuilds ``params.wrist_pose`` straight off these
        handles. Returns the labels of any sliders whose range had to grow.
        """
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        widened = []
        self._restoring = True
        try:
            for handle, value, label in zip(
                    (self.g_tx, self.g_ty, self.g_tz,
                     self.g_roll, self.g_pitch, self.g_yaw),
                    (*T[:3, 3], roll, pitch, yaw),
                    ("x", "y", "z", "roll", "pitch", "yaw")):
                value = float(value)
                if self._fit_wrist_range(handle, value) is not None:
                    widened.append(label)
                handle.value = value
        finally:
            self._restoring = False
        return widened


    def _capture_calibration(self, _=None):
        """Fill the six calibration sliders from where the landmark is NOW.

        This is what makes the orientation sliders usable. Absolute roll/pitch/yaw
        against the table frame is the honest parameterisation, but there is no
        way to guess from cold which triple points the hand at the table rather
        than through it -- so grab the orientation the hand is already in, then
        drive across the grid on x/y alone and the move stays a pure translation.
        """
        T = self._cal_landmark_pose()
        if T is None:
            self.g_cal_status.content = (
                "**nothing solved yet** -- press *FK* first, so there is a hand "
                "pose to capture the landmark from.")
            return
        local = T[:3, 3] - self._corner_viz()
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        self._restoring = True
        try:
            for handle, value in zip(
                    (self.g_cal_x, self.g_cal_y, self.g_cal_z,
                     self.g_cal_roll, self.g_cal_pitch, self.g_cal_yaw),
                    (*local, roll, pitch, yaw)):
                value = float(value)
                self._fit_wrist_range(handle, value)
                handle.value = value
        finally:
            self._restoring = False
        self._refresh_calibration_frame()
        off_grid = not (0.0 <= local[0] <= TABLE_SPAN and 0.0 <= local[1] <= TABLE_SPAN)
        self.g_cal_status.content = (
            f"captured **{self._cal_finger()}** disc {self._cal_disc()} at "
            f"({local[0]:+.4f}, {local[1]:+.4f}, {local[2]:+.4f}) m in the table "
            f"frame" + ("  \n_that is outside the 0.4 x 0.4 m square -- the "
                        "sliders' range was widened to hold it._" if off_grid else ""))


    def _align_to_calibration(self, _=None):
        """Place the hand so the selected landmark lands on the calibration frame.

        A CLOSED-FORM PLACEMENT, not a solve, even though it sits under a button
        that says solve. The landmark is on the metacarpal, which is bolted to the
        palm, so ``T_wrist<-disc`` is a constant of the morphology and the wrist
        pose that puts the disc at the target is one matrix inverse
        (:func:`solvers.wrist_pose_for_disc_target`). Doing it as an IK solve with
        a loosened wrist prior would be slower, approximate, and -- worse -- would
        let whatever contact and pre-grasp constraints happen to be ticked drag
        the landmark off the target it was asked to hit, which is exactly the
        error this feature exists to measure.

        Still goes through the e-stop gate and greys the panel, because it re-poses
        the hand (and in ROS mode that pose is one press away from the arm).
        """
        try:
            gate = self.estop.admit("calibration align")
        except Refused as exc:
            self.g_cal_status.content = f"**refused:** {exc}"
            return
        try:
            with gate:
                self._set_solving(True)
                self._align_to_calibration_admitted()
        except Exception as exc:
            traceback.print_exc()
            self.g_cal_status.content = f"**align failed:** `{exc}`"
        finally:
            self._set_solving()
            self._report_estop()


    def _align_to_calibration_admitted(self):
        complaint = self._cal_check_rigid()
        if complaint:
            self.g_cal_status.content = f"**cannot align:** {complaint}"
            return

        # The placement measures T_wrist<-disc off a SOLVED frame, so there has to
        # be one. A cold app has already solved in __init__; this covers the case
        # where a failed solve left self.result None.
        if self._iter_view() is None:
            self._fk_solve_admitted()

        target = self._calibration_target()
        widened = []
        for _ in range(CAL_REFINE_PASSES):
            res = self._iter_view()
            T_wrist = wrist_pose_for_disc_target(
                self.fk_solver.configs, res.frames[0],
                self._cal_finger(), self._cal_disc(), target)
            widened += [lbl for lbl in self._write_wrist_sliders(T_wrist)
                        if lbl not in widened]
            # Re-solves at the CURRENT tensions, which is what makes the second
            # pass worth its solve: the first pass's transform was measured at the
            # posture the hand was in before it moved.
            self._fk_solve_admitted()

        landed = self._cal_landmark_pose()
        pos_mm, rot_deg = disc_frame_error(landed, target)
        lines = [
            f"**{self._cal_finger()}** disc {self._cal_disc()} "
            f"({CAL_DISCS[self._cal_disc()]}) aligned to "
            f"({self.g_cal_x.value:.3f}, {self.g_cal_y.value:.3f}, "
            f"{self.g_cal_z.value:.3f}) m in the table frame",
            f"residual **{pos_mm:.4f} mm** / **{rot_deg:.4f} deg** "
            f"after {CAL_REFINE_PASSES} passes",
        ]
        # A tenth of a millimetre is an order of magnitude worse than the ~25 um
        # of tension-dependence this is correcting for, so it means the premise
        # broke rather than that the refinement needs another pass.
        if pos_mm > 0.1 or rot_deg > 0.05:
            lines.append(
                "**residual is larger than expected** -- the landmark should be "
                "rigid to the wrist. Check that the FK solve converged.")
        if widened:
            lines.append(
                f"_wrist slider range widened ({', '.join(widened)}) to hold the "
                f"commanded pose._")
        if self.ros_mode:
            lines.append(
                "_To send it: Robot folder, waypoints = *final state only*, tick "
                "*arm*, then *Play solve on robot*._")
        self.g_cal_status.content = "  \n".join(lines)


    # -- GUI construction --

    def _build_calibration_folder(self, gui):
        """The Calibration folder: put a known hand landmark on a known point of
        the table's grid.

        Sits directly under *Table* because everything in it is expressed against
        that landmark -- the x/y sliders ARE grid coordinates on the square drawn
        there, and moving the table (which the object seating does) moves them.
        """
        self._cal_disc_labels = {f"{d} — {label}": d
                                 for d, label in CAL_DISCS.items()}
        default_disc_label = next(k for k, v in self._cal_disc_labels.items()
                                  if v == CAL_DEFAULT_DISC)
        with gui.add_folder("Calibration", expand_by_default=False):
            gui.add_markdown(
                f"Align a hand landmark with a point on the table's "
                f"**{TABLE_SPAN:g} x {TABLE_SPAN:g} m** grid, to test the "
                f"ruler-measured table transform in the URDF against the "
                f"CAD-measured hand. x/y are grid coordinates from the corner "
                f"frame; the lines are every {CAL_GRID_SPACING * 100:.0f} cm.")
            self.g_cal_finger = gui.add_dropdown(
                "finger", FINGER_LABELS, initial_value=FINGER_LABELS[0])
            self.g_cal_disc = gui.add_dropdown(
                "landmark disc", list(self._cal_disc_labels),
                initial_value=default_disc_label,
                hint="Which routing disc's frame is put on the target. Only the "
                     "two metacarpal discs are offered: they are bolted to the "
                     "palm, which is what makes this an exact placement instead "
                     "of an IK solve. Disc 1 is the far end of the metacarpal, "
                     "where the MCP joint starts -- the one you can find on the "
                     "hardware. Turn on *disc frames* in Display to see it.")
            # The square runs 0..TABLE_SPAN from the corner frame, so these ARE
            # the grid coordinates. Step 5 mm: fine enough to sit between lines
            # deliberately, coarse enough that the 10 cm intersections land
            # exactly on a step.
            self.g_cal_x = gui.add_slider("x (m)", 0.0, TABLE_SPAN, 0.005,
                                          TABLE_SPAN / 2)
            self.g_cal_y = gui.add_slider("y (m)", 0.0, TABLE_SPAN, 0.005,
                                          TABLE_SPAN / 2)
            self.g_cal_z = gui.add_slider("z (m)", -0.05, 0.30, 0.005, 0.10)
            self.g_cal_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_cal_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_cal_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_cal_show = gui.add_checkbox("show calibration frame", True)
            self.g_cal_capture = gui.add_button(
                "Capture current", icon=self.viser.Icon.CROSSHAIR,
                hint="Fill the six sliders from where the selected landmark is "
                     "right now. Grab the hand's current orientation this way, "
                     "then drive across the grid on x/y alone and every move is "
                     "a pure translation.")
            self.g_cal_align = gui.add_button(
                "Align hand to frame", icon=self.viser.Icon.TARGET,
                hint="Move the wrist so the selected landmark lands exactly on "
                     "the calibration frame, and re-pose with FK. Closed-form, "
                     "not a solve -- the landmark is rigid to the wrist -- so it "
                     "ignores the constraint set entirely and reports the "
                     "residual it actually achieved.")
            self.g_cal_status = gui.add_markdown("")

        self.g_cal_capture.on_click(self._capture_calibration)
        self.g_cal_align.on_click(self._align_to_calibration)
        for handle in (self.g_cal_finger, self.g_cal_disc, self.g_cal_show,
                       self.g_cal_x, self.g_cal_y, self.g_cal_z,
                       self.g_cal_roll, self.g_cal_pitch, self.g_cal_yaw):
            handle.on_update(self._refresh_calibration_frame)
