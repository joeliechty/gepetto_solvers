"""Interactive viser visualizer for the tendon-hand FK solver and the stepped IK
solve.

Exposes the solver knobs as live web GUI controls -- object picker, wrist start
pose, per-finger flexor tensions, per-finger contact toggles, collision / table
options, AL settings. *FK* re-poses the hand from the current sliders (and the
pose / tension sliders re-solve it live as they move); the *Tensions* folder
prints back the actuated tendon length each solve reached, under the sliders
that commanded it. *Step* advances the IK solve by exactly one Augmented
Lagrangian outer iteration, and *Auto solve* keeps stepping until it converges
or stalls. Every step is kept, so the *Solve steps*
scrubber replays the convergence one iteration at a time (initial guess -> each
outer iteration).

The window docked to the LEFT of the 3D view plots that convergence as a
CONTROL TRAJECTORY: one subplot each for the six things this robot is commanded
with -- the five actuated tendon LENGTHS (what the hand took in, in mm, and what
the hardware is commanded on -- not the tension that was asked for), and the
wrist pose split into x/y/z/roll/pitch/yaw, since a pose is not plottable as a
scalar. Sample 0 is
where the run started (the FK pose on screen, so with *Warm start* on it is the
current kinematics) and every later sample is one AL outer iteration, joined by
straight segments with the knots dotted, so the window fills in live as *Auto
solve* runs and holds the whole path once it stops. A phase-4 *Close* or
phase-5 *Lift* plots its ramp substeps the same way. The white dot marks the
sample the 3D view is showing, so it follows the *Solve steps* scrubber, and the
readout above the plots plus a *trajectory plots* checkbox in *Display* are its
only other controls. Values are re-read from the SOLVE, not from the sliders
that commanded it -- the wrist and the flexor tensions are both variables with
soft priors, so watching them walk away from what was asked for is the point.
Units are whatever already prints the same quantity -- mm for the tendons, as in
the *Tensions* folder's length table, and the wrist sliders' own m and rad -- so
a number can be read off a plot and checked against the readout beside it. One
caveat: the straight line drawn between two rpy samples is
not the path the arm flies between them, which ``robot_plan`` interpolates as a
screw motion (``se3_log``/``se3_exp``).

*Close* is phase 4, and it is not a solve at all. Presets 0-2 shut the fingers as
a SIDE EFFECT of the object equality, so the digits arrive one at a time in
whatever order the optimizer moved them; phase 4 instead COMMANDS the grasping
fingers shut together, each along the same fraction of its own remaining tendon
travel, so they start and finish as one. It is a walk of FK poses rather than an
AL solve -- nothing is enforced, nothing converges, and whatever the fingers meet
on the way, they meet -- and it reports the worst gap between the digits that the
poses actually came back with, so the claim in its name is a measured number
rather than an assertion. It starts from the solve on screen (whichever step the
scrubber is parked on) while *Warm start* is on: the solved wrist pose and flexor
tensions are adopted onto the sliders and the posture is handed to the FK solver,
so a close continues the phase-2 approach instead of jumping back to whatever the
panel was last commanded to. What cannot cross that boundary is the contact
itself -- an FK ramp enforces nothing, so the fingers relax to their tension
equilibrium before the first step. Same carry on the phase-5 *Lift*. Every pose is recorded the way an AL iteration is, so
the *Solve steps* scrubber replays a close and the *Robot* folder plays one. See
``solvers.synchronized_close``.

*E-STOP* is the software emergency stop, and it outranks everything else on the
page. It latches: it breaks a running auto-solve out of its loop and then refuses
every solve -- FK, Step, Auto, and the live re-solve the pose/tension sliders
fire -- until *Rearm* is pressed, so nothing can restart the hand by accident.
The button is never greyed out. Nothing is lost when it trips: the stepper keeps
its multipliers, its penalty weight and its whole history, so *Rearm* + *Auto
solve* resumes the same solve rather than restarting it.

The stop is cooperative and lands at an AL iteration boundary -- one outer
iteration is a single call into the C++ solver with no interrupt hook, so ~1.7 s
is the measured worst case and a floor rather than a tuning knob. What makes the
button nevertheless feel instant is that the solve bindings release the GIL
(``capabilities()["gil_release"]``): the click is serviced while the last
iteration is still running. Against a binding without it the whole interpreter is
frozen for the duration of every iteration and the click cannot even be received
until it ends -- the app says so on the button and in the startup banner.

*ROS mode* (``HandVizApp(server, ros_mode=True, bridge=...)``, which
``gepetto_control``'s ``viz_node`` does) adds a *Robot* folder that commands the
real hardware: **Play solve on robot** exports the WHOLE recorded path -- AL outer
iterations after a solve, the ramp itself after a phase-4 Close -- as waypoints,
interpolates them, and servos the arm (MoveIt Servo twists) and the
hand (``finger_servo_node`` tendon jogs) along them; **Get robot state** reads
the wrist pose and the measured tendon lengths back and makes them the state on
screen. The scene registers against the robot through the table: the drawn square
IS ``lbr_workspace_table_link``, so no transform has to be measured. The folder
opens DISARMED -- one checkbox arms one press of *Play* and clears itself when the
run ends -- and the E-STOP above extends to it: it halts the publishers, not just
the solver, and disarms. Nothing in this module imports rclpy; the bridge is
duck-typed and the conversion lives in ``robot_plan.py``, which is pure numpy and
is covered by ``--smoke``.

The *Calibration* folder tests the one number in this stack that was NOT measured
from CAD: where the workspace table sits, which came off a ruler
(``workspace_table_description.xacro``). Pick a landmark on the hand -- a
metacarpal routing disc of any digit -- and a point on the table's grid, press
*Align hand to frame*, then play it on the robot and look at where the landmark
physically lands. A constant offset across several grid points is a wrong table
origin; an error that grows with distance from the corner is a wrong yaw or a
swapped axis. The x/y sliders ARE grid coordinates, because the viser world and
``lbr_workspace_table_link`` differ by a pure translation (see
``gepetto_control.frames``), and the 10 cm grid is drawn on the square to match
the one on the bench.

That alignment is CLOSED FORM, not a solve, despite the button. The metacarpal is
bolted to the palm -- measured: disc 1 moves 13-29 um in the wrist frame over the
whole 0-2.5 N flexor range, while disc 2, the first past the MCP joint, moves
4.5-13.8 mm -- so the wrist pose that puts the landmark on the target is one
matrix inverse, exact to the micrometre and independent of the constraint set. An
IK solve with a loosened wrist prior would be slower, approximate, and would let
whatever contact and pre-grasp constraints happen to be ticked drag the landmark
off the target, which is precisely the error being measured.

Changing the constraint set (object, contacts, collision, table) restarts the IK
loop, because the Augmented Lagrangian duals it carries describe the constraints
it has been working on. *Warm start* is the way to change one anyway and keep the
posture: while the latch is on, every restart begins from the state on screen
instead of from a straight hand with Q = 0 -- so it also starts an IK solve from
an FK pose you dialled in by hand. Only the posture carries; the penalty schedule
restarts regardless. *Reset defaults* puts every control back and cold-starts.

Run (from the ``crest-sparse`` root, so ``gepetto_solvers`` resolves to the
INSTALLED build -- launching from ``python/`` picks up the stale in-tree
``python/gepetto_solvers/_gepetto_solvers*.so`` instead, and every capability-gated
control silently goes dead):
    python scripts/viz_interactive.py

Every constraint from the paper's Chapter 2 (Eq 2.8-2.19) is an independent
switch in the *Constraints* folder -- object/table contact, object/self/table
collision, drop-normal-row SDF contact, opposition half-space, pre-grasp
centering -- each acting on the shared per-finger mask in its nested *fingers*
sub-folder. A box is the whole story: checked means that constraint family is in
the graph, with no second toggle it silently waits on. That is what makes a
stalled grasp bisectable: solve for the object alone, the table alone, or both,
with or without any of the three avoidances, and see which constraint family is
the one refusing to close. (Their tuning sliders --
collision radius/sigma/cull margin, constraint plane height -- stay in the
*Collision*/*Table* folders alongside the object/primitive picker.)

The support surface is TWO planes, deliberately. The drawn square is the physical
bench: it carries the corner frame, the grid, the calibration target and the
registration against ``lbr_workspace_table_link``, so nothing planning-related is
allowed to move it. The plane the SOLVER constrains against -- the support
equality and the avoidance half-space -- is the *Table* folder's *constraint
plane height* above that surface, drawn as a thin green sheet once it is lifted
off. Raise it to hold the fingers clear of the bench, or sink it below to let
them press in, without touching a single table-frame transform.

The solvers are the reusable ``HandFKSolver`` / ``HandIKStepper`` classes in
``tendon_hand/solvers.py``; the 3D scene is drawn by
``_plotting/viser_hand.ViserHandScene`` and the left-hand plots by
``_plotting/traj_panel.TrajectoryPanel``. The trajectory planner is not part of
this app -- see the ``traj_*`` scripts.

then open the printed http://localhost:8080 URL. The startup line names the
binding that was actually loaded and lists any capability missing from it.

Optional headless self-check of the solver classes (no browser):
    python scripts/viz_interactive.py --smoke
"""

import argparse
import math
import sys
import threading
import traceback

import numpy as np

from gepetto_solvers.core.geometry.scene import (get_primitive_specs, ycb_primitive_specs, proxy_semi_axes,
                    GRASP_FLEXOR_TENSION, TABLE_NORMAL, ELLIPSOID_SET_BETA,
                    TABLE_SPAN, TABLE_THICKNESS, table_corner,
                    object_principal_inplane_axis, INPLANE_DEGENERACY_RATIO,
                    ellipsoid_members, grasp_subset_indices)
from gepetto_solvers.core.solvers import (
    HandSolveParams, HandFKSolver, HandIKStepper,
    resolve_scene, resolve_table_origin, resolve_constraint_plane_origin,
    capabilities,
    euler_to_R, R_to_euler, solved_wrist_pose, plane_witness,
    default_object_center,
    half_space_witness, pregrasp_center_witness, pregrasp_axis_witness,
    pregrasp_centroid_witness, finger_plane_witness, planar_gap_witness,
    default_half_space_axis, PHASE_PRESETS, FLEXOR_IDX,
    synchronized_close, apply_phase_preset, CLOSE_FRACTION,
    lift_wrist, LIFT_HEIGHT_M, LIFT_STEPS,
    disc_pose, wrist_to_disc, wrist_pose_for_disc_target, disc_frame_error,
    DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY)
from gepetto_solvers.core.hand.config import pinch_pose, proximal_disc_flags
from gepetto_solvers.projects.robot_mount.mount import MOUNT_WRIST_XYZ, MOUNT_WRIST_RPY, measured_mount_pose
from gepetto_solvers.core import robot_plan


FINGER_LABELS = ["index", "middle", "ring", "pinky", "thumb"]

# ---------------------------------------------------------------------------
# Table-grid calibration.
# ---------------------------------------------------------------------------
#
# The URDF's hand geometry came from CAD, but where the workspace table SITS was
# measured with a ruler (workspace_table_description.xacro). The Calibration
# folder tests that measurement: command a known hand landmark to a known
# intersection of the grid drawn on the real table, then look at where it
# physically lands. A consistent offset is a wrong table origin; a rotation that
# grows with distance from the corner is a wrong yaw or a swapped axis.

# Spacing of the lines ruled on the physical table, and therefore of the grid
# drawn on the viser square. The square itself is scene.TABLE_SPAN (0.4 m), so
# this gives the 4x4 of 10 cm cells that is actually on the bench.
CAL_GRID_SPACING = 0.1

# The discs offerable as the landmark, disc index -> label.
#
# ONLY THE METACARPAL ONES. config.proximal_disc_flags marks discs 0 and 1
# rigidly attached to the palm, which is what makes the alignment a closed-form
# wrist placement rather than an IK solve: T_wrist<-disc is a constant of the
# morphology. Measured, not assumed -- across the whole 0-2.5 N flexor range disc
# 1 moves 13-29 um in the wrist frame, while disc 2 (the first past the MCP
# joint) moves 4.5-13.8 mm. Disc 1 is the default because it is the one you can
# actually find on the hardware: the far end of the metacarpal, where the MCP
# joint starts. Disc 0 is buried in the palm.
CAL_DISCS = {1: "distal metacarpal", 0: "metacarpal base"}
CAL_DEFAULT_DISC = 1

# How many times the placement re-measures and re-applies. The first pass is
# already micrometre-accurate; the second absorbs the last of the ~25 um of
# tension-dependence, and costs one FK solve.
CAL_REFINE_PASSES = 2

# ---------------------------------------------------------------------------
# ROS-mode constants.
# ---------------------------------------------------------------------------
#
# The two speed sliders in the Robot folder are FRACTIONS of these, so the
# numbers on screen mean "half of what the servo is configured to allow" rather
# than a bare m/s the operator has to hold against a yaml file in their head.

# MoveIt Servo's Cartesian scales, from lbr_bringup/config/moveit_servo.yaml.
# The servo runs command_in_type "unitless", so a twist component of 1.0 IS this
# many m/s (or rad/s) -- which is also why the bridge divides by them before
# publishing.
SERVO_SCALE_LINEAR = 0.4        # m/s
SERVO_SCALE_ROTATIONAL = 0.8    # rad/s


def _max_tendon_speed():
    """HandConfig's tendon speed cap, or the documented value if gepetto_core is
    not installed (this app runs on machines that have never seen the hardware).
    finger_servo_node enforces its own copy regardless, so this only sets what the
    slider means."""
    try:
        try:
            from gepetto_core.config import HandConfig
        except ImportError:
            from gepetto.config import HandConfig
        return float(HandConfig().max_tendon_speed)
    except Exception:
        return 0.065


MAX_TENDON_SPEED = _max_tendon_speed()

# The two playback sources, as they read on the dropdown.
PLAY_HISTORY = "recorded path (waypoints)"
PLAY_FINAL = "final state only"

# This app's own startup object -- see HandVizApp.__init__ for why it's set
# there rather than just changed on the dropdown widget.
#
# The megaminx: a 70 mm dodecahedron the factors see as its circumsphere, so it
# is a single analytic surface (no ellipsoid SET, no fetched/fitted YCB scan
# needed) that still carries hull_vertices -- the true solid is drawn inside the
# shell, and the table seats on the solid rather than on the proxy sphere. It
# also sits inside the graspable band, so the startup scene is one a 5-finger
# grasp can actually close on. Falls back automatically (see _build_gui) when the
# binding cannot build analytic ellipsoid surfaces at all.
DEFAULT_OBJECT_PRIMITIVE = "megaminx"
DEFAULT_OBJECT_FALLBACK = "mid_sphere_ellipsoid"

# Display-only suffix for the baked-SDF spheres in the object dropdown, so they
# read apart from the analytic ``*_sphere_ellipsoid`` look-alikes. The spec keys
# (and the demo scripts' argparse choices) keep the un-suffixed names; only the
# label the user picks from carries "_sdf".
SDF_DROPDOWN_LABELS = {"sphere": "sphere_sdf", "big_sphere": "big_sphere_sdf"}

# The stage the panel opens in, and the one Reset returns it to. Applied through
# the ordinary preset machinery rather than by building each widget at a phase-0
# value, so PHASE_PRESETS stays the single definition of what a phase IS and the
# opening panel cannot drift from the box that claims to describe it.
DEFAULT_PHASE = "phase0"

# Which half of the split the THUMB is sent to -- the opposition axis's sign,
# which the object's own geometry cannot answer (see
# solvers.orient_opposition_axis). Label -> HandSolveParams.half_space_flip.
OPPOSITION_SIDES = {
    "auto (match the hand)": None,
    "as derived": False,
    "flipped": True,
}

# Which shells of an ellipsoid-set object the fingertips may be sent to.
# Label -> HandSolveParams.use_grasp_subset.
#
# A YCB decomposition is not all handles: 5 of the power drill's 6 shells are its
# housing, and the contact equality against the smooth-min of the union is as
# happy landing a fingertip on those as on the grip. Which ones are grasp targets
# is authored per object and travels in the fit as `grasp_subset`.
#
# EITHER WAY, EVERY SHELL STILL COLLIDES -- this narrows the contact target, not
# the object (see EnvironmentConfig::contact_ellipsoid_subset). So "grasp subset"
# is not a way to reach into an object; it is a way to say which part of it the
# hand is reaching for.
CONTACT_SHELL_MODES = {
    "grasp subset": True,
    "all shells": False,
}
# The authored choice is a statement about the object, so honour it by default;
# an object with no authored subset is unaffected either way.
DEFAULT_CONTACT_SHELL_MODE = "grasp subset"


# The wrist sliders and the solvers must agree on what "pitch" means, so the
# convention lives with the params rather than here.
_euler_to_R = euler_to_R


def binding_path():
    """Where the loaded ``gepetto_solvers`` came from.

    Worth reporting because there are two of them: the installed build in
    site-packages, and a stale in-tree ``python/gepetto_solvers/_gepetto_solvers*.so``
    that shadows it whenever the app is launched from the ``python/`` directory.
    A control gated on ``capabilities()`` then goes quietly dead against a build
    the source has long since moved past, which looks like the feature failing
    rather than the import resolving somewhere unexpected."""
    import gepetto_solvers
    return gepetto_solvers.__file__


# ---------------------------------------------------------------------------
# Software e-stop.
# ---------------------------------------------------------------------------

class Refused(Exception):
    """Raised by :meth:`EStop.admit` when a solve is not allowed to start --
    either the latch is engaged or another solve already holds it."""


class EStop:
    """Latching software e-stop, and the single admission gate for every solve.

    LATCHING, not momentary. A momentary stop lets the very next thing that
    touches a slider restart the hand through the live-FK hook, which is
    precisely what someone reaching for a stop button does not want. Tripped
    stays tripped until :meth:`rearm`.

    The stop is COOPERATIVE, and it lands at an Augmented Lagrangian iteration
    boundary. One outer iteration is a single call into C++ with no interrupt
    hook (GTSAM's inner loop, via ``WarmAugmentedLagrangianOptimizer``), so
    nothing in Python can break into it -- ~1.7 s is the measured worst case and
    it is a floor, not a tuning choice. What matters is that this is bounded and
    that NO STATE IS LOST: the stepper keeps its multipliers, its penalty weight
    and its whole history, so a rearm resumes the same solve rather than
    restarting it.

    What makes the button feel instant is the GIL release on the solve binding
    (``capabilities()["gil_release"]``). With it the click is serviced while the
    last iteration is still running -- the latch engages, the controls grey out
    and the status updates at once. Without it the interpreter is frozen for the
    whole iteration and none of that can happen until it ends.

    Also the ONE place that decides whether a solve may start. It used to be a
    bare ``_solving`` bool tested and set without a lock, which the live-FK hook
    read while the auto-solve worker was writing it; folding the latch and the
    busy flag into one lock-guarded object closes that race as a side effect.
    """

    def __init__(self):
        # RLock rather than Lock: _refresh callbacks read the state while
        # holding it, and re-entering must not deadlock the GUI thread.
        self._lock = threading.RLock()
        self._tripped = False
        self._reason = ""
        self._busy = None       # what currently holds the gate, or None
        self._listeners = []    # notified on every trip/rearm; see add_listener

    # -- the latch --

    def add_listener(self, fn):
        """Register ``fn(tripped, reason)``, called on every trip and rearm.

        This is how the latch reaches things that are not solves. In ROS mode the
        robot bridge registers here, so pressing E-STOP does not merely refuse the
        next solve -- it halts the servo publishers, which is the only part of
        this app that can move a physical robot. Anything registered must return
        promptly and must not raise; see :meth:`_notify`.
        """
        with self._lock:
            self._listeners.append(fn)

    def _notify(self, tripped, reason):
        """Fan the latch's new state out to the listeners.

        Called with the lock RELEASED, deliberately. A listener publishes ROS
        messages and may take locks of its own, and holding the e-stop's lock
        across that is how a stop button ends up deadlocked against the thing it
        is trying to stop. An exception is swallowed with a traceback for the
        same reason: one listener failing must not stop the others being told,
        and must never turn the stop button into a crash.
        """
        for fn in list(self._listeners):
            try:
                fn(tripped, reason)
            except Exception:
                traceback.print_exc()

    def trip(self, reason="E-STOP pressed"):
        """Engage the latch. Runs on a viser callback thread, so it must never
        block: it takes the lock only to flip two fields, and deliberately does
        NOT wait for the running solve to notice."""
        with self._lock:
            if not self._tripped:
                self._tripped = True
                self._reason = reason
        self._notify(True, reason)
        return True

    def rearm(self):
        """Release the latch. Refuses while a solve is still winding down, so
        the GUI cannot come back to life around a solve that has not yet
        returned -- the operator would rearm into a hand still moving."""
        with self._lock:
            if self._busy is not None:
                return False
            self._tripped = False
            self._reason = ""
        self._notify(False, "")
        return True

    def is_tripped(self):
        """The poll predicate; handed straight to ``HandIKStepper.run`` as its
        ``should_stop``. A method rather than a property because that is the
        shape run() wants."""
        with self._lock:
            return self._tripped

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def check(self):
        """Raise if the latch is engaged. For polling inside a loop that has
        work of its own between solver calls."""
        if self.is_tripped():
            raise Refused(self._reason)

    # -- admission --

    def admit(self, what):
        """Claim the gate for one solve, or raise :class:`Refused`.

        Claims EAGERLY -- the refusal comes out of this call, not out of a later
        ``__enter__`` -- because the auto-solve hands its gate to a worker
        thread and so cannot express the claim as a ``with`` block. Callers that
        can still use one: ``with estop.admit(...)`` releases on the way out.

        The check and the claim happen under one lock, which is the point: two
        callbacks arriving on different viser worker threads cannot both see a
        free gate and both start solving.
        """
        with self._lock:
            if self._tripped:
                raise Refused(f"E-STOP engaged: {self._reason}")
            if self._busy is not None:
                raise Refused(f"already running: {self._busy}")
            self._busy = what
        return _Gate(self)

    def _release(self):
        with self._lock:
            self._busy = None


class _Gate:
    """The claim :meth:`EStop.admit` hands back. Releasing twice is harmless, so
    a caller may ``release()`` early and still let a ``with`` block unwind."""

    def __init__(self, estop):
        self._estop = estop

    def release(self):
        self._estop._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


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
    ok = _smoke_close() and ok
    ok = _smoke_lift() and ok
    ok = _smoke_calibration() and ok
    ok = _smoke_robot_plan() and ok
    print("Smoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# The claim the phase-4 Close button makes, as a number the smoke test can fail
# on: at every recorded pose, no grasping finger may be more than this fraction
# of the close ahead of or behind any other. Generous against the ~0.1% the walk
# actually lands (see synchronized_close), because what would break this is a
# regression in the FK warm start, not a wobble in the last digit.
_CLOSE_SYNC_TOL = 0.02

# ...and how far each finger may miss the displacement it was commanded to. Ten
# times solvers.CLOSE_TOL_M, i.e. still under the ~1 mm the tendon hardware
# resolves, so a pass here means the ramp is real and not just self-consistent.
_CLOSE_TRACK_TOL_M = 2e-3


def _smoke_close():
    """Check phase 4: that a synchronized close actually closes IN SYNC.

    The whole point of the phase is a claim about several fingers at once, and
    the claim is cheap to check and easy to break -- it rests on the FK solver
    warm-starting well over small upward tension steps, which is a property of
    the binding, not of this file. So it is measured here rather than trusted:
    re-derive each recorded pose's tendon displacements from the result itself,
    and fail on the worst disagreement between digits.
    """
    print("Smoke-testing the phase-4 synchronized close...")
    fingers = ["index", "middle", "thumb"]

    params = apply_phase_preset(HandSolveParams(), "phase4")
    _passive, open_flexors = robot_plan.open_pose_tensions()
    # From the calibrated OPEN hand, which is where the Close button starts from
    # on a freshly opened app and the only starting pose with a fixed meaning.
    params.flexor_tensions = [open_flexors.get(name, GRASP_FLEXOR_TENSION)
                              for name in FINGER_LABELS]
    solver = HandFKSolver(params)
    open_lengths = robot_plan.open_tendon_lengths(params, solver)
    limits = robot_plan.hardware_travel_limits()
    travel = None if limits is None else {name: hi
                                          for name, (_lo, hi) in limits.items()}

    result, notes = synchronized_close(solver, open_lengths, fingers, travel)
    n = result.num_iterates()

    # Progress per finger at every recorded pose, as a fraction of its own close.
    # Read off the ITERATES rather than off what the walk reported, so a bug in
    # the reporting cannot make this pass.
    disp = []
    for i in range(n):
        lengths = dict(zip(result.finger_names, result.at_iterate(i).tendon_lengths(0)))
        disp.append({name: open_lengths[name] - float(lengths[name][FLEXOR_IDX])
                     for name in fingers})
    span = {name: disp[-1][name] - disp[0][name] for name in fingers}
    worst_sync, worst_track = 0.0, 0.0
    for i, row in enumerate(disp):
        progress = [(row[name] - disp[0][name]) / span[name] for name in fingers]
        worst_sync = max(worst_sync, max(progress) - min(progress))
        # ...and against the schedule the walk was supposed to follow: pose i of
        # n-1 is i/(n-1) of the way through.
        want = i / float(n - 1)
        worst_track = max(worst_track,
                          max(abs(p - want) * span[name]
                              for p, name in zip(progress, fingers)))

    synced = worst_sync <= _CLOSE_SYNC_TOL
    tracked = worst_track <= _CLOSE_TRACK_TOL_M
    closed = n > 1 and all(v > 0.0 for v in span.values())
    ok = synced and tracked and closed
    print(f"  [   close] poses={n} "
          f"travel={', '.join(f'{k} {v * 1e3:+.1f}' for k, v in span.items())} mm "
          f"[{'ok' if ok else 'BAD'}]")
    print(f"           - in sync to {worst_sync * 100:.2f}% "
          f"(allow {_CLOSE_SYNC_TOL * 100:.0f}%) "
          f"[{'ok' if synced else 'BAD'}]")
    print(f"           - tracked the ramp to {worst_track * 1e3:.2f} mm "
          f"(allow {_CLOSE_TRACK_TOL_M * 1e3:.1f} mm) "
          f"[{'ok' if tracked else 'BAD'}]")
    for note in notes:
        print(f"           - {note}")
    return ok


# What the phase-5 Lift button claims, as two numbers the smoke test can fail on.
#
# The wrist has to arrive: how far the SOLVED wrist may sit from the height it
# was sent to, over the whole ramp. HandFKSolver already refuses a solve that
# misses its prior by more than 1 mm, so this only has to be loose enough not to
# fail on the microns a healthy solve lands within.
_LIFT_ARRIVE_TOL_M = 1e-3

# ...and the hand has to come with it RIGIDLY. Only the wrist prior moves during
# a lift -- no tension changes, no contact -- so every fingertip must translate
# by the same vector the wrist did, with the posture untouched. This is the check
# worth having: it is what breaks if a step ever outgrows the FK warm-start bound
# and the optimizer starts dragging the hand up instead of moving it.
_LIFT_RIGID_TOL_M = 5e-4


def _smoke_lift():
    """Check phase 5: that a lift raises the whole hand, rigidly, to where it
    was sent.

    Same reasoning as :func:`_smoke_close`: the phase rests on a property of the
    binding (the FK warm start carrying the hand across each step), not of this
    file, so it is measured rather than trusted.
    """
    print("Smoke-testing the phase-5 wrist lift...")

    params = apply_phase_preset(HandSolveParams(), "phase5")
    # Off a CLOSED hand, since that is what the button lifts in practice, and a
    # curled rod is the harder thing to translate rigidly than a straight one.
    params.flexor_tensions = [GRASP_FLEXOR_TENSION] * len(FINGER_LABELS)
    solver = HandFKSolver(params)

    z0 = float(params.wrist_pose[2, 3])
    result, notes = lift_wrist(solver)
    n = result.num_iterates()

    def tips(view):
        """Fingertip positions at a recorded pose -- the node the renderer draws
        the contact sphere on."""
        return {name: np.asarray(
            view.frames[0][name].marginals.rod.states[-1].pose.mean, float)[:3, 3]
            for name in view.finger_names}

    start_tips = tips(result.at_iterate(0))
    worst_arrive, worst_rigid = 0.0, 0.0
    for i in range(n):
        view = result.at_iterate(i)
        want = z0 + i * (LIFT_HEIGHT_M / LIFT_STEPS)
        got = solved_wrist_pose(solver.configs, view.frames[0])
        worst_arrive = max(worst_arrive, abs(float(got[2, 3]) - want))
        # Every tip should sit exactly where it started, plus the rise so far.
        rise = np.array([0.0, 0.0, float(got[2, 3]) - z0])
        worst_rigid = max(worst_rigid,
                          max(float(np.linalg.norm(p - (start_tips[name] + rise)))
                              for name, p in tips(view).items()))

    stepped = n == LIFT_STEPS + 1
    arrived = worst_arrive <= _LIFT_ARRIVE_TOL_M
    rigid = worst_rigid <= _LIFT_RIGID_TOL_M
    ok = stepped and arrived and rigid
    print(f"  [    lift] poses={n} (expect {LIFT_STEPS + 1}) "
          f"rise={(float(solver.params.wrist_pose[2, 3]) - z0) * 1e3:+.1f} mm "
          f"[{'ok' if ok else 'BAD'}]")
    print(f"           - arrived within {worst_arrive * 1e3:.3f} mm of every "
          f"commanded height (allow {_LIFT_ARRIVE_TOL_M * 1e3:.1f} mm) "
          f"[{'ok' if arrived else 'BAD'}]")
    print(f"           - fingertips translated rigidly to {worst_rigid * 1e3:.3f} "
          f"mm (allow {_LIFT_RIGID_TOL_M * 1e3:.1f} mm) "
          f"[{'ok' if rigid else 'BAD'}]")
    for note in notes:
        print(f"           - {note}")
    return ok


# What "landed on the target" has to mean for the Calibration folder to be
# measuring the TABLE rather than its own error. The bench's grid is drawn to
# maybe a millimetre; anything at these tolerances is two orders below that and
# so contributes nothing to what is being calibrated.
_CAL_SMOKE_POS_MM = 0.05
_CAL_SMOKE_ROT_DEG = 0.01

# The premise the closed form rests on: a metacarpal disc does not move in the
# wrist frame when the tendons pull, and the first disc past the MCP does. Both
# halves are asserted -- a test that only checked the rigid one would still pass
# against a build where every disc had been welded to the palm.
_CAL_RIGID_TOL_MM = 0.1
_CAL_ARTICULATED_MIN_MM = 1.0


def _smoke_calibration():
    """Check the closed-form landmark placement the Calibration folder is built on.

    The whole feature is one number -- how far the landmark ends up from where it
    was sent -- so that number is what this measures, with no viser and no
    hardware. It also tests the PREMISE separately: the placement is exact only
    because a metacarpal disc is rigid to the wrist, and if that ever stopped
    being true the residual check alone would not say why.
    """
    print("Smoke-testing the calibration landmark placement...")
    ok = True
    finger, disc = FINGER_LABELS[0], CAL_DEFAULT_DISC

    params = HandSolveParams()
    params.table_burial = 0.0
    solver = HandFKSolver(params)

    # -- the premise: which discs move when the tendons pull --
    def transforms(tension):
        params.flexor_tensions = [tension] * len(FINGER_LABELS)
        frame = solver.solve().frames[0]
        return {name: [wrist_to_disc(solver.configs, frame, name, d)
                       for d in (disc, disc + 1)]
                for name in FINGER_LABELS}

    slack, pulled = transforms(0.0), transforms(2.5)
    rigid_mm = max(np.linalg.norm(slack[n][0][:3, 3] - pulled[n][0][:3, 3])
                   for n in FINGER_LABELS) * 1e3
    moved_mm = min(np.linalg.norm(slack[n][1][:3, 3] - pulled[n][1][:3, 3])
                   for n in FINGER_LABELS) * 1e3
    premise = (rigid_mm < _CAL_RIGID_TOL_MM and moved_mm > _CAL_ARTICULATED_MIN_MM)
    ok = ok and premise
    print(f"  [ rigidity] disc {disc} moves {rigid_mm * 1e3:.1f} um, disc "
          f"{disc + 1} moves {moved_mm:.2f} mm over 0-2.5 N "
          f"[{'ok' if premise else 'BAD'}]")

    # -- the placement itself --
    params.flexor_tensions = list(HandSolveParams().flexor_tensions)
    frame = solver.solve().frames[0]
    # A target well away from where the landmark already is, and rotated, so a
    # placement that silently did nothing could not pass.
    target = disc_pose(frame, finger, disc).copy()
    target[:3, 3] += np.array([0.05, -0.05, 0.05])
    target[:3, :3] = euler_to_R(0.0, 0.0, np.deg2rad(15.0)) @ target[:3, :3]

    for _ in range(CAL_REFINE_PASSES):
        params.wrist_pose = wrist_pose_for_disc_target(
            solver.configs, frame, finger, disc, target)
        frame = solver.solve().frames[0]

    pos_mm, rot_deg = disc_frame_error(disc_pose(frame, finger, disc), target)
    landed = pos_mm < _CAL_SMOKE_POS_MM and rot_deg < _CAL_SMOKE_ROT_DEG
    ok = ok and landed
    print(f"  [    align] residual {pos_mm:.5f} mm / {rot_deg:.5f} deg after "
          f"{CAL_REFINE_PASSES} passes [{'ok' if landed else 'BAD'}]")
    return ok


def _smoke_robot_plan():
    """Check the robot-plan export the way the Robot folder uses it, headlessly.

    This is the half of the ROS integration that can be tested with no ROS, no
    hardware and no browser, and it is the half that decides which way the
    fingers move -- so it is worth running every time the solver changes, not
    only when someone opens the app.
    """
    print("Smoke-testing the robot plan export...")
    ok = True

    params = HandSolveParams()
    open_lengths = robot_plan.open_tendon_lengths(params)
    notes, sign_ok = robot_plan.check_open_lengths(open_lengths, params)
    ok = ok and sign_ok
    print(f"  [    open] {', '.join(f'{k} {v * 1e3:.1f}' for k, v in open_lengths.items())} mm "
          f"[{'ok' if sign_ok else 'BAD'}]")
    for note in notes:
        print(f"           - {note}")

    if not capabilities()["ik_stepping"]:
        print("  [    plan] skipped -- binding cannot step an IK solve")
        return ok

    # A short stepped solve, so the plan has real AL iterates to walk.
    stepper = HandIKStepper(HandSolveParams())
    last = {}
    stepper.run(max_steps=3, on_step=lambda r, s: last.update(res=r))
    result = last.get("res")
    spec, center, _rot, _pose = resolve_scene(stepper.params)
    corner = table_corner(
        resolve_table_origin(stepper.params, spec, center),
        np.asarray(stepper.params.plane_normal, float))

    plan = robot_plan.build_plan(result, stepper.configs, corner, open_lengths)

    # THE WHOLE PATH, one waypoint per recorded iterate. Checked rather than
    # assumed because the failure is silent and was live for a while: build_plan
    # used to take the convergence scrubber's index, the scrubber opens on the
    # LAST iterate, and the "recorded path" therefore collapsed to a single
    # waypoint -- one hop to the final pose with the trajectory dropped. Nothing
    # downstream can tell a one-waypoint plan from a legitimate one.
    n_iterates = result.num_iterates()
    whole = len(plan.waypoints) == n_iterates
    if not whole:
        print(f"  [    plan] BAD -- history gave {len(plan.waypoints)} waypoint(s) "
              f"for {n_iterates} recorded iterates; the path is being truncated")
    ok = ok and whole

    plan, clamp_notes = robot_plan.clamp_to_travel(plan)
    # The approach segment the bridge prepends at play time: pretend the robot is
    # at the hand-open pose, which is the worst case for the first segment.
    plan = robot_plan.prepend_current(
        plan, plan.waypoints[0].wrist_pose, {n: 0.0 for n in plan.finger_names})
    samples = robot_plan.interpolate(plan, hz=100.0)

    # Every sample must be finite and the last must land ON the final waypoint,
    # or the robot would be commanded somewhere the solve never asked for.
    final = plan.waypoints[-1]
    landed = np.allclose(samples[-1].wrist_pose, final.wrist_pose, atol=1e-9)
    finite = all(np.all(np.isfinite(s.wrist_pose)) for s in samples)
    status = "ok" if landed and finite and whole and len(samples) > 1 else "BAD"
    ok = ok and status == "ok"
    print(f"  [    plan] waypoints={len(plan.waypoints)} "
          f"({n_iterates} iterates + 1 approach) samples={len(samples)} "
          f"duration={samples[-1].t:.2f}s [{status}] | {robot_plan.summarize(plan)}")
    for note in clamp_notes:
        print(f"           - {note}")
    return ok


# ---------------------------------------------------------------------------
# Interactive app.
# ---------------------------------------------------------------------------

class HandVizApp:

    def __init__(self, server, ros_mode=False, bridge=None):
        import viser  # local import so --smoke needs no viser
        self.viser = viser
        self.server = server
        # ROS mode adds the Robot folder -- play a solve on the hardware, read the
        # hardware back -- and extends the e-stop to the servo publishers. Off by
        # default so the standalone app is byte-for-byte the app it was; the
        # bridge is DUCK-TYPED (play / read_state / stop / status), so nothing in
        # crest-sparse imports rclpy and the ROS side stays in gepetto_control.
        self.ros_mode = bool(ros_mode) and bridge is not None
        self.bridge = bridge if self.ros_mode else None
        # Cached hand-open tendon lengths, the zero every commanded displacement
        # is measured from. Built lazily on the first robot action (it costs an FK
        # solve plus the sign self-check) and dropped whenever the hand's
        # morphology-independent scene changes cannot affect it -- see
        # _open_lengths.
        self._open_lengths_cache = None
        self._open_notes = []
        # The lower, unchanging half of the Robot folder's readout, cached so
        # playback feedback does not re-poll the bridge ten times a second.
        self._standing_status = None
        self._play_thread = None
        # What this installed binding supports, so we can gate controls a stale
        # .so would crash on (ellipsoid objects, the table, cull margin).
        # Resolved before the params because _fresh_params reads it.
        self.caps = capabilities()
        self.params = self._fresh_params()
        # Which solver produced what is on screen: "FK" for a posed hand, "IK"
        # once the stepper has been driven. Gates the live FK re-solve and labels
        # the status readout; there is no mode picker.
        self.mode = "FK"
        self.result = None
        # The software e-stop, and the gate every solve has to pass to start.
        # Replaces both the old _solving bool and the auto-run's private stop
        # Event -- one object, so "is anything running" and "may anything start"
        # cannot disagree. See EStop.
        self.estop = EStop()
        # Cached IK stepper: it owns the AL outer loop being advanced one
        # iteration per Step, so it has to outlive a single step.
        self.stepper = None
        # Warm-start latch: while on, every (re)build of the stepper starts from
        # the state on screen rather than the cold guess. See _ensure_stepper.
        # On by default -- the staged phase0 -> phase1 -> phase2 pipeline this
        # app is for is a chain of continuations, so cold-starting each stage is
        # the exception. Off with no `solver_seed` binding rather than latched
        # true against a capability that cannot honour it (_toggle_warm_start
        # refuses in that case too, so the latch could never be cleared).
        self.warm_start = self.caps["solver_seed"]
        self._auto_thread = None
        # True while Reset is writing the controls back to their defaults, so the
        # per-handle callbacks (live FK, object rebuild) sit out the restore and
        # the one solve at the end of it is the only one that runs.
        self._restoring = False
        # Latch for the object-contact form guard: it settles the two mutually
        # exclusive boxes by writing the other one, whose callback lands right
        # back in the guard. See _enforce_object_contact.
        self._contact_guard = False
        # Cached YCB catalog/browser state, built lazily by the YCB folder so the
        # app starts without touching the network or the catalog file.
        self._ycb_cache = None
        self._ycb_busy = False

        from gepetto_solvers.core.plotting.viser_hand import ViserHandScene
        self.scene = ViserHandScene(server, FINGER_LABELS)

        # The control-trajectory plots, in their own window docked to the LEFT of
        # the 3D view (the main control panel is on the right, so the two do not
        # compete for the same edge). Built before _build_gui because the Display
        # folder's visibility checkbox needs something to toggle; it is a
        # top-level entity, so it is not placed in any folder that happens to be
        # open. See _plotting/traj_panel.py.
        from gepetto_solvers.core.plotting.traj_panel import TrajectoryPanel
        self.traj = TrajectoryPanel(server, FINGER_LABELS)
        #: Measured robot states from the last playback, keyed by the waypoint
        #: (== iterate) index they were sampled at. None when nothing has been
        #: played, or when what was played cannot be lined up against the plotted
        #: iterates. See `_sample_robot_trace`.
        self._robot_trace = None

        # Park every (current and future) client's camera on the -X/palmar side so
        # the finger curl reads as a grasp instead of bending backwards. Without
        # this viser opens from the opposite side and the correct solve looks wrong.
        server.on_client_connect(lambda client: self._aim_camera(client))

        self._build_gui()
        # Decide up front whether the opening object supports the in-plane
        # contact form, so the box is never offered live for a scene the solve
        # would refuse.
        self._refresh_planar_contact_gate()
        # Same for the contact-shells choice, whose hint counts the opening
        # object's shells and so cannot be written at build time.
        self._refresh_grasp_subset_gate()
        # The panel opens IN a stage, not merely showing its box ticked: the
        # build-time tick fires no callback, so the preset is written here. After
        # _gui_defaults was captured, deliberately -- Reset restores the ticked
        # box and calls this again, rather than snapshotting phase 0's values as
        # if they were the widgets' own.
        self._apply_default_phase()
        # A cached FK solver so wrist/tension tweaks warm-start (rebuilt on object
        # change only).
        self._rebuild_fk()
        self._refresh_object()
        self._fk_solve()

        # Last, so a bridge that publishes on registration cannot fire into a
        # half-built app: the latch now reaches the servo publishers.
        if self.ros_mode:
            self.estop.add_listener(self._on_estop_change)
            self._refresh_robot_status()

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
        return params

    def _object_pose_from_sliders(self):
        """``(center, rotation)`` for the object: its derived pose plus the
        Object-pose slider offsets.

        Resolved against the primitive's OWN default rather than the params'
        current value -- reading back what was written last time would feed the
        control's output into its input, and every sync would compound the offset.
        (The support plane's ``auto_table_origin`` avoids the same trap the same
        way.) All-zero sliders reproduce the derived pose exactly, so this is
        equivalent to the previous ``object_center = None`` behaviour.
        """
        spec = get_primitive_specs()[self.params.primitive]
        center = default_object_center(self.params.primitive, spec)
        rotation = np.asarray(spec.get("rotation", np.eye(3)), float)

        offset = np.array([self.g_obj_dx.value, self.g_obj_dy.value,
                           self.g_obj_dz.value], float)
        delta = _euler_to_R(self.g_obj_roll.value, self.g_obj_pitch.value,
                            self.g_obj_yaw.value)
        return center + offset, delta @ rotation

    def _object_pose_changed(self, _=None):
        """Re-place the object from the sliders.

        The object pose is part of the CONSTRAINT SET -- it is the mean of the
        object prior every contact and collision factor is written against -- so
        moving it invalidates the stepper's Augmented Lagrangian duals, exactly
        like changing the object itself. Re-solving with FK keeps the picture
        honest about that rather than leaving a stale IK pose next to a moved
        object.
        """
        if self._restoring:
            return
        self._sync_params()
        self._refresh_object()
        self._invalidate_stepper()
        if self.mode == "FK":
            self._live_fk()
        else:
            self._render_frame()

    def _refresh_object_mesh(self, spec, center, rotation):
        """Draw (or clear) the object's TRUE geometry behind the analytic surface
        the solver actually sees.

        Two kinds of object have one: a ycb: set, whose shells approximate a
        scanned mesh, and a spec carrying ``hull_vertices`` -- the megaminx,
        whose circumsphere encloses a dodecahedron. Both are the same question
        ("how much object is really inside the surface the fingers stop on"), so
        they share one toggle.

        A YCB object carries BOTH: the scan is what it really looks like, its
        hull is the summary of the scan committed alongside the fit (and what the
        table is seated on -- ``scene.ycb_primitive_specs``). The scan wins when
        the cache has it, since a hull cannot show a mug's handle or a concave
        face; the hull is the fallback rather than nothing at all, because the
        fits are committed and the 1.5 GB of meshes are not, so an object can be
        perfectly loadable on a machine that has never fetched a scan.

        The mesh has to be put in the SAME frame the shells were re-centered
        into, or the two render a few cm apart and the overlay is worse than
        useless. (The hull was written into that frame at spec-build time.)
        """
        if not self.g_show_true_mesh.value:
            self.scene.clear_object_mesh()
            return
        if spec["type"] == "ellipsoid_set":
            try:
                from gepetto_solvers.core.objects.ycb import Catalog, YcbCache, ground_and_center

                cache = YcbCache(Catalog())
                mesh = ground_and_center(
                    cache.load_mesh(spec["ycb"], spec["source"], max_texture=512))
                mesh.apply_translation(-np.asarray(spec["recenter"], float))
                self.scene.set_object_mesh(mesh, center, rotation)
                return
            except Exception as exc:
                print(f"[viz] no scan mesh for {spec.get('ycb')}: {exc}")
        hull = spec.get("hull_vertices")
        if hull is None:
            self.scene.clear_object_mesh()
            return
        # Local point set -> hull; set_object_mesh applies the object pose, so the
        # solid lands inside its own shell however the object is posed.
        self.scene.set_object_mesh(self.scene.hull_mesh(hull), center, rotation)

    # -- YCB objects --------------------------------------------------------

    def _build_ycb_folder(self, gui):
        """Fetch-and-fit controls for the YCB object set.

        The offline path (``python scripts/objects/ycb_browser.py --fit <name>``)
        remains the primary one and writes the same files; this exists so an
        object can be brought in without leaving the app. Everything here is
        gated on ``ellipsoid_set``: without it a fitted object could be written
        but never loaded, so offering the button would be a trap.
        """
        available = self.caps.get("ellipsoid_set", False)
        with gui.add_folder("YCB objects", expand_by_default=False):
            if not available:
                gui.add_markdown(
                    "Needs a rebuilt `_gepetto_solvers` with "
                    "`EnvironmentConfig.ellipsoid_set`.")
                self.g_ycb_object = None
                return

            gui.add_markdown(
                "Fit a YCB object to an ellipsoid **set** and add it to the "
                "object list as `ycb:<name>`. First fetch of an object "
                "downloads 4-12 MB and takes tens of seconds.")
            self.g_ycb_object = gui.add_dropdown(
                "catalog", self._ycb_labels(),
                hint="Every YCB object with a usable textured mesh, with the "
                     "download size of its best mesh.")
            self.g_ycb_backend = gui.add_dropdown(
                "backend", ["gmm", "kmeans", "coacd"], initial_value="gmm",
                hint="gmm handles elongated parts; coacd respects concavity but "
                     "is slow and is an optional dependency.")
            # Same two controls as the browser's Fit panel, not a single
            # "0 = auto" slider: an explicit mode leaves no way for the count to
            # be read as a fit request when automatic was meant.
            self.g_ycb_count = gui.add_dropdown(
                "count", ["auto", "manual"], initial_value="auto",
                hint="auto sweeps k and takes the smallest one near the best "
                     "result. This is what produces a sensible multi-ellipsoid "
                     "decomposition; manual pins the count below.")
            self.g_ycb_k = gui.add_slider(
                "ellipsoids (k)", 1, 15, 1, 4,
                hint="Only used when count is 'manual'.")
            self.g_ycb_coverage = gui.add_slider(
                "coverage target", 0.90, 1.0, 0.005, 0.98)
            self.g_ycb_fit = gui.add_button(
                "Fetch & fit", icon=self.viser.Icon.DOWNLOAD)
            self.g_ycb_status = gui.add_markdown(
                f"{len(self._ycb_fitted())} object(s) already fitted.")
            self.g_ycb_fit.on_click(self._ycb_fit_clicked)

    def _ycb_labels(self):
        """Catalog dropdown labels, or a one-entry placeholder if it cannot load."""
        try:
            from gepetto_solvers.core.objects.ycb import Catalog
            return Catalog().labels()
        except Exception as exc:
            return [f"<catalog unavailable: {exc}>"]

    def _ycb_fitted(self):
        """Object keys already fitted and loadable, i.e. the ``ycb:`` specs."""
        return [k for k in get_primitive_specs() if k.startswith("ycb:")]

    def _ycb_fit_clicked(self, _event=None, name=None):
        """Start a fit. ``name`` overrides the YCB folder's own picker, which is
        how selecting an unfitted object straight from the object dropdown routes
        through the same one worker."""
        if self._ycb_busy:
            return
        self._ycb_busy = True
        threading.Thread(target=self._ycb_fit_worker, args=(name,),
                         daemon=True).start()

    def _ycb_fit_worker(self, name=None):
        """Download + fit one object on a worker thread, then offer it as an object.

        Runs off the GUI thread because a cold fit is a download plus a k-sweep --
        tens of seconds during which viser must stay responsive. It deliberately
        does NOT take the solver's ``_solving`` latch: fitting touches no solver
        state, so there is no reason it should block stepping an unrelated solve.
        """
        self.g_ycb_fit.disabled = True
        try:
            from gepetto_solvers.core.objects.ycb import Catalog, YcbCache
            from gepetto_solvers.core.objects.ycb.fitting import fit_object
            from gepetto_solvers.core.geometry.scene import ycb_primitive_specs

            catalog = Catalog()
            if name is None:
                name = catalog.name_from_label(self.g_ycb_object.value)
            source = catalog.objects[name].sources[0]
            backend = self.g_ycb_backend.value
            coverage = float(self.g_ycb_coverage.value)
            # None => automatic sweep, which is the default and what gives a real
            # multi-ellipsoid decomposition.
            k = (None if self.g_ycb_count.value == "auto"
                 else int(self.g_ycb_k.value))

            def report(fraction, message):
                # Mirrored to the main status bar as well: a fit started by
                # picking an unfitted object from the object dropdown is watched
                # there, not in the (collapsed by default) YCB folder.
                text = f"**{name}** — {message}"
                self.g_ycb_status.content = text
                self._set_status(text)

            # The shared pipeline the browser and the --fit CLI use, so an object
            # fitted here is identical to one fitted there.
            result, _path = fit_object(
                YcbCache(catalog), name, source, backend=backend, k=k,
                coverage=coverage, progress=report,
            )

            # The spec registry caches the directory listing, so a fit written
            # after startup is invisible until that cache is dropped.
            ycb_primitive_specs.cache_clear()
            self._refresh_object_dropdown(select=f"ycb:{name}")
            self.g_ycb_status.content = (
                f"**{name}** · `{backend}`\n\n{result.metrics.summary()}\n\n"
                "_Selected as the current object._")
        except Exception as exc:
            traceback.print_exc()
            self.g_ycb_status.content = f"Fit failed:\n\n`{exc}`"
            self._set_status(f"Fit of **{name}** failed: `{exc}`")
        finally:
            self.g_ycb_fit.disabled = False
            self._ycb_busy = False

    def _refresh_object_dropdown(self, select=None):
        """Rebuild the object dropdown's options after the spec registry changed,
        optionally selecting a key. Assigning ``options`` re-renders the widget in
        place, so the rest of the GUI is untouched."""
        labels, self._label_to_key = self._object_dropdown_labels()
        self.g_object.options = labels
        if select is not None and select in self._label_to_key.values():
            label = SDF_DROPDOWN_LABELS.get(select, select)
            if label in self._label_to_key:
                # Fires on_update -> reloads the scene on the newly fitted object.
                self.g_object.value = label

    # Dropdown suffix for a catalog object with no committed fit yet. Selecting
    # one downloads and fits it on the spot (see _on_object_selected).
    UNFITTED_SUFFIX = "  [fit on select]"

    def _on_object_selected(self, _=None):
        """Load the picked object, fitting it first if it has never been fitted.

        The fit runs on a worker thread and re-enters here once it has written
        the export, so an unfitted pick is a slow version of a fitted one rather
        than a separate mode the user has to know about.
        """
        if self._restoring:
            return
        label = self.g_object.value
        if label in getattr(self, "_unfitted", {}):
            # Leave the scene on the current object; the worker re-selects this
            # one once its fit exists.
            self._set_status(f"Fetching and fitting **{self._unfitted[label]}**… "
                             "(first fetch downloads 4-12 MB)")
            self._ycb_fit_clicked(name=self._unfitted[label])
            return
        self._load_selected_object()

    def _load_selected_object(self):
        self.params.primitive = self._label_to_key[self.g_object.value]
        # The Object-pose sliders are offsets from each primitive's own default,
        # so they carry over to the new object rather than being cleared;
        # _sync_params re-resolves them against the new base pose.
        self.params.object_center, self.params.object_rotation = \
            self._object_pose_from_sliders()
        # Whether the in-plane contact form is even possible is a property of the
        # object (it needs an ellipsoid cross-section), so re-decide it here
        # rather than leaving a live checkbox the next solve would refuse.
        self._refresh_planar_contact_gate()
        # Whether there is a grasp subset to choose is likewise a property of the
        # object, so re-gate it here too -- and before _refresh_object, which
        # greys the excluded shells off the answer.
        self._refresh_grasp_subset_gate()
        self._rebuild_fk()      # FK solver carries the object for its result/spec
        self._refresh_object()
        self._aim_all_cameras()  # re-center on the new object's location
        self._fk_solve()

    def _object_dropdown_labels(self):
        """Every object offerable, fitted or not, as ``(labels, label -> key)``.

        The whole YCB catalog is listed, not just what has been fitted, so the
        picker is the object set rather than a record of what happens to be
        cached. An unfitted entry is marked and fits itself when chosen. The
        alternative -- pre-fitting all ~97 up front -- costs a 0.6 GB download and
        hours of k-sweeps to produce mostly objects that will never be picked.
        """
        keys = self._visible_primitive_keys()
        mapping = {SDF_DROPDOWN_LABELS.get(k, k): k for k in keys}
        self._unfitted = {}
        if self.caps.get("ellipsoid_set", False):
            try:
                from gepetto_solvers.core.objects.ycb import Catalog

                catalog = Catalog()
                for name in catalog.names():
                    if f"ycb:{name}" in keys:
                        continue           # already fitted, listed above
                    label = f"ycb:{name}{self.UNFITTED_SUFFIX}"
                    mapping[label] = f"ycb:{name}"
                    self._unfitted[label] = name
            except Exception as exc:
                print(f"[viz] YCB catalog unavailable: {exc}")
        return list(mapping), mapping

    def _visible_primitive_keys(self):
        """Object keys this binding can actually build, in dropdown order.

        Both analytic surface kinds are gated: a single ellipsoid needs
        ``ellipsoid_semi_axes``, an ellipsoid set needs ``ellipsoid_set``, and a
        stale ``.so`` may have either without the other. Offering an object whose
        env fields do not exist is worse than hiding it -- ``attach_ellipsoid_set``
        raises rather than silently building a surface-less env, so the object
        would simply fail to load with a traceback in the browser.
        """
        gates = {"ellipsoid": self.caps["ellipsoid"],
                 "ellipsoid_set": self.caps.get("ellipsoid_set", False)}
        return [k for k, v in get_primitive_specs().items()
                if gates.get(v["type"], True)]

    def _resolve_default_primitive(self):
        """The startup object, falling back when it is unavailable.

        The default (and its fallback) are analytic-ellipsoid objects, so both
        drop out of the dropdown on a binding without ``ellipsoid_semi_axes``;
        a YCB default would additionally need ``ellipsoid_set`` and a populated
        ``_objects/ycb/fits/``. Resolve here rather than only on the dropdown
        widget, because
        ``_rebuild_fk`` reads ``params.primitive`` directly and would raise a bare
        KeyError on a key the spec registry never produced.
        """
        keys = self._visible_primitive_keys()
        for candidate in (DEFAULT_OBJECT_PRIMITIVE, DEFAULT_OBJECT_FALLBACK):
            if candidate in keys:
                return candidate
        return keys[0]

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

    def _pose_at_mount(self, _=None):
        """Drive the wrist sliders to the measured robot mount and re-pose.

        With the wrist at ``T_flange<-wrist``, the viser world frame IS the flange
        frame, so what you see is the hand as it hangs off the arm in the CAD
        assembly -- the check the measurement actually needs. Turns the mount
        frames on, since arriving here with them off shows nothing new.
        """
        for handle, value in zip(
                (self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw),
                tuple(MOUNT_WRIST_XYZ) + tuple(MOUNT_WRIST_RPY)):
            # Sliders quantize to their step, so the pose actually solved is the
            # rounded one; _render_mount draws from the sliders for that reason.
            handle.value = float(value)
        self.g_show_mount.value = True
        self._sync_params()
        self._fk_solve()

    def _render_mount(self, res=None):
        """Draw or clear the wrist/flange frame pair for the pose on screen.

        Anchored on the SOLVED wrist, not ``params.wrist_pose``: the wrist is a
        variable whose prior is soft, so contact pulling on the hand moves it tens
        of millimetres off the commanded pose. Drawing the commanded pose leaves
        the frames stranded while the hand they describe moves away from them --
        and the flange is rigidly bolted to the wrist, so it must travel with it.
        ``res`` is the iterate being rendered, so the frames track the convergence
        scrubber too. Falls back to the commanded pose before the first solve.
        """
        if not self.g_show_mount.value:
            self.scene.clear_mount_frames()
            return
        T_wrist = (solved_wrist_pose(self.fk_solver.configs, res.frames[0])
                   if res is not None else self.params.wrist_pose)
        self.scene.set_mount_frames(T_wrist, measured_mount_pose())

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
        p.passive_tension = self.g_passive.value
        p.flexor_tensions = [s.value for s in self.g_flexors]
        p.flexor_tension_sigma = 10.0 ** self.g_flexor_sigma.value
        p.passive_tension_sigma = 10.0 ** self.g_passive_sigma.value
        p.ellipsoid_set_beta = float(self.g_set_beta.value)
        p.use_grasp_subset = CONTACT_SHELL_MODES[self.g_contact_shells.value]
        p.contact_fingers = [c.value for c in self.g_contacts]
        p.object_contact = self.g_obj_contact.value or self.g_obj_contact_plane.value
        # Which FORM, once object contact is on at all. The two boxes are kept
        # mutually exclusive by _enforce_object_contact, so this cannot be read
        # as "both": object_contact says whether there is a contact,
        # object_contact_in_plane says which distance it is measured with.
        p.object_contact_in_plane = self.g_obj_contact_plane.value
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
        # rod physics
        p.planar_bending = self.g_planar_bend.value and self.caps["planar_bending"]
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
        # display toggles
        self.scene.show_discs = self.g_show_discs.value
        self.scene.show_disc_frames = self.g_show_disc_frames.value
        self.scene.show_contact_spheres = self.g_show_contact.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self.scene.show_gap_lines = self.g_show_gaps.value
        self.scene.show_finger_planes = self.g_show_finger_planes.value
        self.scene.show_planar_gap = self.g_show_planar_gap.value

    def _table_origin(self):
        """The rendered slab's origin: the TABLE surface, seated from the scene.

        Everything registered against the bench hangs off this -- the slab, its
        corner frame and grid, the calibration target, and ``_corner_viz``, which
        is this app's half of the registration against
        ``lbr_workspace_table_link``. It is therefore deliberately independent of
        the constraint-plane height slider; that moves
        :meth:`_constraint_plane_origin` alone.
        """
        spec, center, _rot, _pose = resolve_scene(self.params)
        return resolve_table_origin(self.params, spec, center)

    def _constraint_plane_origin(self):
        """The origin of the plane the SOLVER sees -- the table raised by the
        Table folder's height slider. Resolved through the solvers module, so the
        picture drawn here is the same plane ``_attach_table`` builds the support
        equality and the avoidance half-space from."""
        spec, center, _rot, _pose = resolve_scene(self.params)
        return resolve_constraint_plane_origin(self.params, spec, center)

    def _refresh_table_readout(self, origin, corner):
        """Publish the landmark's numbers: the square's size and where its corner
        frame currently is, in world coordinates.

        These have to be readable, not inferred. The whole point of the frame is
        to be measured against a real bench, and a triad you can see but whose
        coordinates you cannot read is not a landmark. The table height is quoted
        alongside because the table is seated from the object (see
        ``auto_table_origin``), so it moves when the object changes -- this is
        where you see that it did.

        The CONSTRAINT plane's height is quoted on its own line, because the two
        surfaces are now independent and the whole risk of separating them is
        mistaking one for the other: this says, in world coordinates, exactly
        where the solver's plane sits relative to the bench.
        """
        origin = np.asarray(origin, float).reshape(3)
        corner = np.asarray(corner, float).reshape(3)
        axis = int(np.argmax(np.abs(np.asarray(self.params.plane_normal, float))))
        constraint = np.asarray(self._constraint_plane_origin(), float).reshape(3)
        height = self.params.constraint_plane_height
        self.g_table_status.content = (
            f"square **{TABLE_SPAN:.3f} x {TABLE_SPAN:.3f} m**, "
            f"{TABLE_THICKNESS * 1e3:.0f} mm thick  \n"
            f"table (top face) {'xyz'[axis]} = {origin[axis]:+.4f} m  \n"
            f"constraint plane {'xyz'[axis]} = {constraint[axis]:+.4f} m "
            f"({height * 1e3:+.0f} mm)  \n"
            f"corner frame ({corner[0]:+.4f}, {corner[1]:+.4f}, "
            f"{corner[2]:+.4f}) m")

    # -- rendering --

    def _refresh_object(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        # The shells contact may target, so the rest draw muted. Resolved the same
        # way the solve resolves it, off the same params -- the picture is meant
        # to say which surface the fingertips are being sent to, and a second
        # reading of "which shells" is how it would come to say something else.
        self.scene.set_object(
            spec, center, rotation,
            contact_subset=grasp_subset_indices(spec,
                                                self.params.use_grasp_subset))
        self._refresh_object_mesh(spec, center, rotation)
        # Reference frames. The world triad is fixed, but the object's rides on
        # the pose resolved just above, so it is drawn here -- with the object
        # itself -- rather than in _render_frame: the object moves when the
        # scene is rebuilt, not once per solve iterate.
        self.scene.set_world_frame(self.g_show_world.value)
        if self.g_show_obj_frame.value:
            self.scene.set_object_frame(center, rotation)
        else:
            self.scene.clear_object_frame()
        # The slab is drawn UNCONDITIONALLY -- not gated on params.table, and not
        # on caps["table"] either. It is a physical landmark for real-robot
        # setup, and a landmark that disappears when you switch a constraint off
        # (or when a stale .so cannot build the plane) is not one you can measure
        # against. The checkbox governs the solver's half-space; this is a
        # picture of where that plane is.
        origin = self._table_origin()
        self.scene.set_table(origin, self.params.plane_normal,
                             span=TABLE_SPAN, thickness=TABLE_THICKNESS)
        # Ruled on the slab's top face, matching the grid on the physical bench,
        # so a landmark commanded to an intersection here can be read against the
        # same intersection there. Drawn with the slab for the same reason the
        # slab is drawn unconditionally: it is part of the landmark.
        if self.g_show_grid.value:
            self.scene.set_table_grid(origin, self.params.plane_normal,
                                      span=TABLE_SPAN, spacing=CAL_GRID_SPACING)
        else:
            self.scene.clear_table_grid()
        # The solver's plane, when the slider has lifted it off the bench. Gated
        # on params.table -- unlike the slab above, which is drawn unconditionally
        # because it is a physical landmark, this one is a picture of a constraint
        # and drawing it with that constraint switched off would claim a plane the
        # graph does not have. Also drawn only when it is actually somewhere else:
        # coincident with the slab it is nothing but z-fighting, and a second
        # surface exactly where the table already is would invite the very
        # confusion the split exists to remove.
        constraint = self._constraint_plane_origin()
        if (self.g_show_constraint_plane.value and self.params.table
                and abs(self.params.constraint_plane_height) > 1e-6):
            self.scene.set_constraint_plane(constraint, self.params.plane_normal,
                                            span=TABLE_SPAN)
        else:
            self.scene.clear_constraint_plane()
        corner = table_corner(origin, self.params.plane_normal)
        if self.g_show_table_frame.value:
            self.scene.set_table_frame(
                corner, label=f"table corner  {TABLE_SPAN:g} x {TABLE_SPAN:g} m")
        else:
            self.scene.clear_table_frame()
        self._refresh_table_readout(origin, corner)
        # The calibration target is measured FROM the corner just resolved, so it
        # has to be re-placed whenever the table moves -- which the object seating
        # makes it do on every object change.
        self._refresh_calibration_frame()
        if self.params.half_space:
            axis = (self.params.half_space_axis if self.params.half_space_axis is not None
                   else default_half_space_axis(spec, rotation, self.params.plane_normal))
            split = (self.params.half_space_split if self.params.half_space_split is not None
                    else center)
            self.scene.set_half_space_plane(
                split, axis, margin=self.params.half_space_margin)
        else:
            self.scene.clear_half_space_plane()

    def _render_frame(self, live=False):
        if self.result is None:
            # Nothing solved yet, so the commanded wrist pose is all there is.
            self._render_mount()
            self._report_tendon_lengths(None)
            self._update_traj()
            return
        # Render whichever solve snapshot the convergence scrubber selects; with
        # no scrubber up this is the result itself, so the gap readouts below
        # describe the intermediate state without knowing about iterates at all.
        # Every result here is a single state, so there is only ever frame 0.
        res = self._iter_view(live)
        self._render_mount(res)
        self._report_tendon_lengths(res)
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
        # Purely a picture of the posture on screen, so unlike the overlays
        # above it is gated on its own display checkbox rather than on a
        # constraint being switched on -- the plane is worth looking at exactly
        # when nothing is enforcing anything about it.
        planes = (finger_plane_witness(res, 0)
                  if self.g_show_finger_planes.value else None)
        # Same gating, one step further: the in-plane distance also needs a
        # binding that can evaluate the factor and an object with an analytic
        # ellipsoid form. planar_gap_witness returns None on either, so the
        # overlay simply does not appear rather than the render failing.
        planar = (planar_gap_witness(self.params, res, 0)
                  if (self.g_show_planar_gap.value and self.caps["planar_gap"])
                  else None)
        self._report_iterate(live)
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
                          centroid_gap=centroid_gap,
                          finger_planes=planes,
                          planar_gaps=planar)
        # Last: the plots describe the state just drawn, and they are the one
        # readout here that spans the WHOLE solve rather than this single frame.
        self._update_traj(live)

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
        if not self.g_show_finger_planes.value or self.result is None:
            return []
        names = self.result.contact_names()
        if pinch_pose(names) is None:
            return [f"**finger pinch planes: NOTHING DRAWN** -- no measured "
                    f"pinch pose for ({', '.join(names) or 'no fingers'}); the "
                    f"planes pass through that centroid, and only combinations "
                    f"including the thumb were measured"]
        return []

    def _enforce_object_contact(self, source):
        """Keep the two object-contact FORMS mutually exclusive.

        3D and in-plane are two metrics for one constraint -- one factor per
        contact finger either way -- so "both" is not a state the graph has.
        Rather than silently preferring one at build time, the box you just
        touched wins and the other clears, which is the same rule stated in the
        hints and visible the moment you click.

        ``source`` is the handle that changed. Re-entrant by construction (it
        writes the OTHER handle, whose own callback lands right back here), so
        it is latched; it also sits out :attr:`_restoring`, since Reset and the
        phase presets write both boxes as one batch and settle it themselves at
        the end.
        """
        if self._restoring or self._contact_guard:
            return
        other = (self.g_obj_contact_plane if source is self.g_obj_contact
                 else self.g_obj_contact)
        if not (source.value and other.value):
            return
        self._contact_guard = True
        try:
            other.value = False
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
        if not self.caps["planar_contact"]:
            return False, ("this binding cannot build it (no "
                           "EnvironmentConfig.object_contact_in_plane)")
        # Through resolve_scene, so the answer is read off the SAME spec the next
        # solve will build from rather than a second lookup that could disagree.
        spec = resolve_scene(self.params)[0]
        if ellipsoid_members(spec) is None:
            return False, (f"a `{spec['type']}` object has no ellipsoid "
                           f"cross-section for the pulling plane to cut")
        names = [n for n, c in zip(FINGER_LABELS, self.g_contacts) if c.value]
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
        lines.extend(self._split_axis_note())
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

    # The fingertips ride a shell of roughly this radius about the hand base, and
    # a curl can close on an object only a little smaller than it. Measured, not
    # derived -- see the reachability investigation behind GRASP_SPHERE_CENTER.
    FINGERTIP_SHELL_M = 0.055
    GRASPABLE_MAX_M = 0.050

    def _object_size_note(self):
        """Warn when the selected object is outside what the hand can close on.

        Real scanned objects are 60-300 mm; the hand closes on ~50 mm. That limit
        is GEOMETRIC, so no amount of AL/prior/beta tuning moves it -- a stall on
        a big object is the fingers not reaching, and without this line that
        reads as a solver failure and gets debugged as one. Reported for every
        object, since a hand-authored primitive can be oversized too.
        """
        spec = self.result.spec if self.result is not None else None
        if spec is None:
            return []
        if spec["type"] == "ellipsoid_set":
            largest = float(np.max(spec["extents"])) / 2.0
            smallest = float(np.min(spec["extents"])) / 2.0
        else:
            try:
                semi = proxy_semi_axes(spec)
            except ValueError:
                return []
            largest, smallest = float(np.max(semi)), float(np.min(semi))
        if largest <= self.GRASPABLE_MAX_M:
            return []
        # The narrowest axis is what a grasp actually has to span, so an object
        # that is merely LONG (a screwdriver, a pen) is still graspable across
        # its handle -- say which case this is instead of one blanket warning.
        if smallest <= self.GRASPABLE_MAX_M:
            return [f"_Object spans {2 * largest * 1000:.0f} mm at its longest "
                    f"but only {2 * smallest * 1000:.0f} mm across; grasp it on "
                    "the narrow axis._"]
        return [f"**Object is {2 * smallest * 1000:.0f} mm across its narrowest "
                f"axis** — past the ~{2 * self.GRASPABLE_MAX_M * 1000:.0f} mm the "
                f"fingertips reach off their ~{self.FINGERTIP_SHELL_M * 1000:.0f} "
                "mm shell. A stall here is geometric, not a tuning problem."]

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
                "`TendonHandSolverConfig.initial_state`  \n"
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
        self._refresh_planar_contact_gate()   # restored object may not support it
        self._sync_params()
        self._rebuild_fk()          # also drops the stepper
        self._refresh_object()
        self._aim_all_cameras()
        self._fk_solve()
        self._set_status("**reset** to defaults  \n" + self.g_status.content)

    # -- phase presets --

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
        }[field]

    def _apply_phase_preset(self, name):
        """Write ``PHASE_PRESETS[name]``'s overrides directly onto the
        corresponding GUI widgets, so checking the preset box is a single
        visible action: every affected checkbox/slider jumps to the preset's
        value on screen. One solve-ready sync/invalidate happens at the end --
        Auto solve is a separate, manual next step, not triggered here.

        The one field no preset writes here is ``contact_fingers``: the finger
        mask carries across phases untouched, see the branch below."""
        overrides = PHASE_PRESETS[name].overrides
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
                if field == "object_contact":
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
                    self.g_passive_sigma.value = math.log10(value)
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
        # The batch ran with every per-handle callback suppressed, so the
        # in-plane form's gate -- which is keyed off the object AND the finger
        # mask (it needs a measured thumb pinch pose) -- has to be re-run by
        # hand against whichever object and digits are currently selected.
        self._refresh_planar_contact_gate()
        # ...and that gate may have just cleared an in-plane box the preset asked
        # for (SDF object, or a digit set with no measured pinch pose). Falling
        # back to the 3D metric is right HERE, unlike when the user ticks the box
        # by hand: a phase preset's claim is that the object IS contacted during
        # this phase, and dropping the contact entirely would break the phase
        # rather than substitute a metric. The 3D box ticks visibly, so the panel
        # still says exactly what is in the graph.
        if (overrides.get("object_contact")
                and not (self.g_obj_contact.value or self.g_obj_contact_plane.value)):
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
        """Every phase-preset checkbox, name -> handle. Small and built on
        demand rather than cached, so a future phase3 checkbox only needs
        adding here (and to ``_build_gui``/``_input_handles``)."""
        return {"phase0": self.g_phase0, "phase1": self.g_phase1,
                "phase2": self.g_phase2, "phase4": self.g_phase4,
                "phase5": self.g_phase5}

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

    # -- control-trajectory panel (left-docked plot window) --
    #
    # The six things this robot is commanded with -- five actuated tendons and
    # one wrist pose -- plotted against the iteration the solve is on, in the
    # window traj_panel.TrajectoryPanel owns. Everything below is
    # EXTRACTION: the panel is handed plain numbers and knows nothing about
    # results or iterates, which is what keeps the solver vocabulary on this side
    # of the line and the plotting on that one.

    def _traj_row(self, res):
        """The eleven control numbers of ONE solved state, in panel order:
        five actuated tendon lengths in mm, then the wrist as xyz (m) + rpy (rad).

        The LENGTH, not the tension that produced it. The tension is what the
        solve was asked for; the length is what the hand took in, it is the L
        half of the state a Section 1.8 control tick anchors on, and it is what
        the hardware is actually commanded on -- `robot_plan.build_plan` turns
        each waypoint into ``open_lengths[name] - length``. Same array
        `_report_tendon_lengths` prints under the tension sliders, and in the
        same mm, so the plot and that readout cannot disagree.

        Everything is re-read from the RESULT rather than from the sliders that
        commanded it, for the reason `_report_tendon_lengths` gives: past the
        first iterate neither the wrist nor the tendon is the slider's any more.
        The wrist is a variable with a soft prior, so a contact solve ends a long
        way from the commanded pose -- which is precisely the drift this panel
        exists to make visible. Reading the sliders would draw flat lines.

        The wrist also has to be RECOVERED rather than read: nothing in a result
        reports it directly, so `solved_wrist_pose` inverts finger 0's base
        offset out of its node-0 pose. Split into xyzrpy here because a 4x4 is
        not plottable, using the same ZYX convention (and the same radians) the
        Wrist start pose sliders use, so a number read off a plot goes straight
        back into the slider it came from.
        """
        T = np.asarray(solved_wrist_pose(self.fk_solver.configs, res.frames[0]),
                       float)
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        lengths = [float(np.asarray(length, float)[FLEXOR_IDX]) * 1e3
                   for length in res.tendon_lengths(0)]
        return lengths + [T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw]

    def _robot_traj_row(self, state):
        """One MEASURED robot state as the panel's eleven channels.

        The exact inverse of what `_traj_row` reads off a solve, in the same
        units and the same order, because the whole point is to draw the two on
        one axis: five actuated tendon LENGTHS in mm, then the wrist as xyz (m)
        and rpy (rad) in the viser world frame.

        The hardware reports DISPLACEMENT from the hand-open pose and
        `robot_plan.build_plan` commands it as ``open_lengths[name] - length``,
        so recovering a length is ``open_lengths[name] - displacement`` -- the
        same identity read backwards. Doing it here rather than plotting the
        displacement directly is what makes the measured line comparable to the
        commanded one instead of being a differently-zeroed cousin of it.

        A finger the hardware could not report is left NaN rather than zeroed: a
        motor whose position read failed is a hole in the measurement, and
        `spanGaps: False` on the series draws it as one. Zero would draw as a
        fully open finger, which is a claim about the hand rather than an
        admission that nothing was heard.
        """
        T = np.asarray(state.wrist_pose, float)
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        open_lengths = self._open_lengths()
        # The ORDER has to be the result's own, because that is the order
        # `_traj_row` reads `tendon_lengths(0)` in and therefore the order the
        # panel's first five channels are in. Falling back to FINGER_LABELS only
        # covers the case where nothing is solved, where there is no plot to
        # align with anyway.
        names = (list(self.result.finger_names)
                 if self.result is not None else list(FINGER_LABELS))
        lengths = []
        for name in names:
            disp = state.tendon_disp.get(name)
            lengths.append(np.nan if disp is None or name not in open_lengths
                           else (open_lengths[name] - float(disp)) * 1e3)
        # Fixed to five, so a result carrying a different number of digits can
        # never slide the wrist channels along and draw them on a tendon plot.
        lengths = (lengths + [np.nan] * 5)[:5]
        return lengths + [T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw]

    def _sample_robot_trace(self, feedback):
        """Record where the robot IS, against the waypoint the plan is heading to.

        Called about ten times a second off the playback feedback, on the action
        client's callback thread. Cheap on purpose -- two TF lookups and a cached
        tendon reading -- because it runs while the robot is moving and must not
        become a reason the feedback path falls behind.

        Keyed by WAYPOINT rather than by time, and the last sample for a waypoint
        wins. That makes entry ``k`` "where the machine was as the reference
        finished waypoint k", which is exactly the quantity the plot is being
        asked for: the gap between the two lines at ``k`` is the tracking error
        at that waypoint. Time would have to be resampled onto the iterate grid
        to be plotted at all, and would answer a question nobody asked.

        The feedback's waypoint index is already the CLIENT's -- the executor
        takes its own prepended approach waypoint back off -- so it indexes the
        iterates directly, with no offset to get wrong here.
        """
        trace = self._robot_trace
        if trace is None:
            return          # not collecting: not a history playback
        try:
            state = self.bridge.read_state(self._corner_viz())
            trace[int(feedback.waypoint)] = self._robot_traj_row(state)
        except Exception:
            # Diagnostics on the feedback path of a moving robot. A failed TF
            # lookup mid-run is a missing sample, not a reason to raise into the
            # action client's callback.
            pass

    def _robot_trace_array(self, n):
        """The recorded trace as an ``(n, 11)`` array, NaN where nothing landed.

        Returns None when there is nothing to draw, which the panel takes as "no
        measured line" -- distinct from an all-NaN array, which would mean a
        playback happened and recorded nothing.
        """
        trace = getattr(self, "_robot_trace", None)
        if not trace:
            return None
        out = np.full((n, self.traj.N_CHANNELS), np.nan)
        hit = False
        for index, row in trace.items():
            if 0 <= index < n:
                out[index] = row
                hit = True
        return out if hit else None

    def _traj_samples(self):
        """The whole trajectory on screen as an ``(N, 11)`` array.

        The recorded AL iterates when there are any -- sample 0 is where the run
        started, which under the warm-start latch IS the FK pose that was on
        screen when Step was first pressed, i.e. the current kinematics -- and
        the single solved state when there are none, which is what an FK pose
        is. So the panel shows a lone dot after FK and grows a line from it as
        the solve steps, with no special-casing at the boundary.

        Recomputed in full on every render rather than appended to. Measured at
        1.1 ms over a 26-iterate solve -- 1.6 ms including the eleven uplot
        pushes, against a `_render_frame` that costs 52 ms -- so it is 3% of a
        redraw, and nowhere near worth a cache that would have to know about cold
        restarts, Close/Lift overwriting the history, and the scrubber: three
        separate ways to serve a stale plot.

        Close and Lift record their ramp substeps as iterates too, so those get
        plotted by exactly the same path: a phase-4 close draws as five tendon
        lengths ramping together, which is the claim in its name made visible."""
        res = self.result
        if res is None or getattr(self, "fk_solver", None) is None:
            return np.zeros((0, self.traj.N_CHANNELS))
        n = res.num_iterates()
        views = [res.at_iterate(i) for i in range(n)] if n else [res]
        return np.array([self._traj_row(v) for v in views], float)

    def _traj_cursor(self, n, live):
        """Which sample the 3D view is showing, for the marker dot.

        Follows `_iter_view` exactly, because agreeing with it is the point: the
        dot claims "this plotted sample is the hand you are looking at", and the
        one case where that is easy to get wrong is mid-run, where the scrubber
        still describes the PREVIOUS solve and the render is drawing the newest
        state regardless."""
        if n == 0:
            return None
        if live or getattr(self, "iter_slider", None) is None:
            return n - 1
        return min(self._current_iterate(), n - 1)

    def _update_traj(self, live=False):
        """Redraw the trajectory panel for whatever is on screen.

        Called from `_render_frame`, so it follows the live re-solve, every AL
        step (from the auto-solve worker thread, which is safe -- assigning to a
        uPlot handle only queues a message) and the convergence scrubber alike.

        Exceptions are caught and shown IN the panel rather than raised: this is
        a readout on the render path, and that path runs inside the auto-solve
        loop, so a raise here would end a solve over a plotting bug."""
        panel = getattr(self, "traj", None)
        if panel is None:
            return          # rendered before __init__ built it
        try:
            values = self._traj_samples()
            n = len(values)
            panel.update(
                values, cursor=self._traj_cursor(n, live),
                actual=self._robot_trace_array(n),
                # Worth saying only in the case that looks like a broken panel:
                # one dot and no line is a correct picture of an FK pose.
                note=("current kinematics (FK pose)"
                      if n == 1 and self.result.num_iterates() == 0 else ""))
        except Exception as exc:
            panel.error(exc)

    TENDON_IDLE = "_press **FK** to read the actuated tendon lengths_"

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

    def _corner_viz(self):
        """The table square's minimum corner in viser world coordinates -- this
        app's half of the registration against the physical bench.

        Read live rather than cached: the support plane is seated UNDER the
        object (see ``auto_table_origin``), so switching objects moves it, and a
        stale corner would silently offset every pose sent to the robot by
        however far the table had moved."""
        return table_corner(self._table_origin(),
                            np.asarray(self.params.plane_normal, float))

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
                (f"**ARMED** -- the next *Play* moves the arm and the hand"
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

    # How far past a measured wrist value to grow the slider when the value falls
    # outside its range, as a fraction of how far outside it fell -- headroom
    # enough that the widened slider is still draggable either side of where the
    # robot actually is, rather than pinned against its own new end stop.
    _WRIST_RANGE_MARGIN = 0.25

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

    # Bisection budget for the tension recovery below. 14 halvings of the 0-3 N
    # slider range resolve tension to 0.2 mN, far finer than the ~0.1 mm of
    # displacement the hardware can distinguish, so the tolerance is what
    # actually ends it.
    _TENSION_BISECT_STEPS = 14
    _TENSION_BISECT_TOL_M = 5e-4

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

    def _input_handles(self):
        """Every value-carrying control, in build order. Buttons and markdown are
        deliberately absent -- Reset restores values, not widgets."""
        return ([self.g_object, self.g_contact_shells,
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
                   self.g_show_true_mesh,
                   self.g_show_contact, self.g_show_collision,
                   self.g_show_discs, self.g_show_disc_frames,
                   self.g_show_world, self.g_show_obj_frame,
                   self.g_show_table_frame, self.g_show_grid,
                   self.g_show_gaps, self.g_show_mount,
                   self.g_show_finger_planes, self.g_show_planar_gap,
                   self.g_show_traj])

    def _build_gui(self):
        gui = self.server.gui
        # Map the displayed dropdown label back to the real spec key (identity
        # except for the "_sdf"-suffixed baked spheres, and the "ycb:"-prefixed
        # ellipsoid sets, which keep their prefix as the label so they group
        # together and read as "not one of the hand-authored primitives").
        labels, self._label_to_key = self._object_dropdown_labels()

        step_hint = (None if self.caps["ik_stepping"]
                     else "requires a rebuilt _gepetto_solvers with "
                          "TendonHandSolver.reset_al_duals")

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
                "close depth (fraction)", 0.1, 1.0, 0.05, CLOSE_FRACTION,
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
                "lift height (m)", 0.0, 0.3, 0.01, LIFT_HEIGHT_M,
                hint=f"How far up the wrist goes, along world +Z. Split into "
                     f"{LIFT_STEPS} equal steps whatever the height, so the "
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
                           "TendonHandSolverConfig.initial_state"))
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
                           "TendonHandSolver.set_initial_duals"))
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
            # The sliders above are a demo pose. This is the measured one: put the
            # wrist here and the viser world origin becomes the robot flange, so
            # the hand hangs exactly as it does in the CAD assembly.
            self.g_mount = gui.add_button(
                "Pose at measured robot mount",
                hint="Set the six sliders to mount.MOUNT_WRIST_XYZ/RPY -- the "
                     "wrist pose measured from the Onshape assembly. The world "
                     "origin then IS the flange, and 'mount frames' below draws "
                     "both frames so you can check the hand sits on the arm the "
                     "way it does in CAD.")

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
                for lbl in FINGER_LABELS]
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
                _pinch_default = {"index", "middle", "thumb"}
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
                    for lbl in FINGER_LABELS]

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
                hint="Draw the wrist frame and, offset from it by the measured "
                     "mount transform, the robot flange frame the hand bolts to. "
                     "Use with 'Pose at measured robot mount' to check the "
                     "measurement against the CAD assembly by eye.")
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
        self.g_close.on_click(self._close_hand)
        self.g_lift.on_click(self._lift_hand)

        self.g_object.on_update(self._on_object_selected)

        # Live FK re-solve on the pose / tension sliders (fast, warm-started).
        for h in ([self.g_tx, self.g_ty, self.g_tz, self.g_roll, self.g_pitch,
                   self.g_yaw, self.g_passive] + self.g_flexors):
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
                  self.g_show_disc_frames,
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
    print(f"gepetto_solvers: {binding_path()}")
    missing = [k for k, v in app.caps.items() if not v]
    if missing:
        print(f"  capabilities MISSING from this build: {', '.join(missing)}")
    # The landmark's dimensions, printed so they can be copied into a real-robot
    # setup without opening the browser. The corner frame's position is scene
    # state (it follows the object-seated plane), so that one lives in the GUI.
    print(f"table square: {TABLE_SPAN:.3f} x {TABLE_SPAN:.3f} m, "
          f"{TABLE_THICKNESS * 1e3:.0f} mm thick -- top face is the constraint "
          f"plane, frame on its -X/-Y corner")
    print(f"viser hand visualizer running -- open http://localhost:{args.port}")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
