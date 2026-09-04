""":class:`HandVizApp` -- the composed viser workbench.

Construction and the run loop. Every panel of behaviour lives in one of the
mixins below, which share the attributes ``__init__`` sets up here.
"""



from gepetto_solvers.core.hands import get_hand
from gepetto_solvers.core.solvers import capabilities

from ._calibration import CalibrationMixin
from ._gui import GuiMixin
from ._motion import MotionMixin
from ._notes import NotesMixin
from ._objects import ObjectPanelMixin
from ._params import ParamsSyncMixin
from ._phases import PhaseMixin
from ._render import SceneRenderMixin
from ._robot import RobotMixin
from ._stepping import StepperMixin
from ._trajectory import TrajectoryMixin
from .estop import EStop


class HandVizApp(
    ObjectPanelMixin,
    SceneRenderMixin,
    ParamsSyncMixin,
    NotesMixin,
    StepperMixin,
    MotionMixin,
    PhaseMixin,
    TrajectoryMixin,
    RobotMixin,
    CalibrationMixin,
    GuiMixin,
):
    def _link_meshes(self):
        """The hand's visual link geometry, loaded once and remembered.

        VISUAL ONLY -- collision is the sphere set the solve carries, and the
        factor graph never sees a mesh. A hand that supplies none draws as a
        skeleton, which is a complete drawing in its own right.
        """
        if self._link_mesh_cache is None:
            supplier = getattr(self.hand, "visual_meshes", None)
            self._link_mesh_cache = supplier() if supplier is not None else []
        return self._link_mesh_cache

    def has(self, feature):
        """Whether the hand being posed supports ``feature``.

        The panel-level counterpart of ``self.caps``, and the two answer
        different questions: caps says what the compiled binding can do, this
        says what this ROBOT can do. A control for something the hand does not
        have is not greyed out here -- the whole folder is absent, because a
        tendon-length readout on a hand with no tendons is not a disabled
        feature, it is a category error.
        """
        return feature in self.hand.features

    def _drive_index(self):
        """Index of the actuated actuator in a digit's tendon-length/tension
        vector, from the hand being posed. Every displacement and commanded
        tension the panel reads or writes is at this index."""
        return self.hand.actuation.drive_indices[0]

    def __init__(self, server, ros_mode=False, bridge=None, hand=None):
        import viser  # local import so --smoke needs no viser
        self.viser = viser
        self.server = server
        # The hand this workbench poses. Everything downstream -- the digit
        # checkboxes, the tension sliders, the scene's finger channels, the
        # calibration landmark list -- is sized and named off it, so --hand
        # selects a different mechanism without a control here changing.
        self.hand = hand if hand is not None else get_hand()
        self.digit_names = list(self.hand.digit_names)
        self._link_mesh_cache = None
        # ROS mode adds the Robot folder -- play a solve on the hardware, read the
        # hardware back -- and extends the e-stop to the servo publishers. Off by
        # default so the standalone app is byte-for-byte the app it was; the
        # bridge is DUCK-TYPED (play / read_state / stop / status), so nothing in
        # crest-sparse imports rclpy and the ROS side stays in epfl_hand_control.
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
        self.scene = ViserHandScene(server, self.digit_names)

        # The control-trajectory plots, in their own window docked to the LEFT of
        # the 3D view (the main control panel is on the right, so the two do not
        # compete for the same edge). Built before _build_gui because the Display
        # folder's visibility checkbox needs something to toggle; it is a
        # top-level entity, so it is not placed in any folder that happens to be
        # open. See _plotting/traj_panel.py.
        from gepetto_solvers.core.plotting.traj_panel import TrajectoryPanel
        # Sized by the HAND: one row per driven actuator per digit, in the units
        # `robot_plan` commands that hand in. The tendon hand drives one tendon
        # per digit and gets its five length rows in mm; the Allegro drives four
        # joints and gets sixteen angle rows in rad.
        if self.has("displacement"):
            actuators, unit, fmt = ("tendon",), "mm", "7.2f"
        else:
            actuators = tuple(self.hand.actuation.names[i]
                              for i in self.hand.actuation.drive_indices)
            unit, fmt = "rad", "+7.3f"
        self.traj = TrajectoryPanel(server, self.digit_names, actuators, unit,
                                    digit_format=fmt)
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
        self._refresh_exact_contact_gate()
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
