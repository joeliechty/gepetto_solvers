"""Tunables, labels and tolerances the visualizer is built from.

Separated so a mixin can state exactly which of them it depends on, and so the
smoke tolerances sit next to the numbers they judge.
"""



from gepetto_solvers.core.solvers import euler_to_R


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
