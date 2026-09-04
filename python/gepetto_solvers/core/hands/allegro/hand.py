""":class:`AllegroHand` -- the 16-DOF Allegro hand, posed in joint space."""

from __future__ import annotations

import numpy as np

import gepetto_solvers

from ..base import Actuation, opposing_index_of
from . import meshes as allegro_meshes
from . import spec as allegro


class DigitEnv:
    """Per-digit carrier for the task environment.

    The ``attach_*`` family mutates ``sdf_contact`` / ``sphere_contact`` in
    place, and that is the whole of what it needs from a "config". On the tendon
    hand those fields live on a ``TendonFingerSolverConfig`` alongside the rod
    geometry; a joint-space hand keeps its mechanism in the URDF, so this holds
    the task half and nothing else.

    Deliberately NOT a ``TendonFingerSolverConfig`` with unused fields: the
    environment layer would then be reading rod lengths and tendon counts off a
    hand that has neither, and the first person to trust one of those numbers
    would be badly misled.
    """

    __slots__ = ("sdf_contact", "sphere_contact")

    def __init__(self):
        self.sdf_contact = None
        self.sphere_contact = None


class AllegroHand:
    """Four fingers of four revolute joints each, posed by URDF kinematics.

    Actuation is JOINT POSITION: all four joints of a digit are commanded, so
    unlike the tendon hand there is no single "the" driven value, and no
    displacement readout distinct from it. That is what ``features`` says, and
    what the workbench gates its tendon-shaped panels on.
    """

    name = "allegro"
    kinematics = "rigid_urdf"

    opposing_digit: str | None = allegro.OPPOSING_DIGIT

    #: The three-digit precision grasp, matching the tendon hand's default.
    default_contact_digits = ("index", "middle", "thumb")

    actuation = Actuation(
        n=allegro.DOF_PER_DIGIT,
        names=("j0", "j1", "j2", "j3"),
        # Every joint is motor-driven. This is the first hand to drive more than
        # one actuator per digit, which is why anything reading
        # drive_indices[0] has to be gated on the "single_drive" feature.
        drive_indices=tuple(range(allegro.DOF_PER_DIGIT)),
    )

    #: What this hand has, from hands.base.FEATURES. Almost nothing: no tendons,
    #: no single driven actuator, no displacement distinct from position, no rod
    #: to bend, no measured pinch table, no disc landmarks and no measured close
    #: ramp. Each of those gates a panel off rather than leaving a control
    #: present and dead.
    #:
    #: ``robot_plan`` IS declared: this hand has a robot behind it. A plan for it
    #: carries the four joint positions of each digit in radians, where the tendon
    #: hand's carries one displacement in metres -- see
    #: ``robot_plan.building.command_kind``, which reads exactly the
    #: ``"displacement"`` feature above to tell the two apart.
    features: frozenset[str] = frozenset({"robot_plan"})

    #: The posture the workbench opens on: an OPEN pre-grasp -- the fingers
    #: barely flexed, the thumb swung across the palm to oppose them, and the
    #: hand held around the object rather than already touching it.
    #:
    #: A PRE-GRASP, not a seated grasp, and that is the change from the earlier
    #: default (finger j1 = 0.6, thumb (1.2, 0.3, 0.6, 0.4)): the three grasp
    #: fingertips now ring a circle of radius 84 mm about a point 103 mm out in
    #: front of the palm, so against the workbench's own default object (the
    #: 70 mm megaminx, see viz_interactive.constants.DEFAULT_OBJECT_PRIMITIVE)
    #: every grasp digit opens about 26-28 mm clear of the surface, with the
    #: uncommanded ring finger 50 mm clear. The old posture rang a 49.6 mm
    #: circle, which seated exactly on the smaller ``mid_sphere_ellipsoid`` that
    #: HandSolveParams defaults to but sat 8.5 mm INSIDE the object the app
    #: actually opens on -- a start the phase-0 pre-grasp cannot be run from,
    #: because there is nothing left to approach.
    #:
    #: Round numbers, and on the sliders' own grids (0.01 rad), so the workbench
    #: opens on a value the user can land on again by hand.
    #:
    #: The V5 thumb reaches its opposition from joint_12, whose range is
    #: 0..1.78 rad -- a whole quadrant, where V4 gave it only +-0.47 and could
    #: not truly oppose. 0.99 rad brings it across the palm with room either
    #: way, rather than parking the default on a joint stop where a solve cannot
    #: move.
    DEFAULT_FINGER_Q = (0.0, 0.13, 0.6, 0.4)
    DEFAULT_THUMB_Q = (0.99, 0.38, 0.49, 0.4)

    #: Palm-down hover over the default object.
    #:
    #: Allegro's fingers extend along +z at the identity wrist and flex toward
    #: +x, where the tendon hand's palm lies along -x -- so the two hands need
    #: different poses to face the same object, and the tendon hand's constants
    #: in `solvers.frames` aim this one nowhere near it. A quarter turn about +y
    #: would put the palm's normal straight down the world -z; the extra
    #: 0.33 rad of pitch tips the palm back off vertical so the opened fingers
    #: above reach DOWN AND ACROSS the object rather than past it, and the
    #: translation then holds the centre of their circle on the object.
    #:
    #: Paired with the posture above -- the two were calibrated together in the
    #: workbench, so changing one without the other loses the pre-grasp.
    DEFAULT_WRIST_RPY = (0.0, 1.8984, 0.0)
    DEFAULT_WRIST_XYZ = (-0.017, 0.0673, 0.1004)

    #: log10 of the joint-prior sigma the workbench's "log10 joint sigma" slider
    #: opens on, and returns to on Reset. Optional per hand (the workbench falls
    #: back to its own -2 for a hand that does not say), and 0 here -- a 1 rad
    #: sigma, i.e. a posture that is a suggestion -- because the pre-grasp above
    #: is deliberately OPEN: the joints have to be free to travel most of the
    #: way closed under contact, and a tight prior would hold them open against
    #: it.
    DEFAULT_LOG10_JOINT_SIGMA = 0.0

    #: ``T_flange<-palm_link``: where this hand bolts to the robot arm.
    #:
    #: MEASURED on the robot, not fitted in CAD -- read straight off the live
    #: TF tree with the hand's own driver running::
    #:
    #:     ros2 run tf2_ros tf2_echo lbr_link_ee palm_link
    #:     - Translation: [0.000, 0.000, 0.130]
    #:     - Rotation: in Quaternion (xyzw) [0.000, 0.000, 0.000, 1.000]
    #:
    #: So the hand sits 130 mm straight out the flange's +z with its axes
    #: aligned to it: no rotation to get wrong, which is why this needs no
    #: candidate-scoring machinery like the tendon hand's Onshape fit.
    MOUNT_FLANGE_XYZ = (0.0, 0.0, 0.130)
    MOUNT_FLANGE_RPY = (0.0, 0.0, 0.0)

    #: Ceiling on how fast ONE driven actuator may move, in the units this hand's
    #: plans are commanded in -- radians of joint per second here, the analogue of
    #: the tendon hand's ``max_tendon_speed``.
    #:
    #: Deliberately conservative and NOT a hardware limit: Wonik's driver runs its
    #: own ~333 Hz joint-PD loop and publishes no rate cap for us to read, so this
    #: is a PATH limit -- how fast the reference a plan walks is allowed to move.
    #: 1.0 rad/s takes the widest V5 joint (joint_12's 1.78 rad quadrant) end to
    #: end in under two seconds, which is brisk for a grasp and slow enough to
    #: watch. Raise it once a run has been watched on the real hand.
    max_digit_speed = 1.0

    def driver_joint_names(self):
        """``{digit: [joint name per DOF]}`` for the ROS driver.

        The one table the Allegro bridge needs, and it is NOT re-derivable by
        arithmetic on the digit index: ``/allegroHand/joint_cmd`` is positional,
        and Allegro's own numbering puts the thumb at 12-15 while several other
        orderings would look equally plausible. See
        :data:`.spec.DRIVER_JOINT_ORDER`.
        """
        return allegro.driver_joint_names()

    def __init__(self):
        self.digit_names = list(allegro.DIGIT_NAMES)
        # Fingertip contact radius: the sphere the contact constraint drives
        # onto a surface. MEASURED off the V5 fingertip mesh, in the
        # `link_*_tip` frame the solve actually poses. That frame sits inside
        # the fingertip, and its surface is 14.2 mm away on the PALMAR side --
        # which is the side a grasp closes onto -- against 12.0 mm at the distal
        # pole and 13-15 mm laterally. 14 mm is the pad.
        #
        # V4's tips were 12 mm across and this was 0.006. The gap a contact
        # constraint drives to zero is (distance to the surface - this radius),
        # so the 8 mm difference moves every fingertip 8 mm further out: a
        # default posture calibrated against the old number seats the tips
        # inside the object.
        self.tip_radii = [0.014] * len(self.digit_names)

    # -- digits ------------------------------------------------------------

    def digit_configs(self):
        """A fresh task-environment carrier per digit.

        Fresh every call for the same reason the tendon hand rebuilds its
        configs: ``attach_*`` mutates them in place, so a shared list would leak
        one solve's constraints into the next.
        """
        return [(name, DigitEnv()) for name in self.digit_names]

    def contact_node(self, digit: int) -> int:
        return allegro.contact_site()

    def collision_sites(self, digit: int):
        return allegro.collision_sites()

    # -- measured tables ---------------------------------------------------

    def pinch_pose(self, mask):
        """None: no pinch geometry has been measured for this hand.

        Honest rather than absent -- the pre-grasp centroid constraint asks for
        this and must be able to find out that there is no answer. Running
        ``scripts/fk_pinch_centroids.py`` against this hand would produce one.
        """
        return None

    # -- where it starts ---------------------------------------------------

    def default_pose(self):
        from ...solvers.frames import wrist_pose_from_xyzrpy

        wrist = wrist_pose_from_xyzrpy(self.DEFAULT_WRIST_XYZ,
                                       self.DEFAULT_WRIST_RPY)
        means = [np.array(self.DEFAULT_FINGER_Q, float) for _ in self.digit_names]
        means[self.digit_names.index("thumb")] = np.array(self.DEFAULT_THUMB_Q,
                                                          float)
        return wrist, means

    def mount_pose(self):
        """``T_flange<-wrist`` as a 4x4, from the measurement above.

        The measurement is flange-to-``palm_link``, and this returns
        flange-to-WRIST -- two names for the same frame ONLY because the V5
        description's root (a bare ``world`` link) is joined to ``palm_link`` at
        the origin, so ``T_root<-palm_link`` is the identity. The rigid
        kinematics resolves each digit's mount against the model ROOT, which is
        what the solver's wrist variable is.

        That coincidence is a property of THIS description, not of Allegro: the
        V4 file put ``palm_link`` 95 mm up the root's +z, where this would have
        to compose ``inv(T_root<-palm_link)`` in. ``tests/core/
        test_allegro_hand.py`` asserts the two frames still coincide, so a
        variant that separates them fails there rather than silently mounting
        the hand 95 mm off.

        Fresh array per call: callers assign it into
        ``HandSolveParams.wrist_pose`` and mutate poses in place.
        """
        from ...solvers.frames import wrist_pose_from_xyzrpy

        return wrist_pose_from_xyzrpy(self.MOUNT_FLANGE_XYZ,
                                      self.MOUNT_FLANGE_RPY)

    def visual_meshes(self):
        """``[(attach, path)]`` for the hand's link meshes, or ``[]``.

        ``attach`` is None for the palm (which rides on the wrist) or
        ``(digit, site)`` for a link. Visual only -- the solve uses its own
        sphere set -- so an empty list costs the picture its skin and nothing
        else, and the renderer falls back to the skeleton.
        """
        return allegro_meshes.visual_meshes()

    def joint_limits(self):
        """Per-digit ``[(lo, hi)]`` from the URDF, one pair per joint.

        Read through the joint NAMES, never by arithmetic on the digit index:
        Pinocchio orders Allegro's configuration index, thumb, middle, ring --
        not the digit order -- so ``digit * 4 + joint`` silently returns another
        digit's limits. The solver has always looked these up by name; this is
        here so the workbench does too.

        The limits are per DIGIT on the V5 hand, not shared: the index finger
        abducts +-0.3 rad where the middle and ring get +-0.26, and their
        flexion ranges differ too. A caller that reads one digit's pair and
        applies it to the others clips a joint that had room.
        """
        chain = gepetto_solvers.RigidChainModel.from_urdf_file(
            str(allegro.URDF_PATH))
        lo, hi = chain.lower_position_limits, chain.upper_position_limits
        out = []
        for spec in allegro.digit_specs():
            out.append([(lo[chain.joint_indices(j)[0]],
                         hi[chain.joint_indices(j)[0]]) for j in spec.joints])
        return out

    # -- actuation ---------------------------------------------------------

    def actuation_means(self, params):
        """Per-digit commanded joint positions, i.e. q_S in p(q).

        Reads ``params.joint_targets`` -- ``flexor_tensions`` is one scalar per
        digit and cannot say where four independent joints should go. Falls back
        to the neutral configuration when nothing is commanded, which is the
        open hand.
        """
        n = len(self.digit_names)
        targets = getattr(params, "joint_targets", None)
        if targets is None:
            return [np.zeros(self.actuation.n) for _ in range(n)]
        if len(targets) != n:
            raise ValueError(
                f"joint_targets has {len(targets)} entries but this hand has "
                f"{n} digits")
        means = []
        for i, q in enumerate(targets):
            q = np.asarray(q, float)
            if q.shape != (self.actuation.n,):
                raise ValueError(
                    f"joint_targets[{i}] has shape {q.shape}, expected "
                    f"({self.actuation.n},) -- one value per joint of digit "
                    f"{self.digit_names[i]!r}")
            means.append(q)
        return means

    # -- the C++ side ------------------------------------------------------

    @property
    def opposing_index(self) -> int:
        return opposing_index_of(self.digit_names, self.opposing_digit)

    def build_spec(self, configs, params=None):
        """The C++ ``HandSpec``: the task envs off ``configs``, and a
        ``rigid_urdf`` payload naming this hand's URDF, joints and site frames.

        ``params`` seeds the configuration. Seeding at the SAME posture the
        joint prior is centred on matters: the solve then starts at zero
        kinematics residual and converges in one iteration, where a seed a few
        tenths of a radian away costs tens of iterations for the same answer.
        """
        q_init = (self.actuation_means(params) if params is not None
                  else [np.zeros(self.actuation.n) for _ in self.digit_names])

        spec = gepetto_solvers.HandSpec()
        spec.kinematics = self.kinematics
        spec.digit_names = list(self.digit_names)
        spec.opposing_digit = self.opposing_index
        spec.env = [cfg.sdf_contact for _, cfg in configs]
        spec.sphere_contact = [cfg.sphere_contact for _, cfg in configs]
        spec.kinematics_config = allegro.kinematics_config(
            q_init=[list(map(float, q)) for q in q_init])
        return spec
