""":class:`AllegroHand` -- the 16-DOF Allegro hand, posed in joint space."""

from __future__ import annotations

import numpy as np

import gepetto_solvers

from ..base import Actuation, opposing_index_of
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

    #: What this hand has, from hands.base.FEATURES. Nothing: no tendons, no
    #: single driven actuator, no displacement distinct from position, no rod to
    #: bend, no measured pinch table, no disc landmarks, no hardware bridge and
    #: no measured close ramp. Each of those gates a panel off rather than
    #: leaving a control present and dead.
    features: frozenset[str] = frozenset()

    #: The posture the workbench opens on: fingers flexed, thumb abducted across
    #: the palm to oppose them.
    #:
    #: MEASURED, within the URDF's joint limits and deliberately off them. The
    #: thumb has to come a long way to oppose -- joint_12 abducts only +-0.47 rad
    #: -- and driving it to the stop closes the three-digit spread to 33 mm but
    #: leaves the default sitting exactly on two limits, with no room for a solve
    #: to move. Backed off to round numbers inside them, the spread is 37 mm,
    #: which against a 35 mm-radius object puts every grasp fingertip within a
    #: few millimetres of the surface.
    DEFAULT_FINGER_Q = (0.0, 0.8, 0.8, 0.8)
    DEFAULT_THUMB_Q = (0.45, 0.80, 1.50, 0.50)

    #: Palm-down hover over the default object.
    #:
    #: Allegro's fingers extend along +z at the identity wrist, where the tendon
    #: hand's palm lies along -x -- so the two hands need different poses to face
    #: the same object, and the tendon hand's constants in `solvers.frames` aim
    #: this one nowhere near it. Pi about +y turns the fingers to -z; the
    #: translation then puts the centroid of the grasp digits on the object at
    #: the posture above.
    DEFAULT_WRIST_RPY = (0.0, np.pi, 0.0)
    DEFAULT_WRIST_XYZ = (0.0624, 0.0625, 0.0970)

    def __init__(self):
        self.digit_names = list(allegro.DIGIT_NAMES)
        # Fingertip contact radius. Allegro's tips are ~12 mm across; half of
        # that is the sphere the contact constraint drives onto a surface.
        self.tip_radii = [0.006] * len(self.digit_names)

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

    def joint_limits(self):
        """Per-digit ``[(lo, hi)]`` from the URDF, one pair per joint.

        Read through the joint NAMES, never by arithmetic on the digit index:
        Pinocchio orders Allegro's configuration index, thumb, middle, ring --
        not the digit order -- so ``digit * 4 + joint`` silently returns another
        digit's limits. The solver has always looked these up by name; this is
        here so the workbench does too.
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
