""":class:`TendonHand5F` -- the five-digit, six-tendon anatomical hand."""

from __future__ import annotations

import numpy as np

import gepetto_solvers

from ...geometry.scene import GRASP_FLEXOR_TENSION
from ..base import Actuation, HardwareMap, MotionProfile, opposing_index_of
from .dimensions import (
    default_hand_tip_radii,
    load_hand_dimensions,
)
from .discs import disc_node_indices, proximal_disc_flags
from .morphology import get_default_hand_configs, tip_node_index
from .pinch import pinch_pose_for_mask

#: The six-tendon routing, proximal to distal. Five spring-backed passives and
#: one motor-driven flexor -- see :mod:`.finger_config` for the routing matrix
#: that produces them.
TENDON_NAMES = ("Lateral+", "Lateral-", "Abduct+", "Abduct-", "Extensor", "Flexor")

#: Index 5 is the actuated flexor. Declared once, here, on the hand it belongs
#: to; it used to be a module constant named twice in two packages and
#: cross-checked at runtime because nothing else could keep them in agreement.
FLEXOR_INDEX = 5


class TendonHand5F:
    """Four fingers and a thumb, each a Cosserat rod driven by six tendons.

    The digits are built from a CAD dimension table (``gepetto_core`` when it is
    importable, a pinned fallback otherwise -- see :func:`load_hand_dimensions`),
    and the thumb is appended LAST. That order is load-bearing everywhere a
    per-digit list is indexed, so it is fixed here and read from
    :attr:`digit_names` rather than re-derived.
    """

    name: str = "tendon_5f"
    kinematics: str = "tendon"

    # Annotated as optional, not inferred as `str`: the Hand protocol declares it
    # `str | None` (a hand may have no opposing digit) and a protocol's attributes
    # are invariant, so a bare `= "thumb"` would narrow this class out of
    # satisfying its own interface.
    opposing_digit: str | None = "thumb"

    #: Which digits a grasp starts with when a caller names none. The measured
    #: three-digit pinch (index, middle, thumb) -- the combination the pinch
    #: table is tightest for.
    default_contact_digits = ("index", "middle", "thumb")

    actuation = Actuation(
        n=len(TENDON_NAMES),
        names=TENDON_NAMES,
        drive_indices=(FLEXOR_INDEX,),
    )

    #: Measured on this hand: at 2.0 N the index has taken in 5.8 mm, the middle
    #: 7.4 mm and the thumb 10.2 mm, and the thumb stops at ~8.0 mm around 1.7 N.
    #: The ramp is spaced off a probed slope rather than these numbers, but they
    #: are what set the step count and tolerance.
    motion = MotionProfile()

    #: Everything in the vocabulary: this is the hand the whole workbench was
    #: built around, so every panel applies to it.
    features = frozenset({
        "tendons", "single_drive", "displacement", "planar_bending",
        "pinch_table", "calibration", "robot_plan", "close_ramp",
        "pregrasp", "normal_row_choice",
    })

    #: How many discs from the base are on the rigidly co-mounted metacarpal.
    #: The metacarpal is the first bone in the 7-segment spec and spans discs 0
    #: and 1, so self-collision skips pairs where both spheres are proximal.
    num_proximal_discs = 2

    def __init__(self, dims=None):
        self.dims = load_hand_dimensions() if dims is None else dims
        self._configs = get_default_hand_configs(self.dims)
        self.digit_names = [name for name, _ in self._configs]
        self.tip_radii = default_hand_tip_radii(self.dims)

        self.hardware = HardwareMap(
            actuator_names={n: f"{n}_flex" for n in self.digit_names},
            open_passive=0.5,
            # Measured per digit at the open pose; the pinky sits higher because
            # its shorter, stiffer rod needs more tension to reach the same
            # zero-bend length.
            open_drive={"index": 0.84, "middle": 0.84, "ring": 0.84,
                        "pinky": 1.03, "thumb": 0.84},
            open_length_warn=0.005,
            flexion_probe=1.5,
        )

    # -- digits ------------------------------------------------------------

    def digit_configs(self):
        """Freshly built ``[(name, TendonFingerSolverConfig)]``, thumb last.

        Rebuilt every call rather than handing out ``self._configs``: the
        ``attach_*`` environment family mutates these configs IN PLACE, so two
        solvers sharing one list would see each other's constraints.
        """
        return get_default_hand_configs(self.dims)

    def contact_node(self, digit: int) -> int:
        return tip_node_index(self._configs[digit][1])

    def collision_sites(self, digit: int):
        cfg = self._configs[digit][1]
        nodes = disc_node_indices(cfg)
        flags = [bool(f) for f in
                 proximal_disc_flags(cfg, self.num_proximal_discs)]
        return nodes, flags

    # -- measured tables ---------------------------------------------------

    def pinch_pose(self, mask):
        """The measured pinch pose for the digits ``mask`` selects, or None.

        None for any set without the thumb and for fewer than two digits: those
        combinations are all on one side of the palm, so their closest approach
        is a fist curl rather than a pinch, and the scan deliberately does not
        cover them. Callers must handle None rather than substituting a default.
        """
        return pinch_pose_for_mask(self._configs, mask)

    # -- the C++ side ------------------------------------------------------

    @property
    def opposing_index(self) -> int:
        return opposing_index_of(self.digit_names, self.opposing_digit)

    def default_pose(self):
        """The measured hover: 75 mm up the support normal, pitched -1.22 rad so
        the palm (which lies along the base frame's -x) faces down over the
        object. See ``solvers.frames`` for the numbers and why."""
        from ...solvers.frames import default_wrist_pose

        means = [np.full(self.actuation.n, 0.5) for _ in self.digit_names]
        for m in means:
            self.actuation.set_drive(m, GRASP_FLEXOR_TENSION)
        return default_wrist_pose(), means

    def mount_pose(self):
        """``T_flange<-wrist``: where this hand bolts to the robot.

        The Onshape measurement, which lives in
        :mod:`gepetto_solvers.projects.robot_mount.mount` along with the
        convention derivation and the fitting script that produced it. Held
        there rather than inlined here because it is a fit against a CAD
        assembly that has to be RE-RUN when the assembly or the morphology
        changes, and the module says so.

        Imported inside the method: ``robot_mount.mount`` reads this package's
        dimension tables, so a module-level import would close a cycle.
        """
        from ....projects.robot_mount.mount import measured_mount_pose

        return measured_mount_pose()

    def actuation_means(self, params):
        """Per-digit tendon tensions: the passive background hold everywhere,
        with that digit's commanded value at the driven index."""
        means = []
        for i in range(len(self.digit_names)):
            mean = np.full(self.actuation.n, params.passive_tension)
            self.actuation.set_drive(mean, params.flexor_tensions[i])
            means.append(mean)
        return means

    def build_spec(self, configs, params=None):
        """The C++ ``HandSpec`` for ``configs``.

        ``params`` is ignored: this hand's kinematics payload is the rod and
        tendon geometry, which does not depend on what is being commanded.

        ``configs`` is a :meth:`digit_configs` list that the solver has already
        attached its environment to; ``make_tendon_hand_spec`` splits each
        config's ``sdf_contact`` / ``sphere_contact`` off into the spec's task
        half and keeps the rod and tendon geometry as the ``"tendon"``
        kinematics payload.
        """
        return gepetto_solvers.make_tendon_hand_spec(
            configs, opposing_digit=self.opposing_index)
