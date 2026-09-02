"""The :class:`Hand` interface: everything the solvers need to know about a hand.

The solvers, planners and readback paths used to reach for module-level facts --
``FINGER_NAMES``, ``FLEXOR_IDX``, ``HAND_PINCH_POSES``, ``HARDWARE_FINGER_NAMES``,
the literal string ``"thumb"``. Each of those is a property of ONE hand, and none
of them is checkable: a second hand would have silently inherited the first
hand's flexor index and pinch table.

They live here instead, on an object the solver is handed. A hand supplies:

* its digits, their solver configs and their contact radii,
* which digit opposes the others (the thumb on an anatomical hand),
* how it is actuated -- how many actuators, which are driven, how to build a
  prior covariance over them,
* the measured tables that belong to its morphology (pinch poses, ramp
  constants, hardware travel),
* and :meth:`Hand.build_spec`, the C++ :class:`HandSpec` naming the kinematics
  the graph builder should load.

:mod:`gepetto_solvers.core.hands.tendon_5f` is the implementation for the
five-digit tendon hand this repository was built around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Actuation:
    """How one digit is driven.

    ``n``              number of actuation variables per digit (6 tendons on the
                       tendon hand).
    ``names``          one label per actuation variable, for plots and overlays.
    ``drive_indices``  which of them a motor commands. The rest are passive
                       (spring-backed tendons here) and behave differently in
                       every prior: a passive holds roughly constant, so it is
                       pinned tight, while a driven one must be free to be
                       pushed off its commanded value by contact.

    This replaces the module-level ``FLEXOR_IDX = 5`` that was declared in two
    places and cross-checked at runtime because nothing else could keep the two
    in agreement.
    """

    n: int
    names: tuple[str, ...]
    drive_indices: tuple[int, ...]

    def __post_init__(self):
        if len(self.names) != self.n:
            raise ValueError(
                f"Actuation: {len(self.names)} names for {self.n} actuators")
        for i in self.drive_indices:
            if not 0 <= i < self.n:
                raise ValueError(
                    f"Actuation: drive index {i} out of range for {self.n} actuators")

    @property
    def passive_indices(self) -> tuple[int, ...]:
        driven = set(self.drive_indices)
        return tuple(i for i in range(self.n) if i not in driven)

    def prior_cov(self, passive_sigma: float, drive_sigma: float) -> np.ndarray:
        """Diagonal covariance with a distinct entry for the DRIVEN actuators.

        Every actuation prior on a hand is anisotropic in the same way: passives
        that behave one way and motor-driven actuators that behave another.
        Writing it once keeps the split from drifting apart between the priors
        that have to agree about it.
        """
        d = np.full(self.n, float(passive_sigma) ** 2)
        for i in self.drive_indices:
            d[i] = float(drive_sigma) ** 2
        return np.diag(d)

    def uniform_cov(self, sigma: float) -> np.ndarray:
        """Isotropic covariance over the actuators.

        A contact-free (FK) solve needs this: the tight-passive/loose-driven
        split is underdetermined with nothing to trade against, and GTSAM raises
        IndeterminantLinearSystem on the actuation variable.
        """
        return float(sigma) ** 2 * np.eye(self.n)

    def set_drive(self, vector: np.ndarray, value: float) -> np.ndarray:
        """Write ``value`` into every driven entry of ``vector``, in place."""
        for i in self.drive_indices:
            vector[i] = value
        return vector

    def drive_value(self, vector) -> float:
        """The driven entry of ``vector``. Raises when more than one is driven --
        a caller reading "the" commanded value has to say which on such a hand."""
        if len(self.drive_indices) != 1:
            raise ValueError(
                f"drive_value: this hand drives {len(self.drive_indices)} "
                f"actuators per digit, so there is no single value to read; "
                f"index with drive_indices explicitly.")
        return vector[self.drive_indices[0]]


# ---------------------------------------------------------------------------
# Hardware and motion, both measured per hand
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareMap:
    """How this hand's digits map onto the physical robot's actuators.

    ``actuator_names``  ``{digit: hardware actuator name}``.
    ``open_passive``    the background tension the open hand holds (N).
    ``open_drive``      ``{digit: driven tension at the open pose}`` (N).
    ``open_length_warn`` how far a measured open length may sit from the
                        commanded one before it is worth warning about (m).
    ``flexion_probe``   the tension used to MEASURE which way flexion runs,
                        rather than assuming a sign (N).
    """

    actuator_names: dict[str, str] = field(default_factory=dict)
    open_passive: float = 0.5
    open_drive: dict[str, float] = field(default_factory=dict)
    open_length_warn: float = 0.005
    flexion_probe: float = 1.5


@dataclass(frozen=True)
class MotionProfile:
    """Measured ramp constants for the open-loop close and lift.

    These are properties of one hand's tendon travel and stiffness -- how much
    displacement a given tension buys, and where each digit stops. They are not
    tunables: they were measured, and a different hand's numbers are different.
    """

    close_steps: int = 12
    close_fraction: float = 0.9
    close_probe_step: float = 0.1
    close_tol_m: float = 2e-4
    close_refine: int = 3
    lift_height_m: float = 0.15
    lift_steps: int = 12


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

@runtime_checkable
class Hand(Protocol):
    """What a hand must supply for the solvers to pose it.

    A Protocol rather than a base class so a hand can be assembled however suits
    it -- the tendon hand builds itself out of a CAD dimension table, another
    might parse a URDF -- and still satisfy this by structure. ``isinstance``
    works against it (``runtime_checkable``), which is what the interface test
    uses; note that only checks the NAMES are present, so the test also exercises
    a stub hand end to end.
    """

    #: Registry key this hand is fetched by (e.g. ``"tendon_5f"``).
    name: str

    #: Registry key of the C++ HandKinematics to load (e.g. ``"tendon"``).
    #: Several hands may share one kinematics: two tendon hands of different
    #: morphology are different Hands over the same mechanism type.
    kinematics: str

    #: Digit names, in the order every per-digit list in the solvers uses.
    digit_names: list[str]

    #: Contact-sphere radius per digit, same order.
    tip_radii: list[float]

    #: How each digit is driven.
    actuation: Actuation

    #: Name of the digit that opposes the others in the pre-grasp constraints,
    #: or None for a hand with no opposition (which then never builds those
    #: constraints).
    opposing_digit: str | None

    def digit_configs(self) -> list[tuple[str, object]]:
        """``[(name, config)]`` per digit, freshly built.

        Fresh every call, because the environment ``attach_*`` family MUTATES
        these configs in place: two solvers sharing one list would see each
        other's constraints.
        """
        ...

    def contact_node(self, digit: int) -> int:
        """Index of the site a task constraint contacts with (the tip)."""
        ...

    def collision_sites(self, digit: int) -> tuple[list[int], list[bool]]:
        """``(node indices, is_proximal flags)`` for this digit's collision
        spheres. Proximal sites are rigidly co-mounted with each other, so pairs
        of them are skipped by self-collision."""
        ...

    def pinch_pose(self, mask: list[bool]):
        """The measured pinch pose for the digits selected by ``mask``, or None
        when this hand has no measurement for that combination."""
        ...

    #: What this hand supports, from the vocabulary in :data:`FEATURES`. The
    #: workbench gates whole panels on it, so a control that would do nothing on
    #: this hand is ABSENT rather than present and dead.
    features: frozenset[str]

    def build_spec(self, configs: list[tuple[str, object]], params=None):
        """The C++ ``HandSpec`` for ``configs`` (as returned by
        :meth:`digit_configs`, after the environment has been attached to it).

        Taking the configs rather than rebuilding them is the point: by this
        stage they carry the task environment the solver attached, and that is
        what the spec's task half is made of.

        ``params`` is the solve's :class:`HandSolveParams`, for a hand whose
        SPEC depends on it -- a joint-space hand seeds its configuration from
        the commanded one, so that the solve starts at zero kinematics residual.
        A hand that does not need it ignores it.
        """
        ...

    def default_pose(self):
        """``(wrist_pose 4x4, actuation means)`` -- where this hand starts.

        A hand's neutral posture and the wrist pose that aims it at the default
        scene are properties of ITS geometry: the tendon hand's palm lies along
        its base frame's -x, the Allegro hand's fingers extend +z, so the pose
        that hovers one palm-down over an object points the other away from it.
        The workbench seeds its sliders from this.
        """
        ...

    def actuation_means(self, params) -> list:
        """One actuation mean vector per digit -- the q_S of p(q), or the
        commanded tensions.

        Exists because ``params.flexor_tensions`` is one SCALAR per digit, which
        cannot command four independent joints. Each hand turns the params it
        cares about into the vector its actuation variable actually takes.
        """
        ...

    @property
    def opposing_index(self) -> int:
        """Index of :attr:`opposing_digit`, or -1 when there is none."""
        ...


#: The feature vocabulary. A hand declares the subset it supports, and the
#: workbench and the solvers gate on it.
#:
#: These are capabilities of the HAND, distinct from ``solvers.capabilities()``,
#: which reports what the compiled binding can do. Both gate controls; they
#: answer different questions ("can this robot do it" vs "can this build do it").
FEATURES = frozenset({
    # Tendon routing exists: the tendon/disc overlays and the routing readouts.
    "tendons",
    # One actuator per digit is motor-driven, so "the" commanded value is a
    # single number -- what every drive_indices[0] reader assumes.
    "single_drive",
    # A displacement readout distinct from the actuation (tendon length), which
    # the hardware plan is expressed in.
    "displacement",
    # The rod planar-bending approximation.
    "planar_bending",
    # A measured pinch table, so the pre-grasp centroid constraint has a target.
    "pinch_table",
    # Disc-addressed calibration landmarks.
    "calibration",
    # A hardware bridge to play a plan on.
    "robot_plan",
    # Measured close/lift ramp constants.
    "close_ramp",
})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def opposing_index_of(digit_names, opposing_digit) -> int:
    """Index of ``opposing_digit`` in ``digit_names``, or -1 when it is None.

    Raises when the name is given but absent -- a hand that names an opposing
    digit it does not have would otherwise silently drop every pre-grasp
    constraint.
    """
    if opposing_digit is None:
        return -1
    try:
        return list(digit_names).index(opposing_digit)
    except ValueError:
        raise ValueError(
            f"opposing_digit {opposing_digit!r} is not one of this hand's "
            f"digits {list(digit_names)!r}") from None
