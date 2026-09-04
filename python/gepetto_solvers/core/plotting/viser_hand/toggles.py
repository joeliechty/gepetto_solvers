"""Which constraint-distance overlays are drawn.

One flag per constraint FAMILY, deliberately separate from whether that
constraint is in the factor graph. The two questions are different and the
useful answers rarely coincide: the distance a collision inequality would see is
worth reading precisely while collision is OFF (it says what turning it on would
cost), and a contact gap is worth reading before the equality that closes it is
ever attached. Riding the overlays on the constraint switches -- which is what
the panel used to do -- made every one of those inspections impossible without
changing the solve you were trying to inspect.

Held in one object rather than as nine attributes on
:class:`~gepetto_solvers.core.plotting.viser_hand.scene.ViserHandScene` so the
group can be passed, defaulted and reset as a unit, and so adding a family is
one field here rather than a constructor argument threaded through every caller.
"""

from dataclasses import dataclass


@dataclass
class DistanceOverlays:
    """Per-family switches for the measurement overlays.

    The defaults are the ones worth having on while posing a hand: the contact
    and pre-grasp distances, which are what a solve is aiming at. The collision
    families default OFF because they are per SPHERE rather than per finger --
    five digits of four spheres is twenty lines and twenty labels, which is a
    thing to switch on with a question in mind, not scene furniture. The metric
    comparison and the grasp wrench default off for the same reason.
    """

    #: Fingertip -> object surface (the ``h_contact`` equality's own distance).
    object_contact: bool = True
    #: Fingertip -> support plane.
    table_contact: bool = True
    #: Every non-contact sphere -> object surface (``h_pen``).
    object_collision: bool = False
    #: Every non-contact sphere -> support plane (``h_pen`` against the plane).
    table_collision: bool = False
    #: Cross-digit sphere pairs near touching.
    self_collision: bool = False
    #: Opposition half-space signed margin.
    half_space: bool = True
    #: Pre-grasp centering, pinch centroid and short-axis alignment.
    pregrasp: bool = True
    #: Exact vs. Taubin ellipsoid distance, side by side.
    ellipsoid_metric: bool = False
    #: The net virtual wrench of the contacts (``h_grasp``).
    grasp_wrench: bool = False

    def any_on(self):
        """Whether anything in the group is switched on.

        Lets a caller skip computing the witnesses altogether -- they walk every
        sphere pair in the hand -- rather than computing them and drawing
        nothing.
        """
        return any(getattr(self, f) for f in self.__dataclass_fields__)
