"""Hands: what the solvers are told about the mechanism they are posing.

:mod:`base` defines the :class:`Hand` interface -- digits, actuation, the
opposing digit, the measured tables, and the C++ ``HandSpec`` naming the
kinematics to load. :mod:`registry` fetches one by name. :mod:`tendon_5f` is the
five-digit tendon hand.

Adding a hand means two registrations: a ``HandKinematics`` on the C++ side under
some ``kinematics`` name, and a :class:`Hand` here that names it. See
``docs/adding_a_hand.md``.
"""

from .base import (
    Actuation,
    Hand,
    HardwareMap,
    MotionProfile,
    opposing_index_of,
)
from .registry import DEFAULT_HAND, get_hand, register_hand, registered_hands
from .tendon_5f import TendonHand5F

register_hand(TendonHand5F.name, TendonHand5F)

__all__ = [
    "Actuation",
    "DEFAULT_HAND",
    "Hand",
    "HardwareMap",
    "MotionProfile",
    "TendonHand5F",
    "get_hand",
    "opposing_index_of",
    "register_hand",
    "registered_hands",
]
