"""The Allegro hand: 16 revolute joints, four 4-DOF chains off a common palm.

The mechanism comes from a URDF and is posed by the ``"rigid_urdf"`` kinematics,
which is generic -- this package is only the description of one particular hand
built on it, the way :mod:`gepetto_solvers.core.hands.tendon_5f` is for the
tendon hand.

===================  ===================================================
:mod:`hand`          :class:`AllegroHand`, the Hand implementation
:mod:`spec`          digits, joints, sites, and the kinematics config
``urdf/``            the vendored Wonik V5 right-B URDF; see its NOTICE.md
===================  ===================================================
"""

from . import spec
from .hand import AllegroHand, DigitEnv

__all__ = ["AllegroHand", "DigitEnv", "spec"]
