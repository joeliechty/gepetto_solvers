"""The five-digit tendon hand: the hand this repository was built around.

:class:`TendonHand5F` is the :class:`~gepetto_solvers.core.hands.base.Hand`
implementation, and the object the solvers are handed. Everything it is made of
lives beside it:

===================  ===================================================
:mod:`rotations`     elementary rotations and the digit base pose
:mod:`dimensions`    physical morphology, the bone/joint spec, tip radii
:mod:`morphology`    digit placement and the per-digit solver configs
:mod:`discs`         which rod nodes carry a disc; the contact mask
:mod:`pinch`         the measured pinch table
:mod:`finger_config` the 6-tendon routing matrix and rod compliance
===================  ===================================================

THE MEASURED TABLES HERE BELONG TO THIS HAND. ``DEFAULT_HAND_DIMENSIONS``,
``HAND_PINCH_POSES``, the 6x7 tendon routing and the close/lift ramp constants
are properties of these bone lengths, this palm layout and this tendon scheme.
Nothing in the code can detect a mismatch with a different morphology, which is
why they are reached through a :class:`TendonHand5F` instance rather than
imported as global truth. The module-level names below are re-exported for the
code that genuinely works ON this hand's geometry (the CAD comparison, the pinch
regeneration script, the calibration overlay); a SOLVER should read
``self.hand`` instead.
"""

from .dimensions import (
    DEFAULT_HAND_DIMENSIONS,
    FINGER_NAMES,
    bone_joint_spec_from_bones,
    default_hand_tip_radii,
    load_hand_dimensions,
)

# Underscore-prefixed but genuinely shared, like _Rx/_Ry/_Rz below: the whole
# attach_* family validates its per-digit mask with this. Re-exported with the
# redundant-alias form so it survives an __all__ that cannot name it.
from .discs import (
    _resolve_contact_mask as _resolve_contact_mask,
)
from .discs import disc_node_indices, proximal_disc_flags
from .finger_config import get_6tendon_config, get_K_inv
from .hand import FLEXOR_INDEX, TENDON_NAMES, TendonHand5F
from .morphology import (
    finger_base_offset,
    get_default_hand_configs,
    tip_node_index,
)
from .pinch import (
    DIGIT_ORDER,
    HAND_PINCH_POSES,
    PinchPose,
    pinch_pose,
    pinch_pose_for_mask,
)

# _Rx/_Ry/_Rz are underscore-prefixed but genuinely shared: robot_mount.mount
# builds its candidate mounting rotations from them. Re-exported explicitly
# rather than renamed, so the CAD-comparison code keeps reading the same way.
from .rotations import (
    _Rx as _Rx,
)
from .rotations import (
    _Ry as _Ry,
)
from .rotations import (
    _Rz as _Rz,
)
from .rotations import (
    default_base_rotation,
    default_finger_base_pose,
    hand_growth_axis,
)

__all__ = [
    "DEFAULT_HAND_DIMENSIONS",
    "DIGIT_ORDER",
    "FINGER_NAMES",
    "FLEXOR_INDEX",
    "HAND_PINCH_POSES",
    "PinchPose",
    "TENDON_NAMES",
    "TendonHand5F",
    "bone_joint_spec_from_bones",
    "default_base_rotation",
    "default_finger_base_pose",
    "default_hand_tip_radii",
    "disc_node_indices",
    "finger_base_offset",
    "get_6tendon_config",
    "get_K_inv",
    "get_default_hand_configs",
    "hand_growth_axis",
    "load_hand_dimensions",
    "pinch_pose",
    "pinch_pose_for_mask",
    "proximal_disc_flags",
    "tip_node_index",
]
