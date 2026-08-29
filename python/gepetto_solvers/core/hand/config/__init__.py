"""Build the hand: physical morphology in, solver configs out.

Split out of what used to be one 1112-line module. The ``attach_*`` environment
builders that also lived there now have their own package,
:mod:`gepetto_solvers.core.environment` -- they define the *problem*, while this
one defines the *robot*. They are re-exported below so every existing
``from ...hand.config import attach_contact`` keeps working.

================  ===================================================
:mod:`rotations`  elementary rotations and the digit base pose
:mod:`dimensions` physical morphology, the bone/joint spec, tip radii
:mod:`morphology` digit placement and the per-finger solver configs
:mod:`discs`      which rod nodes carry a disc; the contact mask
:mod:`pinch`      the measured pinch table
================  ===================================================
"""

from ...environment.collision import attach_collision
from ...environment.contact import attach_contact
from ...environment.pregrasp import (
    attach_pregrasp_axis_alignment,
    attach_pregrasp_center,
    attach_pregrasp_centroid,
)
from ...environment.support import (
    attach_half_space,
    attach_table,
    opposition_axis_from_object,
    opposition_directions,
)
from .dimensions import (
    DEFAULT_HAND_DIMENSIONS,
    FINGER_NAMES,
    bone_joint_spec_from_bones,
    default_hand_tip_radii,
    load_hand_dimensions,
)
from .discs import disc_node_indices, proximal_disc_flags
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
    # dimensions
    "DEFAULT_HAND_DIMENSIONS",
    "FINGER_NAMES",
    "bone_joint_spec_from_bones",
    "default_hand_tip_radii",
    "load_hand_dimensions",
    # rotations / morphology
    "default_base_rotation",
    "default_finger_base_pose",
    "finger_base_offset",
    "get_default_hand_configs",
    "hand_growth_axis",
    "tip_node_index",
    # discs
    "disc_node_indices",
    "proximal_disc_flags",
    # pinch
    "DIGIT_ORDER",
    "HAND_PINCH_POSES",
    "PinchPose",
    "pinch_pose",
    "pinch_pose_for_mask",
    # re-exported from core.environment -- see the module docstring
    "attach_collision",
    "attach_contact",
    "attach_half_space",
    "attach_pregrasp_axis_alignment",
    "attach_pregrasp_center",
    "attach_pregrasp_centroid",
    "attach_table",
    "opposition_axis_from_object",
    "opposition_directions",
]
