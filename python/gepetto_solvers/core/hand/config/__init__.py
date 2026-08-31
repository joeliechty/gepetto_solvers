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


# The attach_* family lives in gepetto_solvers.core.environment; it is re-exported
# here because every existing caller imports it from this module.
#
# LAZILY, via PEP 562, and that is load-bearing rather than clever. Those modules
# import back into this package (`..hand.config.discs`, `..hand.config.morphology`),
# so importing them at module scope closes a cycle: anyone whose first touch is
# `core.environment.collision` gets this package half-initialized and the import
# fails. Deferring to first attribute access breaks the cycle in one direction
# without changing what `from ...hand.config import attach_contact` returns.
_ENVIRONMENT_EXPORTS = {
    "attach_collision": "collision",
    "attach_contact": "contact",
    "attach_pregrasp_axis_alignment": "pregrasp",
    "attach_pregrasp_center": "pregrasp",
    "attach_pregrasp_centroid": "pregrasp",
    "attach_half_space": "support",
    "attach_table": "support",
    "opposition_axis_from_object": "support",
    "opposition_directions": "support",
}


def __getattr__(name):
    module = _ENVIRONMENT_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"...environment.{module}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_ENVIRONMENT_EXPORTS))
