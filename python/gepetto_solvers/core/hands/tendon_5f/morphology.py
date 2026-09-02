"""Assembling the anatomical hand: digit placement, and the solver configs.

:func:`finger_base_offset` is the SE(3) ``hand_base_offset`` that puts a digit on
the palm, including the OpenSCAD ``a_print`` conjugation sandwich.
:func:`get_default_hand_configs` turns the morphology into the per-finger
``TendonFingerSolverConfig`` list the solvers take, thumb last.
"""

import numpy as np

from .dimensions import (
    FINGER_NAMES,
    bone_joint_spec_from_bones,
    load_hand_dimensions,
)
from .finger_config import get_6tendon_config
from .rotations import _Rx, _Ry, _Rz, default_finger_base_pose


def tip_node_index(config):
    """Index of the last rod node (the tip) for a finger config."""
    num_nodes = config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    return num_nodes - 1


# def bone_joint_spec_from_bones(bone_lengths_mm):
#     """Interleave 4 physical bone lengths (mm) with the 3 standard joint lengths.

#     Produces the 7-segment ``[(type, length_m), ...]`` spec (4 bones + 3 joints)
#     that ``get_6tendon_config`` requires. ``bone_lengths_mm`` must have length 4.

#     Each raw bone length is the full rigid CAD length between joint centers;
#     half of each bordering joint's length (see ``_STANDARD_JOINT_LENGTHS``) is
#     carved off the adjacent bone ends into that joint's flexible segment, so
#     the metacarpal and distal phalanx (one bordering joint each) lose half of
#     that one joint, and the proximal/middle phalanges (a joint at each end)
#     lose half of each of their two bordering joints.
#     """
#     if len(bone_lengths_mm) != 4:
#         raise ValueError(
#             f"expected 4 bone lengths, got {len(bone_lengths_mm)}: {bone_lengths_mm}")
#     mcp_e, pip_e, dip_e = (j * 1000.0 / 2.0 for j in _STANDARD_JOINT_LENGTHS)
#     adjusted_mm = [
#         bone_lengths_mm[0] - mcp_e,
#         bone_lengths_mm[1] - mcp_e - pip_e,
#         bone_lengths_mm[2] - pip_e - dip_e,
#         bone_lengths_mm[3] - dip_e,
#     ]
#     bones = [b / 1000.0 for b in adjusted_mm]
#     j = _STANDARD_JOINT_LENGTHS
#     return [
#         ("bone", bones[0]),   # metacarpal
#         ("joint", j[0]),      # MCP
#         ("bone", bones[1]),   # proximal phalanx
#         ("joint", j[1]),      # PIP
#         ("bone", bones[2]),   # middle phalanx
#         ("joint", j[2]),      # DIP
#         ("bone", bones[3]),   # distal phalanx
#     ]


def finger_base_offset(o_mm, a_deg, a_print_deg=45.0):
    """
    SE(3) `hand_base_offset` placing a digit on the palm.
    Restores the full 3D rotation and the a_print sandwich to match the physical CAD,
    with an inverted print angle to cup the fingers inwards.
    """
    # 1. Extract local digit angles
    rx, ry, rz = (np.deg2rad(v) for v in a_deg)
    
    # Invert the print angle to account for the solver's coordinate frame
    aprint = np.deg2rad(-a_print_deg)
    
    # 2. Map local CAD rotations to Solver axes
    R_cad_z = _Rz(rz)
    R_cad_y = _Rx(-ry)
    R_cad_x = _Ry(rx)
    R_local = R_cad_z @ R_cad_y @ R_cad_x
    
    # 3. Construct the OpenSCAD 'a_print' sandwich (conjugating through Z)
    R_print = _Rz(aprint)
    R_print_inv = _Rz(-aprint)
    
    # 4. Apply the transformation
    palm = np.eye(4)
    palm[:3, :3] = R_print @ R_local @ R_print_inv
    palm[:3, 3] = np.asarray(o_mm, dtype=float) / 1000.0
    
    return palm @ default_finger_base_pose()


def get_default_hand_configs(dims=None):
    if dims is None:
        dims = load_hand_dimensions()

    configs = []
    for i, name in enumerate(FINGER_NAMES):
        spec = bone_joint_spec_from_bones(
            dims["bl_finger"][i],
            joint_lengths_mm=dims["jd_finger"][i]
        )
        cfg = get_6tendon_config(bone_joint_spec=spec)
        cfg.hand_base_offset = finger_base_offset(dims["o_finger"][i], dims["a_finger"][i])
        configs.append((name, cfg))

    # Thumb configuration
    spec_thumb = bone_joint_spec_from_bones(
        dims["bl_thumb"][0],
        joint_lengths_mm=dims["jd_thumb"][0]
    )
    cfg_thumb = get_6tendon_config(bone_joint_spec=spec_thumb)
    cfg_thumb.hand_base_offset = finger_base_offset(dims["o_thumb"][0], dims["a_thumb"][0])

    configs.append(("thumb", cfg_thumb))
    return configs
