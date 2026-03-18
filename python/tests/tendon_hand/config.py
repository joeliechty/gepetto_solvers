import numpy as np

from crest_sparse import TendonRobotSolverConfig
from ..tendon_finger.config import (
    get_K_inv, build_K_inv_per_segment,
    build_disc_positions_and_segments, get_6tendon_per_disc_input,
    _build_default_tendon_routing_radii,
)


# ---- Per-finger anatomy specifications ----
# All fingers use the same 7-segment structure (4 bones + 3 joints):
#   metacarpal, MCP, proximal phalanx, PIP, middle phalanx, DIP, distal phalanx

DEFAULT_BONE_JOINT_SPECS = {
    "index": [
        ("bone", 0.050), ("joint", 0.010), ("bone", 0.040),
        ("joint", 0.008), ("bone", 0.025), ("joint", 0.008), ("bone", 0.018),
    ],
    "middle": [
        ("bone", 0.055), ("joint", 0.010), ("bone", 0.045),
        ("joint", 0.008), ("bone", 0.028), ("joint", 0.008), ("bone", 0.020),
    ],
    "ring": [
        ("bone", 0.050), ("joint", 0.010), ("bone", 0.042),
        ("joint", 0.008), ("bone", 0.026), ("joint", 0.008), ("bone", 0.018),
    ],
    "pinky": [
        ("bone", 0.045), ("joint", 0.008), ("bone", 0.033),
        ("joint", 0.007), ("bone", 0.020), ("joint", 0.007), ("bone", 0.015),
    ],
    "thumb": [
        ("bone", 0.035), ("joint", 0.010), ("bone", 0.035),
        ("joint", 0.010), ("bone", 0.025), ("joint", 0.008), ("bone", 0.020),
    ],
}


# ---- Rotation helpers ----

def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])


def _rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _default_base_rotation():
    """The hardcoded base rotation used by the solver when base_pose is zeros.

    Rx(-pi/2) * Rz(pi) maps local +z (rod growth) to world +y.
    """
    return _rotation_x(-np.pi / 2) @ _rotation_z(np.pi)


# ---- Base pose computation ----

def compute_finger_base_pose(fan_angle_rad, palm_offset=None, fan_axis='x'):
    """Compute a 4x4 SE(3) base pose for a finger fanned at a given angle.

    The fan rotation is applied on top of the default base rotation
    (Rx(-pi/2)*Rz(pi)).

    Parameters
    ----------
    fan_angle_rad : float
        Fan rotation angle in radians.
    palm_offset : array-like (3,), optional
        Translation offset from the palm origin.
    fan_axis : str
        Axis to rotate around for fanning: 'x' (vertical spread, default)
        or 'z' (horizontal spread, used for thumb).
    """
    R_base = _default_base_rotation()
    if fan_axis == 'x':
        R_fan = _rotation_x(fan_angle_rad)
    elif fan_axis == 'z':
        R_fan = _rotation_z(fan_angle_rad)
    else:
        raise ValueError(f"Unknown fan_axis: {fan_axis!r}")
    R = R_fan @ R_base

    T = np.eye(4)
    T[:3, :3] = R
    if palm_offset is not None:
        T[:3, 3] = palm_offset
    return T


def compute_thumb_base_pose(side="right",
                            yaw_deg=-45.0, roll_deg=30.0, pitch_deg=120.0,
                            x_offset=0.02, z_offset=0.04, y_offset=-0.03):
    """Compute base pose for a thumb.

    Two rotations are applied on top of the default base rotation to orient
    the thumb for opposition:

        R = Rz(yaw) @ Rx(roll) @ R_base

    Parameters
    ----------
    side : str
        "left" or "right".
    yaw_deg : float
        Rotation around world z-axis (degrees).  Positive angles turn the
        thumb growth direction from +y toward -x (i.e. inward across the
        palm for a right hand).
    roll_deg : float
        Rotation around world x-axis (degrees).  Positive angles tilt the
        thumb growth direction upward (+z).
    x_offset : float
        Lateral distance from palm center along world x.
    z_offset : float
        Height offset along world z (near the index finger level).
    """
    sign = 1.0 if side == "right" else -1.0
    yaw = np.deg2rad(yaw_deg)
    roll = sign * np.deg2rad(roll_deg)
    pitch = sign * np.deg2rad(pitch_deg)

    R_base = _default_base_rotation()
    R = _rotation_z(yaw) @ _rotation_x(roll) @ _rotation_y(pitch) @ R_base

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x_offset, y_offset, sign * z_offset]
    return T


# ---- Config builder for a single finger ----

def build_finger_config(bone_joint_spec, base_pose_4x4):
    """Build a TendonRobotSolverConfig for a single 6-tendon finger.

    Parameters
    ----------
    bone_joint_spec : list of (str, float) tuples
        The 7-segment bone/joint specification.
    base_pose_4x4 : np.ndarray (4, 4)
        SE(3) base pose for this finger.
    """
    config = TendonRobotSolverConfig()

    config.base.use_dense = False
    config.base.linear_solver_type = "MULTIFRONTAL_QR"
    config.num_tendons = 6
    config.num_between_nodes = 3

    config.base_pose = base_pose_4x4

    disc_positions, segment_types, num_discs, total_length = \
        build_disc_positions_and_segments(bone_joint_spec)

    config.rod_length = total_length
    config.num_discs = num_discs
    config.disc_positions_normalized = disc_positions

    config.K_inv = get_K_inv()
    config.K_inv_per_segment = build_K_inv_per_segment(
        num_discs=num_discs,
        num_between_nodes=config.num_between_nodes,
        segment_types=segment_types,
        bone_stiffness_scale=1e-6,
    )

    tendon_routing_radii = _build_default_tendon_routing_radii(bone_joint_spec)
    config.per_disc_tendon_input = get_6tendon_per_disc_input(
        bone_joint_spec, disc_positions, num_discs,
        tendon_routing_radii=tendon_routing_radii,
    )

    config.sigma_twist_rot = 1.0e-2
    config.sigma_twist_pos = 1.0e-4
    config.sigma_stress_force = 1.0e-4
    config.sigma_stress_moment = 1.0e-5
    config.sigma_base_pos = 1.0e-4
    config.sigma_base_rot = 1.0e-3

    return config


# ---- Hand config builder ----

def get_hand_config(num_fingers=4, thumb_side="right", finger_spread_angle_deg=30.0):
    """Build solver configs for all fingers of a hand.

    Parameters
    ----------
    num_fingers : int
        Number of non-thumb fingers (1-4). Finger types are selected from
        [index, middle, ring, pinky] in order.
    thumb_side : str or None
        "left", "right", "both", or None for no thumb.
    finger_spread_angle_deg : float
        Total angular spread of non-thumb fingers in degrees.

    Returns
    -------
    list of (str, TendonRobotSolverConfig)
        Ordered list of (name, config) tuples.
    """
    finger_type_order = ["index", "middle", "ring", "pinky"]
    finger_names = finger_type_order[:num_fingers]

    spread_rad = np.deg2rad(finger_spread_angle_deg)
    if num_fingers == 1:
        fan_angles = [0.0]
    else:
        # Positive fan angle (Rx) tilts toward +z; index is at top (+z), pinky at bottom (-z)
        fan_angles = np.linspace(spread_rad / 2, -spread_rad / 2, num_fingers)

    finger_spacing_m = 0.018
    if num_fingers == 1:
        z_offsets = [0.0]
    else:
        z_offsets = np.linspace(
            (num_fingers - 1) / 2.0 * finger_spacing_m,
            -(num_fingers - 1) / 2.0 * finger_spacing_m,
            num_fingers,
        )

    configs = []
    for name, angle, z_off in zip(finger_names, fan_angles, z_offsets):
        bone_joint_spec = DEFAULT_BONE_JOINT_SPECS[name]
        base_pose = compute_finger_base_pose(angle, palm_offset=[0, 0, z_off], fan_axis='x')
        config = build_finger_config(bone_joint_spec, base_pose)
        configs.append((name, config))

    if thumb_side in ("left", "both"):
        spec = DEFAULT_BONE_JOINT_SPECS["thumb"]
        pose = compute_thumb_base_pose("left")
        configs.append(("thumb_left", build_finger_config(spec, pose)))
    if thumb_side in ("right", "both"):
        spec = DEFAULT_BONE_JOINT_SPECS["thumb"]
        pose = compute_thumb_base_pose("right")
        configs.append(("thumb_right", build_finger_config(spec, pose)))

    return configs
