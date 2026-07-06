"""Config builders for a multi-finger tendon *hand* whose fingers share one
floating wrist base.

Each finger is a standard 6-tendon finger (reusing
``tendon_finger.config.get_6tendon_config``); the only hand-specific piece is the
per-finger ``hand_base_offset`` that places that finger relative to the shared
wrist. The C++ ``TendonHandSolver`` gives every finger the *same* wrist variable
``T_base`` with ``T_0 = T_base o hand_base_offset``, so changing an offset just
repositions that finger on the common wrist. Add a finger by appending one more
``(name, config)`` entry — no code changes required.
"""

import numpy as np

from ..tendon_finger.config import get_6tendon_config


def _Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def _Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s,  c]])


def default_base_rotation():
    """Rx(-pi/2) @ Rz(pi): maps local +z (rod growth) to world +y.

    This is the legacy single-finger mounting used throughout the tendon_finger
    tests, so a finger with this offset (and an identity wrist) behaves exactly
    like the standalone single-finger solve.
    """
    return _Rx(-np.pi / 2) @ _Rz(np.pi)


def default_finger_base_pose():
    T = np.eye(4)
    T[:3, :3] = default_base_rotation()
    return T


def point_reflection_about_z(center):
    """4x4 SE(3) for a 180 deg rotation about the world Z axis through ``center``.

    In the z=0 plane this is a point reflection through ``center``: it maps a
    finger (and its whole in-plane curl) to a mirrored finger that approaches the
    same point from the opposite side — i.e. opposition.
    """
    Rz = np.eye(4)
    Rz[:3, :3] = _Rz(np.pi)
    Tc = np.eye(4); Tc[:3, 3] = np.asarray(center, dtype=float)
    Tc_inv = np.eye(4); Tc_inv[:3, 3] = -np.asarray(center, dtype=float)
    return Tc @ Rz @ Tc_inv


def tip_node_index(config):
    """Index of the last rod node (the tip) for a finger config."""
    num_nodes = config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    return num_nodes - 1


def get_two_finger_opposition_configs(object_center, bone_joint_spec=None):
    """Two identical 6-tendon fingers placed in opposition about ``object_center``.

    Finger A uses the legacy single-finger mounting (identity wrist, default base
    rotation) so it reaches the object exactly like the standalone test. Finger B
    is A reflected 180 deg about the vertical axis through ``object_center``, so it
    curls into the object from the opposite side. Both share the wrist variable.

    Returns a list ``[("finger_a", cfg_a), ("finger_b", cfg_b)]``. The caller is
    responsible for attaching each finger's ``sdf_contact`` / ``sphere_contact``.
    """
    cfg_a = get_6tendon_config(bone_joint_spec=bone_joint_spec)
    cfg_b = get_6tendon_config(bone_joint_spec=bone_joint_spec)

    base_a = default_finger_base_pose()
    offset_a = base_a                                    # wrist is identity here
    offset_b = point_reflection_about_z(object_center) @ base_a

    cfg_a.hand_base_offset = offset_a
    cfg_b.hand_base_offset = offset_b

    return [("finger_a", cfg_a), ("finger_b", cfg_b)]


def rotate_about_z_through(center, theta):
    """4x4 SE(3) for a rotation by ``theta`` (radians) about world +Z through
    ``center``. ``point_reflection_about_z`` is the ``theta = pi`` special case.

    Applied to a finger's mount, this is a rigid isometry of the whole
    finger+object geometry: a finger placed with
    ``rotate_about_z_through(object_center, theta) @ default_finger_base_pose()``
    reaches the object exactly like the default-mounted single-finger solve, but
    from an approach azimuth rotated by ``theta`` about the object center.
    """
    Rz = np.eye(4)
    Rz[:3, :3] = _Rz(theta)
    Tc = np.eye(4); Tc[:3, 3] = np.asarray(center, dtype=float)
    Tc_inv = np.eye(4); Tc_inv[:3, 3] = -np.asarray(center, dtype=float)
    return Tc @ Rz @ Tc_inv


# Standard 7-segment finger anatomy (4 bones + 3 joints); mirrors the default in
# tendon_finger.config.get_6tendon_config. The 6-tendon routing is hardcoded for
# exactly 7 segments, so different-sized fingers must keep this structure and
# only scale the segment lengths.
_STANDARD_FINGER_SPEC = [
    ("bone", 0.06),     # metacarpal
    ("joint", 0.01),    # MCP
    ("bone", 0.03),     # proximal phalanx
    ("joint", 0.005),   # PIP
    ("bone", 0.015),    # middle phalanx
    ("joint", 0.005),   # DIP
    ("bone", 0.012),    # distal phalanx
]


def _scaled_finger_spec(scale):
    """Uniformly scale every segment length of the standard finger, keeping the
    7-segment (4 bone / 3 joint) structure the tendon routing requires."""
    return [(t, length * scale) for (t, length) in _STANDARD_FINGER_SPEC]


def get_five_finger_grasp_configs(object_center):
    """Five 6-tendon fingers on one shared floating wrist, all grasping the
    object at ``object_center``: four fingers plus an opposable thumb.

    The four fingers are identical full-length fingers, each mounted via
    ``rotate_about_z_through(object_center, theta)`` — an exact isometry of the
    known-good single-finger solve — so every fingertip is guaranteed to reach
    the surface. Their azimuths space the mounts ~20 mm apart (real MCP spacing)
    and fan them across one side of the object:

        index -30 deg, middle -10 deg, ring +10 deg, pinky +30 deg.

    The thumb is anatomically distinct: a shorter (scaled) finger opposing the
    others from the radial side (~130 deg about the object, not a dead-opposite
    180 deg mirror) and *pronated* so its pad turns toward the finger pads (this
    pronation is what makes a thumb opposable). It stays on the same isometry
    circle as the four fingers, so its tip reaches the object like the working
    two-finger opposition mount. Its shaping constants (opposition angle,
    pronation, length) are the empirically-tuned part of the layout.

    Returns ``[(name, cfg), ...]``; the caller attaches each finger's
    ``sdf_contact`` environment (all referencing the same shared object).
    """
    center = np.asarray(object_center, dtype=float)
    base = default_finger_base_pose()

    configs = []

    # --- Four fingers: identical full fingers, rotated about the object center.
    finger_azimuths = [
        ("index", np.deg2rad(-30.0)),
        ("middle", np.deg2rad(-10.0)),
        ("ring", np.deg2rad(10.0)),
        ("pinky", np.deg2rad(30.0)),
    ]
    for name, theta in finger_azimuths:
        cfg = get_6tendon_config()
        cfg.hand_base_offset = rotate_about_z_through(center, theta) @ base
        configs.append((name, cfg))

    # --- Opposable thumb: shorter finger, opposing from the radial side, with a
    # modest pronation roll so its pad turns toward the finger pads.
    #
    # We keep the thumb *on the isometry circle* (the opposition rotation about
    # the object center is the only in-plane placement), so its tip is guaranteed
    # to reach the surface exactly like the working two-finger opposition mount
    # (that mount is this same family at 180 deg). Three tunables shape how
    # thumb-like it looks; they are deliberately conservative because an
    # aggressive roll or an off-circle CMC offset can rotate the finger's
    # (essentially planar) curl away from the object and stall the solve:
    #   * opposition angle ~130 deg  -> comes across from the radial side, not a
    #     dead-opposite 180 deg mirror of a finger.
    #   * pronation roll ~30 deg     -> pad turns toward the finger pads. This is
    #     capped: the model finger has a single free DOF (the flexor) curling in
    #     one plane, so rolling that plane too far (>~30 deg here) tilts the tip's
    #     reachable arc off the in-plane object and the flexor tension goes
    #     unconstrained -> IndeterminantLinearSystem on the thumb's Q variable.
    #   * length scale 0.85          -> ~116 mm, a shorter thumb with reach margin
    #     over the ~71 mm base-to-object distance.
    # A proximal/radial CMC world-offset would be more anatomically faithful but
    # pushes the base off the isometry circle (further from the object), so it is
    # left at zero by default; raise it only alongside a solve-in-the-loop tune.
    thumb_opposition = np.deg2rad(130.0)
    thumb_pronation = np.deg2rad(30.0)
    thumb_cmc_offset = np.array([0.0, 0.0, 0.0])

    roll = np.eye(4)
    roll[:3, :3] = _Rz(thumb_pronation)    # local +Z is the rod growth axis
    cmc = np.eye(4); cmc[:3, 3] = thumb_cmc_offset

    cfg_thumb = get_6tendon_config(bone_joint_spec=_scaled_finger_spec(0.85))
    cfg_thumb.hand_base_offset = (
        cmc @ rotate_about_z_through(center, thumb_opposition) @ base @ roll)
    configs.append(("thumb", cfg_thumb))

    return configs
