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


def _Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


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
    # Sign is negative to match the 180-deg CAD tendon-routing flip: the x=0 mirror
    # of the old +30 deg pronation is -30 deg, keeping the thumb pad turned toward the
    # finger pads now that the flexor curls to the opposite side.
    thumb_pronation = np.deg2rad(-30.0)
    thumb_cmc_offset = np.array([0.0, 0.0, 0.0])

    roll = np.eye(4)
    roll[:3, :3] = _Rz(thumb_pronation)    # local +Z is the rod growth axis
    cmc = np.eye(4); cmc[:3, 3] = thumb_cmc_offset

    cfg_thumb = get_6tendon_config(bone_joint_spec=_scaled_finger_spec(0.85))
    cfg_thumb.hand_base_offset = (
        cmc @ rotate_about_z_through(center, thumb_opposition) @ base @ roll)
    configs.append(("thumb", cfg_thumb))

    return configs


# ---------------------------------------------------------------------------
# Anthropomorphic hand from physical (gepetto_core) dimensions
# ---------------------------------------------------------------------------
#
# The four fingers + opposable thumb above are laid out around a grasp object
# purely for a contact demo. The helpers below instead build a hand from the
# *physical morphology* of a printed hand: per-digit bone lengths, palm origins,
# and base angles. These come from ``gepetto_core``'s default hand configuration
# (parsed from its bundled ``parameters.scad``) when that package is installed,
# and fall back to the hard-coded ``DEFAULT_HAND_DIMENSIONS`` below so crest-sparse
# still runs standalone.

# Each joint's flexible segment eats half its length from each adjacent bone
# end, so the raw CAD/physical bone length isn't also counted as rigid right up
# to the joint center. This keeps total digit length (sum of raw bone lengths)
# unchanged -- length just moves from "rigid" to "flexible" at each joint
# boundary. MCP / PIP / DIP total flexible lengths, largest at the base joint:
_STANDARD_JOINT_LENGTHS = [0.010, 0.006, 0.004]  # MCP / PIP / DIP

# Fallback copy of the gepetto_core default hand (bundled parameters.scad),
# used when gepetto_core is not importable. Fingers: index, middle, ring, pinky.
# Only the first thumb is used (n_thumbs = 1). Lengths/origins in mm, angles in deg.
DEFAULT_HAND_DIMENSIONS = {
    "bl_finger": [[80, 38, 20, 20],
                  [82, 41, 20.5, 20.5],
                  [79, 38, 19, 19],
                  [74, 35, 17, 17]],
    "o_finger": [[0, 1.5, 0],
                 [0, 0, -12],
                 [0, 1, -24],
                 [0, 3.5, -34]],
    "a_finger": [[0, -10, 0],
                 [0, 3, 0],
                 [0, 16, 0],
                 [0, 30, 0]],
    "bl_thumb": [[25, 35, 25, 25]],
    "o_thumb": [[-3, 5.5, 15]],
    "a_thumb": [[0, -100, 0]],
}

FINGER_NAMES = ["index", "middle", "ring", "pinky"]


def load_hand_dimensions():
    """Return the hand morphology dict, preferring ``gepetto_core``.

    Tries ``gepetto_core.geometry.HandGeometry.default()`` and reads the
    per-digit bone lengths / origins / angles from it. Any import failure
    (notably ``dynamixel_sdk`` missing, which ``gepetto_core.__init__`` imports)
    falls back to :data:`DEFAULT_HAND_DIMENSIONS` so this works standalone.

    Returns a dict with the same keys as :data:`DEFAULT_HAND_DIMENSIONS`.
    """
    try:
        from gepetto_core.geometry import HandGeometry
        g = HandGeometry.default()
        dims = {
            "bl_finger": g.bl_finger,
            "o_finger": g.o_finger,
            "a_finger": g.a_finger,
            "bl_thumb": g.bl_thumb[:1],
            "o_thumb": g.o_thumb[:1],
            "a_thumb": g.a_thumb[:1],
        }
        print("[hand dims] loaded from gepetto_core HandGeometry.default()")
        return dims
    except Exception as exc:  # noqa: BLE001 - any import/parse failure -> fallback
        print(f"[hand dims] gepetto_core unavailable ({exc.__class__.__name__}: "
              f"{exc}); using DEFAULT_HAND_DIMENSIONS")
        return DEFAULT_HAND_DIMENSIONS


def bone_joint_spec_from_bones(bone_lengths_mm):
    """Interleave 4 physical bone lengths (mm) with the 3 standard joint lengths.

    Produces the 7-segment ``[(type, length_m), ...]`` spec (4 bones + 3 joints)
    that ``get_6tendon_config`` requires. ``bone_lengths_mm`` must have length 4.

    Each raw bone length is the full rigid CAD length between joint centers;
    half of each bordering joint's length (see ``_STANDARD_JOINT_LENGTHS``) is
    carved off the adjacent bone ends into that joint's flexible segment, so
    the metacarpal and distal phalanx (one bordering joint each) lose half of
    that one joint, and the proximal/middle phalanges (a joint at each end)
    lose half of each of their two bordering joints.
    """
    if len(bone_lengths_mm) != 4:
        raise ValueError(
            f"expected 4 bone lengths, got {len(bone_lengths_mm)}: {bone_lengths_mm}")
    mcp_e, pip_e, dip_e = (j * 1000.0 / 2.0 for j in _STANDARD_JOINT_LENGTHS)
    adjusted_mm = [
        bone_lengths_mm[0] - mcp_e,
        bone_lengths_mm[1] - mcp_e - pip_e,
        bone_lengths_mm[2] - pip_e - dip_e,
        bone_lengths_mm[3] - dip_e,
    ]
    bones = [b / 1000.0 for b in adjusted_mm]
    j = _STANDARD_JOINT_LENGTHS
    return [
        ("bone", bones[0]),   # metacarpal
        ("joint", j[0]),      # MCP
        ("bone", bones[1]),   # proximal phalanx
        ("joint", j[1]),      # PIP
        ("bone", bones[2]),   # middle phalanx
        ("joint", j[2]),      # DIP
        ("bone", bones[3]),   # distal phalanx
    ]


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

# def finger_base_offset(o_mm, a_deg):
#     """SE(3) ``hand_base_offset`` placing a digit on the palm.

#     ``o_mm`` is the digit's (x, y, z) base origin in mm and ``a_deg`` is its
#     (rx, ry, rz) base rotation in degrees, in the gepetto_core /
#     ``parameters.scad`` palm convention (``o_finger`` / ``a_finger``). In that
#     CAD frame a finger grows along +X, the knuckle row is spread along Z (the
#     ``o_finger`` z entries 0, -12, -24, -34) and ``a_finger``'s ``ry`` is the
#     side-to-side splay about the growth axis.

#     ``default_finger_base_pose()`` maps the rod's local frame into the port
#     world frame so a zero-angle finger grows along world +Y, with the knuckle
#     row along world Z (``o_mm`` is used directly as the world offset -- the CAD
#     palm and port world share axes here). The splay must therefore be applied
#     as a rotation about the world **X** axis: the axis perpendicular to both
#     the growth (Y) and the knuckle-row (Z) directions, so the fingers tilt
#     apart *within* the flat plane of the hand rather than stacking out of it.

#     Two earlier attempts got this wrong: applying ``ry`` about the growth axis
#     (Y) only spun each finger about its own shaft (no visible splay), and
#     conjugating it through Z (the ``a_print`` sandwich from ``hand.scad``)
#     both flipped the splay inward *and* skewed the knuckle row out of plane so
#     the fingers looked stacked. Only the world-X splay below keeps the bases in
#     a flat row while fanning the tips apart.

#     The splay sign is ``-ry``: with the ``a_finger`` values increasing
#     index->pinky (-10, 3, 16, 30), the index tip tilts toward +Z and the pinky
#     tip toward -Z -- away from each other, matching a real hand's fan. Only
#     ``ry`` drives the splay (every reference digit, thumb included, specifies
#     only ``ry``); a zero-angle digit reduces to ``default_finger_base_pose()``
#     translated to ``o_mm``, i.e. the legacy single-finger mount.
#     """
#     _, ry, _ = (np.deg2rad(v) for v in a_deg)
#     palm = np.eye(4)
#     palm[:3, :3] = _Rx(-ry)
#     palm[:3, 3] = np.asarray(o_mm, dtype=float) / 1000.0
#     return palm @ default_finger_base_pose()


def get_default_hand_configs(dims=None):
    """Build ``[(name, cfg), ...]`` for a 4-finger + thumb hand from morphology.

    Each digit is a standard 6-tendon finger sized from its physical bone lengths
    and placed via ``hand_base_offset`` from its palm origin/angle. ``dims`` defaults
    to :func:`load_hand_dimensions` (gepetto_core, else fallback). No contact is
    attached, so the caller gets a pure-kinematics hand.
    """
    if dims is None:
        dims = load_hand_dimensions()

    configs = []
    for i, name in enumerate(FINGER_NAMES):
        cfg = get_6tendon_config(
            bone_joint_spec=bone_joint_spec_from_bones(dims["bl_finger"][i]))
        cfg.hand_base_offset = finger_base_offset(
            dims["o_finger"][i], dims["a_finger"][i])
        configs.append((name, cfg))

    cfg_thumb = get_6tendon_config(
        bone_joint_spec=bone_joint_spec_from_bones(dims["bl_thumb"][0]))
    cfg_thumb.hand_base_offset = finger_base_offset(
        dims["o_thumb"][0], dims["a_thumb"][0])
    configs.append(("thumb", cfg_thumb))

    return configs
