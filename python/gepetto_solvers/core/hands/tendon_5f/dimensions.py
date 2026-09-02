"""Physical hand morphology: where the numbers come from, and the bone/joint spec.

:func:`load_hand_dimensions` prefers ``gepetto_core``'s CAD-derived geometry and
falls back to :data:`DEFAULT_HAND_DIMENSIONS` so this repo runs standalone. THE
TWO ARE NOT THE SAME HAND -- the middle finger's first joint diameter is 9.8 mm
from the CAD and 14.0 mm in the fallback -- so anything asserting a number should
pin one source rather than take whichever is installed.
"""


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


# # Fallback copy of the gepetto_core default hand (bundled parameters.scad),
# # used when gepetto_core is not importable. Fingers: index, middle, ring, pinky.
# # Only the first thumb is used (n_thumbs = 1). Lengths/origins in mm, angles in deg.
# DEFAULT_HAND_DIMENSIONS = {
#     "bl_finger": [[80, 38, 20, 20],
#                   [82, 41, 20.5, 20.5],
#                   [79, 38, 19, 19],
#                   [74, 35, 17, 17]],
#     "o_finger": [[0, 1.5, 0],
#                  [0, 0, -12],
#                  [0, 1, -24],
#                  [0, 3.5, -34]],
#     "a_finger": [[0, -10, 0],
#                  [0, 3, 0],
#                  [0, 16, 0],
#                  [0, 30, 0]],
#     "bl_thumb": [[25, 35, 25, 25]],
#     "o_thumb": [[-3, 5.5, 15]],
#     "a_thumb": [[0, -100, 0]],
# }

# FINGER_NAMES = ["index", "middle", "ring", "pinky"]


# def load_hand_dimensions():
#     """Return the hand morphology dict, preferring ``gepetto_core``.

#     Tries ``gepetto_core.geometry.HandGeometry.default()`` and reads the
#     per-digit bone lengths / origins / angles from it. Any import failure
#     (notably ``dynamixel_sdk`` missing, which ``gepetto_core.__init__`` imports)
#     falls back to :data:`DEFAULT_HAND_DIMENSIONS` so this works standalone.

#     Returns a dict with the same keys as :data:`DEFAULT_HAND_DIMENSIONS`.
#     """
#     try:
#         from gepetto_core.geometry import HandGeometry
#         g = HandGeometry.default()
#         dims = {
#             "bl_finger": g.bl_finger,
#             "o_finger": g.o_finger,
#             "a_finger": g.a_finger,
#             "bl_thumb": g.bl_thumb[:1],
#             "o_thumb": g.o_thumb[:1],
#             "a_thumb": g.a_thumb[:1],
#         }
#         print("[hand dims] loaded from gepetto_core HandGeometry.default()")
#         return dims
#     except Exception as exc:  # noqa: BLE001 - any import/parse failure -> fallback
#         print(f"[hand dims] gepetto_core unavailable ({exc.__class__.__name__}: "
#               f"{exc}); using DEFAULT_HAND_DIMENSIONS")
#         return DEFAULT_HAND_DIMENSIONS


# Fallback copy of the gepetto_core default hand (bundled parameters.scad),
# used when gepetto_core is not importable. Fingers: index, middle, ring, pinky.
# Only the first thumb is used (n_thumbs = 1). Lengths/origins in mm, angles in deg.
DEFAULT_HAND_DIMENSIONS = {
    "bl_finger": [[80, 38, 20, 20],
                  [82, 41, 20.5, 20.5],
                  [79, 38, 19, 19],
                  [74, 35, 17, 17]],
    "o_finger": [[0, 1.5, 0], [0, 0, -12], [0, 1, -24], [0, 3.5, -34]],
    "a_finger": [[0, -10, 0], [0, 3, 0], [0, 16, 0], [0, 30, 0]],
    "jd_finger": [[8.75, 12.5, 7.5, 6.25, 5.0], 
                  [14.0, 14.0, 8.4, 7.0, 5.6], 
                  [8.75, 12.5, 7.5, 6.25, 5.0], 
                  [7.7, 11.0, 6.6, 5.5, 4.4]],
    "w_finger": [[10.5, 15.0, 12.0, 10.5, 10.5],
                 [11.2, 16.0, 12.8, 11.2, 11.2],
                 [10.5, 15.0, 12.0, 10.5, 10.5],
                 [9.1, 13.0, 10.4, 9.1, 9.1]],
    "bl_thumb": [[25, 35, 25, 25]],
    "o_thumb": [[-3, 5.5, 15]],
    "a_thumb": [[0, -100, 0]],
    "jd_thumb": [[11.2, 16.0, 9.6, 8.0, 6.4]],
    "w_thumb": [[14.0, 20.0, 16.0, 14.0, 14.0]]
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
            "jd_finger": g.jd_finger,
            "w_finger": g.w_finger,
            "bl_thumb": g.bl_thumb[:1],
            "o_thumb": g.o_thumb[:1],
            "a_thumb": g.a_thumb[:1],
            "jd_thumb": g.jd_thumb[:1],
            "w_thumb": g.w_thumb[:1],
        }
        print("[hand dims] loaded from gepetto_core HandGeometry.default()")
        return dims
    except Exception as exc:  # noqa: BLE001 - any import/parse failure -> fallback
        print(f"[hand dims] gepetto_core unavailable ({exc.__class__.__name__}: "
              f"{exc}); using DEFAULT_HAND_DIMENSIONS")
        return DEFAULT_HAND_DIMENSIONS


def bone_joint_spec_from_bones(bone_lengths_mm, joint_lengths_mm=None):
    """Interleave 4 physical bone lengths (mm) with the 3 standard joint lengths.
    Updated to accept dynamic CAD joint dimensions."""
    if len(bone_lengths_mm) != 4:
        raise ValueError(
            f"expected 4 bone lengths, got {len(bone_lengths_mm)}: {bone_lengths_mm}")

    # Use CAD joint diameters if provided, otherwise fallback to standard
    if joint_lengths_mm is not None and len(joint_lengths_mm) >= 3:
        # Assuming the 3 middle elements of the jd array correspond to the MCP, PIP, DIP joints
        j_mm = joint_lengths_mm[1:4]
    else:
        j_mm = [j * 1000.0 for j in _STANDARD_JOINT_LENGTHS]

    mcp_e, pip_e, dip_e = (j / 2.0 for j in j_mm)

    adjusted_mm = [
        bone_lengths_mm[0] - mcp_e,
        bone_lengths_mm[1] - mcp_e - pip_e,
        bone_lengths_mm[2] - pip_e - dip_e,
        bone_lengths_mm[3] - dip_e,
    ]

    bones = [b / 1000.0 for b in adjusted_mm]
    j = [x / 1000.0 for x in j_mm]

    return [
        ("bone", bones[0]),   # metacarpal
        ("joint", j[0]),      # MCP
        ("bone", bones[1]),   # proximal phalanx
        ("joint", j[1]),      # PIP
        ("bone", bones[2]),   # middle phalanx
        ("joint", j[2]),      # DIP
        ("bone", bones[3]),   # distal phalanx
    ]


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


# def get_default_hand_configs(dims=None):
#     """Build ``[(name, cfg), ...]`` for a 4-finger + thumb hand from morphology.

#     Each digit is a standard 6-tendon finger sized from its physical bone lengths
#     and placed via ``hand_base_offset`` from its palm origin/angle. ``dims`` defaults
#     to :func:`load_hand_dimensions` (gepetto_core, else fallback). No contact is
#     attached, so the caller gets a pure-kinematics hand.
#     """
#     if dims is None:
#         dims = load_hand_dimensions()

#     configs = []
#     for i, name in enumerate(FINGER_NAMES):
#         cfg = get_6tendon_config(
#             bone_joint_spec=bone_joint_spec_from_bones(dims["bl_finger"][i]))
#         cfg.hand_base_offset = finger_base_offset(
#             dims["o_finger"][i], dims["a_finger"][i])
#         configs.append((name, cfg))

#     cfg_thumb = get_6tendon_config(
#         bone_joint_spec=bone_joint_spec_from_bones(dims["bl_thumb"][0]))
#     cfg_thumb.hand_base_offset = finger_base_offset(
#         dims["o_thumb"][0], dims["a_thumb"][0])
#     configs.append(("thumb", cfg_thumb))
#
#     return configs

def _tip_radius_from_width(width_mm):
    """CAD tip *width* (mm, diameter) -> tip sphere *radius* (m) for contact."""
    return (width_mm / 2.0) / 1000.0


def default_hand_tip_radii(dims=None):
    """Per-digit fingertip contact radius (m), aligned with the digit order of
    :func:`get_default_hand_configs` (index, middle, ring, pinky, thumb).

    Derived from each digit's CAD distal tip *width* (``w_finger`` / ``w_thumb``
    last column, a diameter in mm). The ``TendonFingerSolverConfig`` C++ object
    has no ``tip_radius`` field, so the radius is returned alongside the configs
    for the caller to feed into its contact env (``EnvironmentConfig`` /
    ``SphereContact``) instead of being stored on the config.
    """
    if dims is None:
        dims = load_hand_dimensions()
    radii = [_tip_radius_from_width(dims["w_finger"][i][-1])
             for i in range(len(FINGER_NAMES))]
    radii.append(_tip_radius_from_width(dims["w_thumb"][0][-1]))
    return radii
