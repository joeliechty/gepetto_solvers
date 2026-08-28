"""Config builders for a multi-finger tendon *hand* whose fingers share one
floating wrist base.

Each finger is a standard 6-tendon finger (reusing
``finger_config.get_6tendon_config``); the only hand-specific piece is the
per-finger ``hand_base_offset`` that places that finger relative to the shared
wrist. The C++ ``TendonHandSolver`` gives every finger the *same* wrist variable
``T_base`` with ``T_0 = T_base o hand_base_offset``, so changing an offset just
repositions that finger on the common wrist. Add a finger by appending one more
``(name, config)`` entry — no code changes required.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .finger_config import get_6tendon_config


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


def hand_growth_axis(configs):
    """Mean rod-growth direction in the hand BASE frame, as a unit vector.

    Each finger's rod grows along the local +z of its ``hand_base_offset``, so
    this is ``mean_i(offset_i[:3,:3] @ [0,0,1])`` normalized. Purely analytic — no
    solve — which makes it a cheap cross-check on the growth axis measured from a
    forward-kinematics tip centroid (the two agree to ~9 deg on the default hand;
    they differ because the fingers fan out and curl).
    """
    axes = [np.asarray(cfg.hand_base_offset, dtype=float)[:3, :3] @ np.array([0.0, 0.0, 1.0])
            for _, cfg in configs]
    g = np.mean(axes, axis=0)
    return g / np.linalg.norm(g)


def tip_node_index(config):
    """Index of the last rod node (the tip) for a finger config."""
    num_nodes = config.num_discs + (config.num_discs - 1) * config.num_between_nodes
    return num_nodes - 1


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


# ---------------------------------------------------------------------------
# Measured pinch geometry for THIS hand
# ---------------------------------------------------------------------------
#
# Where each combination of digits actually meets when it closes, measured
# offline by tests/tendon_hand/fk_pinch_centroids.py (--q-max 4.5) against the
# hand get_default_hand_configs() builds above.
#
# THESE NUMBERS BELONG TO THIS MORPHOLOGY. They are a property of the bone
# lengths, palm origins and base angles in DEFAULT_HAND_DIMENSIONS /
# gepetto_core, and of finger_base_offset()'s mounting convention -- change any
# of those and every entry here is silently wrong, because nothing in the code
# can detect the mismatch. That is exactly why they live in this module, next
# to the dimensions they were derived from, rather than in a solver or a demo
# script. Regenerate with:
#
#     python -m tests.tendon_hand.fk_pinch_centroids --q-max 4.5
#
# The centroid is in the WRIST / HAND-BASE frame, which is what makes it usable
# as a constraint: PreGraspCentroidFactor pushes it through the wrist pose to
# get a world point. Measured with the wrist pinned at identity (the solved
# wrist held to 7e-16, so the tip poses were already base-frame).

# The digit order get_default_hand_configs() returns, thumb last. FINGER_NAMES
# above is the four non-thumb fingers only, so it cannot be reused here.
DIGIT_ORDER = FINGER_NAMES + ["thumb"]


@dataclass(frozen=True)
class PinchPose:
    """Where one combination of digits meets, and what closes it.

    ``centroid``   (x, y, z) in meters, WRIST/HAND-BASE frame: the centroid of
                   the combination's fingertip contact spheres at their closest
                   approach.
    ``tensions``   ``{finger: flexor tension N}`` that produces that pose --
                   what to command to actually close this pinch. Note some
                   exceed the interactive viewer's 3 N slider maximum (the
                   pinky needs up to 3.55 N).
    ``gap``        meters, the closest tip-sphere pair's SURFACE separation
                   there. <= 0 means the spheres genuinely touch; positive
                   means this combination never closes and ``centroid`` is a
                   closest-approach point rather than a contact point. Quoted
                   to the source log's 0.1 mm precision.
    """
    centroid: Tuple[float, float, float]
    tensions: Dict[str, float]
    gap: float

    def touches(self, tol=2e-4):
        """Whether these digits actually reach each other (vs merely getting
        as close as the hand allows). 7 of the 15 combinations do not."""
        return self.gap <= tol


def _pinch_key(finger_names):
    """Canonical lookup key: the given digits in ``DIGIT_ORDER``, deduplicated.

    Order-insensitive so a caller can pass a contact mask, a set, or whatever
    order the GUI checkboxes happen to be read in and still hit the same entry.
    """
    wanted = set(finger_names)
    return tuple(n for n in DIGIT_ORDER if n in wanted)


HAND_PINCH_POSES: Dict[Tuple[str, ...], PinchPose] = {
    ("index", "thumb"): PinchPose(
        (-0.07212, 0.07190, 0.00335), {"thumb": 1.25, "index": 2.70}, -0.0002),
    ("middle", "thumb"): PinchPose(
        (-0.07260, 0.07270, -0.00795), {"thumb": 1.35, "middle": 2.20}, 0.0013),
    ("ring", "thumb"): PinchPose(
        (-0.06388, 0.06684, -0.02104), {"thumb": 1.50, "ring": 2.75}, 0.0009),
    ("pinky", "thumb"): PinchPose(
        (-0.05512, 0.06067, -0.02966), {"thumb": 1.60, "pinky": 3.55}, 0.0000),
    ("index", "middle", "thumb"): PinchPose(
        (-0.07223, 0.06981, -0.00553),
        {"thumb": 1.35, "index": 2.70, "middle": 2.25}, -0.0030),
    ("index", "ring", "thumb"): PinchPose(
        (-0.06792, 0.06592, -0.01107),
        {"thumb": 1.40, "index": 2.80, "ring": 2.80}, 0.0024),
    ("index", "pinky", "thumb"): PinchPose(
        (-0.06178, 0.06110, -0.01766),
        {"thumb": 1.50, "index": 2.95, "pinky": 3.50}, 0.0050),
    ("middle", "ring", "thumb"): PinchPose(
        (-0.06911, 0.06667, -0.01631),
        {"thumb": 1.45, "middle": 2.30, "ring": 2.75}, 0.0012),
    ("middle", "pinky", "thumb"): PinchPose(
        (-0.06443, 0.06283, -0.02112),
        {"thumb": 1.50, "middle": 2.40, "pinky": 3.40}, 0.0041),
    ("ring", "pinky", "thumb"): PinchPose(
        (-0.06135, 0.06281, -0.02616),
        {"thumb": 1.55, "ring": 2.85, "pinky": 3.40}, 0.0009),
    ("index", "middle", "ring", "thumb"): PinchPose(
        (-0.07047, 0.06670, -0.01122),
        {"thumb": 1.40, "index": 2.75, "middle": 2.30, "ring": 2.75}, 0.0008),
    ("index", "middle", "pinky", "thumb"): PinchPose(
        (-0.06446, 0.05954, -0.01608),
        {"thumb": 1.50, "index": 2.95, "middle": 2.45, "pinky": 3.50}, 0.0001),
    ("index", "ring", "pinky", "thumb"): PinchPose(
        (-0.06282, 0.05976, -0.01868),
        {"thumb": 1.50, "index": 2.95, "ring": 2.95, "pinky": 3.50}, 0.0024),
    ("middle", "ring", "pinky", "thumb"): PinchPose(
        (-0.06508, 0.06159, -0.02133),
        {"thumb": 1.50, "middle": 2.40, "ring": 2.90, "pinky": 3.40}, 0.0017),
    ("index", "middle", "ring", "pinky", "thumb"): PinchPose(
        (-0.06628, 0.06096, -0.01636),
        {"thumb": 1.45, "index": 2.90, "middle": 2.40, "ring": 2.90,
         "pinky": 3.45}, 0.0004),
}


def pinch_pose(finger_names) -> Optional[PinchPose]:
    """The measured :class:`PinchPose` for a set of digits, or None.

    None means the combination was never measured, which is the honest answer
    for anything the scan did not cover: fewer than two digits, or any set
    WITHOUT the thumb. Non-thumb sets are excluded on purpose -- those fingers
    are all on the same side of the palm, so their "closest approach" is a
    fist curl rather than a pinch, and calling that a grasp centroid would be
    wrong. Callers must handle None rather than substituting a default.
    """
    return HAND_PINCH_POSES.get(_pinch_key(finger_names))


def pinch_pose_for_mask(configs, contact_fingers) -> Optional[PinchPose]:
    """:func:`pinch_pose` driven by a per-finger bool mask in ``configs``
    order -- the form the solver params and the GUI checkboxes carry."""
    mask = _resolve_contact_mask(configs, contact_fingers)
    return pinch_pose([name for (name, _), on in zip(configs, mask) if on])


# ---------------------------------------------------------------------------
# Collision avoidance (Section 1.5)
# ---------------------------------------------------------------------------


def disc_node_indices(config):
    """Rod node index of each disc, proximal (0) -> distal (tip)."""
    return [i * (config.num_between_nodes + 1) for i in range(config.num_discs)]


def proximal_disc_flags(config, num_proximal_discs=2):
    """Parallel to :func:`disc_node_indices`: 1 for disc nodes on the rigidly-
    attached proximal (metacarpal) bone, else 0.

    The metacarpal is the first bone in the ``bone_joint`` spec and spans discs
    0 and 1, so ``num_proximal_discs`` defaults to 2. Finger-finger collision in
    the hand skips a sphere pair iff *both* spheres are proximal, so marking the
    metacarpal discs proximal keeps the rigidly-attached bases from being checked
    against each other (they cannot move relative to one another).
    """
    return [1 if d < num_proximal_discs else 0 for d in range(config.num_discs)]


def _resolve_contact_mask(configs, contact_fingers):
    """Validate a per-finger contact mask against ``configs`` and normalize it to
    a list of bools (``None`` => every finger contacts, the legacy behavior)."""
    if contact_fingers is None:
        return [True] * len(configs)
    mask = [bool(f) for f in contact_fingers]
    if len(mask) != len(configs):
        raise ValueError(
            f"contact_fingers has {len(mask)} entries but there are "
            f"{len(configs)} fingers; pass one flag per finger.")
    return mask


def attach_contact(configs, spec, objects_dir, primitive, object_pose, *,
                   tip_radii=None, radius=None, contact_fingers=None,
                   object_pose_cov=None, proxy_and_exact=False,
                   drop_normal_row=False, ellipsoid_set_beta=None,
                   in_plane=False, pinch_centroid=None, contact_subset=None):
    """Attach the shared object surface + a terminal tip contact to every finger
    of a hand config list, in place. Returns ``configs`` for chaining.

    This is the block the contact demo scripts (``ik_5f_contact.py``,
    ``traj_5f_contact.py``, ...) write inline, factored out so the solver classes
    and any caller share one definition of "the fingertip touches the object".

    ``contact_fingers`` (None = all, the legacy behavior) is a per-finger bool
    mask: a finger whose flag is False still gets the env — so
    :func:`attach_collision` / :func:`attach_table` can hang off it and keep that
    finger out of the object — but *without* ``target_contact_node``, so the C++
    layer treats it as a collision-only env: ``TendonHandModel::build_graph``
    adds no witness contact factor for it and ``get_initial_values`` seeds no
    witness point. That is the same shape the trajectory planner already builds
    for every step before k=K, so it is a well-trodden path. Use it to solve for
    a pinch/subset grasp instead of forcing all five fingertips onto the object.

    ``radius`` overrides the contact sphere radius for all fingers; otherwise
    ``tip_radii[i]`` is used. The radius is written even for a masked-off finger
    (it is inert without a contact node, and keeps the env self-describing).

    ``proxy_and_exact`` (Section 1.8 controller) attaches the bounding-ellipsoid
    proxy *and* the baked SDF, so the controller can swap from the proxy it
    slides against in phase 2 to the exact geometry it servos on in phase 3.
    Default False keeps the single-surface behavior every existing caller relies
    on.

    ``ellipsoid_set_beta`` overrides the LogSumExp sharpness for an
    ``ellipsoid_set`` object (None = the spec's own value). Only the smooth-min
    STANDOFF changes with it, not the geometry: the constraint surface sits up to
    ln(K)/beta outside the true union. Inert for every other surface kind.

    ``contact_subset`` (``scene.grasp_subset_indices``) restricts which members
    of an ``ellipsoid_set`` the fingertips may be driven onto -- the authored
    "these shells are handles, those only bound the shape" choice that travels
    with a YCB fit. None = every member, the pre-existing behavior, and inert for
    a surface with no members to choose between.

    It narrows CONTACT ONLY. :func:`attach_collision` shares this very env, and
    the whole set stays on it, so the excluded shells keep pushing the fingers
    out while nothing is sent to touch them. That is the point: they are the
    drill's housing, not a handle, and a hand allowed to pass through them would
    be planning against an object that is not there.

    ``drop_normal_row`` (Eq 2.12-2.15) selects the 4-row witness contact form
    [c_R, c_O, c_T1, c_T2] (c_N dropped) instead of the default 5-row form.
    Written for every finger regardless of ``contact_fingers`` -- it is a
    property of the contact FORM, like ``radius``, not gated by the mask. Only
    affects the witness-point contact factor; inert for a center-direct
    ellipsoid contact, which has no normal row to begin with.

    ``in_plane`` (Eq 13) swaps the object contact equality from the full 3D
    distance to the distance measured inside each finger's pulling plane (Eq 11).
    It needs ``pinch_centroid``: the wrist-frame point where the participating
    digits meet (:func:`pinch_pose_for_mask` off the SAME mask), which is the
    plane's third point. Also a property of the contact FORM, so written to every
    finger's env.

    Three ways to ask for something that cannot be built, all of which RAISE
    rather than quietly falling back to the 3D form -- the same reasoning
    :func:`attach_ellipsoid_set` documents. Degrading silently here would leave
    the caller believing a constraint is in the graph that is not, and the
    resulting grasp would look like a solver failure rather than a mis-request:

      * a binding with no ``object_contact_in_plane`` field,
      * an object with no ellipsoid form (cube/cylinder/capsule, and a baked SDF
        with no analytic look-alike): no cross-section for the plane to cut,
      * no ``pinch_centroid``: Eq 11 has no plane without it, which is what a
        thumbless digit set gives you.
    """
    import crest_sparse

    from .scene import (configure_object_proxy_and_exact, configure_object_surface,
                        ellipsoid_members)

    mask = _resolve_contact_mask(configs, contact_fingers)
    centroid = None
    if in_plane:
        probe = crest_sparse.EnvironmentConfig()
        if not hasattr(probe, "object_contact_in_plane"):
            raise AttributeError(
                "this crest_sparse build has no EnvironmentConfig."
                "object_contact_in_plane, so the Eq 13 in-plane contact cannot be "
                "built -- rebuild it (pip install . from the crest-sparse root)")
        if ellipsoid_members(spec) is None:
            raise ValueError(
                f"in-plane contact (Eq 13) needs an ellipsoid surface to cut, but "
                f"the {spec['type']!r} object {primitive!r} has none -- use a "
                f"sphere, an ellipsoid or a ycb: set, or contact it in 3D")
        if pinch_centroid is None:
            raise ValueError(
                "in-plane contact (Eq 13) needs pinch_centroid, the wrist-frame "
                "point Eq 11 spans the pulling plane with; the checked digits have "
                "no measured pinch pose (only combinations INCLUDING THE THUMB "
                "were measured -- see HAND_PINCH_POSES)")
        centroid = np.asarray(pinch_centroid, dtype=float).reshape(3)
    if object_pose_cov is None:
        object_pose_cov = 1e-8 * np.eye(6)
    setup_surface = (configure_object_proxy_and_exact if proxy_and_exact
                     else configure_object_surface)

    for i, (_, cfg) in enumerate(configs):
        env = crest_sparse.EnvironmentConfig()
        setup_surface(env, spec, objects_dir, primitive,
                      contact_subset=contact_subset)
        if ellipsoid_set_beta is not None and hasattr(env, "ellipsoid_set_beta"):
            env.ellipsoid_set_beta = float(ellipsoid_set_beta)
        env.object_pose_mean = object_pose
        env.object_pose_cov = object_pose_cov
        env.object_pose_per_step = False
        if radius is not None:
            env.contact_node_radius = radius
        elif tip_radii is not None:
            env.contact_node_radius = tip_radii[i]
        env.contact_drop_normal_row = drop_normal_row
        if centroid is not None:
            env.object_contact_in_plane = True
            env.contact_plane_centroid = centroid
        if mask[i]:
            env.target_contact_node = tip_node_index(cfg)
        cfg.sdf_contact = env
    return configs


def attach_collision(configs, vdb_path, object_pose, *,
                     radius=0.003, sigma=1e-4, num_proximal_discs=2,
                     object_pose_cov=None, cull_margin=None, avoidance=True,
                     self_collision=True):
    """Declare the Section 1.5 collision spheres on every finger of a hand config
    list, in place. Returns ``configs`` for chaining.

    Each finger gets collision spheres on its disc nodes (radius ``radius``),
    with the metacarpal discs flagged proximal. If a finger already has an
    ``sdf_contact`` env (e.g. a terminal tip contact), the collision fields are
    added to that same env so contact and collision share one object; otherwise
    a fresh collision-only env is created and the SDF loaded.

    The C++ hand builder then adds, at every trajectory step, sphere-to-SDF
    inequalities keeping each finger out of the object and sphere-to-sphere
    inequalities keeping distinct fingers apart (skipping proximal-proximal
    pairs).

    The sphere SET is one thing; the three constraint families built on it are
    three others, each with its own switch -- ``avoidance`` (finger-OBJECT,
    ``env.collision_avoidance``), ``self_collision`` (FINGER-FINGER,
    ``env.self_collision``) and :func:`attach_table`'s ``avoidance``
    (finger-PLANE, ``env.plane_avoidance``). Declaring the spheres builds
    nothing on its own; every family is gated on its own field alone, so any
    combination of the three is available. ``avoidance=False`` with a support
    plane is how a caller turns table collision on with object collision off;
    ``self_collision`` defaults True because keeping the fingers out of each
    other is wanted in nearly every solve.

    ``cull_margin`` (m, None = keep all pairs): drop finger-finger sphere pairs
    whose gap at the initial values exceeds this margin. Heuristic speedup —
    roughly half the 5-finger trajectory graph is inequality constraints that
    never activate — but a culled pair is unprotected, so rely on the tests'
    all-pairs penetration report to validate the chosen margin. Finger-object
    constraints are never culled.
    """
    import crest_sparse

    if object_pose_cov is None:
        object_pose_cov = 1e-8 * np.eye(6)

    for _, cfg in configs:
        env = cfg.sdf_contact            # copy (or None) via the optional binding
        if env is None:
            env = crest_sparse.EnvironmentConfig()
            env.load_sdf(vdb_path)
            env.object_pose_mean = object_pose
            env.object_pose_cov = object_pose_cov
            env.object_pose_per_step = False

        nodes = disc_node_indices(cfg)
        env.collision_avoidance = bool(avoidance)
        if not hasattr(env, "self_collision"):
            if not self_collision:
                raise AttributeError(
                    "this crest_sparse build has no "
                    "EnvironmentConfig.self_collision, so finger-finger "
                    "avoidance cannot be turned off -- rebuild it "
                    "(pip install .)")
        else:
            env.self_collision = bool(self_collision)
        env.collision_sigma = sigma
        env.collision_node_indices = nodes
        env.collision_node_radii = [radius] * len(nodes)
        env.collision_node_is_proximal = proximal_disc_flags(cfg, num_proximal_discs)
        if cull_margin is not None:
            env.collision_cull_margin = cull_margin
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_table(configs, plane_origin, plane_normal, *,
                 avoidance=True, contact_node=None, radius=None,
                 tip_radii=None, dims=None, contact_fingers=None):
    """Attach a Section 1.6 world-fixed analytic support plane ("table") to every
    finger of a hand config list, in place. Returns ``configs`` for chaining.

    The table is a half-space with origin ``plane_origin`` and OUTWARD unit normal
    ``plane_normal`` (``SDF_table(p) = (p - origin) . normal``). The plane fields
    are written onto each finger's existing ``sdf_contact`` env (created if absent)
    so contact/collision/table share one env. Per finger this sets:
      * ``plane_origin`` / ``plane_normal`` — the support surface,
      * ``plane_avoidance`` = ``avoidance`` — the free-space approach collision
        (Eq 1.59): every non-tip collision sphere is kept out of the half-space,
      * ``table_contact_node`` — the fingertip node that slides on the plane
        (defaults to ``tip_node_index(cfg)``, and see ``contact_fingers`` below).
        That node gets a SINGLE-residual equality on its sphere CENTER,
        ``Dist_plane(c) = 0`` (``PlaneCollisionGapFactor`` as a
        ``ZeroCostConstraint``) — not the original §1.6 five-residual witness
        form, which introduced a free contact point whose gauge four of its rows
        existed only to pin. The C++ planner *schedules* this field per step
        around ``k_touch`` (cleared during the approach phase, kept during the
        slide phase), so it is safe to set it for every step here,
      * ``table_contact_radius`` — that tip's contact sphere radius.

    ``contact_fingers`` (None = all, the legacy behavior) is the same per-finger
    bool mask :func:`attach_contact` takes: a finger that is not solving for
    contact gets the plane and its avoidance inequality but *no*
    ``table_contact_node``, so nothing asks it to touch the table either. Where
    ``avoidance`` is active that fingertip is then held *above* the plane rather
    than pinned to it — the C++ layer exempts the table contact node from plane
    avoidance (its collision would fight the sliding equality it is pinned by),
    and a masked-off finger no longer has one. During the planner's slide phase
    (k >= ``k_touch``) plane avoidance is off for every finger by design, so
    there a masked-off fingertip is simply unconstrained by the plane.

    ``radius`` overrides the per-finger tip radius for all fingers; otherwise
    ``tip_radii[i]`` (if given) or the env's existing ``contact_node_radius`` is
    used. The plane is treated as absent by the C++ layer whenever the normal has
    zero norm, so this is a no-op-safe opt-in that leaves plane-free runs unchanged.
    """
    import crest_sparse

    mask = _resolve_contact_mask(configs, contact_fingers)
    origin = np.asarray(plane_origin, dtype=float).reshape(3)
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal = normal / np.linalg.norm(normal)

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            env = crest_sparse.EnvironmentConfig()
        env.plane_origin = origin
        env.plane_normal = normal
        env.plane_avoidance = avoidance
        if not mask[i]:
            # No sliding equality for this finger; clear rather than skip, in case
            # the env already carried a contact node from an earlier attach.
            env.table_contact_node = None
        else:
            env.table_contact_node = (contact_node if contact_node is not None
                                      else tip_node_index(cfg))
            if radius is not None:
                env.table_contact_radius = radius
            elif tip_radii is not None:
                env.table_contact_radius = tip_radii[i]
            else:
                env.table_contact_radius = env.contact_node_radius
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def opposition_axis_from_object(plane_normal, e_long):
    """``m_hat = n_hat x e_long``: the opposition axis implied by splitting the
    support surface ALONG the object's longest in-plane axis.

    Putting the split *line* along ``e_long`` (e.g. lengthwise along a pen)
    means the half-space normal -- the direction that actually separates thumb
    from fingers -- is perpendicular to it within the plane. That is exactly
    ``n_hat x e_long``: thumb and fingers end up opposed ACROSS the object's
    width, not split along its length. Get ``e_long`` from
    :func:`scene.object_principal_inplane_axis`, which already handles
    degenerate (in-plane-isotropic) objects with a documented fallback.

    NOTE this generally differs from :func:`opposition_directions`'s legacy
    default of world +X, which is only correct by coincidence when it happens
    to already be perpendicular to the object's long axis -- for an elongated
    object (a pen) oriented so its length runs along world +X, using world +X
    as ``m_hat`` directly splits the two groups ACROSS the object's length
    (bisecting its short axis) instead of along it, putting the thumb near one
    end and the fingers near the other.

    For an object with no long axis at all (a ball) ``e_long`` falls back to
    world +Y, so ``m_hat`` comes out -X: thumb on the -X side of the object,
    the opposing fingers on +X.
    """
    n = np.asarray(plane_normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    e = np.asarray(e_long, dtype=float).reshape(3)
    e = e - (e @ n) * n
    ne = np.linalg.norm(e)
    if ne < 1e-9:
        raise ValueError(
            "opposition_axis_from_object: e_long is parallel to the plane "
            "normal, so it defines no in-plane split direction")
    m = np.cross(n, e / ne)
    return m / np.linalg.norm(m)


def opposition_directions(configs, *, thumb_index=-1, axis=None):
    """Per-finger in-plane unit vectors ``m_hat`` for the Eq 2.16-2.17 (Eq 1.92)
    half-space split.

    Divide the support surface in half along ``axis`` and put the thumb on one
    half, the grasping fingers on the other. This returns ``+axis`` for the
    thumb and ``-axis`` for every other finger, so the two groups are driven to
    opposite halves.

    ``axis`` (default world +X, which is thumb-on-+X and so the MIRROR of what
    the derived path now produces for a shapeless object -- see
    :func:`opposition_axis_from_object`; every live caller passes an explicit
    axis from ``solvers.default_half_space_axis``, and new ones should too)
    must lie IN the support plane -- the
    constraint is only radius-independent when ``n_table . m_hat = 0``, which
    is what makes its Jacobian constant. ``thumb_index`` defaults to the last
    config, matching :func:`get_default_hand_configs` (four fingers, then the
    thumb).
    """
    if axis is None:
        axis = np.array([1.0, 0.0, 0.0])
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    n = len(configs)
    thumb = thumb_index % n
    return [axis if i == thumb else -axis for i in range(n)]


def attach_half_space(configs, split_point, directions, *, contact_fingers=None,
                      margin=0.0, contact_node=None):
    """Attach the Eq 2.16-2.17 (Eq 1.92) opposition half-space to every masked-in
    finger's env, in place. Returns ``configs`` for chaining.

    ``split_point`` is a point on the splitting line (e.g. the object centroid
    projected onto the support surface); ``directions`` is one in-plane unit
    vector per finger, as produced by :func:`opposition_directions`. A finger
    masked off by ``contact_fingers`` gets no half-space.

    ``contact_node`` is the node whose sphere center is constrained (default
    ``tip_node_index(cfg)``, the same fingertip :func:`attach_table` slides),
    written onto this constraint's OWN field ``env.half_space_node``. Standing
    on its own field is the point: the constraint used to be built off
    ``table_contact_node``, so it silently did nothing without table contact.
    It needs no support plane and no contact of any kind, and can be attached
    before or after anything else -- it does need an env to write onto, so call
    it after :func:`attach_contact` or :func:`attach_collision` has made one.

    ``margin`` (m, >= 0) is the MINIMUM STANDOFF each finger must keep from the
    splitting line, written onto ``env.half_space_margin`` -- the constraint the
    C++ ``HalfSpaceGapFactor`` builds is then

        -(c - p_split) . m_hat + margin <= 0 ,

    so the thumb's side and the opposing fingers' side are each held ``margin``
    off the split (a corridor of width ``2 * margin`` between them). At 0 -- the
    default, and the original constraint -- a fingertip sitting exactly ON the
    split is already legal, so opposition alone does not stop the digits closing
    onto each other. Raises on a binding too old to carry the field rather than
    silently dropping the standoff; call
    :func:`solvers.capabilities`'s ``half_space_margin`` to gate on it.
    """
    mask = _resolve_contact_mask(configs, contact_fingers)
    if len(directions) != len(configs):
        raise ValueError(
            f"directions has {len(directions)} entries but there are "
            f"{len(configs)} fingers; pass one m_hat per finger.")
    p_split = np.asarray(split_point, dtype=float).reshape(3)
    margin = float(margin)

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        if mask[i]:
            m = np.asarray(directions[i], dtype=float).reshape(3)
            env.half_space_enabled = True
            env.half_space_split_point = p_split
            env.half_space_normal = m / np.linalg.norm(m)
            if hasattr(env, "half_space_node"):
                env.half_space_node = (contact_node if contact_node is not None
                                       else tip_node_index(cfg))
            if margin != 0.0 and not hasattr(env, "half_space_margin"):
                raise AttributeError(
                    "this crest_sparse build has no "
                    "EnvironmentConfig.half_space_margin -- rebuild it "
                    "(pip install .) to use an opposition standoff")
            if hasattr(env, "half_space_margin"):
                env.half_space_margin = margin
        else:
            env.half_space_enabled = False
            # Clear rather than skip, in case the env already carried a node
            # from an earlier attach -- same reason attach_table clears
            # table_contact_node for a masked-off finger.
            if hasattr(env, "half_space_node"):
                env.half_space_node = None
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_pregrasp_center(configs, *, clearance_height=0.0, clearance_normal=None,
                           contact_fingers=None, contact_node=None):
    """Attach the pre-grasp hand-centering constraint (Eq 2.18-2.19) to every
    PARTICIPATING finger's env, in place. Returns ``configs`` for chaining.

    A HAND-LEVEL constraint: the C++ layer collects every finger with
    ``pregrasp_center_node`` set, groups the one named "thumb" against the
    rest, and adds ONE Vector3 equality centering their sphere-center midpoint
    over the object (raised by ``clearance_height`` along ``clearance_normal``).
    Requires the thumb AND at least one other finger to participate, and a
    nonzero ``clearance_normal``, or the C++ layer silently skips the
    constraint.

    ``contact_fingers`` (None = all) selects which fingers participate, the
    same per-finger bool mask :func:`attach_contact`/:func:`attach_table` take.
    Call AFTER attach_contact (needs an existing env with ``object_pose_mean``/
    ``object_pose_cov`` set, so this constraint can anchor the object pose even
    when no other block does).
    """
    mask = _resolve_contact_mask(configs, contact_fingers)
    normal = (np.asarray(clearance_normal, dtype=float).reshape(3)
             if clearance_normal is not None else np.zeros(3))

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        env.pregrasp_clearance_height = clearance_height
        env.pregrasp_clearance_normal = normal
        env.pregrasp_center_node = (
            (contact_node if contact_node is not None else tip_node_index(cfg))
            if mask[i] else None)
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_pregrasp_axis_alignment(configs, axis, *, contact_fingers=None, contact_node=None):
    """Attach the pre-grasp short-axis alignment constraint (companion to
    Eq 2.16-2.17) to every PARTICIPATING finger's env, in place. Returns
    ``configs`` for chaining.

    A HAND-LEVEL constraint, same shape as :func:`attach_pregrasp_center`: the
    C++ layer collects every finger with ``pregrasp_align_node`` set, groups
    the one named "thumb" against the rest, and adds ONE scalar equality
    aligning the vector between their sphere-center centroids with ``axis``,
    direction-agnostically (squared cosine). Requires the thumb AND at least
    one other finger to participate, and a nonzero ``axis``, or the C++ layer
    silently skips the constraint.

    ``axis`` is a caller-supplied world-frame direction -- typically
    ``solvers.default_half_space_axis(...)``, the SAME axis the opposition
    half-space uses (perpendicular to the object's longest in-plane axis).
    Passed in rather than derived here so this stays a pure env-mutation
    helper, matching :func:`attach_half_space`/:func:`attach_pregrasp_center`.

    ``contact_fingers`` (None = all) selects which fingers participate, the
    same per-finger bool mask every other ``attach_*`` helper here takes.
    Call AFTER attach_contact (needs an existing ``cfg.sdf_contact`` env).
    """
    mask = _resolve_contact_mask(configs, contact_fingers)
    m_hat = np.asarray(axis, dtype=float).reshape(3)
    if np.linalg.norm(m_hat) > 0:
        m_hat = m_hat / np.linalg.norm(m_hat)

    for i, (_, cfg) in enumerate(configs):
        env = cfg.sdf_contact
        if env is None:
            continue
        env.pregrasp_align_axis = m_hat
        env.pregrasp_align_node = (
            (contact_node if contact_node is not None else tip_node_index(cfg))
            if mask[i] else None)
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


def attach_pregrasp_centroid(configs, centroid, *, clearance_height=0.0,
                             clearance_normal=None):
    """Attach the pre-grasp PINCH-CENTROID centering constraint to every
    finger's env, in place. Returns ``configs`` for chaining.

    A HAND-LEVEL constraint like :func:`attach_pregrasp_center`, but with one
    structural difference that shows up in this signature: there is no
    per-finger mask and no ``contact_node``. ``centroid`` is a point FIXED in
    the wrist frame (from :data:`HAND_PINCH_POSES`, chosen by the caller from
    whichever digits are participating), so no finger opts in and no fingertip
    pose enters the residual -- the C++ layer keys the factor off the shared
    wrist variable and the object, and reads these fields off whichever env it
    finds them on first. They are written to every finger's env anyway, the
    same way ``plane_origin``/``plane_normal`` are, so the envs stay uniform.

    The C++ layer silently skips the constraint when ``clearance_normal`` has
    zero norm, so leaving it unset is a safe no-op. Call AFTER attach_contact
    (needs an existing ``cfg.sdf_contact`` env carrying the object pose, which
    this constraint can end up anchoring).
    """
    c = np.asarray(centroid, dtype=float).reshape(3)
    normal = (np.asarray(clearance_normal, dtype=float).reshape(3)
             if clearance_normal is not None else np.zeros(3))

    for _, cfg in configs:
        env = cfg.sdf_contact
        if env is None:
            continue
        env.pregrasp_centroid_point = c
        env.pregrasp_centroid_clearance = float(clearance_height)
        env.pregrasp_centroid_normal = normal
        cfg.sdf_contact = env            # write the (mutated) env back
    return configs


