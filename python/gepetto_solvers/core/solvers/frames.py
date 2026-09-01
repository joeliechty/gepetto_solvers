"""Pose conventions and frame arithmetic.

ZYX (yaw-pitch-roll) throughout, defined once here so a pose typed into a CLI, a
slider and a test all mean the same rotation.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Hand base / wrist start pose.
# ---------------------------------------------------------------------------

def euler_to_R(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) rotation matrix from radians.

    The convention the wrist-pose sliders and the headless harnesses all quote
    poses in, defined once here so a pose typed into a CLI, a slider and a test
    all mean the same rotation."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    """(roll, pitch, yaw) in radians from a ZYX rotation matrix. Inverse of
    :func:`euler_to_R`, so a pose can round-trip through the sliders."""
    R = np.asarray(R, float)
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) > 1.0 - 1e-9:      # gimbal lock: fold roll into yaw
        return 0.0, float(pitch), float(np.arctan2(-R[0, 1], R[1, 1]))
    return (float(np.arctan2(R[2, 1], R[2, 2])), float(pitch),
            float(np.arctan2(R[1, 0], R[0, 0])))


def wrist_pose_from_xyzrpy(xyz, rpy):
    """4x4 base pose from a translation (m) and ZYX euler angles (rad)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_R(*rpy)
    T[:3, 3] = np.asarray(xyz, float)
    return T


def solved_wrist_pose(configs, frame):
    """The wrist pose a solved frame actually ended at, as a 4x4, recovered from
    a FRAME alone.

    The wrist is a VARIABLE, not a fixed input: its prior is soft (sigma_wrist_*)
    and contact pulls against it, so a solve that presses the hand onto a table
    moves the base tens of millimetres away from the commanded pose.

    This inverts digit 0's mounting offset out of its base site, which is exact
    for any hand: ``HandKinematics::digit_base_offset`` defines site 0 as the
    digit's FIXED attachment, ``T_0 = T_wrist o T_offset``, so the relation holds
    whatever the mechanism does downstream of it. All digits agree to machine
    precision -- they share the one wrist variable.

    Prefer :meth:`HandResult.wrist_pose` where a result is in hand: the state
    bundle carries the wrist directly, so it needs no configs and no inversion.
    This exists for the callers that only have a frame."""
    name, cfg = configs[0]
    T0 = np.asarray(frame[name].marginals.sites[0].pose.mean, float)
    return T0 @ np.linalg.inv(np.asarray(cfg.hand_base_offset, float))


def disc_pose(frame, finger_name, disc):
    """World pose of one routing disc's body frame, as a 4x4.

    The same ``disc_pose_idx`` walk ``_plotting/viser_hand._update_disc_frames``
    does, so this returns exactly the frame the *disc frames* display toggle
    draws a triad on -- which is what makes it usable as a landmark you can point
    at on screen and then find on the physical hand.
    """
    fm = frame[finger_name].marginals
    node = fm.extras.tendon_config.disc_pose_idx[disc]
    return np.asarray(fm.sites[node].pose.mean, float)


def wrist_to_disc(configs, frame, finger_name, disc):
    """``T_wrist<-disc`` for one disc, measured off a solved frame."""
    return (np.linalg.inv(solved_wrist_pose(configs, frame))
            @ disc_pose(frame, finger_name, disc))


def wrist_pose_for_disc_target(configs, frame, finger_name, disc, T_target):
    """The wrist pose that puts ``disc``'s frame at ``T_target``.

    ONLY VALID FOR A DISC ON THE METACARPAL -- one ``config.proximal_disc_flags``
    marks rigid, i.e. disc 0 or 1. Those two are bolted to the palm, so
    ``T_wrist<-disc`` is a constant of the morphology rather than a function of
    the posture, and inverting it is an exact placement:

        T_wrist = T_target o inv(T_wrist<-disc)

    Measured, not assumed: across the full 0 -> 2.5 N flexor range disc 1 moves
    13-29 um in the wrist frame on every digit, while disc 2 (the first one past
    the MCP joint) moves 4.5-13.8 mm. So the residual left by a single
    application of this is micrometres, and one refinement pass -- re-solve,
    re-measure ``T_wrist<-disc``, re-apply -- clears even that.

    Handed a disc past the MCP it would still return a pose, but the transform it
    inverted describes only the posture it was measured in; callers gate on
    ``proximal_disc_flags`` rather than letting that pass silently.
    """
    return np.asarray(T_target, float) @ np.linalg.inv(
        wrist_to_disc(configs, frame, finger_name, disc))


def disc_frame_error(T_actual, T_target):
    """``(position_mm, rotation_deg)`` between two frames, for a residual readout.

    The rotation is the geodesic angle of ``R_target^T R_actual`` -- one number
    for "how far off is the orientation", which is what a calibration readout
    wants, rather than three Euler components that trade against each other.
    """
    A = np.asarray(T_actual, float)
    B = np.asarray(T_target, float)
    pos = float(np.linalg.norm(A[:3, 3] - B[:3, 3])) * 1e3
    # Clipped because a trace a hair outside [-1, 3] from round-off makes arccos
    # return nan, which reads as a broken solve rather than a perfect one.
    cos = (np.trace(B[:3, :3].T @ A[:3, :3]) - 1.0) / 2.0
    return pos, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# The default hand base pose: lifted 75 mm along the support normal and pitched
# -1.22 rad about +Y. The mount puts the palm along the base frame's -x, so that
# pitch swings the palm to face roughly -z -- i.e. the hand hovers palm-down over
# the object at the default grasp locus, fingers already aimed at it, instead of
# standing at the identity pose with the palm pointing sideways and the fingers
# through the scene.
#
# This is the posing the interactive visualizer opens on. Keep the two in sync:
# the visualizer seeds its sliders from these numbers rather than repeating them.
DEFAULT_WRIST_XYZ = (0.0, 0.0, 0.075)


DEFAULT_WRIST_RPY = (0.0, -1.22, 0.0)


def default_wrist_pose():
    """The default hand base pose as a 4x4 (fresh array per call, since it is a
    dataclass field default and callers mutate poses in place)."""
    return wrist_pose_from_xyzrpy(DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY)
