"""Elementary rotations and the digit base pose they build.

``default_base_rotation`` is the orientation a finger is mounted at before its
per-digit angles are applied; ``finger_base_offset`` in :mod:`morphology` adds
those on top.
"""

import numpy as np


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
