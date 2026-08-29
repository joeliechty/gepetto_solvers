"""Small pure helpers: quaternion conversion and object mesh construction."""

import numpy as np
import trimesh

from .palette import _OBJECT_RGB


def _wxyz_from_R(R):
    """viser's (w, x, y, z) quaternion from a 3x3 rotation.

    Shepperd's method: pick the largest of the four diagonal combinations so the
    square root is never taken of something near zero. The naive w-first formula
    loses all precision at 180 deg rotations, and the mount transform is exactly
    that kind of pose.
    """
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return tuple(q)


def _recenter(mesh):
    """Translate a trimesh primitive so its bounding box is centered on the origin
    (trimesh's capsule/cylinder are not all origin-centered)."""
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def _object_trimesh(spec, center):
    """Best-effort trimesh for a grasp object, in final world orientation --
    mirrors ``ik_5f_contact._add_object_mesh`` (the spec's rotation is already
    baked into how each primitive is drawn, so it is not re-applied here)."""
    t = spec["type"]
    if t == "cylinder":
        mesh = trimesh.creation.cylinder(radius=spec["radius"], height=spec["height"])
    elif t == "capsule":
        mesh = trimesh.creation.capsule(height=spec["height"], radius=spec["radius"])
    else:
        return None
    _recenter(mesh)
    mesh.apply_translation(np.asarray(center, float))
    mesh.visual.vertex_colors = np.array([*_OBJECT_RGB, 120], dtype=np.uint8)
    return mesh
