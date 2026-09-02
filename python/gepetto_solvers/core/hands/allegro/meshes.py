"""The Allegro hand's visual meshes, and where each one attaches.

The URDF names a mesh per link. This resolves those names to the vendored files
and pairs each with the SITE it rides on, so a renderer can draw the hand's
actual shape by putting each mesh at the pose the solve already computed for that
site -- no separate kinematic pass.

Visual geometry only. Collision in this URDF is boxes and spheres, and the solve
uses its own sphere set, so nothing here is load-bearing: a missing mesh costs
the picture its skin, not its correctness.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .spec import _SITE_FRAMES, DIGIT_NAMES, URDF_PATH

#: Where the vendored .gltf/.bin pairs live. Committed rather than fetched --
#: they total under a megabyte, so the workbench works offline on a fresh clone.
MESH_DIR = URDF_PATH.parent / "meshes"

#: The URDF link the palm mesh belongs to. It hangs off the model root, so it
#: rides on the WRIST rather than on any digit.
_PALM_LINK = "palm_link"


def _gltf_to_urdf():
    """glTF is Y-UP; URDF and ROS are Z-up. R_x(+90) is the difference.

    The conversion is IMPLICIT in the format -- these files carry an identity
    node transform, so nothing in the data says which way is up and a loader
    that just reads vertices gets a hand lying on its side. Drake applies this
    internally; we have to apply it ourselves.

    Both the axis and the SIGN are checked against the URDF's own collision
    boxes, which are written in the link frame and so are independent ground
    truth. After R_x(+90) every link's long axis agrees with its box's, and the
    mesh centres sit a mean 3.2 mm from the collision origins -- the difference
    between a mesh and the box approximating it. R_x(-90) puts them 36.8 mm out,
    with the links extending backwards out of their joints.
    """
    T = np.eye(4)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]])
    return T


#: Mesh-local correction applied to every mesh from this URDF. See above.
GLTF_TO_URDF = _gltf_to_urdf()


def _rpy_xyz(origin):
    """A URDF ``<origin>`` element as a 4x4. Missing element or attribute means
    identity, which is what the URDF spec says."""
    T = np.eye(4)
    if origin is None:
        return T
    r, p_, y = (float(v) for v in origin.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p_),
                              np.sin(p_), np.cos(y), np.sin(y))
    T[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                 @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                 @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    T[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return T


def _fixed_placement(link_name):
    """A link's constant placement in the model ROOT frame, as a 4x4.

    Only meaningful for a link reached from the root through fixed joints --
    which is what the palm is. Raises if a movable joint is on the way, because
    then there is no constant answer and a caller composing one would be drawing
    the palm somewhere it only sometimes is.

    This is why the palm needs it at all: it does not sit at the root.
    Allegro's URDF puts `palm_link` 95 mm up the root's +Z through the fixed
    `root_to_base` joint, so a palm mesh drawn at the wrist lands a whole palm
    height low, hanging below the finger bases instead of behind them.
    """
    root = ET.parse(URDF_PATH).getroot()

    # child link -> the joint that attaches it. A joint missing either end is
    # malformed URDF and is skipped rather than crashing the walk; the loop
    # below then simply stops there, which is the same answer as reaching the
    # root.
    child_of = {}
    for j in root.findall("joint"):
        child = j.find("child")
        if child is not None and child.get("link"):
            child_of[child.get("link")] = j

    T = np.eye(4)
    name = link_name
    seen = set()
    while name in child_of:
        if name in seen:
            raise ValueError(f"cycle in the URDF joint tree at {name!r}")
        seen.add(name)

        joint = child_of[name]
        if joint.get("type") != "fixed":
            raise ValueError(
                f"{link_name} sits below the movable joint "
                f"{joint.get('name')!r}, so it has no constant placement in the "
                f"root frame")
        T = _rpy_xyz(joint.find("origin")) @ T

        parent = joint.find("parent")
        if parent is None or not parent.get("link"):
            break
        name = parent.get("link")
    return T


def _link_mesh_files():
    """``{link name: filename}`` from the URDF's ``<visual>`` blocks.

    Parsed rather than tabulated because the mapping is many-to-one -- the three
    identical fingers share one set of mesh files -- and a table would be a
    second place for that to be written down wrongly.
    """
    root = ET.parse(URDF_PATH).getroot()
    out: dict[str, str] = {}
    for link in root.findall("link"):
        mesh = link.find("visual/geometry/mesh")
        name = link.get("name")
        filename = mesh.get("filename") if mesh is not None else None
        # A <mesh> with no filename, or a link with no name, is malformed URDF.
        # Skip it: this is scenery, and losing one link's skin beats refusing to
        # draw the hand.
        if name is None or filename is None:
            continue
        out[name] = filename.rsplit("/", 1)[-1]
    return out


def visual_meshes():
    """``[(attach, path, T_local)]`` for every link mesh this hand has.

    ``attach`` is ``None`` for the palm, which rides on the wrist, or
    ``(digit_index, site_index)`` for a link, indexing the same sites
    ``HandKinematics::site_pose_key`` addresses.

    ``T_local`` takes the mesh's own coordinates into the frame it attaches to.
    It is carried PER MESH rather than assumed by the renderer, because it is a
    property of the asset and not of the drawing: these files are glTF, which is
    Y-up (see :data:`GLTF_TO_URDF`), whereas a hand shipping Z-up STL or OBJ
    would give the identity here. Every URDF visual origin on this hand is the
    identity, so the axis correction is all there is; a hand whose URDF offsets
    its visuals would compose that in too.

    Returns only files that are actually present, so a checkout missing them
    degrades to the skeleton rather than raising.
    """
    files = _link_mesh_files()
    found: list[tuple[tuple[int, int] | None, Path, np.ndarray]] = []

    palm = files.get(_PALM_LINK)
    if palm and (MESH_DIR / palm).exists():
        # The palm rides on the WRIST, but its mesh belongs to `palm_link`,
        # which is not at the wrist -- so its own placement composes in ahead of
        # the axis correction.
        found.append((None, MESH_DIR / palm,
                      _fixed_placement(_PALM_LINK) @ GLTF_TO_URDF))

    for d, digit in enumerate(DIGIT_NAMES):
        # Site 0 is the fixed mount and carries no link of its own, so the
        # frames listed for sites 1..N line up with index + 1.
        for s, link in enumerate(_SITE_FRAMES[digit], start=1):
            name = files.get(link)
            if name and (MESH_DIR / name).exists():
                found.append(((d, s), MESH_DIR / name, GLTF_TO_URDF))
    return found


def available() -> bool:
    """Whether the meshes are on disk. False degrades the workbench to the
    skeleton, which is a complete and correct drawing in its own right."""
    return MESH_DIR.is_dir() and any(MESH_DIR.glob("*.gltf"))
