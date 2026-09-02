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

from .spec import _SITE_FRAMES, DIGIT_NAMES, URDF_PATH

#: Where the vendored .gltf/.bin pairs live. Committed rather than fetched --
#: they total under a megabyte, so the workbench works offline on a fresh clone.
MESH_DIR = URDF_PATH.parent / "meshes"

#: The URDF link the palm mesh belongs to. It hangs off the model root, so it
#: rides on the WRIST rather than on any digit.
_PALM_LINK = "palm_link"


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
    """``[(attach, path)]`` for every link mesh this hand has.

    ``attach`` is ``None`` for the palm, which rides on the wrist, or
    ``(digit_index, site_index)`` for a link, indexing the same sites
    ``HandKinematics::site_pose_key`` addresses. Every URDF visual origin on this
    hand is the identity, so a mesh is drawn at its site's pose directly.

    Returns only files that are actually present, so a checkout missing them
    degrades to the skeleton rather than raising.
    """
    files = _link_mesh_files()
    found: list[tuple[tuple[int, int] | None, Path]] = []

    palm = files.get(_PALM_LINK)
    if palm and (MESH_DIR / palm).exists():
        found.append((None, MESH_DIR / palm))

    for d, digit in enumerate(DIGIT_NAMES):
        # Site 0 is the fixed mount and carries no link of its own, so the
        # frames listed for sites 1..N line up with index + 1.
        for s, link in enumerate(_SITE_FRAMES[digit], start=1):
            name = files.get(link)
            if name and (MESH_DIR / name).exists():
                found.append(((d, s), MESH_DIR / name))
    return found


def available() -> bool:
    """Whether the meshes are on disk. False degrades the workbench to the
    skeleton, which is a complete and correct drawing in its own right."""
    return MESH_DIR.is_dir() and any(MESH_DIR.glob("*.gltf"))
