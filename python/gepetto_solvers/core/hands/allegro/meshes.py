"""The Allegro hand's visual meshes, and where each one attaches.

The URDF names a mesh per link. This resolves those names to the vendored files
and pairs each with the SITE it rides on, so a renderer can draw the hand's
actual shape by putting each mesh at the pose the solve already computed for that
site -- no separate kinematic pass.

Visual geometry only. Nothing here is load-bearing: collision in this repository
is the sphere set the solve carries on each digit's sites, the factor graph never
sees a mesh, and a missing file costs the picture its skin, not its correctness.
That is doubly worth stating for this hand, because upstream gives the SAME mesh
as both ``<visual>`` and ``<collision>`` -- we draw it and ignore the claim.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .spec import _SITE_FRAMES, DIGIT_NAMES, URDF_PATH

#: Where the vendored meshes live. Committed rather than fetched -- the set is
#: about 1.5 MB, so the workbench works offline on a fresh clone.
MESH_DIR = URDF_PATH.parent / "meshes"

#: Upstream ships STL; we vendor decimated binary glTF under the same stems (see
#: ``scripts/vendor_allegro_v5_meshes.py``). The URDF is kept VERBATIM so it can
#: be diffed against the manufacturer's, which means the names it gives are the
#: STL ones and the extension is swapped here rather than in the file.
_URDF_SUFFIX = ".stl"
_VENDORED_SUFFIX = ".glb"

#: The URDF link the palm mesh belongs to. It hangs off the model root, so it
#: rides on the WRIST rather than on any digit.
_PALM_LINK = "palm_link"


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


def _scale(mesh):
    """A ``<mesh scale=...>`` attribute as a 4x4. Missing means unity.

    THIS HAND'S MESHES ARE IN MILLIMETRES and the URDF scales them by 1e-3. The
    vendoring script deliberately does not bake that in: the URDF stays the one
    authority on placement, so a mesh drawn without reading its ``scale`` comes
    out a thousand times too large rather than subtly wrong.
    """
    T = np.eye(4)
    if mesh is None:
        return T
    values = [float(v) for v in mesh.get("scale", "1 1 1").split()]
    T[:3, :3] = np.diag(values)
    return T


def _visual_placement(visual):
    """A ``<visual>`` block's mesh-local transform: its origin times its scale.

    NON-IDENTITY ON EVERY LINK OF THIS HAND, and that is the whole reason this
    function exists. Wonik authors all the V5 STLs in one shared assembly frame
    and then writes each link's ``<origin>`` to bring its own part back to its
    joint -- so the palm's mesh carries ``xyz="0.02 0 -0.1" rpy="0 3.14 1.57"``,
    and a renderer that assumes an identity visual origin (which is what a hand
    whose meshes are authored per link gets away with) draws twenty-one parts
    scattered over a 200 mm cube instead of a hand.

    Order matters: scale acts in the mesh's own coordinates, so it goes on the
    right.
    """
    if visual is None:
        return np.eye(4)
    return _rpy_xyz(visual.find("origin")) @ _scale(visual.find("geometry/mesh"))


def _fixed_placement(link_name):
    """A link's constant placement in the model ROOT frame, as a 4x4.

    Only meaningful for a link reached from the root through fixed joints --
    which is what the palm is. Raises if a movable joint is on the way, because
    then there is no constant answer and a caller composing one would be drawing
    the palm somewhere it only sometimes is.

    The rigid kinematics resolves a digit's mount to the WRIST variable, so the
    wrist frame is the model root; a mesh riding on the wrist therefore needs its
    link's offset from that root composed in. On this description the offset
    happens to be the identity -- V5's root is a bare ``world`` link joined to
    ``palm_link`` at the origin -- but it is computed rather than assumed,
    because V4's was not: it put ``palm_link`` 95 mm up the root's +Z, and a palm
    drawn at the wrist landed a whole palm-height low, hanging below the finger
    bases instead of behind them.
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


def _link_visuals():
    """``{link name: (vendored filename, T_local)}`` from the ``<visual>`` blocks.

    Parsed rather than tabulated because the mapping is many-to-one -- the three
    identical fingers share one set of mesh files, and so do their tips -- and a
    table would be a second place for that to be written down wrongly.
    """
    root = ET.parse(URDF_PATH).getroot()
    out: dict[str, tuple[str, np.ndarray]] = {}
    for link in root.findall("link"):
        visual = link.find("visual")
        mesh = link.find("visual/geometry/mesh")
        name = link.get("name")
        filename = mesh.get("filename") if mesh is not None else None
        # A <mesh> with no filename, or a link with no name, is malformed URDF.
        # Skip it: this is scenery, and losing one link's skin beats refusing to
        # draw the hand.
        if name is None or filename is None:
            continue
        stem = filename.rsplit("/", 1)[-1]
        if stem.lower().endswith(_URDF_SUFFIX):
            stem = stem[:-len(_URDF_SUFFIX)]
        out[name] = (stem + _VENDORED_SUFFIX, _visual_placement(visual))
    return out


def visual_meshes():
    """``[(attach, path, T_local)]`` for every link mesh this hand has.

    ``attach`` is ``None`` for the palm, which rides on the wrist, or
    ``(digit_index, site_index)`` for a link, indexing the same sites
    ``HandKinematics::site_pose_key`` addresses.

    ``T_local`` takes the mesh's own coordinates into the frame it attaches to.
    It is carried PER MESH rather than assumed by the renderer, because it is a
    property of the asset and of what the URDF says about it, not of the
    drawing. On this hand it is the link's ``<visual><origin>`` times the mesh
    ``scale`` (see :func:`_visual_placement`), and for the palm the link's own
    placement in the root frame composes in ahead of both. A hand whose URDF
    authors each mesh in its own link frame at metre scale would return the
    identity here.

    Returns only files that are actually present, so a checkout missing them
    degrades to the skeleton rather than raising.
    """
    visuals = _link_visuals()
    found: list[tuple[tuple[int, int] | None, Path, np.ndarray]] = []

    palm = visuals.get(_PALM_LINK)
    if palm and (MESH_DIR / palm[0]).exists():
        # The palm rides on the WRIST, but its mesh belongs to `palm_link` --
        # so that link's placement in the root frame composes in ahead of the
        # mesh's own visual origin.
        found.append((None, MESH_DIR / palm[0],
                      _fixed_placement(_PALM_LINK) @ palm[1]))

    for d, digit in enumerate(DIGIT_NAMES):
        # Site 0 is the fixed mount and carries no link of its own, so the
        # frames listed for sites 1..N line up with index + 1.
        for s, link in enumerate(_SITE_FRAMES[digit], start=1):
            entry = visuals.get(link)
            if entry and (MESH_DIR / entry[0]).exists():
                found.append(((d, s), MESH_DIR / entry[0], entry[1]))
    return found


def available() -> bool:
    """Whether the meshes are on disk. False degrades the workbench to the
    skeleton, which is a complete and correct drawing in its own right."""
    return MESH_DIR.is_dir() and any(MESH_DIR.glob(f"*{_VENDORED_SUFFIX}"))
