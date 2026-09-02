"""Vendor one Allegro Hand V5 description from Wonik Robotics' ROS 2 package.

Copies the chosen URDF **verbatim** into
``core/hands/allegro/urdf/`` and converts every mesh it names from the
upstream millimetre STL to a decimated binary glTF beside it.

Run when the vendored description needs refreshing against upstream::

    python scripts/vendor_allegro_v5_meshes.py                  # clones upstream
    python scripts/vendor_allegro_v5_meshes.py --src ~/allegro_hand_ros2_v5

Needs the ``vendor`` extra (``pip install -e ".[vendor]"``) for ``trimesh`` and
``fast-simplification``. Neither is a runtime dependency of the package: this
script is run by hand when upstream changes and its output is committed.

WHY CONVERT AT ALL. Upstream ships ~21 MB of STL for one variant, against the
1.5 MB of decimated glTF committed here. The meshes are VISUAL ONLY -- collision
in this repository is the sphere set the solve carries, and the factor graph
never sees a mesh -- so the geometry loss (p99 under 2 mm on a 128 mm palm) costs
the picture nothing a viewer can see, and buys a repository that clones fast and
draws the hand offline.

WHY MILLIMETRES ARE KEPT. The conversion decimates and nothing else: no scaling,
no re-centring, no axis change. The URDF's own ``scale`` and ``<visual><origin>``
stay the single authority on where a mesh goes, so the vendored URDF can be
diffed against upstream byte for byte and ``meshes.py`` reads placement from it
rather than from an assumption baked into an asset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

UPSTREAM = "https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5.git"

#: Where the description lives inside the upstream workspace.
PKG = Path("src") / "allegro_hand_controllers"

#: Face budget per mesh. 8000 holds every visible feature -- below ~6000 the
#: palm's fingertip mounts start to round off -- and lands the set at ~1.5 MB.
DEFAULT_FACES = 8000

#: Vendored into the package, next to the module that reads it.
DEST = (Path(__file__).resolve().parent.parent / "python" / "gepetto_solvers"
        / "core" / "hands" / "allegro" / "urdf")


def _clone(into: Path) -> Path:
    """Shallow-clone upstream and return the workspace root."""
    subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(into)],
                   check=True)
    return into


def _upstream_revision(src: Path) -> str:
    """The commit the meshes came from, for the NOTICE. Empty if not a clone."""
    try:
        out = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _mesh_names(urdf: Path) -> list[str]:
    """Every distinct mesh basename the URDF names, visual and collision.

    Both are read even though only the visuals are drawn: on this hand they are
    the same files, and a variant that ever diverged should still vendor a
    complete set rather than a silently partial one.
    """
    root = ET.parse(urdf).getroot()
    names = []
    for mesh in root.iter("mesh"):
        name = (mesh.get("filename") or "").rsplit("/", 1)[-1]
        if name and name not in names:
            names.append(name)
    return names


def _convert(stl: Path, glb: Path, faces: int) -> tuple[int, int]:
    """Decimate one STL to at most ``faces`` triangles and write it as glTF.

    Returns ``(faces before, faces after)``. Units and frame are untouched.
    """
    import trimesh

    mesh = trimesh.load(stl, force="mesh")
    before = len(mesh.faces)
    # STL repeats a vertex per face; welding first is what makes the decimator's
    # quadrics meaningful and roughly halves the file on its own.
    mesh.merge_vertices()
    if len(mesh.faces) > faces:
        mesh = mesh.simplify_quadric_decimation(face_count=faces)
    mesh.export(glb)
    return before, len(mesh.faces)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None,
                    help="an existing clone of allegro_hand_ros2_v5; cloned to "
                         "a temporary directory when omitted")
    ap.add_argument("--variant", default="right_B",
                    help="which description to vendor (right_B, right_A, "
                         "left_B, left_A)")
    ap.add_argument("--faces", type=int, default=DEFAULT_FACES,
                    help=f"face budget per mesh (default {DEFAULT_FACES})")
    ap.add_argument("--dest", type=Path, default=DEST,
                    help="where to write the URDF and meshes/")
    args = ap.parse_args()

    tmp = None
    src = args.src
    if src is None:
        tmp = tempfile.TemporaryDirectory()
        src = _clone(Path(tmp.name) / "allegro_hand_ros2_v5")

    urdf = src / PKG / "urdf" / f"allegro_hand_description_{args.variant}.urdf"
    if not urdf.is_file():
        raise SystemExit(f"no such description: {urdf}")

    mesh_dir = args.dest / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    out_urdf = args.dest / f"allegro_hand_v5_{args.variant}.urdf"
    shutil.copyfile(urdf, out_urdf)
    print(f"urdf   {out_urdf.relative_to(args.dest.parent)}  (verbatim)")

    total = 0
    for name in _mesh_names(urdf):
        stl = src / PKG / "meshes" / name
        if not stl.is_file():
            print(f"  SKIP {name}: not in {stl.parent}")
            continue
        glb = mesh_dir / (Path(name).stem + ".glb")
        before, after = _convert(stl, glb, args.faces)
        total += glb.stat().st_size
        print(f"  mesh {name:24s} {before:7d} -> {after:5d} faces  "
              f"{stl.stat().st_size / 1e6:5.2f} -> {glb.stat().st_size / 1e6:4.2f} MB")

    print(f"total  {total / 1e6:.2f} MB in {mesh_dir}")
    rev = _upstream_revision(src)
    if rev:
        print(f"upstream {UPSTREAM} @ {rev}")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
