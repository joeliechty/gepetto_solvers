"""Download, cache, and load meshes from the YCB Object and Model Set.

The YCB archives live in a public S3 bucket. Each object ships in a couple of
flavours; we care about the two that carry a textured mesh:

    google/<obj>_google_16k.tgz          -> <obj>/google_16k/textured.obj
    berkeley/<obj>/<obj>_berkeley_meshes.tgz -> <obj>/tsdf/textured.obj
                                             (and sometimes <obj>/poisson/)

``ycb_catalog.json`` records which of the two exist per object, so the UI can
show download sizes and never offer an object that has no usable mesh.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import trimesh

CATALOG_PATH = Path(__file__).parent / "catalog.json"

# Downloaded archives and extracted meshes. Gitignored -- this reaches ~0.6 GB
# for the whole set, and every byte of it is reproducible from the bucket.
DEFAULT_CACHE = Path(__file__).parent / "ycb_data"

# Committed ellipsoid fits, which is what the hand scripts actually read. Kept
# apart from DEFAULT_CACHE precisely because these are NOT reproducible on
# demand: a fit is the output of a stochastic GMM sweep that takes tens of
# seconds, so the chosen decomposition is checked in rather than re-derived.
FITS_DIR = Path(__file__).parent / "fits"

# Objects whose archives download fine but whose meshes are not usable geometry.
# Kept here rather than stripped from the catalog JSON so the reason travels with
# the exclusion.
EXCLUDED: dict[str, str] = {
    "023_wine_glass": (
        "84-face shell with inverted winding (negative volume) — transparent "
        "objects scan badly; there is no solid to approximate"
    ),
}

# Progress callbacks receive (fraction_complete, human_readable_status).
ProgressFn = Callable[[float, str], None]


def _noop(fraction: float, message: str) -> None:
    del fraction, message


@dataclass(frozen=True)
class ObjectInfo:
    """What the catalog knows about one YCB object."""

    name: str
    google_bytes: int | None
    berkeley_bytes: int | None

    @property
    def sources(self) -> list[str]:
        """Available sources, best-looking mesh first."""
        out = []
        if self.google_bytes is not None:
            out.append("google_16k")
        if self.berkeley_bytes is not None:
            out.append("berkeley")
        return out

    def size_bytes(self, source: str) -> int | None:
        return self.google_bytes if source == "google_16k" else self.berkeley_bytes

    def label(self) -> str:
        """Dropdown label: '011_banana (5.5 MB)'."""
        best = self.size_bytes(self.sources[0])
        return f"{self.name} ({best / 1e6:.1f} MB)"


class Catalog:
    """The set of YCB objects that have a downloadable textured mesh."""

    def __init__(self, path: Path = CATALOG_PATH):
        raw = json.loads(path.read_text())
        self.base_url: str = raw["base_url"]
        self.objects: dict[str, ObjectInfo] = {
            name: ObjectInfo(
                name=name,
                google_bytes=(entry["google_16k"] or {}).get("bytes")
                if entry["google_16k"]
                else None,
                berkeley_bytes=(entry["berkeley"] or {}).get("bytes")
                if entry["berkeley"]
                else None,
            )
            for name, entry in sorted(raw["objects"].items())
            if name not in EXCLUDED
        }

    def names(self) -> list[str]:
        return list(self.objects)

    def labels(self) -> list[str]:
        return [o.label() for o in self.objects.values()]

    def name_from_label(self, label: str) -> str:
        return label.split(" (")[0]

    def archive_url(self, name: str, source: str) -> str:
        if source == "google_16k":
            return f"{self.base_url}/google/{name}_google_16k.tgz"
        return f"{self.base_url}/berkeley/{name}/{name}_berkeley_meshes.tgz"


class YcbCache:
    """Downloads YCB archives on demand and keeps the extracted meshes on disk."""

    def __init__(self, catalog: Catalog, root: Path = DEFAULT_CACHE):
        self.catalog = catalog
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- download -----------------------------------------------------------

    def _extract_dir(self, name: str, source: str) -> Path:
        return self.root / source / name

    def is_cached(self, name: str, source: str) -> bool:
        """Is this object's mesh already extracted locally?

        The same test :meth:`ensure` uses to decide whether to download, exposed
        so a caller can ask WITHOUT triggering one. The SDF setup script is the
        reason: it distinguishes objects it can bake right now from objects that
        would first cost a download, and a user who has not asked for ~0.6 GB of
        scans should not get them from a status query."""
        target = self._extract_dir(name, source)
        return target.exists() and any(target.rglob("textured.obj"))

    def _download(self, url: str, dest: Path, progress: ProgressFn) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 18):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        progress(
                            done / total,
                            f"Downloading… {done / 1e6:.1f} / {total / 1e6:.1f} MB",
                        )
                    else:
                        progress(0.0, f"Downloading… {done / 1e6:.1f} MB")
        tmp.replace(dest)

    def ensure(self, name: str, source: str, progress: ProgressFn = _noop) -> Path:
        """Return the extracted directory for an object, fetching it if needed."""
        target = self._extract_dir(name, source)
        if self.is_cached(name, source):
            progress(1.0, "Using cached download.")
            return target

        url = self.catalog.archive_url(name, source)
        archive = self.root / "archives" / f"{name}_{source}.tgz"
        if not archive.exists():
            self._download(url, archive, progress)

        progress(1.0, "Extracting…")
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            # Archives are laid out as <obj>/<variant>/..., so strip the leading
            # object directory and extract straight into `target`.
            members = [m for m in tar.getmembers() if _is_safe_member(m)]
            for member in members:
                parts = Path(member.name).parts
                if len(parts) <= 1:
                    continue
                member.name = str(Path(*parts[1:]))
                tar.extract(member, target)
        return target

    # -- loading ------------------------------------------------------------

    def load_mesh(
        self,
        name: str,
        source: str,
        max_texture: int | None = 1024,
        progress: ProgressFn = _noop,
    ) -> trimesh.Trimesh:
        """Download if needed, then load the textured mesh for an object."""
        directory = self.ensure(name, source, progress)

        obj_path = _pick_obj(directory)
        if obj_path is None:
            raise FileNotFoundError(f"No textured.obj found under {directory}")

        progress(1.0, "Loading mesh…")
        mesh = trimesh.load(obj_path, process=False, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"{obj_path} did not load as a single mesh")

        _bind_texture(mesh, obj_path)
        if max_texture is not None:
            _downsample_texture(mesh, max_texture)
        return mesh


# The YCB tarballs come from a known bucket, but extraction should still refuse
# absolute paths and `..` escapes rather than trusting the archive.
def _is_safe_member(member: tarfile.TarInfo) -> bool:
    if member.issym() or member.islnk():
        return False
    path = Path(member.name)
    return not path.is_absolute() and ".." not in path.parts


def _pick_obj(directory: Path) -> Path | None:
    """Choose the best textured.obj in an extracted archive.

    google_16k has exactly one. Berkeley archives carry `tsdf/` (watertight,
    reliably textured) and sometimes `poisson/`, so prefer tsdf.
    """
    candidates = sorted(directory.rglob("textured.obj"))
    if not candidates:
        return None
    for preferred in ("google_16k", "tsdf", "poisson"):
        for candidate in candidates:
            if preferred in candidate.parts:
                return candidate
    return candidates[0]


def _find_texture(obj_path: Path) -> Path | None:
    """Locate the colour map for an OBJ, preferring what its MTL names."""
    directory = obj_path.parent
    for mtl in directory.glob("*.mtl"):
        for line in mtl.read_text(errors="ignore").splitlines():
            if line.strip().lower().startswith("map_kd"):
                candidate = directory / line.split(None, 1)[1].strip()
                if candidate.exists():
                    return candidate
    images = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return images[0] if images else None


def _bind_texture(mesh: trimesh.Trimesh, obj_path: Path) -> None:
    """Attach the texture when trimesh failed to bind it.

    Several YCB OBJs (the Berkeley ones especially) declare a material in their
    MTL but never emit a `usemtl` statement, so trimesh falls back to a 2x2
    placeholder even though the real texture sits right next to the file.
    """
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)
    if image is not None and max(image.size) > 4:
        return  # Already textured.

    uv = getattr(mesh.visual, "uv", None)
    texture_path = _find_texture(obj_path)
    if uv is None or texture_path is None:
        return  # Nothing to bind; caller still gets a usable untextured mesh.

    try:
        from PIL import Image

        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv, image=Image.open(texture_path).convert("RGB")
        )
    except Exception:
        pass


def _downsample_texture(mesh: trimesh.Trimesh, max_size: int) -> None:
    """Shrink oversized texture maps in place.

    YCB ships 4096x4096 textures. Those turn into very large GLB payloads for
    the browser, and the difference is invisible at normal viewing distance.
    """
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)
    if image is None or max(image.size) <= max_size:
        return
    scale = max_size / max(image.size)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    try:
        from PIL import Image

        material.image = image.resize(new_size, Image.LANCZOS)
    except Exception:
        pass  # Keep the full-resolution texture rather than failing the load.


def ground_offset(mesh: trimesh.Trimesh) -> np.ndarray:
    """Translation that centers a mesh in XY and rests it on z=0.

    Exposed separately so callers that fit geometry to the displayed mesh can
    record the shift and map results back to the original mesh frame.
    """
    bounds = mesh.bounds
    return np.array(
        [
            -(bounds[0][0] + bounds[1][0]) / 2.0,
            -(bounds[0][1] + bounds[1][1]) / 2.0,
            -bounds[0][2],
        ]
    )


def ground_and_center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Center the mesh in XY and rest its lowest point on z=0."""
    out = mesh.copy()
    out.apply_translation(ground_offset(mesh))
    return out


def describe(mesh: trimesh.Trimesh) -> str:
    """A short human-readable summary of a loaded mesh."""
    extents_cm = np.asarray(mesh.extents) * 100.0
    return (
        f"{len(mesh.vertices):,} verts · {len(mesh.faces):,} faces\n"
        f"{extents_cm[0]:.1f} × {extents_cm[1]:.1f} × {extents_cm[2]:.1f} cm"
    )


def prefetch(cache: YcbCache, names: Iterable[str], source: str = "google_16k") -> None:
    """Warm the cache for a list of objects (handy before an offline demo)."""
    for name in names:
        info = cache.catalog.objects[name]
        use = source if source in info.sources else info.sources[0]
        print(f"  {name} [{use}]", flush=True)
        cache.ensure(name, use)
