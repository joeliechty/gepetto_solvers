"""A viser server for browsing the YCB object set and fitting ellipsoids to it.

Pick an object from the dropdown, hit Load, and it shows up in the 3D view.
Archives are pulled from the YCB S3 bucket on first use and cached under
`ycb_data/`, so the second look at an object is instant.

Run from the ``crest-sparse/python`` directory::

    python -m tests._objects.ycb.browser          # then open http://localhost:8082
    python -m tests._objects.ycb.browser --fit 043_phillips_screwdriver
    python -m tests._objects.ycb.browser --prefetch 011_banana 013_apple

This is the AUTHORING tool: browse the catalog, tune backend / k / coverage, and
eyeball a decomposition before committing it. What the hand scripts consume is the
exported JSON it writes into ``fits/`` -- see ``tendon_hand/scene.py``'s
``ycb_primitive_specs()``, which turns each of those into an ``ellipsoid_set``
object primitive, and the C++ ``EllipsoidSetCollisionGapFactor`` that evaluates it.

Meshes are placed in metres, centered in XY, resting on the z=0 grid.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import trimesh
import viser

from . import ellipsoids as ye
from .data import (
    FITS_DIR,
    Catalog,
    YcbCache,
    describe,
    ground_and_center,
    ground_offset,
    prefetch,
)
from .fitting import DEFAULT_K_MAX, fit_object

TEXTURE_CHOICES: dict[str, int | None] = {
    "512 px": 512,
    "1024 px": 1024,
    "2048 px": 2048,
    "Full (4096 px)": None,
}

# Objects are laid out along X at this spacing when several are shown at once.
LAYOUT_SPACING = 0.25

# Distinct colours cycled per ellipsoid so the decomposition is readable.
ELLIPSOID_COLORS = [
    (239, 108, 96),
    (96, 178, 239),
    (140, 214, 122),
    (247, 196, 92),
    (186, 132, 233),
    (94, 219, 205),
    (240, 142, 197),
    (168, 180, 196),
]


class YcbViewer:
    def __init__(self, cache: YcbCache, port: int = 8082):
        self.cache = cache
        self.catalog = cache.catalog
        self.server = viser.ViserServer(port=port, label="YCB browser")

        # Loaded geometry is kept separately from the scene handles, because
        # switching render mode means re-adding the nodes from scratch --
        # viser's mesh handles don't expose a mutable `wireframe` flag.
        self._meshes: dict[str, trimesh.Trimesh] = {}
        self._handles: dict[str, viser.SceneNodeHandle] = {}
        # Ellipsoid fits and the grounding shift applied to each loaded mesh, so
        # exported centers can be mapped back to the original mesh frame.
        self._fits: dict[str, ye.EllipsoidFit] = {}
        self._offsets: dict[str, np.ndarray] = {}
        self._sources: dict[str, str] = {}
        self._ellipsoid_handles: list[viser.SceneNodeHandle] = []
        self._lock = threading.Lock()
        self._busy = False

        self._build_scene()
        self._build_gui()

    # -- scene --------------------------------------------------------------

    def _build_scene(self) -> None:
        self.server.scene.set_up_direction("+z")
        self.grid = self.server.scene.add_grid(
            "/grid",
            width=1.0,
            height=1.0,
            cell_size=0.05,
            section_size=0.25,
            plane="xy",
        )
        self.axes = self.server.scene.add_frame(
            "/origin", show_axes=True, axes_length=0.1, axes_radius=0.002
        )

    def _render(self) -> None:
        """Rebuild every scene node from the loaded meshes and their fits."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        for handle in self._ellipsoid_handles:
            handle.remove()
        self._ellipsoid_handles.clear()

        names = list(self._meshes)
        start = -LAYOUT_SPACING * (len(names) - 1) / 2.0
        wireframe = self.wireframe.value
        show_ellipsoids = self.show_ellipsoids.value
        # Hide the texture when ellipsoids are up, otherwise the shells are
        # buried inside an opaque mesh and the fit cannot be judged.
        mesh_hidden = show_ellipsoids and self.hide_mesh.value

        for index, name in enumerate(names):
            mesh = self._meshes[name]
            position = (start + index * LAYOUT_SPACING, 0.0, 0.0)
            if wireframe or mesh_hidden:
                handle = self.server.scene.add_mesh_simple(
                    f"/objects/{name}",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    wireframe=True,
                    color=(150, 150, 150),
                    position=position,
                )
            else:
                handle = self.server.scene.add_mesh_trimesh(
                    f"/objects/{name}", mesh, position=position
                )
            self._handles[name] = handle

            if show_ellipsoids and name in self._fits:
                self._render_ellipsoids(name, self._fits[name], position)

    def _render_ellipsoids(
        self, name: str, fit_result: ye.EllipsoidFit, position: tuple[float, float, float]
    ) -> None:
        opacity = float(self.ellipsoid_opacity.value)
        for index, ellipsoid in enumerate(fit_result.ellipsoids):
            shell = ellipsoid.as_mesh(subdivisions=3)
            handle = self.server.scene.add_mesh_simple(
                f"/ellipsoids/{name}/{index}",
                vertices=shell.vertices,
                faces=shell.faces,
                color=ELLIPSOID_COLORS[index % len(ELLIPSOID_COLORS)],
                opacity=opacity,
                side="double",  # transparent shells need both faces drawn
                flat_shading=False,
                position=position,
            )
            self._ellipsoid_handles.append(handle)

    # -- gui ----------------------------------------------------------------

    def _build_gui(self) -> None:
        gui = self.server.gui

        gui.add_markdown(
            f"### YCB browser\n{len(self.catalog.objects)} objects · "
            "meshes stream from the YCB S3 bucket on first load."
        )

        self.object_dropdown = gui.add_dropdown(
            "Object",
            options=self.catalog.labels(),
            initial_value=self.catalog.labels()[0],
            hint="Size shown is the download for this object's best mesh.",
        )
        self.source_dropdown = gui.add_dropdown(
            "Source",
            options=["best available", "google_16k", "berkeley"],
            initial_value="best available",
            hint="google_16k is the cleaner scan; berkeley covers more objects.",
        )
        self.texture_dropdown = gui.add_dropdown(
            "Texture",
            options=list(TEXTURE_CHOICES),
            initial_value="1024 px",
            hint="Textures are downsampled before being sent to the browser.",
        )
        self.replace_checkbox = gui.add_checkbox(
            "Replace current object",
            initial_value=True,
            hint="Uncheck to lay several objects out side by side.",
        )

        self.load_button = gui.add_button("Load object", icon=viser.Icon.DOWNLOAD)
        self.clear_button = gui.add_button("Clear scene", icon=viser.Icon.TRASH)

        self.progress = gui.add_progress_bar(0.0, visible=False, animated=True)
        self.status = gui.add_markdown("Pick an object and press **Load object**.")

        with gui.add_folder("Ellipsoid fit", expand_by_default=False):
            self.show_ellipsoids = gui.add_checkbox(
                "Show ellipsoids",
                initial_value=False,
                hint="Overlay the ellipsoid approximation on the loaded object.",
            )
            self.hide_mesh = gui.add_checkbox(
                "Mesh as wireframe",
                initial_value=True,
                hint="Otherwise the opaque mesh hides the ellipsoids inside it.",
            )
            self.backend_dropdown = gui.add_dropdown(
                "Backend",
                options=["gmm", "kmeans", "coacd"],
                initial_value="gmm",
                hint="gmm handles elongated parts; coacd respects concavity but is slow.",
            )
            self.count_dropdown = gui.add_dropdown(
                "Count",
                options=["auto", "manual"],
                initial_value="auto",
                hint="auto sweeps k and takes the smallest one near the best result.",
            )
            self.k_slider = gui.add_slider(
                "Ellipsoids (k)", min=1, max=15, step=1, initial_value=4
            )
            self.coverage_slider = gui.add_slider(
                "Coverage target", min=0.90, max=1.0, step=0.005, initial_value=0.98,
                hint="Fraction of the surface the union must contain. 1.0 fully encloses.",
            )
            self.ellipsoid_opacity = gui.add_slider(
                "Opacity", min=0.1, max=1.0, step=0.05, initial_value=0.45
            )
            self.fit_button = gui.add_button("Fit ellipsoids", icon=viser.Icon.CIRCLES)
            self.export_button = gui.add_button("Export JSON", icon=viser.Icon.FILE_EXPORT)
            self.fit_status = gui.add_markdown("No fit yet.")

        with gui.add_folder("View", expand_by_default=False):
            self.show_grid = gui.add_checkbox("Show grid", initial_value=True)
            self.show_axes = gui.add_checkbox("Show origin axes", initial_value=True)
            self.wireframe = gui.add_checkbox("Wireframe", initial_value=False)

        @self.fit_button.on_click
        def _(_event) -> None:
            self._start(self._fit_worker)

        @self.export_button.on_click
        def _(_event) -> None:
            self._export()

        @self.show_ellipsoids.on_update
        def _(_event) -> None:
            self._render()

        @self.hide_mesh.on_update
        def _(_event) -> None:
            self._render()

        @self.ellipsoid_opacity.on_update
        def _(_event) -> None:
            self._render()

        @self.load_button.on_click
        def _(_event) -> None:
            self._start_load()

        @self.clear_button.on_click
        def _(_event) -> None:
            self._meshes.clear()
            self._render()
            self.status.content = "Scene cleared."

        @self.show_grid.on_update
        def _(_event) -> None:
            self.grid.visible = self.show_grid.value

        @self.show_axes.on_update
        def _(_event) -> None:
            self.axes.visible = self.show_axes.value

        @self.wireframe.on_update
        def _(_event) -> None:
            self._render()

    # -- loading ------------------------------------------------------------

    def _resolve_source(self, name: str) -> str:
        """Honour the requested source, falling back if the object lacks it."""
        available = self.catalog.objects[name].sources
        requested = self.source_dropdown.value
        if requested == "best available" or requested not in available:
            return available[0]
        return requested

    def _start(self, worker) -> None:
        """Run a long job on a worker thread so the UI stays responsive.

        One lock covers downloads and fitting alike -- CoACD fits can run for
        tens of seconds, and overlapping them with a load would race the scene.
        """
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=worker, daemon=True).start()

    def _start_load(self) -> None:
        self._start(self._load_worker)

    def _load_worker(self) -> None:
        name = self.catalog.name_from_label(self.object_dropdown.value)
        source = self._resolve_source(name)
        max_texture = TEXTURE_CHOICES[self.texture_dropdown.value]

        self.load_button.disabled = True
        self.progress.visible = True
        self.progress.value = 0.0

        def report(fraction: float, message: str) -> None:
            self.progress.value = max(0.0, min(1.0, fraction)) * 100.0
            self.status.content = f"**{name}** — {message}"

        try:
            report(0.0, f"Fetching from `{source}`…")
            raw = self.cache.load_mesh(name, source, max_texture, progress=report)
            offset = ground_offset(raw)
            mesh = ground_and_center(raw)

            if self.replace_checkbox.value:
                self._meshes.clear()
                self._fits.clear()
                self._offsets.clear()
                self._sources.clear()
            self._meshes[name] = mesh
            self._offsets[name] = offset
            self._sources[name] = source
            self._fits.pop(name, None)  # a new mesh invalidates any old fit
            self._render()

            requested = self.source_dropdown.value
            note = ""
            if requested != "best available" and requested != source:
                note = f"\n\n_(no `{requested}` mesh for this object; used `{source}`)_"
            self.status.content = f"**{name}** · `{source}`\n\n{describe(mesh)}{note}"
        except Exception as exc:  # Surface it in the browser, keep serving.
            traceback.print_exc()
            self.status.content = f"**{name}** — failed to load:\n\n`{exc}`"
        finally:
            self.progress.visible = False
            self.load_button.disabled = False
            with self._lock:
                self._busy = False

    # -- fitting ------------------------------------------------------------

    def _fit_target(self) -> str | None:
        """The object to fit: the selected one if loaded, else the only one."""
        selected = self.catalog.name_from_label(self.object_dropdown.value)
        if selected in self._meshes:
            return selected
        return next(iter(self._meshes), None)

    def _fit_worker(self) -> None:
        name = self._fit_target()
        if name is None:
            self.fit_status.content = "Load an object first."
            with self._lock:
                self._busy = False
            return

        backend = self.backend_dropdown.value
        coverage = float(self.coverage_slider.value)
        automatic = self.count_dropdown.value == "auto"
        k = None if automatic else int(self.k_slider.value)

        self.fit_button.disabled = True
        self.progress.visible = True
        self.progress.value = 0.0

        def report(fraction: float, message: str) -> None:
            self.progress.value = max(0.0, min(1.0, fraction)) * 100.0
            self.fit_status.content = f"**{name}** — {message}"

        try:
            # Shared with the --fit CLI and the hand visualizer's fit-on-select,
            # so all three produce the same decomposition for the same settings.
            result, _path = fit_object(
                self.cache, name, self._sources.get(name, ""),
                backend=backend, k=k, coverage=coverage, progress=report,
            )
            self._fits[name] = result
            if not self.show_ellipsoids.value:
                self.show_ellipsoids.value = True  # triggers a re-render
            else:
                self._render()

            metrics = result.metrics
            caveat = (
                ""
                if metrics.volume_reliable
                else "\n\n_Volume ratio is unreliable: this mesh is not a closed solid._"
            )
            self.fit_status.content = (
                f"**{name}** · `{backend}`{' · auto' if automatic else ''}\n\n"
                f"{metrics.k} ellipsoid{'s' if metrics.k != 1 else ''}\n\n"
                f"Excess volume: **{metrics.excess_volume_ratio:.2f}x**\n\n"
                f"Surface covered: **{metrics.surface_coverage * 100:.1f}%**{caveat}"
            )
        except Exception as exc:
            traceback.print_exc()
            self.fit_status.content = f"**{name}** — fit failed:\n\n`{exc}`"
        finally:
            self.progress.visible = False
            self.fit_button.disabled = False
            with self._lock:
                self._busy = False

    def _export(self) -> None:
        name = self._fit_target()
        if name is None or name not in self._fits:
            self.fit_status.content = "Nothing to export — fit an object first."
            return
        try:
            path = ye.export_json(
                FITS_DIR, name, self._sources.get(name, ""), self._fits[name]
            )
            self.fit_status.content = (
                f"Exported **{name}** to\n\n`{path}`\n\n"
                "_The hand scripts pick this up as an `ycb:` object._"
            )
        except Exception as exc:
            traceback.print_exc()
            self.fit_status.content = f"Export failed:\n\n`{exc}`"

    # -- run ----------------------------------------------------------------

    def run(self) -> None:
        print("Serving — open the URL above in a browser. Ctrl-C to stop.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\nShutting down.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).parent / "ycb_data",
        help="Where downloaded YCB archives and meshes are kept.",
    )
    parser.add_argument(
        "--prefetch",
        nargs="+",
        metavar="OBJECT",
        help="Download these objects (or 'all') and exit, instead of serving.",
    )
    parser.add_argument(
        "--fit",
        nargs="+",
        metavar="OBJECT",
        help="Fit ellipsoids for these objects (or 'all') and exit.",
    )
    parser.add_argument("--backend", default="gmm", choices=list(ye.BACKENDS))
    parser.add_argument(
        "--k", type=int, default=None, help="Ellipsoid count; omit for automatic."
    )
    parser.add_argument(
        "--k-max", type=int, default=DEFAULT_K_MAX,
        help=f"Largest k the automatic sweep will try (default {DEFAULT_K_MAX}). "
             "Changes WHICH fit is chosen, not just how long the search takes: "
             "the sweep returns the smallest k within tolerance of the best k it "
             "found, so a lower ceiling can move the answer. Leave it alone "
             "unless you know why you are changing it.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Leave objects that already have a fit alone. For resuming a batch.",
    )
    parser.add_argument("--coverage", type=float, default=0.98)
    args = parser.parse_args()

    catalog = Catalog()
    cache = YcbCache(catalog, args.cache_dir)

    if args.fit:
        names = _resolve_names(parser, catalog, args.fit)
        print(f"Fitting {len(names)} object(s) with {args.backend}…", flush=True)
        done = failed = skipped = 0
        for index, name in enumerate(names, 1):
            info = catalog.objects[name]
            source = info.sources[0]
            if args.skip_existing and (FITS_DIR / f"{name}__{source}.json").exists():
                skipped += 1
                continue
            started = time.time()
            try:
                result, _path = fit_object(
                    cache, name, source, backend=args.backend, k=args.k,
                    coverage=args.coverage, k_max=args.k_max,
                )
                done += 1
                print(f"  [{index}/{len(names)}] {name}: {result.metrics.summary()}"
                      f"  ({time.time() - started:.0f}s)", flush=True)
            except Exception as exc:
                failed += 1
                # Keep going: a batch over the whole catalog will hit a few scans
                # that are shells rather than solids, and one bad object must not
                # cost the other 96.
                print(f"  [{index}/{len(names)}] {name}: FAILED — {exc}", flush=True)
        print(f"\n{done} fitted, {failed} failed, {skipped} skipped -> {FITS_DIR}")
        return

    if args.prefetch:
        names = _resolve_names(parser, catalog, args.prefetch)
        print(f"Prefetching {len(names)} object(s) into {args.cache_dir}…")
        prefetch(cache, names)
        return

    YcbViewer(cache, port=args.port).run()


def _resolve_names(parser, catalog: Catalog, requested: list[str]) -> list[str]:
    """Expand 'all' and reject names that are not in the catalog."""
    if requested == ["all"]:
        return catalog.names()
    unknown = [n for n in requested if n not in catalog.objects]
    if unknown:
        parser.error(f"unknown object(s): {', '.join(sorted(unknown))}")
    return list(requested)


if __name__ == "__main__":
    main()
