"""Choosing and placing the grasp object, including the YCB catalogue.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""

import threading
import traceback

import numpy as np

from gepetto_solvers.core.geometry.scene import (
    TABLE_SPAN,
    TABLE_THICKNESS,
    get_primitive_specs,
    grasp_subset_indices,
    proxy_semi_axes,
    table_corner,
)
from gepetto_solvers.core.solvers import (
    default_half_space_axis,
    default_object_center,
    resolve_scene,
)

from .constants import (
    CAL_GRID_SPACING,
    DEFAULT_OBJECT_FALLBACK,
    DEFAULT_OBJECT_PRIMITIVE,
    SDF_DROPDOWN_LABELS,
    _euler_to_R,
)


class ObjectPanelMixin:
    # Dropdown suffix for a catalog object with no committed fit yet. Selecting
    # one downloads and fits it on the spot (see _on_object_selected).
    UNFITTED_SUFFIX = "  [fit on select]"

    # The fingertips ride a shell of roughly this radius about the hand base, and
    # a curl can close on an object only a little smaller than it. Measured, not
    # derived -- see the reachability investigation behind GRASP_SPHERE_CENTER.
    FINGERTIP_SHELL_M = 0.055
    GRASPABLE_MAX_M = 0.050

    def _object_pose_from_sliders(self):
        """``(center, rotation)`` for the object: its derived pose plus the
        Object-pose slider offsets.

        Resolved against the primitive's OWN default rather than the params'
        current value -- reading back what was written last time would feed the
        control's output into its input, and every sync would compound the offset.
        (The support plane's ``auto_table_origin`` avoids the same trap the same
        way.) All-zero sliders reproduce the derived pose exactly, so this is
        equivalent to the previous ``object_center = None`` behaviour.
        """
        spec = get_primitive_specs()[self.params.primitive]
        center = default_object_center(self.params.primitive, spec)
        rotation = np.asarray(spec.get("rotation", np.eye(3)), float)

        offset = np.array([self.g_obj_dx.value, self.g_obj_dy.value,
                           self.g_obj_dz.value], float)
        delta = _euler_to_R(self.g_obj_roll.value, self.g_obj_pitch.value,
                            self.g_obj_yaw.value)
        return center + offset, delta @ rotation


    def _object_pose_changed(self, _=None):
        """Re-place the object from the sliders.

        The object pose is part of the CONSTRAINT SET -- it is the mean of the
        object prior every contact and collision factor is written against -- so
        moving it invalidates the stepper's Augmented Lagrangian duals, exactly
        like changing the object itself. Re-solving with FK keeps the picture
        honest about that rather than leaving a stale IK pose next to a moved
        object.

        IN EITHER MODE, deliberately. A stepped IK result describes the object
        where it WAS -- its factors were built against that pose, and every
        distance drawn off it is measured to it -- so re-rendering it beside a
        moved object draws lines to a position nothing is at, which is exactly
        the confusion this method exists to prevent. The stepper has just been
        invalidated, so the IK loop is over regardless; re-solving FK is what
        "start over" already means and leaves the picture describing the scene
        that is actually on screen.
        """
        if self._restoring:
            return
        self._sync_params()
        self._refresh_object()
        self._invalidate_stepper()
        self._fk_solve()


    def _refresh_object_mesh(self, spec, center, rotation):
        """Draw (or clear) the object's TRUE geometry behind the analytic surface
        the solver actually sees.

        Two kinds of object have one: a ycb: set, whose shells approximate a
        scanned mesh, and a spec carrying ``hull_vertices`` -- the megaminx,
        whose circumsphere encloses a dodecahedron. Both are the same question
        ("how much object is really inside the surface the fingers stop on"), so
        they share one toggle.

        A YCB object carries BOTH: the scan is what it really looks like, its
        hull is the summary of the scan committed alongside the fit (and what the
        table is seated on -- ``scene.ycb_primitive_specs``). The scan wins when
        the cache has it, since a hull cannot show a mug's handle or a concave
        face; the hull is the fallback rather than nothing at all, because the
        fits are committed and the 1.5 GB of meshes are not, so an object can be
        perfectly loadable on a machine that has never fetched a scan.

        The mesh has to be put in the SAME frame the shells were re-centered
        into, or the two render a few cm apart and the overlay is worse than
        useless. (The hull was written into that frame at spec-build time.)
        """
        if not self.g_show_true_mesh.value:
            self.scene.clear_object_mesh()
            return
        if spec["type"] == "ellipsoid_set":
            try:
                from gepetto_solvers.core.objects.ycb import (
                    Catalog,
                    YcbCache,
                    ground_and_center,
                )

                cache = YcbCache(Catalog())
                mesh = ground_and_center(
                    cache.load_mesh(spec["ycb"], spec["source"], max_texture=512))
                mesh.apply_translation(-np.asarray(spec["recenter"], float))
                self.scene.set_object_mesh(mesh, center, rotation)
                return
            except Exception as exc:
                print(f"[viz] no scan mesh for {spec.get('ycb')}: {exc}")
        hull = spec.get("hull_vertices")
        if hull is None:
            self.scene.clear_object_mesh()
            return
        # Local point set -> hull; set_object_mesh applies the object pose, so the
        # solid lands inside its own shell however the object is posed.
        self.scene.set_object_mesh(self.scene.hull_mesh(hull), center, rotation)


    # -- YCB objects --------------------------------------------------------

    def _build_ycb_folder(self, gui):
        """Fetch-and-fit controls for the YCB object set.

        The offline path (``python scripts/objects/ycb_browser.py --fit <name>``)
        remains the primary one and writes the same files; this exists so an
        object can be brought in without leaving the app. Everything here is
        gated on ``ellipsoid_set``: without it a fitted object could be written
        but never loaded, so offering the button would be a trap.
        """
        available = self.caps.get("ellipsoid_set", False)
        with gui.add_folder("YCB objects", expand_by_default=False):
            if not available:
                gui.add_markdown(
                    "Needs a rebuilt `_gepetto_solvers` with "
                    "`EnvironmentConfig.ellipsoid_set`.")
                self.g_ycb_object = None
                return

            gui.add_markdown(
                "Fit a YCB object to an ellipsoid **set** and add it to the "
                "object list as `ycb:<name>`. First fetch of an object "
                "downloads 4-12 MB and takes tens of seconds.")
            self.g_ycb_object = gui.add_dropdown(
                "catalog", self._ycb_labels(),
                hint="Every YCB object with a usable textured mesh, with the "
                     "download size of its best mesh.")
            self.g_ycb_backend = gui.add_dropdown(
                "backend", ["gmm", "kmeans", "coacd"], initial_value="gmm",
                hint="gmm handles elongated parts; coacd respects concavity but "
                     "is slow and is an optional dependency.")
            # Same two controls as the browser's Fit panel, not a single
            # "0 = auto" slider: an explicit mode leaves no way for the count to
            # be read as a fit request when automatic was meant.
            self.g_ycb_count = gui.add_dropdown(
                "count", ["auto", "manual"], initial_value="auto",
                hint="auto sweeps k and takes the smallest one near the best "
                     "result. This is what produces a sensible multi-ellipsoid "
                     "decomposition; manual pins the count below.")
            self.g_ycb_k = gui.add_slider(
                "ellipsoids (k)", 1, 15, 1, 4,
                hint="Only used when count is 'manual'.")
            self.g_ycb_coverage = gui.add_slider(
                "coverage target", 0.90, 1.0, 0.005, 0.98)
            self.g_ycb_fit = gui.add_button(
                "Fetch & fit", icon=self.viser.Icon.DOWNLOAD)
            self.g_ycb_status = gui.add_markdown(
                f"{len(self._ycb_fitted())} object(s) already fitted.")
            self.g_ycb_fit.on_click(self._ycb_fit_clicked)


    def _ycb_labels(self):
        """Catalog dropdown labels, or a one-entry placeholder if it cannot load."""
        try:
            from gepetto_solvers.core.objects.ycb import Catalog
            return Catalog().labels()
        except Exception as exc:
            return [f"<catalog unavailable: {exc}>"]


    def _ycb_fitted(self):
        """Object keys already fitted and loadable, i.e. the ``ycb:`` specs."""
        return [k for k in get_primitive_specs() if k.startswith("ycb:")]


    def _ycb_fit_clicked(self, _event=None, name=None):
        """Start a fit. ``name`` overrides the YCB folder's own picker, which is
        how selecting an unfitted object straight from the object dropdown routes
        through the same one worker."""
        if self._ycb_busy:
            return
        self._ycb_busy = True
        threading.Thread(target=self._ycb_fit_worker, args=(name,),
                         daemon=True).start()


    def _ycb_fit_worker(self, name=None):
        """Download + fit one object on a worker thread, then offer it as an object.

        Runs off the GUI thread because a cold fit is a download plus a k-sweep --
        tens of seconds during which viser must stay responsive. It deliberately
        does NOT take the solver's ``_solving`` latch: fitting touches no solver
        state, so there is no reason it should block stepping an unrelated solve.
        """
        self.g_ycb_fit.disabled = True
        try:
            from gepetto_solvers.core.geometry.scene import ycb_primitive_specs
            from gepetto_solvers.core.objects.ycb import Catalog, YcbCache
            from gepetto_solvers.core.objects.ycb.fitting import fit_object

            catalog = Catalog()
            if name is None:
                name = catalog.name_from_label(self.g_ycb_object.value)
            source = catalog.objects[name].sources[0]
            backend = self.g_ycb_backend.value
            coverage = float(self.g_ycb_coverage.value)
            # None => automatic sweep, which is the default and what gives a real
            # multi-ellipsoid decomposition.
            k = (None if self.g_ycb_count.value == "auto"
                 else int(self.g_ycb_k.value))

            def report(fraction, message):
                # Mirrored to the main status bar as well: a fit started by
                # picking an unfitted object from the object dropdown is watched
                # there, not in the (collapsed by default) YCB folder.
                text = f"**{name}** — {message}"
                self.g_ycb_status.content = text
                self._set_status(text)

            # The shared pipeline the browser and the --fit CLI use, so an object
            # fitted here is identical to one fitted there.
            result, _path = fit_object(
                YcbCache(catalog), name, source, backend=backend, k=k,
                coverage=coverage, progress=report,
            )

            # The spec registry caches the directory listing, so a fit written
            # after startup is invisible until that cache is dropped.
            ycb_primitive_specs.cache_clear()
            self._refresh_object_dropdown(select=f"ycb:{name}")
            self.g_ycb_status.content = (
                f"**{name}** · `{backend}`\n\n{result.metrics.summary()}\n\n"
                "_Selected as the current object._")
        except Exception as exc:
            traceback.print_exc()
            self.g_ycb_status.content = f"Fit failed:\n\n`{exc}`"
            self._set_status(f"Fit of **{name}** failed: `{exc}`")
        finally:
            self.g_ycb_fit.disabled = False
            self._ycb_busy = False


    def _refresh_object_dropdown(self, select=None):
        """Rebuild the object dropdown's options after the spec registry changed,
        optionally selecting a key. Assigning ``options`` re-renders the widget in
        place, so the rest of the GUI is untouched."""
        labels, self._label_to_key = self._object_dropdown_labels()
        self.g_object.options = labels
        if select is not None and select in self._label_to_key.values():
            label = SDF_DROPDOWN_LABELS.get(select, select)
            if label in self._label_to_key:
                # Fires on_update -> reloads the scene on the newly fitted object.
                self.g_object.value = label


    def _on_object_selected(self, _=None):
        """Load the picked object, fitting it first if it has never been fitted.

        The fit runs on a worker thread and re-enters here once it has written
        the export, so an unfitted pick is a slow version of a fitted one rather
        than a separate mode the user has to know about.
        """
        if self._restoring:
            return
        label = self.g_object.value
        if label in getattr(self, "_unfitted", {}):
            # Leave the scene on the current object; the worker re-selects this
            # one once its fit exists.
            self._set_status(f"Fetching and fitting **{self._unfitted[label]}**… "
                             "(first fetch downloads 4-12 MB)")
            self._ycb_fit_clicked(name=self._unfitted[label])
            return
        self._load_selected_object()


    def _load_selected_object(self):
        self.params.primitive = self._label_to_key[self.g_object.value]
        # The Object-pose sliders are offsets from each primitive's own default,
        # so they carry over to the new object rather than being cleared;
        # _sync_params re-resolves them against the new base pose.
        self.params.object_center, self.params.object_rotation = \
            self._object_pose_from_sliders()
        # Whether the in-plane contact form is even possible is a property of the
        # object (it needs an ellipsoid cross-section), so re-decide it here
        # rather than leaving a live checkbox the next solve would refuse.
        self._refresh_planar_contact_gate()
        self._refresh_exact_contact_gate()
        # ...and so is whether the contact has a witness point to lay rows out
        # on: a `ycb:` object is an ellipsoid SET, which has none.
        self._refresh_normal_row_gate()
        # Whether there is a grasp subset to choose is likewise a property of the
        # object, so re-gate it here too -- and before _refresh_object, which
        # greys the excluded shells off the answer.
        self._refresh_grasp_subset_gate()
        self._rebuild_fk()      # FK solver carries the object for its result/spec
        self._refresh_object()
        self._aim_all_cameras()  # re-center on the new object's location
        self._fk_solve()


    def _object_dropdown_labels(self):
        """Every object offerable, fitted or not, as ``(labels, label -> key)``.

        The whole YCB catalog is listed, not just what has been fitted, so the
        picker is the object set rather than a record of what happens to be
        cached. An unfitted entry is marked and fits itself when chosen. The
        alternative -- pre-fitting all ~97 up front -- costs a 0.6 GB download and
        hours of k-sweeps to produce mostly objects that will never be picked.
        """
        keys = self._visible_primitive_keys()
        mapping = {SDF_DROPDOWN_LABELS.get(k, k): k for k in keys}
        self._unfitted = {}
        if self.caps.get("ellipsoid_set", False):
            try:
                from gepetto_solvers.core.objects.ycb import Catalog

                catalog = Catalog()
                for name in catalog.names():
                    if f"ycb:{name}" in keys:
                        continue           # already fitted, listed above
                    label = f"ycb:{name}{self.UNFITTED_SUFFIX}"
                    mapping[label] = f"ycb:{name}"
                    self._unfitted[label] = name
            except Exception as exc:
                print(f"[viz] YCB catalog unavailable: {exc}")
        return list(mapping), mapping


    def _visible_primitive_keys(self):
        """Object keys this binding can actually build, in dropdown order.

        Both analytic surface kinds are gated: a single ellipsoid needs
        ``ellipsoid_semi_axes``, an ellipsoid set needs ``ellipsoid_set``, and a
        stale ``.so`` may have either without the other. Offering an object whose
        env fields do not exist is worse than hiding it -- ``attach_ellipsoid_set``
        raises rather than silently building a surface-less env, so the object
        would simply fail to load with a traceback in the browser.
        """
        gates = {"ellipsoid": self.caps["ellipsoid"],
                 "ellipsoid_set": self.caps.get("ellipsoid_set", False)}
        return [k for k, v in get_primitive_specs().items()
                if gates.get(v["type"], True)]


    def _resolve_default_primitive(self):
        """The startup object, falling back when it is unavailable.

        The default (and its fallback) are analytic-ellipsoid objects, so both
        drop out of the dropdown on a binding without ``ellipsoid_semi_axes``;
        a YCB default would additionally need ``ellipsoid_set`` and a populated
        ``_objects/ycb/fits/``. Resolve here rather than only on the dropdown
        widget, because
        ``_rebuild_fk`` reads ``params.primitive`` directly and would raise a bare
        KeyError on a key the spec registry never produced.
        """
        keys = self._visible_primitive_keys()
        for candidate in (DEFAULT_OBJECT_PRIMITIVE, DEFAULT_OBJECT_FALLBACK):
            if candidate in keys:
                return candidate
        return keys[0]


    # -- rendering --

    def _refresh_object(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        # The shells contact may target, so the rest draw muted. Resolved the same
        # way the solve resolves it, off the same params -- the picture is meant
        # to say which surface the fingertips are being sent to, and a second
        # reading of "which shells" is how it would come to say something else.
        self.scene.set_object(
            spec, center, rotation,
            contact_subset=grasp_subset_indices(spec,
                                                self.params.use_grasp_subset))
        self._refresh_object_mesh(spec, center, rotation)
        # Reference frames. The world triad is fixed, but the object's rides on
        # the pose resolved just above, so it is drawn here -- with the object
        # itself -- rather than in _render_frame: the object moves when the
        # scene is rebuilt, not once per solve iterate.
        self.scene.set_world_frame(self.g_show_world.value)
        if self.g_show_obj_frame.value:
            self.scene.set_object_frame(center, rotation)
        else:
            self.scene.clear_object_frame()
        # The slab is drawn UNCONDITIONALLY -- not gated on params.table, and not
        # on caps["table"] either. It is a physical landmark for real-robot
        # setup, and a landmark that disappears when you switch a constraint off
        # (or when a stale .so cannot build the plane) is not one you can measure
        # against. The checkbox governs the solver's half-space; this is a
        # picture of where that plane is.
        origin = self._table_origin()
        self.scene.set_table(origin, self.params.plane_normal,
                             span=TABLE_SPAN, thickness=TABLE_THICKNESS)
        # Ruled on the slab's top face, matching the grid on the physical bench,
        # so a landmark commanded to an intersection here can be read against the
        # same intersection there. Drawn with the slab for the same reason the
        # slab is drawn unconditionally: it is part of the landmark.
        if self.g_show_grid.value:
            self.scene.set_table_grid(origin, self.params.plane_normal,
                                      span=TABLE_SPAN, spacing=CAL_GRID_SPACING)
        else:
            self.scene.clear_table_grid()
        # The solver's plane, when the slider has lifted it off the bench. Gated
        # on params.table -- unlike the slab above, which is drawn unconditionally
        # because it is a physical landmark, this one is a picture of a constraint
        # and drawing it with that constraint switched off would claim a plane the
        # graph does not have. Also drawn only when it is actually somewhere else:
        # coincident with the slab it is nothing but z-fighting, and a second
        # surface exactly where the table already is would invite the very
        # confusion the split exists to remove.
        constraint = self._constraint_plane_origin()
        if (self.g_show_constraint_plane.value and self.params.table
                and abs(self.params.constraint_plane_height) > 1e-6):
            self.scene.set_constraint_plane(constraint, self.params.plane_normal,
                                            span=TABLE_SPAN)
        else:
            self.scene.clear_constraint_plane()
        corner = table_corner(origin, self.params.plane_normal)
        if self.g_show_table_frame.value:
            self.scene.set_table_frame(
                corner, label=f"table corner  {TABLE_SPAN:g} x {TABLE_SPAN:g} m")
        else:
            self.scene.clear_table_frame()
        self._refresh_table_readout(origin, corner)
        # The calibration target is measured FROM the corner just resolved, so it
        # has to be re-placed whenever the table moves -- which the object seating
        # makes it do on every object change.
        self._refresh_calibration_frame()
        if self.params.half_space:
            axis = (self.params.half_space_axis if self.params.half_space_axis is not None
                   else default_half_space_axis(spec, rotation, self.params.plane_normal))
            split = (self.params.half_space_split if self.params.half_space_split is not None
                    else center)
            self.scene.set_half_space_plane(
                split, axis, margin=self.params.half_space_margin)
        else:
            self.scene.clear_half_space_plane()


    def _object_size_note(self):
        """Warn when the selected object is outside what the hand can close on.

        Real scanned objects are 60-300 mm; the hand closes on ~50 mm. That limit
        is GEOMETRIC, so no amount of AL/prior/beta tuning moves it -- a stall on
        a big object is the fingers not reaching, and without this line that
        reads as a solver failure and gets debugged as one. Reported for every
        object, since a hand-authored primitive can be oversized too.
        """
        spec = self.result.spec if self.result is not None else None
        if spec is None:
            return []
        if spec["type"] == "ellipsoid_set":
            largest = float(np.max(spec["extents"])) / 2.0
            smallest = float(np.min(spec["extents"])) / 2.0
        else:
            try:
                semi = proxy_semi_axes(spec)
            except ValueError:
                return []
            largest, smallest = float(np.max(semi)), float(np.min(semi))
        if largest <= self.GRASPABLE_MAX_M:
            return []
        # The narrowest axis is what a grasp actually has to span, so an object
        # that is merely LONG (a screwdriver, a pen) is still graspable across
        # its handle -- say which case this is instead of one blanket warning.
        if smallest <= self.GRASPABLE_MAX_M:
            return [f"_Object spans {2 * largest * 1000:.0f} mm at its longest "
                    f"but only {2 * smallest * 1000:.0f} mm across; grasp it on "
                    "the narrow axis._"]
        return [f"**Object is {2 * smallest * 1000:.0f} mm across its narrowest "
                f"axis** — past the ~{2 * self.GRASPABLE_MAX_M * 1000:.0f} mm the "
                f"fingertips reach off their ~{self.FINGERTIP_SHELL_M * 1000:.0f} "
                "mm shell. A stall here is geometric, not a tuning problem."]
