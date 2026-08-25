"""Interactive viser visualizer for the tendon-hand FK solver and the stepped IK
solve.

Exposes the solver knobs as live web GUI controls -- object picker, wrist start
pose, per-finger flexor tensions, per-finger contact toggles, collision / table
options, AL settings. *FK* re-poses the hand from the current sliders (and the
pose / tension sliders re-solve it live as they move); *Step* advances the IK
solve by exactly one Augmented Lagrangian outer iteration, and *Auto solve* keeps
stepping until it converges or stalls. Every step is kept, so the *Solve steps*
scrubber replays the convergence one iteration at a time (initial guess -> each
outer iteration).

*E-STOP* is the software emergency stop, and it outranks everything else on the
page. It latches: it breaks a running auto-solve out of its loop and then refuses
every solve -- FK, Step, Auto, and the live re-solve the pose/tension sliders
fire -- until *Rearm* is pressed, so nothing can restart the hand by accident.
The button is never greyed out. Nothing is lost when it trips: the stepper keeps
its multipliers, its penalty weight and its whole history, so *Rearm* + *Auto
solve* resumes the same solve rather than restarting it.

The stop is cooperative and lands at an AL iteration boundary -- one outer
iteration is a single call into the C++ solver with no interrupt hook, so ~1.7 s
is the measured worst case and a floor rather than a tuning knob. What makes the
button nevertheless feel instant is that the solve bindings release the GIL
(``capabilities()["gil_release"]``): the click is serviced while the last
iteration is still running. Against a binding without it the whole interpreter is
frozen for the duration of every iteration and the click cannot even be received
until it ends -- the app says so on the button and in the startup banner.

Changing the constraint set (object, contacts, collision, table) restarts the IK
loop, because the Augmented Lagrangian duals it carries describe the constraints
it has been working on. *Warm start* is the way to change one anyway and keep the
posture: while the latch is on, every restart begins from the state on screen
instead of from a straight hand with Q = 0 -- so it also starts an IK solve from
an FK pose you dialled in by hand. Only the posture carries; the penalty schedule
restarts regardless. *Reset defaults* puts every control back and cold-starts.

Run (from the ``crest-sparse`` root, so ``crest_sparse`` resolves to the
INSTALLED build -- launching from ``python/`` picks up the stale in-tree
``python/crest_sparse/_crest_sparse*.so`` instead, and every capability-gated
control silently goes dead):
    python -m python.tests.tendon_hand.viz_interactive

Every constraint from the paper's Chapter 2 (Eq 2.8-2.19) is an independent
switch in the *Constraints* folder -- object/table contact, object/self/table
collision, drop-normal-row SDF contact, opposition half-space, pre-grasp
centering -- each acting on the shared per-finger mask in its nested *fingers*
sub-folder. A box is the whole story: checked means that constraint family is in
the graph, with no second toggle it silently waits on. That is what makes a
stalled grasp bisectable: solve for the object alone, the table alone, or both,
with or without any of the three avoidances, and see which constraint family is
the one refusing to close. (Their tuning sliders --
collision radius/sigma/cull margin, table height offset -- stay in the
*Collision*/*Table* folders alongside the object/primitive picker.)

The solvers are the reusable ``HandFKSolver`` / ``HandIKStepper`` classes in
``tendon_hand/solvers.py``; the 3D scene is drawn by
``_plotting/viser_hand.ViserHandScene``. The trajectory planner is not part of
this app -- see the ``traj_*`` scripts.

then open the printed http://localhost:8080 URL. The startup line names the
binding that was actually loaded and lists any capability missing from it.

Optional headless self-check of the solver classes (no browser):
    python -m python.tests.tendon_hand.viz_interactive --smoke
"""

import argparse
import math
import sys
import threading
import traceback

import numpy as np

from .scene import (get_primitive_specs, ycb_primitive_specs, proxy_semi_axes,
                    GRASP_FLEXOR_TENSION, TABLE_NORMAL, ELLIPSOID_SET_BETA,
                    TABLE_SPAN, TABLE_THICKNESS, table_corner,
                    object_principal_inplane_axis, INPLANE_DEGENERACY_RATIO,
                    ellipsoid_members)
from .solvers import (
    HandSolveParams, HandFKSolver, HandIKStepper,
    resolve_scene, resolve_table_origin, capabilities,
    euler_to_R, R_to_euler, solved_wrist_pose, plane_witness,
    default_object_center,
    half_space_witness, pregrasp_center_witness, pregrasp_axis_witness,
    pregrasp_centroid_witness, finger_plane_witness, planar_gap_witness,
    default_half_space_axis, PHASE_PRESETS, FLEXOR_IDX,
    DEFAULT_WRIST_XYZ, DEFAULT_WRIST_RPY)
from .config import pinch_pose
from .mount import MOUNT_WRIST_XYZ, MOUNT_WRIST_RPY, measured_mount_pose


FINGER_LABELS = ["index", "middle", "ring", "pinky", "thumb"]

# This app's own startup object -- see HandVizApp.__init__ for why it's set
# there rather than just changed on the dropdown widget.
#
# The megaminx: a 70 mm dodecahedron the factors see as its circumsphere, so it
# is a single analytic surface (no ellipsoid SET, no fetched/fitted YCB scan
# needed) that still carries hull_vertices -- the true solid is drawn inside the
# shell, and the table seats on the solid rather than on the proxy sphere. It
# also sits inside the graspable band, so the startup scene is one a 5-finger
# grasp can actually close on. Falls back automatically (see _build_gui) when the
# binding cannot build analytic ellipsoid surfaces at all.
DEFAULT_OBJECT_PRIMITIVE = "megaminx"
DEFAULT_OBJECT_FALLBACK = "mid_sphere_ellipsoid"

# Display-only suffix for the baked-SDF spheres in the object dropdown, so they
# read apart from the analytic ``*_sphere_ellipsoid`` look-alikes. The spec keys
# (and the demo scripts' argparse choices) keep the un-suffixed names; only the
# label the user picks from carries "_sdf".
SDF_DROPDOWN_LABELS = {"sphere": "sphere_sdf", "big_sphere": "big_sphere_sdf"}

# Which half of the split the THUMB is sent to -- the opposition axis's sign,
# which the object's own geometry cannot answer (see
# solvers.orient_opposition_axis). Label -> HandSolveParams.half_space_flip.
OPPOSITION_SIDES = {
    "auto (match the hand)": None,
    "as derived": False,
    "flipped": True,
}


# The wrist sliders and the solvers must agree on what "pitch" means, so the
# convention lives with the params rather than here.
_euler_to_R = euler_to_R


def binding_path():
    """Where the loaded ``crest_sparse`` came from.

    Worth reporting because there are two of them: the installed build in
    site-packages, and a stale in-tree ``python/crest_sparse/_crest_sparse*.so``
    that shadows it whenever the app is launched from the ``python/`` directory.
    A control gated on ``capabilities()`` then goes quietly dead against a build
    the source has long since moved past, which looks like the feature failing
    rather than the import resolving somewhere unexpected."""
    import crest_sparse
    return crest_sparse.__file__


# ---------------------------------------------------------------------------
# Software e-stop.
# ---------------------------------------------------------------------------

class Refused(Exception):
    """Raised by :meth:`EStop.admit` when a solve is not allowed to start --
    either the latch is engaged or another solve already holds it."""


class EStop:
    """Latching software e-stop, and the single admission gate for every solve.

    LATCHING, not momentary. A momentary stop lets the very next thing that
    touches a slider restart the hand through the live-FK hook, which is
    precisely what someone reaching for a stop button does not want. Tripped
    stays tripped until :meth:`rearm`.

    The stop is COOPERATIVE, and it lands at an Augmented Lagrangian iteration
    boundary. One outer iteration is a single call into C++ with no interrupt
    hook (GTSAM's inner loop, via ``WarmAugmentedLagrangianOptimizer``), so
    nothing in Python can break into it -- ~1.7 s is the measured worst case and
    it is a floor, not a tuning choice. What matters is that this is bounded and
    that NO STATE IS LOST: the stepper keeps its multipliers, its penalty weight
    and its whole history, so a rearm resumes the same solve rather than
    restarting it.

    What makes the button feel instant is the GIL release on the solve binding
    (``capabilities()["gil_release"]``). With it the click is serviced while the
    last iteration is still running -- the latch engages, the controls grey out
    and the status updates at once. Without it the interpreter is frozen for the
    whole iteration and none of that can happen until it ends.

    Also the ONE place that decides whether a solve may start. It used to be a
    bare ``_solving`` bool tested and set without a lock, which the live-FK hook
    read while the auto-solve worker was writing it; folding the latch and the
    busy flag into one lock-guarded object closes that race as a side effect.
    """

    def __init__(self):
        # RLock rather than Lock: _refresh callbacks read the state while
        # holding it, and re-entering must not deadlock the GUI thread.
        self._lock = threading.RLock()
        self._tripped = False
        self._reason = ""
        self._busy = None       # what currently holds the gate, or None

    # -- the latch --

    def trip(self, reason="E-STOP pressed"):
        """Engage the latch. Runs on a viser callback thread, so it must never
        block: it takes the lock only to flip two fields, and deliberately does
        NOT wait for the running solve to notice."""
        with self._lock:
            if not self._tripped:
                self._tripped = True
                self._reason = reason
            return True

    def rearm(self):
        """Release the latch. Refuses while a solve is still winding down, so
        the GUI cannot come back to life around a solve that has not yet
        returned -- the operator would rearm into a hand still moving."""
        with self._lock:
            if self._busy is not None:
                return False
            self._tripped = False
            self._reason = ""
            return True

    def is_tripped(self):
        """The poll predicate; handed straight to ``HandIKStepper.run`` as its
        ``should_stop``. A method rather than a property because that is the
        shape run() wants."""
        with self._lock:
            return self._tripped

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def check(self):
        """Raise if the latch is engaged. For polling inside a loop that has
        work of its own between solver calls."""
        if self.is_tripped():
            raise Refused(self._reason)

    # -- admission --

    def admit(self, what):
        """Claim the gate for one solve, or raise :class:`Refused`.

        Claims EAGERLY -- the refusal comes out of this call, not out of a later
        ``__enter__`` -- because the auto-solve hands its gate to a worker
        thread and so cannot express the claim as a ``with`` block. Callers that
        can still use one: ``with estop.admit(...)`` releases on the way out.

        The check and the claim happen under one lock, which is the point: two
        callbacks arriving on different viser worker threads cannot both see a
        free gate and both start solving.
        """
        with self._lock:
            if self._tripped:
                raise Refused(f"E-STOP engaged: {self._reason}")
            if self._busy is not None:
                raise Refused(f"already running: {self._busy}")
            self._busy = what
        return _Gate(self)

    def _release(self):
        with self._lock:
            self._busy = None


class _Gate:
    """The claim :meth:`EStop.admit` hands back. Releasing twice is harmless, so
    a caller may ``release()`` early and still let a ``with`` block unwind."""

    def __init__(self, estop):
        self._estop = estop

    def release(self):
        self._estop._release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# ---------------------------------------------------------------------------
# Headless smoke test -- validates the solver classes independently of viser.
# ---------------------------------------------------------------------------

def _smoke():
    print(f"Smoke-testing the hand solver classes "
          f"({HandSolveParams().primitive}, defaults)...")
    ok = True
    caps = capabilities()

    # FK: one frame, no contact.
    res = HandFKSolver(HandSolveParams()).solve()
    status = "ok" if len(res.frames) == 1 else "BAD"
    ok = ok and status == "ok"
    print(f"  [{'FK':>8}] frames={len(res.frames)} (expect 1) [{status}] | "
          f"iters={res.meta.iterations} err={res.meta.error:.3g}")

    # The stepper, in the three contact configurations the split toggles exist
    # to bisect. A handful of steps is enough to prove the loop runs and carries;
    # convergence is what the GUI's Auto solve is for.
    if not caps["ik_stepping"]:
        print("  [IK-step] skipped -- binding has no TendonHandSolver.reset_al_duals")
    else:
        cases = [("IK", False, False)]
        if caps["table"]:
            cases += [("IK-table", True, False), ("IK-both", True, True)]
        else:
            print("  [IK-table] skipped -- binding has no support-plane env fields")
        for label, table, obj in cases:
            params = HandSolveParams()
            if table:
                params.table = True
                params.table_contact = True
                params.object_contact = obj
            # The last stepped result, captured the way the GUI does -- run()
            # returns only the status, and the gaps live on the result.
            last = {}
            st = HandIKStepper(params).run(
                max_steps=5, on_step=lambda r, s: last.update(res=r))
            res = last.get("res")
            # One snapshot per step plus the initial guess.
            n = res.num_iterates() if res is not None else 0
            status = "ok" if st.steps > 0 and n == st.steps + 1 else "BAD"
            if status == "BAD":
                ok = False
            extra = ""
            if res is not None:
                if params.object_contact:
                    extra += f" | worst object gap {res.worst_gap(0):+.5f} m"
                if res.table_contact_names():
                    extra += (f" | worst table gap "
                              f"{res.worst_table_gap(params, 0):+.5f} m")
            print(f"  [{label:>8}] steps={st.steps} state={st.state} "
                  f"snapshots={n} [{status}] | violation={st.violation:.3e} "
                  f"cost={st.cost:.4g}{extra}")
    print("Smoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Interactive app.
# ---------------------------------------------------------------------------

class HandVizApp:

    def __init__(self, server):
        import viser  # local import so --smoke needs no viser
        self.viser = viser
        self.server = server
        # What this installed binding supports, so we can gate controls a stale
        # .so would crash on (ellipsoid objects, the table, cull margin).
        # Resolved before the params because _fresh_params reads it.
        self.caps = capabilities()
        self.params = self._fresh_params()
        # Which solver produced what is on screen: "FK" for a posed hand, "IK"
        # once the stepper has been driven. Gates the live FK re-solve and labels
        # the status readout; there is no mode picker.
        self.mode = "FK"
        self.result = None
        # The software e-stop, and the gate every solve has to pass to start.
        # Replaces both the old _solving bool and the auto-run's private stop
        # Event -- one object, so "is anything running" and "may anything start"
        # cannot disagree. See EStop.
        self.estop = EStop()
        # Cached IK stepper: it owns the AL outer loop being advanced one
        # iteration per Step, so it has to outlive a single step.
        self.stepper = None
        # Warm-start latch: while on, every (re)build of the stepper starts from
        # the state on screen rather than the cold guess. See _ensure_stepper.
        self.warm_start = False
        self._auto_thread = None
        # True while Reset is writing the controls back to their defaults, so the
        # per-handle callbacks (live FK, object rebuild) sit out the restore and
        # the one solve at the end of it is the only one that runs.
        self._restoring = False
        # Latch for the object-contact form guard: it settles the two mutually
        # exclusive boxes by writing the other one, whose callback lands right
        # back in the guard. See _enforce_object_contact.
        self._contact_guard = False
        # Cached YCB catalog/browser state, built lazily by the YCB folder so the
        # app starts without touching the network or the catalog file.
        self._ycb_cache = None
        self._ycb_busy = False

        from .._plotting.viser_hand import ViserHandScene
        self.scene = ViserHandScene(server, FINGER_LABELS)

        # Park every (current and future) client's camera on the -X/palmar side so
        # the finger curl reads as a grasp instead of bending backwards. Without
        # this viser opens from the opposite side and the correct solve looks wrong.
        server.on_client_connect(lambda client: self._aim_camera(client))

        self._build_gui()
        # Decide up front whether the opening object supports the in-plane
        # contact form, so the box is never offered live for a scene the solve
        # would refuse.
        self._refresh_planar_contact_gate()
        # A cached FK solver so wrist/tension tweaks warm-start (rebuilt on object
        # change only).
        self._rebuild_fk()
        self._refresh_object()
        self._fk_solve()

    def _fresh_params(self):
        """A cold ``HandSolveParams`` plus this app's OWN scene defaults.

        The single source of "defaults" for the fields no GUI control owns, so
        startup and *Reset defaults* cannot drift apart. Reset used to build a
        bare ``HandSolveParams()``, which restored the headless
        ``table_burial = 0.5`` under sliders that still read 0 and re-seated the
        table -- moving the object and its ellipsoids with it.

        Only the not-GUI-backed fields belong here; everything a widget drives
        is written by :meth:`_sync_params` from the (already restored) handles.
        """
        params = HandSolveParams()
        # Not HandSolveParams' own default, which stays whatever headless
        # callers/other scripts expect. Set on the params (not just on the
        # dropdown widget) because _rebuild_fk()/_refresh_object() run before
        # the first _sync_params() and read params.primitive directly; the
        # dropdown's own default_label computation reads it too, so the widget
        # follows automatically.
        params.primitive = self._resolve_default_primitive()
        # Seat the object ON the table rather than half-buried in it. Another of
        # this app's own defaults (like the object above): HandSolveParams keeps
        # 0.5 for the §1.8 low-profile-object case its docstring argues for, but
        # for browsing arbitrary objects -- YCB scans included -- an object sunk
        # halfway through the table reads as a bug in the scene. The seating rule
        # derives the height from the object's own along-normal extent, so this
        # is correct per object with nothing to re-tune per pick.
        params.table_burial = 0.0
        return params

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
        """
        if self._restoring:
            return
        self._sync_params()
        self._refresh_object()
        self._invalidate_stepper()
        if self.mode == "FK":
            self._live_fk()
        else:
            self._render_frame()

    def _refresh_object_mesh(self, spec, center, rotation):
        """Draw (or clear) the object's TRUE geometry behind the analytic surface
        the solver actually sees.

        Two kinds of object have one: a ycb: set, whose shells approximate a
        scanned mesh, and a spec carrying ``hull_vertices`` -- the megaminx,
        whose circumsphere encloses a dodecahedron. Both are the same question
        ("how much object is really inside the surface the fingers stop on"), so
        they share one toggle.

        The mesh has to be put in the SAME frame the shells were re-centered
        into (see ``scene.ycb_primitive_specs``), or the two render a few cm
        apart and the overlay is worse than useless. A missing mesh cache is not
        an error -- the fits are committed but the meshes are not, so an object
        can be perfectly loadable with nothing to draw behind it.
        """
        if not self.g_show_true_mesh.value:
            self.scene.clear_object_mesh()
            return
        hull = spec.get("hull_vertices")
        if hull is not None:
            # Local point set -> hull; set_object_mesh applies the object pose,
            # so the solid lands inside its own shell however the object is posed.
            self.scene.set_object_mesh(self.scene.hull_mesh(hull), center, rotation)
            return
        if spec["type"] != "ellipsoid_set":
            self.scene.clear_object_mesh()
            return
        try:
            from .._objects.ycb import Catalog, YcbCache, ground_and_center

            cache = YcbCache(Catalog())
            mesh = ground_and_center(
                cache.load_mesh(spec["ycb"], spec["source"], max_texture=512))
            mesh.apply_translation(-np.asarray(spec["recenter"], float))
            self.scene.set_object_mesh(mesh, center, rotation)
        except Exception as exc:
            self.scene.clear_object_mesh()
            print(f"[viz] no scan mesh for {spec.get('ycb')}: {exc}")

    # -- YCB objects --------------------------------------------------------

    def _build_ycb_folder(self, gui):
        """Fetch-and-fit controls for the YCB object set.

        The offline path (``python -m tests._objects.ycb.browser --fit <name>``)
        remains the primary one and writes the same files; this exists so an
        object can be brought in without leaving the app. Everything here is
        gated on ``ellipsoid_set``: without it a fitted object could be written
        but never loaded, so offering the button would be a trap.
        """
        available = self.caps.get("ellipsoid_set", False)
        with gui.add_folder("YCB objects", expand_by_default=False):
            if not available:
                gui.add_markdown(
                    "Needs a rebuilt `_crest_sparse` with "
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
            from .._objects.ycb import Catalog
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
            from .._objects.ycb import Catalog, YcbCache
            from .._objects.ycb.fitting import fit_object
            from .scene import ycb_primitive_specs

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

    # Dropdown suffix for a catalog object with no committed fit yet. Selecting
    # one downloads and fits it on the spot (see _on_object_selected).
    UNFITTED_SUFFIX = "  [fit on select]"

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
                from .._objects.ycb import Catalog

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

    def _aim_camera(self, client):
        """Point one client's camera at the current object from the demo viewpoint."""
        _spec, center, _rot, _pose = resolve_scene(self.params)
        pos, look = self.scene.grasp_camera(center)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(float(v) for v in pos)
        client.camera.look_at = tuple(float(v) for v in look)

    def _aim_all_cameras(self):
        for client in self.server.get_clients().values():
            self._aim_camera(client)

    # -- solver plumbing --

    def _rebuild_fk(self):
        self.fk_solver = HandFKSolver(self.params)
        # The FK solver is rebuilt whenever the object changes, and the object is
        # part of the stepper's constraint set too.
        self._invalidate_stepper()

    def _pose_at_mount(self, _=None):
        """Drive the wrist sliders to the measured robot mount and re-pose.

        With the wrist at ``T_flange<-wrist``, the viser world frame IS the flange
        frame, so what you see is the hand as it hangs off the arm in the CAD
        assembly -- the check the measurement actually needs. Turns the mount
        frames on, since arriving here with them off shows nothing new.
        """
        for handle, value in zip(
                (self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw),
                tuple(MOUNT_WRIST_XYZ) + tuple(MOUNT_WRIST_RPY)):
            # Sliders quantize to their step, so the pose actually solved is the
            # rounded one; _render_mount draws from the sliders for that reason.
            handle.value = float(value)
        self.g_show_mount.value = True
        self._sync_params()
        self._fk_solve()

    def _render_mount(self, res=None):
        """Draw or clear the wrist/flange frame pair for the pose on screen.

        Anchored on the SOLVED wrist, not ``params.wrist_pose``: the wrist is a
        variable whose prior is soft, so contact pulling on the hand moves it tens
        of millimetres off the commanded pose. Drawing the commanded pose leaves
        the frames stranded while the hand they describe moves away from them --
        and the flange is rigidly bolted to the wrist, so it must travel with it.
        ``res`` is the iterate being rendered, so the frames track the convergence
        scrubber too. Falls back to the commanded pose before the first solve.
        """
        if not self.g_show_mount.value:
            self.scene.clear_mount_frames()
            return
        T_wrist = (solved_wrist_pose(self.fk_solver.configs, res.frames[0])
                   if res is not None else self.params.wrist_pose)
        self.scene.set_mount_frames(T_wrist, measured_mount_pose())

    def _sync_wrist(self):
        R = _euler_to_R(self.g_roll.value, self.g_pitch.value, self.g_yaw.value)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.g_tx.value, self.g_ty.value, self.g_tz.value]
        self.params.wrist_pose = T

    def _sync_params(self):
        p = self.params
        p.primitive = self._label_to_key[self.g_object.value]
        p.object_center, p.object_rotation = self._object_pose_from_sliders()
        self._sync_wrist()
        p.passive_tension = self.g_passive.value
        p.flexor_tensions = [s.value for s in self.g_flexors]
        p.flexor_tension_sigma = 10.0 ** self.g_flexor_sigma.value
        p.passive_tension_sigma = 10.0 ** self.g_passive_sigma.value
        p.ellipsoid_set_beta = float(self.g_set_beta.value)
        p.contact_fingers = [c.value for c in self.g_contacts]
        p.object_contact = self.g_obj_contact.value or self.g_obj_contact_plane.value
        # Which FORM, once object contact is on at all. The two boxes are kept
        # mutually exclusive by _enforce_object_contact, so this cannot be read
        # as "both": object_contact says whether there is a contact,
        # object_contact_in_plane says which distance it is measured with.
        p.object_contact_in_plane = self.g_obj_contact_plane.value
        p.table_contact = self.g_tbl_contact.value
        p.contact_drop_normal_row = self.g_drop_normal_row.value
        p.half_space = self.g_half_space.value
        p.half_space_split = None   # derive from object_center
        # Cleared every sync so the SIGN is re-resolved against the posture on
        # screen: _attach_opposition writes the oriented axis back here, and a
        # stale one would keep sending the thumb to the side it was on two
        # solves ago.
        p.half_space_axis = None
        p.half_space_flip = OPPOSITION_SIDES[self.g_half_sides.value]
        p.half_space_margin = self.g_half_margin.value
        p.pregrasp_center = self.g_pregrasp_center.value
        p.h_clear = self.g_h_clear.value
        p.pregrasp_axis_align = self.g_axis_align.value
        p.pregrasp_centroid = self.g_pregrasp_centroid.value
        p.sigma_wrist_pos = 10.0 ** self.g_sig_pos.value
        p.sigma_wrist_rot = 10.0 ** self.g_sig_rot.value
        # AL
        p.al_mu = self.g_al_mu.value
        p.al_rate = self.g_al_rate.value
        p.al_iters = self.g_al_iters.value
        p.ik_settle_steps = int(self.g_ik_settle.value)
        # collision
        p.collision = self.g_collision.value
        p.self_collision = self.g_self_collision.value
        p.collision_radius = self.g_coll_radius.value
        p.collision_sigma = 10.0 ** self.g_coll_sigma.value
        p.cull_margin = (None if not self.caps["collision_cull"] or self.g_cull.value <= 0
                         else self.g_cull.value)
        # table
        p.table = self.g_table.value and self.caps["table"]
        p.plane_normal = np.array(TABLE_NORMAL, float)
        p.plane_avoidance = self.g_plane_avoid.value
        # Resolve the auto (under-object) seating once, then bake the "height
        # offset" slider into an EXPLICIT plane_origin -- shared by the factor
        # graph (via this params object) and the rendered slab (_table_origin
        # just reads it back). Previously this stayed None and the offset was
        # applied only in _table_origin() for the visual scene, so dragging the
        # slider moved the drawn table without moving the solver's plane.
        p.plane_origin = None
        spec, center, _rot, _pose = resolve_scene(p)
        p.plane_origin = np.asarray(
            resolve_table_origin(p, spec, center), float) + (
                self.g_plane_offset.value * np.array(TABLE_NORMAL, float))
        # display toggles
        self.scene.show_discs = self.g_show_discs.value
        self.scene.show_contact_spheres = self.g_show_contact.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self.scene.show_gap_lines = self.g_show_gaps.value
        self.scene.show_finger_planes = self.g_show_finger_planes.value
        self.scene.show_planar_gap = self.g_show_planar_gap.value

    def _table_origin(self):
        """The rendered slab's origin -- reads straight off ``params.plane_origin``,
        which ``_sync_params`` bakes the height-offset slider into, so the drawn
        table and the factor graph's plane always agree."""
        spec, center, _rot, _pose = resolve_scene(self.params)
        return resolve_table_origin(self.params, spec, center)

    def _refresh_table_readout(self, origin, corner):
        """Publish the landmark's numbers: the square's size and where its corner
        frame currently is, in world coordinates.

        These have to be readable, not inferred. The whole point of the frame is
        to be measured against a real bench, and a triad you can see but whose
        coordinates you cannot read is not a landmark. The plane height is quoted
        alongside because the table is seated from the object (see
        ``auto_table_origin``), so it moves when the object changes -- this is
        where you see that it did.
        """
        origin = np.asarray(origin, float).reshape(3)
        corner = np.asarray(corner, float).reshape(3)
        axis = int(np.argmax(np.abs(np.asarray(self.params.plane_normal, float))))
        self.g_table_status.content = (
            f"square **{TABLE_SPAN:.3f} x {TABLE_SPAN:.3f} m**, "
            f"{TABLE_THICKNESS * 1e3:.0f} mm thick  \n"
            f"plane (top face) {'xyz'[axis]} = {origin[axis]:+.4f} m  \n"
            f"corner frame ({corner[0]:+.4f}, {corner[1]:+.4f}, "
            f"{corner[2]:+.4f}) m")

    # -- rendering --

    def _refresh_object(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        self.scene.set_object(spec, center, rotation)
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
        corner = table_corner(origin, self.params.plane_normal)
        if self.g_show_table_frame.value:
            self.scene.set_table_frame(
                corner, label=f"table corner  {TABLE_SPAN:g} x {TABLE_SPAN:g} m")
        else:
            self.scene.clear_table_frame()
        self._refresh_table_readout(origin, corner)
        if self.params.half_space:
            axis = (self.params.half_space_axis if self.params.half_space_axis is not None
                   else default_half_space_axis(spec, rotation, self.params.plane_normal))
            split = (self.params.half_space_split if self.params.half_space_split is not None
                    else center)
            self.scene.set_half_space_plane(
                split, axis, margin=self.params.half_space_margin)
        else:
            self.scene.clear_half_space_plane()

    def _render_frame(self, live=False):
        if self.result is None:
            # Nothing solved yet, so the commanded wrist pose is all there is.
            self._render_mount()
            return
        # Render whichever solve snapshot the convergence scrubber selects; with
        # no scrubber up this is the result itself, so the gap readouts below
        # describe the intermediate state without knowing about iterates at all.
        # Every result here is a single state, so there is only ever frame 0.
        res = self._iter_view(live)
        self._render_mount(res)
        # Only the fingers this solve drove onto a surface get a gap line for it;
        # a distance readout on a finger nothing asked to touch is just noise.
        # The two sets are independent, so a finger can carry both lines, one, or
        # neither.
        gaps = res.contact_witness(0)
        names = set(res.contact_names() if self.params.object_contact else [])
        gaps = {name: v for name, v in gaps.items() if name in names}
        table_names = res.table_contact_names()
        table_gaps = (plane_witness(self.params, res, 0, names=table_names)
                      if table_names else None)
        half_gaps = (half_space_witness(self.params, res, 0)
                    if self.params.half_space else None)
        center_gap = (pregrasp_center_witness(self.params, res, 0)
                     if self.params.pregrasp_center else None)
        axis_align = (pregrasp_axis_witness(self.params, res, 0)
                     if self.params.pregrasp_axis_align else None)
        centroid_gap = (pregrasp_centroid_witness(self.params, res, 0)
                       if self.params.pregrasp_centroid else None)
        # Purely a picture of the posture on screen, so unlike the overlays
        # above it is gated on its own display checkbox rather than on a
        # constraint being switched on -- the plane is worth looking at exactly
        # when nothing is enforcing anything about it.
        planes = (finger_plane_witness(res, 0)
                  if self.g_show_finger_planes.value else None)
        # Same gating, one step further: the in-plane distance also needs a
        # binding that can evaluate the factor and an object with an analytic
        # ellipsoid form. planar_gap_witness returns None on either, so the
        # overlay simply does not appear rather than the render failing.
        planar = (planar_gap_witness(self.params, res, 0)
                  if (self.g_show_planar_gap.value and self.caps["planar_gap"])
                  else None)
        self._report_iterate(live)
        self.scene.update(res.frames[0],
                          tip_radii=res.tip_radii,
                          collision_radius=self.params.collision_radius,
                          # The spheres are drawn whenever ANY of the three
                          # consumers is using them, matching what the solve
                          # actually built.
                          collision=(self.params.collision
                                     or self.params.self_collision
                                     or (self.params.table
                                         and self.params.plane_avoidance)),
                          gaps=gaps,
                          table_gaps=table_gaps,
                          half_space_gaps=half_gaps,
                          center_gap=center_gap,
                          axis_align=axis_align,
                          centroid_gap=centroid_gap,
                          finger_planes=planes,
                          planar_gaps=planar)

    def _set_status(self, text):
        self.g_status.content = text

    def _error_status(self, exc):
        """Show a solver exception with the reason attached where we can name it.

        GTSAM's IndeterminantLinearSystem text explains what an ill-posed system
        IS, not which of the controls on this page caused one. In a pre-grasp
        solve there is essentially one answer -- the wrist prior stopped fixing
        the wrist -- so say that instead of leaving a Doxygen link on screen."""
        note = []
        if "Indeterminant" in str(exc):
            note = self._wrist_gauge_note() or [
                "*Indeterminant means some variable has no information left. In "
                "a pre-grasp solve the usual culprit is the wrist prior: "
                "inequalities (collision, opposition) contribute nothing while "
                "satisfied, so the prior is what fixes the wrist.*"]
        self._set_status("  \n".join([f"**Error:** {exc}"] + note))

    # -- solve --

    def _set_solving(self, solving=None):
        """Put the control panel into the state the e-stop says it should be in.

        Two things grey the panel out: a solve running (it is busy) and the
        e-stop being engaged (it is refusing). Both are read off the latch, so
        this can be called with no argument from anywhere -- the trip and rearm
        handlers do exactly that -- and cannot disagree with what admit() will
        actually allow. ``solving`` is accepted only to spare the callers that
        already know the answer.
        """
        if solving is None:
            solving = self.estop.busy is not None
        tripped = self.estop.is_tripped()
        # While the latch is engaged NOTHING may start, so every control that
        # begins work greys out, not just the ones a running solve blocks.
        blocked = solving or tripped
        self.g_fk.disabled = blocked
        # Names the state, not the button that caused it: labelling this one
        # "E-STOP" would read as a second stop button sitting next to the real
        # one.
        self.g_fk.label = ("FK (stopped)" if tripped
                           else "Solving..." if solving else "FK")
        for btn in (getattr(self, "g_ik_step", None), getattr(self, "g_ik_auto", None)):
            if btn is not None:
                btn.disabled = blocked or not self.caps["ik_stepping"]
        # Reset would pull the params out from under a running step, and flipping
        # the warm start mid-run cannot affect the loop already built, so both
        # wait it out rather than looking like they did something.
        if getattr(self, "g_warm", None) is not None:
            self.g_warm.disabled = blocked or not self.caps["solver_seed"]
        if getattr(self, "g_reset", None) is not None:
            self.g_reset.disabled = blocked
        # Rearm is live only when there is something to rearm FROM, and only
        # once the stopped solve has actually returned -- rearming around a
        # solve still winding down would hand the controls back while the hand
        # is still moving. EStop.rearm() enforces the same rule; this just makes
        # the button agree with it.
        if getattr(self, "g_rearm", None) is not None:
            self.g_rearm.disabled = not tripped or solving

    def _fk_solve(self, _=None):
        """Re-pose the hand from the current sliders with the FK solver.

        Also the "start over" action for the IK loop: it drops any partially
        stepped solve. What the next Step then starts from is the warm-start
        latch's business -- off, the cold guess; on, this FK pose.

        Refused outright while the e-stop is engaged: an FK solve re-poses the
        hand, which is exactly what a stopped app must not do."""
        try:
            gate = self.estop.admit("FK solve")
        except Refused:
            return
        try:
            with gate:
                self._set_solving(True)
                try:
                    self._sync_params()
                    self._refresh_object()
                    self._invalidate_stepper()
                    self.mode = "FK"
                    self._set_status("Solving (FK)...")
                    # Reuse the cached FK solver (shares self.params) so this
                    # warm-starts.
                    self.result = self.fk_solver.solve()
                    self._rebuild_iter_slider()
                    self._render_frame()
                    self._report()
                except Exception as exc:  # surface it in the GUI, keep serving
                    self._error_status(exc)
                    raise
        finally:
            # Outside the gate, so it reads a released latch and can re-enable
            # the controls (or leave them grey, if the e-stop tripped meanwhile).
            self._set_solving()
            self._report_estop()

    def _pinch_note(self):
        """Warn when pinch-centroid centering is checked but the selected
        digits have no measured pinch pose.

        Only combinations INCLUDING THE THUMB were measured, so a selection
        like index+middle silently attaches nothing -- the C++ layer skips an
        unconfigured constraint without complaint, which is the same trap the
        other pre-grasp toggles set (they need the thumb plus one other finger
        or they vanish too). Say it out loud instead."""
        if not self.params.pregrasp_centroid:
            return []
        names = [n for n, c in zip(FINGER_LABELS, self.g_contacts) if c.value]
        pose = pinch_pose(names)
        if pose is None:
            return [f"**pinch-centroid: INACTIVE** -- no measured pinch pose "
                    f"for ({', '.join(names) or 'no fingers'}); only "
                    f"combinations including the thumb were measured"]
        if not pose.touches():
            # A real measurement, but of a closest approach rather than a
            # contact -- the centroid is still well-defined, the digits just
            # never actually meet there.
            return [f"pinch-centroid: these digits close to "
                    f"{pose.gap * 1000:.1f} mm apart (they never touch)"]
        return []

    def _finger_plane_note(self):
        """Say when the pinch-plane overlay is switched on but has no plane to
        draw -- the same measured-pinch-pose dependency :meth:`_pinch_note`
        covers, hit from the display side.

        Worth its own line because here there is no constraint to blame: the
        checkbox is ticked, the hand is on screen, and nothing appears. The
        planes are anchored on the centroid of the digits the SOLVE designated,
        so a thumbless selection leaves them undefined.

        Reads the result's digits, not the checkboxes -- unlike
        :meth:`_pinch_note`, which describes the constraint the NEXT solve will
        attach. This describes an overlay on the posture already drawn, and
        that overlay is keyed off the same ``contact_names`` the result carries,
        so a contact box unticked since the last solve must not change this line
        while the hand it describes is still on screen."""
        if not self.g_show_finger_planes.value or self.result is None:
            return []
        names = self.result.contact_names()
        if pinch_pose(names) is None:
            return [f"**finger pinch planes: NOTHING DRAWN** -- no measured "
                    f"pinch pose for ({', '.join(names) or 'no fingers'}); the "
                    f"planes pass through that centroid, and only combinations "
                    f"including the thumb were measured"]
        return []

    def _enforce_object_contact(self, source):
        """Keep the two object-contact FORMS mutually exclusive.

        3D and in-plane are two metrics for one constraint -- one factor per
        contact finger either way -- so "both" is not a state the graph has.
        Rather than silently preferring one at build time, the box you just
        touched wins and the other clears, which is the same rule stated in the
        hints and visible the moment you click.

        ``source`` is the handle that changed. Re-entrant by construction (it
        writes the OTHER handle, whose own callback lands right back here), so
        it is latched; it also sits out :attr:`_restoring`, since Reset and the
        phase presets write both boxes as one batch and settle it themselves at
        the end.
        """
        if self._restoring or self._contact_guard:
            return
        other = (self.g_obj_contact_plane if source is self.g_obj_contact
                 else self.g_obj_contact)
        if not (source.value and other.value):
            return
        self._contact_guard = True
        try:
            other.value = False
        finally:
            self._contact_guard = False

    def _planar_contact_available(self):
        """``(ok, reason)`` for whether Eq 13 in-plane contact can be built for
        the scene AS SET UP IN THE GUI -- checked before a solve rather than
        after, so an impossible request never reaches the solver.

        Mirrors the three refusals in :func:`config.attach_contact` exactly. The
        two live here as well because the GUI knows the answer while the box is
        still being offered, and greying a control is a better way to say "not
        for this object" than an exception after Auto solve."""
        if not self.caps["planar_contact"]:
            return False, ("this binding cannot build it (no "
                           "EnvironmentConfig.object_contact_in_plane)")
        # Through resolve_scene, so the answer is read off the SAME spec the next
        # solve will build from rather than a second lookup that could disagree.
        spec = resolve_scene(self.params)[0]
        if ellipsoid_members(spec) is None:
            return False, (f"a `{spec['type']}` object has no ellipsoid "
                           f"cross-section for the pulling plane to cut")
        names = [n for n, c in zip(FINGER_LABELS, self.g_contacts) if c.value]
        if pinch_pose(names) is None:
            return False, (f"no measured pinch pose for "
                           f"({', '.join(names) or 'no fingers'}), so Eq 11 has "
                           f"no centroid to span the plane with")
        return True, ""

    def _refresh_planar_contact_gate(self):
        """Grey the in-plane contact box -- and clear it if it was on -- whenever
        the current object or digit set cannot support it.

        Clearing rather than leaving it checked-but-disabled is deliberate: a
        ticked box that the next solve would refuse is a lie about what is in the
        graph. The status line says why (see :meth:`_planar_contact_note`)."""
        ok, _reason = self._planar_contact_available()
        self.g_obj_contact_plane.disabled = not ok
        if not ok and self.g_obj_contact_plane.value:
            # Only this box is cleared -- the 3D box is deliberately NOT ticked
            # in compensation. Substituting the other metric for the one that was
            # asked for is exactly the silent fallback attach_contact refuses to
            # make; object contact simply goes off, and the status line says why.
            self._contact_guard = True
            try:
                self.g_obj_contact_plane.value = False
            finally:
                self._contact_guard = False

    def _planar_contact_note(self):
        """Say why the in-plane contact box is greyed, when it is."""
        ok, reason = self._planar_contact_available()
        if ok:
            return []
        return [f"*in-plane object contact unavailable: {reason}*"]

    def _planar_gap_note(self):
        """Say when the in-plane overlay is on but has nothing to measure.

        Three ways to get an empty overlay, none of them a failure: a binding
        that cannot evaluate the factor, an object with no ellipsoid form
        (cube/cylinder/capsule -- the factor takes an ellipsoid set), or solved
        digits with no measured pinch pose, which leaves Eq 11 without its
        centroid and so without a plane."""
        if not self.g_show_planar_gap.value or self.result is None:
            return []
        if not self.caps["planar_gap"]:
            return ["**in-plane distance: UNAVAILABLE** -- this binding has no "
                    "`ellipsoid_set_planar_gap`; rebuild it (`pip install .` "
                    "from the crest-sparse root)"]
        if ellipsoid_members(self.result.spec) is None:
            return [f"**in-plane distance: NOTHING DRAWN** -- a "
                    f"`{self.result.spec['type']}` object has no ellipsoid "
                    f"cross-section; use a sphere, an ellipsoid or a ycb: object"]
        if pinch_pose(self.result.contact_names()) is None:
            return ["**in-plane distance: NOTHING DRAWN** -- no measured pinch "
                    "pose for the solved digits, so Eq 11 has no plane to cut "
                    "the object with"]
        return []

    def _half_space_note(self):
        """The opposition half-space's own status line: inert when no finger is
        checked for it, and the standoff it is holding when there is one.

        The constraint is built per finger off its own ``half_space_node``, so
        the only way to check the box and get nothing is to check no fingers --
        the finger mask is the one dependency it has left."""
        if not self.params.half_space:
            return []
        names = [n for n, c in zip(FINGER_LABELS, self.g_contacts) if c.value]
        if not names:
            return ["**opposition half-space: INACTIVE** -- no fingers checked "
                    "in the *fingers* folder, so there is nothing to oppose"]
        if not self.caps["half_space_margin"]:
            return ["opposition half-space: this binding has no "
                    "`EnvironmentConfig.half_space_margin`, so the standoff "
                    "slider is inert (rebuild with `pip install .`)"]
        lines = []
        if self.params.half_space_margin > 0.0:
            lines.append(f"opposition standoff: each side held "
                         f"{self.params.half_space_margin * 1000:.0f} mm off "
                         f"the split "
                         f"({self.params.half_space_margin * 2000:.0f} mm "
                         f"corridor)")
        lines.extend(self._split_axis_note())
        lines.extend(self._opposition_side_note())
        lines.extend(self._rotation_driver_note())
        return lines

    # Wrist priors looser than this (sigma, m and rad) stop fixing the wrist's
    # gauge in any solve whose other constraints are all INEQUALITIES. Measured
    # on the pen pre-grasp scene (pinch-centroid + collision + opposition): every
    # cell at sigma >= 10 on EITHER prior throws IndeterminantLinearSystem near
    # W0, every cell at <= 1 solves, and adding the other pre-grasp constraints
    # does not move the boundary.
    WRIST_PRIOR_GAUGE_LIMIT = 10.0

    def _wrist_gauge_note(self):
        """Warn when the wrist prior is the only thing fixing the wrist, and is
        too loose to do it.

        An inequality that is SATISFIED contributes no rows to the linearized
        system -- collision, table avoidance and the opposition half-space are
        all inequalities, and in a pre-grasp scene they sit slack. Contact
        equalities are what would otherwise pin the hand, and a pre-grasp solve
        has none by definition. So the 6 dof of the wrist are held by: the
        pre-grasp constraints that touch it (pinch-centroid, 3 rows on a point;
        centering, 3 rows on a fingertip midpoint; short-axis alignment, ONE
        scalar row) and the prior. Loosen the prior far enough and what is left
        is rank-deficient -- which surfaces as GTSAM's IndeterminantLinearSystem
        near W0, not as anything that names the prior."""
        pos, rot = self.params.sigma_wrist_pos, self.params.sigma_wrist_rot
        loose = [n for n, v in (("position", pos), ("rotation", rot))
                 if v >= self.WRIST_PRIOR_GAUGE_LIMIT]
        if not loose:
            return []
        equality_backed = (self.params.object_contact or self.params.table_contact)
        if equality_backed:
            return []
        return [f"**wrist prior ({' and '.join(loose)}) is very loose "
                f"(sigma {pos:g} m / {rot:g} rad)** -- with no contact "
                f"equalities on, nothing else fixes the wrist: collision and "
                f"the opposition half-space are inequalities and contribute "
                f"nothing while satisfied. Expect "
                f"*IndeterminantLinearSystem near W0*; keep sigma at or below "
                f"1 (log10 0)."]

    def _rotation_driver_note(self):
        """Say when nothing in the constraint set can rotate the hand.

        The half-space is a one-sided inequality on POSITIONS: once each digit
        is on its own side it goes slack and contributes no gradient at all, so
        it cannot turn the wrist however large the standoff. Pinch-centroid is
        three rows on a single point -- satisfiable by translation alone.
        Short-axis alignment is the only constraint here that says anything
        about ORIENTATION. Measured on this scene: 1.5 degrees of wrist rotation
        without it, 45 with."""
        if not (self.params.half_space or self.params.pregrasp_centroid):
            return []
        if self.params.pregrasp_axis_align or self.params.object_contact:
            return []
        return ["*nothing here rotates the hand*: the half-space is a "
                "one-sided inequality on positions (slack => no gradient) and "
                "pinch-centroid is satisfiable by translation alone. Tick "
                "**short-axis alignment** for the orientation constraint."]

    def _split_axis_note(self):
        """Which way the split LINE is pointing, and whether the object chose it.

        The line is derived silently from the object's silhouette on the support
        plane, and a derivation with no readout is one nobody can check: the
        degenerate case (every ball, can and bowl) hands back a fixed world
        direction that looks identical on screen to a measured one, so a wrong
        axis and a defaulted axis are indistinguishable without the ratio."""
        spec, _center, rotation, _pose = resolve_scene(self.params)
        try:
            e_long, ratio = object_principal_inplane_axis(
                spec, rotation, self.params.plane_normal)
        except ValueError:
            return []
        # Reported as the LINE (mod 180 deg), since the sign shown to the user is
        # the side assignment, and _opposition_side_note already covers that.
        deg = np.degrees(np.arctan2(e_long[1], e_long[0])) % 180.0
        if ratio < INPLANE_DEGENERACY_RATIO:
            return [f"split line: object is in-plane isotropic ({ratio:.2f}x), "
                    f"so the default +Y split is used — thumb on the -X side"]
        return [f"split line: {deg:.0f}° from +X in the table plane "
                f"(object is {ratio:.1f}x longer that way)"]

    def _opposition_side_note(self):
        """How far the current posture is from the side assignment being asked
        for -- the one number that says whether this constraint is a nudge or a
        demand that the hand turn itself inside out.

        Worth a line of its own because the failure is silent otherwise: with
        the sides the wrong way up the solve stalls on iteration 3 with the hand
        visibly untouched, which reads as the solver giving up rather than as
        the constraint asking for a 180 degree roll."""
        res = self._iter_view()
        if res is None or "thumb" not in res.finger_names:
            return []
        gaps = half_space_witness(self.params, res, 0)
        if not gaps or "thumb" not in gaps:
            return []
        worst = min(v[2] for v in gaps.values())
        mode = self.g_half_sides.value
        if worst >= 0.0:
            return [f"opposition sides ({mode}): satisfied, worst digit "
                    f"{worst * 1000:+.0f} mm inside its half"]
        return [f"**opposition sides ({mode}): the hand is on the WRONG side "
                f"by up to {-worst * 1000:.0f} mm** -- it has to trade thumb "
                f"and fingers over to satisfy this, which normally stalls the "
                f"solve. Try *auto* (or *flipped*) in the sides dropdown."]

    def _report(self):
        m = self.result.meta
        lines = [f"**{self.mode}** &nbsp; iters={m.iterations} &nbsp; "
                 f"err={m.error:.3g} &nbsp; {m.total_time_ms:.0f} ms"]
        if self.mode != "FK":
            lines.extend(self._contact_lines(-1))
        lines.extend(self._half_space_note())
        lines.extend(self._wrist_gauge_note())
        lines.extend(self._pinch_note())
        lines.extend(self._finger_plane_note())
        lines.extend(self._planar_contact_note())
        lines.extend(self._planar_gap_note())
        lines.extend(self._object_size_note())
        self._set_status("  \n".join(lines))

    # The fingertips ride a shell of roughly this radius about the hand base, and
    # a curl can close on an object only a little smaller than it. Measured, not
    # derived -- see the reachability investigation behind GRASP_SPHERE_CENTER.
    FINGERTIP_SHELL_M = 0.055
    GRASPABLE_MAX_M = 0.050

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

    def _fingers_label(self, names):
        return ("none" if not names
                else ", ".join(names)
                if len(names) < len(self.result.finger_names) else "all")

    def _contact_lines(self, k):
        """The per-surface contact readout: which fingers were driven onto each
        surface and how far they ended up from it. Reported per surface because
        the whole point of splitting the toggles is telling the two apart -- one
        combined number cannot say which family is the one refusing to close."""
        lines = []
        if self.params.object_contact:
            names = self.result.contact_names()
            lines.append(f"object contact: {self._fingers_label(names)}")
            lines.append(f"terminal worst object gap: "
                         f"{self.result.worst_gap(k):+.5f} m")
        table_names = self.result.table_contact_names()
        if table_names:
            lines.append(f"table contact: {self._fingers_label(table_names)}")
            lines.append(f"terminal worst table gap: "
                         f"{self.result.worst_table_gap(self.params, k):+.5f} m")
        if not lines:
            lines.append("contact: none (no surface targeted)")
        return lines

    # -- IK stepping: drive the AL outer loop one iteration at a time --

    def _invalidate_stepper(self):
        """Drop the cached stepper so the next Step cold-starts.

        Only for changes to the CONSTRAINT SET (object, contact mask, collision,
        table): the stepper carries Lagrange multipliers and a penalty weight
        that describe the constraints it has been working on, and those mean
        nothing against a different set. Tensions and the wrist pose are passed
        fresh every step, so they deliberately do NOT invalidate -- nudging the
        flexor mid-solve and continuing is the point."""
        self.stepper = None

    def _ensure_stepper(self):
        """The cached stepper, built on demand.

        This is the ONE place the warm start is applied, and it is applied at
        BUILD time: a rebuilt solver is exactly the thing that would otherwise
        cold-start, and reading the on-screen state here means the posture handed
        over is whatever is showing at that instant -- no capture to go stale, no
        ordering to get right."""
        if self.stepper is None:
            if self.warm_start:
                self.params.initial_state = self._seed_state()
                # The other half: the multipliers holding each constraint. The
                # posture alone restarts the penalty schedule at mu = al_mu with
                # every multiplier at zero, so the solve lets go of everything it
                # had already satisfied and hauls it back over the next few
                # iterations -- the fingers lifting off the table. Matched onto
                # the new constraint set by identity, so the shared constraints
                # keep their multipliers and the new ones start at zero.
                self.params.initial_duals = (
                    self._seed_duals() if self.g_carry_duals.value else None)
                # Carrying the posture is only half of it. The wrist and the
                # flexor tensions are VARIABLES with soft priors commanded from
                # the sliders, and a contact solve moves both a long way from
                # what is commanded; rebuilding those priors at the old slider
                # values hauls the hand straight back. Adopt where the solve
                # actually ended up -- and show it on the sliders.
                self._adopt_solved_wrist()
                self._adopt_solved_tensions()
            else:
                self.params.initial_state = None
                self.params.initial_duals = None
            self.stepper = HandIKStepper(self.params)
        return self.stepper

    def _seed_duals(self):
        """The AL multipliers of the solve on screen, or None.

        Taken from ``self.result`` rather than the scrubbed iterate view: the
        multipliers belong to the solve as a whole (the AL outer loop's running
        state), not to the snapshot the scrubber happens to be showing."""
        return None if self.result is None else self.result.duals

    def _adopt_solved_wrist(self):
        """Re-aim the wrist prior at the pose the last solve reached, and move
        the sliders to match.

        The sliders are the prior's only input -- every step re-commands it from
        ``params.wrist_pose``, which ``_sync_params`` reads straight off them --
        so leaving them on the old numbers would undo this on the very next step.
        Moving them is also the honest thing: after a solve they no longer
        describe where the hand is."""
        res = self._iter_view()
        if res is None:
            return
        T = solved_wrist_pose(self.fk_solver.configs, res.frames[0])
        roll, pitch, yaw = R_to_euler(T[:3, :3])
        self._restoring = True   # these are OUR writes; no live-FK re-solve
        try:
            for handle, value in zip(
                    (self.g_tx, self.g_ty, self.g_tz,
                     self.g_roll, self.g_pitch, self.g_yaw),
                    (*T[:3, 3], roll, pitch, yaw)):
                handle.value = float(value)
        finally:
            self._restoring = False
        # Exact, not the round trip through the sliders (they hold the same
        # numbers; viser does not snap a programmatic write to the step grid).
        self.params.wrist_pose = np.asarray(T, float)

    def _adopt_solved_tensions(self):
        """Move the flexor sliders to the tensions the last solve reached.

        Same failure as the wrist, one variable over: the flexor prior is soft by
        design (``_IK_TENSION_COV`` gives it variance 1e-1 so contact can drive
        it), so a grasp ends with the flexor far from what the slider commands --
        1.28 N against a commanded 0.6 N here. Rebuild the prior at the slider
        value and it pulls the fingers back open even without a settling step.

        Only the flexors: the five passives are pinned at 1e-6 and come back
        within 7e-6 N of what was commanded, so the one shared passive slider is
        already telling the truth and moving it would be noise."""
        res = self._iter_view()
        if res is None:
            return
        lo, hi = self.g_flexors[0].min, self.g_flexors[0].max
        solved, clamped = [], False
        for name in res.finger_names:
            q = float(np.asarray(
                res.frames[0][name].marginals.tensions.mean, float)[FLEXOR_IDX])
            clamped = clamped or not (lo <= q <= hi)
            solved.append(min(max(q, lo), hi))
        self._restoring = True
        try:
            for handle, q in zip(self.g_flexors, solved):
                handle.value = q
        finally:
            self._restoring = False
        self.params.flexor_tensions = solved
        if clamped:
            # A clamped value is a prior commanded somewhere the hand is not, so
            # say so rather than let it look like the warm start misbehaving.
            self._set_status(
                f"**warm start:** a solved flexor tension fell outside the "
                f"slider range [{lo}, {hi}] N and was clamped -- the hand will "
                f"move toward the clamped value.")

    def _show_step(self, result, status):
        """Render one stepped state and update both status readouts. Called from
        the auto-run thread as well as the Step button."""
        self.result = result
        self._render_frame(live=True)
        # Deliberately not _report(): during stepping the AL numbers below are
        # the interesting readout, and writing both just overwrites one with the
        # other every frame.
        self._report_step_status(status)

    def _report_step_status(self, status):
        # "stalled" is the solver's own stopping rule (a step that changed
        # nothing), not a dead end: mu still grows, so pressing Auto solve again
        # resumes and often unwedges. Say so, or it reads as a failure.
        verdict = {"running": "stepping",
                   "converged": "**converged**",
                   "stalled": "**stalled** -- last step changed nothing; "
                              "Auto solve again to continue, FK to restart",
                   }[status.state]
        if self.estop.is_tripped() and status.state == "running":
            verdict = "**E-STOP** -- press Rearm to resume from here"
        gaps = [f"object {self.result.worst_gap(0):+.5f} m"] \
            if self.params.object_contact else []
        if self.result.table_contact_names():
            gaps.append(f"table {self.result.worst_table_gap(self.params, 0):+.5f} m")
        # What the rebuild inherited. Worth a line of its own: 0 matched against
        # a non-empty carry is a tag drift (a constraint built without one),
        # which otherwise just looks like the warm start not working.
        rep = self.result.dual_transfer
        carried = ("" if rep is None else
                   f"  \nduals carried: {rep.matched}/{rep.total} constraints")
        pinch = (self._half_space_note() + self._wrist_gauge_note()
                 + self._pinch_note())
        self._set_status(
            f"**IK step {status.steps}** &nbsp; {verdict}  \n"
            f"violation={status.violation:.3e} &nbsp; cost={status.cost:.4g} "
            f"&nbsp; mu={status.mu:.3g}  \n"
            f"worst gap: {' &nbsp; '.join(gaps) or 'n/a'}{carried}"
            + ("  \n" + "  \n".join(pinch) if pinch else ""))

    def _ik_step(self, _=None):
        """One Augmented Lagrangian outer iteration, continuing the last one.

        Refused while the e-stop is engaged. The iteration itself is not
        interruptible once started -- one step is one uninterruptible call into
        C++ -- so the latch's job here is purely to refuse to begin."""
        if not self.caps["ik_stepping"]:
            return
        try:
            gate = self.estop.admit("IK step")
        except Refused:
            return
        try:
            with gate:
                self._set_solving(True)
                try:
                    self._sync_params()
                    self._refresh_object()
                    self.mode = "IK"
                    stepper = self._ensure_stepper()
                    self._show_step(stepper.step(), stepper.status())
                    self._rebuild_iter_slider()
                except Exception as exc:
                    self._error_status(exc)
                    raise
        finally:
            self._set_solving()
            self._report_estop()

    def _ik_auto(self, _=None):
        """Step to convergence on a worker thread, redrawing after each iteration.

        The loop has to leave viser's callback thread free: run inline and the
        E-STOP click would sit behind the whole solve, which is precisely when
        it stops being useful.

        This is the one path the e-stop can interrupt rather than merely refuse.
        ``run()`` polls ``should_stop`` between iterations, so a trip breaks the
        loop at the next boundary -- bounded by one AL iteration (~1.7 s worst
        case measured), because that is a single call into C++ with no interrupt
        hook. Everything the stepper holds (multipliers, mu, the full history)
        survives, so Rearm + Auto solve resumes rather than restarts."""
        if not self.caps["ik_stepping"]:
            return
        try:
            gate = self.estop.admit("IK auto solve")
        except Refused:
            return
        try:
            self._set_solving(True)
            self._sync_params()
            self._refresh_object()
            self.mode = "IK"
            stepper = self._ensure_stepper()
        except Exception as exc:
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise

        def worker():
            try:
                status = stepper.run(max_steps=self.g_ik_max.value,
                                     on_step=self._show_step,
                                     should_stop=self.estop.is_tripped)
                self._report_step_status(status)
                # Only now: rebuilding the slider once per frame mid-animation
                # would tear a GUI handle down and re-add it every iteration.
                self._rebuild_iter_slider()
            except Exception as exc:
                self._error_status(exc)
            finally:
                # Release the gate FIRST, so the refresh below sees an idle
                # latch and can hand the controls back (or offer Rearm).
                gate.release()
                self._set_solving()
                self._report_estop()

        self._auto_thread = threading.Thread(target=worker, daemon=True)
        try:
            self._auto_thread.start()
        except Exception as exc:
            # Nothing else will ever release the gate if the worker never runs,
            # and a gate held forever means the app refuses every solve for the
            # rest of the session. Rare, but the failure mode is permanent.
            gate.release()
            self._error_status(exc)
            self._set_solving()
            raise

    # -- e-stop --

    def _estop(self, _=None):
        """Engage the e-stop. Runs on a viser callback thread and must never
        block: it flips the latch and repaints, and does NOT wait for the
        running solve to notice. A running auto-solve breaks out at its next
        iteration boundary; anything else is simply refused from here on."""
        self.estop.trip("E-STOP pressed")
        self._set_solving()
        self._report_estop()

    def _rearm(self, _=None):
        """Release the e-stop and hand the controls back.

        Refused while a stopped solve is still winding down -- rearming around
        a solve that has not yet returned would re-enable the panel while the
        hand is still moving, which is the one thing the button exists to
        prevent."""
        if not self.estop.rearm():
            self._set_status(
                "**E-STOP still engaged** -- the running solve has not returned "
                "yet (it stops at the end of the current AL iteration). Try "
                "again in a moment.")
            return
        self._set_solving()
        # Put the real readout back: whatever the solve was showing when it was
        # stopped is still on screen and still true.
        if self.mode == "IK" and self.stepper is not None:
            self._report_step_status(self.stepper.status())
        elif self.result is not None:
            self._report()
        else:
            self._set_status("**Rearmed.**")

    def _report_estop(self):
        """Write the e-stop banner over the status line, if engaged.

        Called after every solve path returns so the last thing written is the
        latch's state rather than a step readout that has been overtaken."""
        if not self.estop.is_tripped():
            return
        note = ("" if self.caps["gil_release"] else
                "  \n*this binding does not release the GIL during a solve, so "
                "this click landed only when the iteration ended -- rebuild "
                "with `pip install .` from the crest-sparse root*")
        self._set_status(
            f"# &#9888; E-STOP ENGAGED  \n{self.estop.reason} -- every solve is "
            f"refused until you press **Rearm**. Nothing was lost: the solve is "
            f"paused where it stopped and resumes from there." + note)

    # -- warm start and reset --

    def _seed_state(self):
        """The posture on screen right now, as a solver seed (None if nothing has
        been solved). Follows the *iterate* scrubber, so rewinding to an earlier
        iteration and branching from there works."""
        res = self._iter_view()
        return None if res is None else res.state(0)

    def _toggle_warm_start(self, _=None):
        """Flip the warm-start latch.

        A LATCH, not a one-shot capture: the state is read at the moment the
        stepper is (re)built, so it cannot go stale and the order you press
        things in does not matter. Capturing on the button press instead meant
        that anything re-posing the hand in between -- pressing FK, or just
        dragging a tension slider, which re-solves FK live -- silently threw the
        captured state away and the next Step cold-started."""
        if not self.caps["solver_seed"]:
            return
        self.warm_start = not self.warm_start
        self._refresh_warm_start()

    def _refresh_warm_start(self):
        """Put the latch's state on the button and in its readout."""
        on = self.warm_start
        self.g_warm.label = f"Warm start: {'ON' if on else 'off'}"
        self.g_warm.color = "blue" if on else None
        if not self.caps["solver_seed"]:
            # Name the .so that is actually loaded: the usual cause is not a
            # missing rebuild but the STALE IN-TREE copy at
            # python/crest_sparse/_crest_sparse*.so shadowing the installed one,
            # which happens whenever the app is run from the python/ directory.
            self.g_warm_status.content = (
                "**unavailable** -- this binding has no "
                "`TendonHandSolverConfig.initial_state`  \n"
                f"loaded from `{binding_path()}`  \n"
                "*(run from the crest-sparse root -- `python -m "
                "python.tests.tendon_hand.viz_interactive` -- so the installed "
                "build is used, and rebuild with `pip install .`)*")
        elif on:
            self.g_warm_status.content = (
                "the next **Step** starts from the state on screen"
                + (", carrying the AL multipliers"
                   if self.caps["dual_transfer"] and self.g_carry_duals.value
                   else ""))
        else:
            self.g_warm_status.content = (
                "the next **Step** cold-starts (straight hand, Q = 0)")

    def _reset_defaults(self, _=None):
        """Put every control back to the value it was built with and cold-start.

        A full reset rather than a re-solve: fresh params (so the warm-start
        posture and any derived scene state go too), fresh FK solver, no
        stepper, camera back on the default object.

        Refused while a solve is running, and while the e-stop is engaged: a
        reset cold-starts, which would throw away exactly the state the stop was
        protecting. Rearm first, deliberately."""
        if self.estop.busy is not None or self.estop.is_tripped():
            return
        self._restoring = True
        try:
            for handle, value in self._gui_defaults:
                handle.value = value
        finally:
            self._restoring = False
        # _fresh_params, not a bare HandSolveParams(): the app's own scene
        # defaults (object seating on the table) are not GUI-backed, so a bare
        # one would put the table/object/ellipsoids somewhere the restored
        # sliders do not describe.
        self.params = self._fresh_params()
        self.warm_start = False     # a button, so not in _gui_defaults
        self._refresh_warm_start()
        self._refresh_planar_contact_gate()   # restored object may not support it
        self._sync_params()
        self._rebuild_fk()          # also drops the stepper
        self._refresh_object()
        self._aim_all_cameras()
        self._fk_solve()
        self._set_status("**reset** to defaults  \n" + self.g_status.content)

    # -- phase presets --

    def _preset_widget(self, field):
        """The GUI handle a ``PHASE_PRESETS`` override field writes onto, for
        the plain 1:1 cases (everything except ``contact_fingers``,
        ``sigma_wrist_pos``/``sigma_wrist_rot`` and ``flexor_tension_sigma``,
        which :meth:`_apply_phase_preset` special-cases itself)."""
        return {
            "object_contact": self.g_obj_contact,
            "object_contact_in_plane": self.g_obj_contact_plane,
            "table_contact": self.g_tbl_contact,
            "collision": self.g_collision,
            "self_collision": self.g_self_collision,
            "table": self.g_table,
            "plane_avoidance": self.g_plane_avoid,
            "half_space": self.g_half_space,
            "half_space_margin": self.g_half_margin,
            "pregrasp_center": self.g_pregrasp_center,
            "pregrasp_axis_align": self.g_axis_align,
            "pregrasp_centroid": self.g_pregrasp_centroid,
            "h_clear": self.g_h_clear,
            "contact_drop_normal_row": self.g_drop_normal_row,
        }[field]

    def _apply_phase_preset(self, name):
        """Write ``PHASE_PRESETS[name]``'s overrides directly onto the
        corresponding GUI widgets, so checking the preset box is a single
        visible action: every affected checkbox/slider jumps to the preset's
        value on screen. One solve-ready sync/invalidate happens at the end --
        Auto solve is a separate, manual next step, not triggered here."""
        overrides = PHASE_PRESETS[name].overrides
        self._restoring = True   # batch write; no live-FK/other side effects
        try:
            for field, value in overrides.items():
                if field == "contact_fingers":
                    for handle, v in zip(self.g_contacts, value):
                        handle.value = bool(v)
                elif field == "object_contact":
                    # A preset says WHETHER the object is contacted; it has no
                    # opinion on which metric, so the form the user picked
                    # survives it. Off clears both boxes (otherwise a checked
                    # in-plane form would keep contact alive through a phase that
                    # asked for none); on writes the 3D box only if no form is
                    # selected yet.
                    if not value:
                        self.g_obj_contact.value = False
                        self.g_obj_contact_plane.value = False
                    elif not self.g_obj_contact_plane.value:
                        self.g_obj_contact.value = True
                elif field == "sigma_wrist_pos":
                    self.g_sig_pos.value = math.log10(value)
                elif field == "sigma_wrist_rot":
                    self.g_sig_rot.value = math.log10(value)
                elif field == "flexor_tension_sigma":
                    self.g_flexor_sigma.value = math.log10(value)
                else:
                    self._preset_widget(field).value = value
        finally:
            self._restoring = False
        # The batch write above ran with every per-handle callback suppressed, so
        # The batch ran with every per-handle callback suppressed, and a preset's
        # contact_fingers may have just taken the thumb away -- which is what the
        # in-plane form's plane is keyed off.
        self._refresh_planar_contact_gate()
        self._sync_params()
        self._invalidate_stepper()
        self._refresh_object()
        self._render_frame()

    def _phase_checkboxes(self):
        """Every phase-preset checkbox, name -> handle. Small and built on
        demand rather than cached, so a future phase3 checkbox only needs
        adding here (and to ``_build_gui``/``_input_handles``)."""
        return {"phase0": self.g_phase0, "phase1": self.g_phase1,
                "phase2": self.g_phase2}

    def _on_phase_toggle(self, name, _=None):
        """Checking a phase preset applies it and unchecks every OTHER phase
        checkbox first -- they are mutually exclusive stages of the same
        pipeline, and leaving two checked at once would show a state whose
        settings actually contradict each other (e.g. phase 0's
        pregrasp_centroid=True vs. phase 1's False). Unchecking is a no-op --
        the controls a
        preset wrote stay exactly where it left them, freely editable
        afterward; there is nothing to "undo" back to."""
        if self._restoring:
            return
        checkbox = self._phase_checkboxes()[name]
        if not checkbox.value:
            return
        self._restoring = True
        try:
            for other_name, other_box in self._phase_checkboxes().items():
                if other_name != name:
                    other_box.value = False
        finally:
            self._restoring = False
        self._apply_phase_preset(name)

    def _live_fk(self, _=None):
        """FK is fast and warm-starts, so re-solve live as sliders move.

        Only while the hand is FK-posed: once the stepper is running, the same
        sliders are read live by every step, and re-solving FK here would throw
        that loop away mid-solve.

        The busy/e-stop test is _fk_solve's gate anyway, so this is belt and
        braces -- but it is the reason the latch has to LATCH: this fires on
        every slider drag, and a momentary stop would let the next twitch of a
        tension slider re-pose the hand straight after the button was hit."""
        if self.mode == "FK" and not self._restoring:
            self._fk_solve()

    # -- solve-iteration scrubber (IK) --

    def _current_iterate(self):
        """The scrubbed iterate index, clamped to the CURRENT result.

        The slider is rebuilt after the render that follows a step, so between
        the two it can still hold an index from a longer history -- a cold
        restart drops from ~30 snapshots to 2 and would otherwise index off the
        end."""
        if getattr(self, "iter_slider", None) is None or self.result is None:
            return 0
        return int(np.clip(int(self.iter_slider.value), 0,
                           max(self.result.num_iterates() - 1, 0)))

    def _rebuild_iter_slider(self):
        """Rebuild the convergence scrubber for the steps taken so far.

        Torn down and re-created rather than resized because the iteration count
        is whatever the AL outer loop has run to date -- and an FK pose (or a
        freshly restarted loop) must leave no slider at all."""
        for name in ("iter_slider", "g_iter_status"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.remove()
            setattr(self, name, None)
        n = self.result.num_iterates() if self.result else 0
        if n <= 1:
            # Say why the folder is empty rather than leaving a bare header the
            # user has to guess at.
            with self.iter_folder:
                self.g_iter_status = self.server.gui.add_markdown(
                    "press **Step** or **Auto solve** to record IK iterations")
        else:
            with self.iter_folder:
                self.iter_slider = self.server.gui.add_slider(
                    "iterate", min=0, max=n - 1, step=1, initial_value=n - 1,
                    hint="0 is the initial guess, the last is the most recent "
                         "step; in between, one Augmented Lagrangian outer "
                         "iteration each.")
                self.g_iter_status = self.server.gui.add_markdown("")
            self.iter_slider.on_update(lambda _: self._render_frame())

    def _iter_view(self, live=False):
        """The result to render: the selected snapshot when the scrubber is up,
        otherwise the solve's own final state.

        ``live`` is the animation talking. While a solve is actually stepping,
        the newest state is the thing to draw -- not whatever index the scrubber
        is parked on. The scrubber CANNOT follow along mid-run: a viser slider's
        max is fixed at construction (there is no setter), which is why
        _rebuild_iter_slider tears it down and re-creates it, and why that is
        deliberately not done per frame. So during a run the slider still
        describes the PREVIOUS run. Honouring it here pinned every frame of the
        new solve to that stale index and the hand sat frozen until the run
        ended and the slider was rebuilt.

        The default stays scrubber-first, because _seed_state reads this to warm
        start from a rewound iterate -- that is the rewind-and-branch feature,
        and it must keep seeing the scrubbed state even though it is called from
        inside a running solve.
        """
        if self.result is None or live:
            return self.result
        if getattr(self, "iter_slider", None) is None:
            return self.result
        return self.result.at_iterate(self._current_iterate())

    def _report_iterate(self, live=False):
        """The AL convergence numbers for the iteration on screen, as the stepper
        labelled them when it took the step.

        ``live`` follows _iter_view: mid-run the newest snapshot is being drawn,
        so report THAT one. Reading the slider instead would label the picture
        with the index of a frame it is not showing."""
        # Gated on the SLIDER, not the markdown: with no slider up the markdown
        # is carrying the "nothing recorded" note, which must survive re-renders.
        if getattr(self, "iter_slider", None) is None:
            return
        n = self.result.num_iterates()
        i = max(n - 1, 0) if live else self._current_iterate()
        notes = self.result.iterate_notes
        body = notes[i] if notes is not None else ""
        self.g_iter_status.content = f"iterate {i} / {n - 1}  \n{body}"

    # -- GUI construction --

    def _input_handles(self):
        """Every value-carrying control, in build order. Buttons and markdown are
        deliberately absent -- Reset restores values, not widgets."""
        return ([self.g_object, self.g_ik_max, self.g_ik_settle, self.g_carry_duals,
                 self.g_obj_dx, self.g_obj_dy, self.g_obj_dz,
                 self.g_obj_roll, self.g_obj_pitch, self.g_obj_yaw,
                 self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw,
                 self.g_sig_pos, self.g_sig_rot, self.g_passive]
                + self.g_flexors
                + [self.g_flexor_sigma, self.g_passive_sigma,
                   self.g_phase0, self.g_phase1, self.g_phase2]
                + [self.g_obj_contact, self.g_obj_contact_plane,
                   self.g_tbl_contact, self.g_drop_normal_row,
                   self.g_half_space, self.g_half_sides, self.g_half_margin,
                   self.g_pregrasp_center, self.g_h_clear,
                   self.g_pregrasp_centroid, self.g_axis_align]
                + self.g_contacts
                + [self.g_collision, self.g_self_collision,
                   self.g_coll_radius, self.g_coll_sigma, self.g_cull,
                   self.g_set_beta,
                   self.g_table, self.g_plane_offset, self.g_plane_avoid,
                   self.g_al_mu, self.g_al_rate, self.g_al_iters,
                   self.g_show_true_mesh,
                   self.g_show_contact, self.g_show_collision,
                   self.g_show_discs, self.g_show_world, self.g_show_obj_frame,
                   self.g_show_table_frame, self.g_show_gaps, self.g_show_mount,
                   self.g_show_finger_planes, self.g_show_planar_gap])

    def _build_gui(self):
        gui = self.server.gui
        # Map the displayed dropdown label back to the real spec key (identity
        # except for the "_sdf"-suffixed baked spheres, and the "ycb:"-prefixed
        # ellipsoid sets, which keep their prefix as the label so they group
        # together and read as "not one of the hand-authored primitives").
        labels, self._label_to_key = self._object_dropdown_labels()

        step_hint = (None if self.caps["ik_stepping"]
                     else "requires a rebuilt _crest_sparse with "
                          "TendonHandSolver.reset_al_duals")

        with gui.add_folder("Solver"):
            # Opens on whatever __init__ resolved (see _resolve_default_primitive,
            # which already handled the object being unavailable), so the widget
            # and self.params cannot disagree about which scene is loaded.
            default_label = SDF_DROPDOWN_LABELS.get(self.params.primitive,
                                                    self.params.primitive)
            if default_label not in self._label_to_key:
                default_label = labels[0]
            self.g_object = gui.add_dropdown(
                "object", labels, initial_value=default_label)
            self.g_fk = gui.add_button(
                "FK", icon=self.viser.Icon.PLAYER_PLAY,
                hint="Re-pose the hand from the current wrist / tension sliders "
                     "with the FK solver -- no contact, no collision. Also the "
                     "restart button for the IK loop: it drops any stepped solve "
                     "in progress.")
            self.g_ik_step = gui.add_button(
                "Step", icon=self.viser.Icon.PLAYER_TRACK_NEXT,
                disabled=not self.caps["ik_stepping"],
                hint=step_hint or (
                    "Advance the IK solve by exactly one Augmented Lagrangian "
                    "outer iteration, continuing the previous one. Tensions and "
                    "the wrist pose are re-read every step, so you can nudge "
                    "them mid-solve; changing the object, contacts, collision "
                    "or table starts over."))
            self.g_ik_auto = gui.add_button(
                "Auto solve", icon=self.viser.Icon.PLAYER_PLAY,
                disabled=not self.caps["ik_stepping"],
                hint=step_hint or (
                    "Keep stepping until the solve converges, stalls or hits "
                    "the cap, redrawing after every iteration."))
            # NEVER disabled -- not while solving, not while idle, not on a
            # binding missing every other capability. A stop button that can be
            # greyed out is not a stop button; this one is always available and
            # always the highest-priority thing on the page.
            self.g_ik_stop = gui.add_button(
                "E-STOP", icon=self.viser.Icon.ALERT_TRIANGLE, color="red",
                hint="Software e-stop. Latches: it halts a running auto-solve "
                     "and then REFUSES every solve -- FK, Step, Auto, and the "
                     "live re-solve on the pose/tension sliders -- until you "
                     "press Rearm, so nothing can restart the hand by accident. "
                     "Nothing is lost: the solve pauses where it stopped, "
                     "keeping its multipliers, penalty weight and full step "
                     "history, and Rearm resumes from there rather than "
                     "restarting. A running auto-solve stops at the end of the "
                     "current AL outer iteration (~1.7 s worst case) -- one "
                     "iteration is a single call into the C++ solver with no "
                     "interrupt hook, so that is a floor, not a setting."
                     + ("" if self.caps["gil_release"] else
                        " WARNING: this binding does not release the GIL during "
                        "a solve, so this click cannot even be received until "
                        "the current iteration ends -- rebuild with `pip "
                        "install .` from the crest-sparse root."))
            self.g_rearm = gui.add_button(
                "Rearm", icon=self.viser.Icon.LOCK_OPEN, disabled=True,
                hint="Release the e-stop and hand the controls back. Live only "
                     "while the latch is engaged AND the stopped solve has "
                     "actually returned -- rearming around a solve still "
                     "winding down would re-enable the panel while the hand is "
                     "still moving.")
            self.g_ik_max = gui.add_slider(
                "max steps", 5, 300, 5, 200,
                disabled=not self.caps["ik_stepping"],
                hint="Backstop for Auto solve. A converging grasp takes ~30 "
                     "iterations, so hitting this means it is not converging.")
            self.g_ik_settle = gui.add_slider(
                "settle steps", 0, 5, 1, HandSolveParams().ik_settle_steps,
                disabled=not self.caps["ik_stepping"],
                hint="Leading steps that pin the flexor prior as tightly as the "
                     "passives, to settle the cold start. The solver seeds every "
                     "tension at zero on an already-curled rod, and the flexor -- "
                     "by far the softest variable -- absorbs that inconsistency by "
                     "swinging NEGATIVE, i.e. the fingers hyperextend and then "
                     "spend ~13 steps crawling back to the FK pose. One settling "
                     "step lands step 1 on the FK pose instead, and reaches the "
                     "same grasp. Set 0 to watch the old behaviour. Ignored "
                     "entirely on a warm start -- a seeded posture is already "
                     "consistent, and pinning its tendons back to the commanded "
                     "means is exactly what would undo it.")
            self.g_status = gui.add_markdown("")

        # The scrubber over the steps taken. The slider and its readout are built
        # per step by _rebuild_iter_slider().
        self.iter_folder = gui.add_folder("Solve steps")

        with gui.add_folder("Warm start / reset"):
            self.g_warm = gui.add_button(
                "Warm start: off", icon=self.viser.Icon.PIN,
                disabled=not self.caps["solver_seed"],
                hint=("Latch: while it is ON, every restart of the IK loop "
                      "begins from the state on screen instead of from a "
                      "straight hand with Q = 0. Changing the object, contacts, "
                      "collision, table or an AL knob restarts the loop (the "
                      "carried multipliers describe the old constraint set), so "
                      "this is what lets you change a setting and carry on from "
                      "the posture you had. It also starts an IK solve from an "
                      "FK pose you dialled in, and it follows the iterate "
                      "scrubber, so you can rewind and branch. Only the POSTURE "
                      "carries -- the penalty schedule restarts either way."
                      if self.caps["solver_seed"]
                      else "requires a rebuilt _crest_sparse with "
                           "TendonHandSolverConfig.initial_state"))
            self.g_carry_duals = gui.add_checkbox(
                "carry AL duals", True, disabled=not self.caps["dual_transfer"],
                hint=("Also carry the Augmented Lagrangian multipliers, matched "
                      "to the new constraint set by identity. This is what stops "
                      "the hand letting go of constraints it had already "
                      "satisfied: without it the rebuilt solve restarts the "
                      "penalty schedule at mu = al_mu with every multiplier at "
                      "zero. Untick to see the difference."
                      if self.caps["dual_transfer"]
                      else "requires a rebuilt _crest_sparse with "
                           "TendonHandSolver.set_initial_duals"))
            self.g_reset = gui.add_button(
                "Reset defaults", icon=self.viser.Icon.ROTATE,
                hint="Put every control back to the value it opened with, turn "
                     "the warm start off, drop the stepped solve, and re-pose "
                     "with FK.")
            self.g_warm_status = gui.add_markdown("")

        with gui.add_folder("Object pose"):
            # OFFSETS from whatever the primitive resolves to on its own, not
            # absolute coordinates. Two reasons: every object keeps its own
            # sensible default placement (the grasp locus for the graspable ones,
            # OBJECT_CENTER for the small ones), and switching objects therefore
            # cannot fling the new one somewhere arbitrary just because the
            # previous one had been dragged. All-zero == the derived pose, so the
            # scene opens exactly as it did before these existed.
            self.g_obj_dx = gui.add_slider("dx (m)", -0.15, 0.15, 0.001, 0.0)
            self.g_obj_dy = gui.add_slider("dy (m)", -0.15, 0.15, 0.001, 0.0)
            self.g_obj_dz = gui.add_slider("dz (m)", -0.15, 0.15, 0.001, 0.0)
            self.g_obj_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_obj_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_obj_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, 0.0)
            gui.add_markdown(
                "_Rotation is applied about the object's own centre, on top of "
                "the primitive's base orientation. Moving the object changes the "
                "constraint set, so it restarts the IK loop._")

        with gui.add_folder("Wrist start pose"):
            # Seeded from the shared default (solvers.DEFAULT_WRIST_*) so the
            # pose the GUI opens on IS HandSolveParams' default -- a headless
            # repro of what is on screen needs no numbers copied across.
            x0, y0, z0 = DEFAULT_WRIST_XYZ
            r0, p0, yw0 = DEFAULT_WRIST_RPY
            self.g_tx = gui.add_slider("x (m)", -0.1, 0.1, 0.001, x0)
            self.g_ty = gui.add_slider("y (m)", -0.1, 0.1, 0.001, y0)
            self.g_tz = gui.add_slider("z (m)", -0.1, 0.1, 0.001, z0)
            self.g_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, r0)
            self.g_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, p0)
            self.g_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, yw0)
            self.g_sig_pos = gui.add_slider("log10 sigma_pos", -6, 2, 0.5, -2)
            self.g_sig_rot = gui.add_slider("log10 sigma_rot", -6, 2, 0.5, -2)
            # The sliders above are a demo pose. This is the measured one: put the
            # wrist here and the viser world origin becomes the robot flange, so
            # the hand hangs exactly as it does in the CAD assembly.
            self.g_mount = gui.add_button(
                "Pose at measured robot mount",
                hint="Set the six sliders to mount.MOUNT_WRIST_XYZ/RPY -- the "
                     "wrist pose measured from the Onshape assembly. The world "
                     "origin then IS the flange, and 'mount frames' below draws "
                     "both frames so you can check the hand sits on the arm the "
                     "way it does in CAD.")

        with gui.add_folder("Tensions (N)"):
            self.g_passive = gui.add_slider("passive", 0.0, 3.0, 0.05, 0.5)
            self.g_flexors = [
                gui.add_slider(lbl, 0.0, 3.0, 0.05, GRASP_FLEXOR_TENSION)
                for lbl in FINGER_LABELS]
            self.g_flexor_sigma = gui.add_slider(
                "log10 flexor tension sigma", -3.0, 5.0, 0.1,
                math.log10(HandSolveParams().flexor_tension_sigma),
                hint="How loose the ACTUATED (flexor) tendon's tension prior "
                     "is once contact is expected to move it away from its "
                     "commanded value above. Read live every IK step, so a "
                     "drag takes effect on the next Step/Auto solve with no "
                     "rebuild. Has no effect on FK.")
            self.g_passive_sigma = gui.add_slider(
                "log10 passive tension sigma", -6.0, 1.0, 0.1,
                math.log10(HandSolveParams().passive_tension_sigma),
                hint="How loose the five PASSIVE tendons' tension prior is -- "
                     "normally left tight (their physics is a spring holding "
                     "roughly constant tension), unlike the actuated flexor "
                     "above. Below ~1e-3 (variance ~1e-6) risks an "
                     "IndeterminantLinearSystem against the flexor's much "
                     "looser scale. Read live every IK step, so a drag takes "
                     "effect on the next Step/Auto solve with no rebuild. "
                     "Has no effect on FK.")

        # One-click constraint-set presets, backed by solvers.PHASE_PRESETS so
        # the same data is usable headlessly. Checking a box writes its whole
        # preset onto the Constraints controls below in one go; press Auto
        # solve afterward to run it. Mutually exclusive -- checking one
        # unchecks the other, see _on_phase_toggle.
        with gui.add_folder("Presets"):
            self.g_phase0 = gui.add_checkbox(
                PHASE_PRESETS["phase0"].label, False,
                hint="Apply the phase-0 preset: no object/table contact yet, "
                     "collision avoidance on, pinch-centroid centering + "
                     "short-axis alignment on (the opposition half-space and "
                     "fingertip-midpoint centering stay OFF -- the pinch "
                     "centroid already positions the hand and the other two "
                     "fight it), a loose wrist prior (this is a big "
                     "repositioning move), and a 3-finger pinch "
                     "(index/middle/thumb). Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve. Unchecking is a no-op.")
            self.g_phase1 = gui.add_checkbox(
                PHASE_PRESETS["phase1"].label, False,
                hint="Apply the phase-1 preset: table contact ON (object "
                     "contact stays off), table COLLISION avoidance OFF -- a "
                     "deliberate departure from the paper, since this phase "
                     "drives the fingers onto the plane the avoidance "
                     "half-space would push them off (the table itself stays "
                     "on; object/self collision are untouched) -- the three "
                     "pre-grasp constraints (opposition half-space, "
                     "centering, short-axis alignment) turned back OFF now "
                     "that they've done their job, and a tighter wrist prior "
                     "than phase 0 (held closer to where it ended up, not "
                     "free to roam). Same 3-finger pinch. Writes straight "
                     "onto the Constraints/Wrist controls -- check this, then "
                     "press Auto solve. Unchecking is a no-op.")
            self.g_phase2 = gui.add_checkbox(
                PHASE_PRESETS["phase2"].label, False,
                hint="Apply the phase-2 preset: object contact turned back "
                     "ON alongside table contact (approaching the object "
                     "while still sliding on the table), table collision "
                     "avoidance still OFF as in phase 1 (contact with the "
                     "plane is maintained here, so the half-space would "
                     "fight it; object collision stays on), pre-grasp "
                     "constraints still off, and the wrist prior loosened "
                     "back to phase 0's level -- object approach is another "
                     "significant motion, not phase 1's small settle. Same "
                     "3-finger pinch. Writes straight onto the "
                     "Constraints/Wrist controls -- check this, then press "
                     "Auto solve. Unchecking is a no-op.")

        # Every constraint on/off toggle lives here (Chapter 2, Eq 2.8-2.19),
        # grouped by the paper's structure. Numeric tuning sliders that go with
        # a toggle (collision radius/sigma/cull margin, table height offset)
        # stay behind in Collision/Table below -- only the booleans move.
        with gui.add_folder("Constraints"):
            with gui.add_folder("Collision (Eq 2.8-2.9)"):
                self.g_collision = gui.add_checkbox(
                    "object collision", True,
                    hint="Keep every non-contact sphere out of the OBJECT. "
                         "Independent of the other two collision families "
                         "below -- the three share one set of collision "
                         "spheres but each is its own switch. Sliders in the "
                         "Collision folder below.")
                self.g_self_collision = gui.add_checkbox(
                    "self collision", True,
                    disabled=not self.caps["self_collision"],
                    hint="Keep the FINGERS out of each other: one inequality "
                         "per cross-finger sphere pair, minus the pairs the "
                         "cull margin drops. On by default -- a hand passing "
                         "through itself is never wanted -- and needs neither "
                         "the object nor the table. The most expensive of the "
                         "three families by factor count, so this is the one "
                         "to untick when bisecting a slow solve."
                         if self.caps["self_collision"]
                         else "requires a rebuilt _crest_sparse with "
                              "EnvironmentConfig.self_collision")
                self.g_plane_avoid = gui.add_checkbox(
                    "table collision", True,
                    hint="Keep every non-contact sphere out of the "
                         "half-space. Independent of the other two collision "
                         "families. Needs the table enabled below -- with no "
                         "plane there is no half-space to stay out of.")

            with gui.add_folder("Contact (Eq 2.11-2.15)"):
                self.g_table = gui.add_checkbox(
                    "table enabled", True, disabled=not self.caps["table"],
                    hint="Put the support plane in the factor graph. Affects the "
                         "SOLVER only -- the table square is always drawn, since "
                         "it doubles as the scene's landmark (see 'table frame' "
                         "under Display)."
                    if self.caps["table"]
                    else "requires a newer _crest_sparse build (plane env fields)")
                self.g_obj_contact = gui.add_checkbox(
                    "object contact (3D)", True,
                    hint="Drive the checked fingertips onto the OBJECT "
                         "surface, measuring the full 3D distance to it. Turn "
                         "off to leave the object as pure collision geometry -- "
                         "the way to see what the table constraints do on their "
                         "own. Mutually exclusive with the in-plane form below: "
                         "they are two metrics for the SAME constraint, so "
                         "checking one clears the other.")
                self.g_obj_contact_plane = gui.add_checkbox(
                    "object contact (in-plane)", False,
                    disabled=not self.caps["planar_contact"],
                    hint="Eq 13: the same fingertip-onto-object equality, but "
                         "with the distance measured inside each finger's "
                         "pulling plane (Eq 11: metacarpal base, fingertip, "
                         "pinch centroid) -- the plane a tendon can actually "
                         "pull along, so the solve is not asked for "
                         "out-of-plane torsion the hand cannot produce. Same "
                         "factor count and the same zero set (distance = tip "
                         "radius); only the metric differs. Needs an ellipsoid "
                         "or ycb: object and a digit set INCLUDING THE THUMB, "
                         "and greys out when the scene cannot support it. "
                         "Watch it with 'in-plane distance' under Display.")
                self.g_tbl_contact = gui.add_checkbox(
                    "table contact", False,
                    hint="Drive the checked fingertips onto the SUPPORT "
                         "PLANE (one equality per finger on the distance "
                         "from its contact sphere to the plane). Needs "
                         "*table enabled*; combine with object contact to "
                         "solve for both at once.")
                self.g_drop_normal_row = gui.add_checkbox(
                    "drop contact normal row", False,
                    disabled=not self.caps["drop_normal_row"],
                    hint="Eq 2.12-2.15: use the 4-row [c_R, c_O, c_T1, "
                         "c_T2] SDF witness contact form (c_N dropped) "
                         "instead of the default 5-row form. Only affects "
                         "non-ellipsoid (SDF) object contact.")

            with gui.add_folder("Pre-grasp (Eq 2.16-2.19)"):
                self.g_half_space = gui.add_checkbox(
                    "opposition half-space", False,
                    disabled=not self.caps["opposition"],
                    hint="Eq 2.16-2.17: keep each checked finger on its "
                         "designated side of the splitting plane, thumb "
                         "opposite the rest. Independent of table contact and "
                         "of the table itself -- it constrains a fingertip "
                         "against a line, not against the plane.")
                self.g_half_sides = gui.add_dropdown(
                    "opposition sides", list(OPPOSITION_SIDES),
                    initial_value="auto (match the hand)",
                    hint="Which half of the split the THUMB is sent to. The "
                         "object fixes the split LINE; its sign -- the side "
                         "assignment -- is an arbitrary object-frame "
                         "convention, and the wrong one asks the thumb and the "
                         "fingers to TRADE sides, i.e. to roll the hand ~180 "
                         "degrees about the object. That is infeasible from any "
                         "normal start pose and the solve stalls immediately, "
                         "at any standoff and any wrist sigma. *auto* orients "
                         "it by where the digits already are (the nearer "
                         "opposition); *flipped* deliberately asks for the "
                         "other one; *as derived* is the old object-only "
                         "behaviour, kept for comparison.")
                self.g_half_margin = gui.add_slider(
                    "half-space standoff (m)", 0.0, 0.05, 0.001, 0.0,
                    disabled=not self.caps["half_space_margin"],
                    hint="Minimum distance each contact finger must keep from "
                         "the splitting plane, along its own side's direction: "
                         "HalfSpaceGapFactor's d_min, making the constraint "
                         "-(c - p_split).m_hat + standoff <= 0 rather than "
                         "<= 0 alone. At 0 a fingertip sitting exactly ON the "
                         "split is already legal, so opposition alone does not "
                         "stop the thumb and the fingers closing onto each "
                         "other; a positive standoff holds them 2x this far "
                         "apart, which is what makes it a pre-grasp opening. "
                         "Drawn as the two fainter planes either side of the "
                         "split."
                         if self.caps["half_space_margin"]
                         else "requires a rebuilt _crest_sparse with "
                              "EnvironmentConfig.half_space_margin")
                self.g_pregrasp_center = gui.add_checkbox(
                    "pre-grasp centering", False,
                    disabled=not self.caps["pregrasp_center"],
                    hint="Eq 2.18-2.19: center the midpoint of the thumb + "
                         "opposing fingers' contact points over the object, "
                         "raised by the clearance slider along the table "
                         "normal. Needs the thumb AND at least one other "
                         "finger checked below.")
                self.g_h_clear = gui.add_slider(
                    "clearance (m)", 0.0, 0.08, 0.002, 0.07,
                    hint="Pre-grasp centering's height above the object "
                         "centroid, along the table normal.")
                self.g_pregrasp_centroid = gui.add_checkbox(
                    "pinch-centroid centering", False,
                    disabled=not self.caps["pregrasp_centroid"],
                    hint="Drive the point where the CHECKED digits are "
                         "measured to meet -- a constant in the hand frame, "
                         "looked up from config.HAND_PINCH_POSES -- onto the "
                         "object centroid, raised by the clearance slider "
                         "above. Unlike pre-grasp centering (which averages "
                         "where the fingertips actually are, so it only says "
                         "something once they are nearly closed) this "
                         "constrains the WRIST alone: it positions the hand "
                         "so that closing those digits would close them ON "
                         "the object, whatever the fingers are doing now. "
                         "Only combinations INCLUDING THE THUMB were "
                         "measured; any other selection leaves it inert and "
                         "says so in the status line."
                         if self.caps["pregrasp_centroid"]
                         else "requires a rebuilt _crest_sparse with "
                              "EnvironmentConfig.pregrasp_centroid_point")
                self.g_axis_align = gui.add_checkbox(
                    "short-axis alignment", False,
                    disabled=not self.caps["pregrasp_axis_align"],
                    hint="Align the thumb-vs-opposing-fingers connecting "
                         "vector with the perpendicular to the opposition "
                         "split plane (the same axis opposition half-space "
                         "uses), direction-agnostic -- it doesn't matter "
                         "which way it points, just that it's colinear. "
                         "Needs the thumb AND at least one other finger "
                         "checked below. Independent of opposition "
                         "half-space and pre-grasp centering -- computes its "
                         "own copy of the axis either way.")

            with gui.add_folder("fingers"):
                # Default to a 3-finger pinch (thumb, index, middle) rather
                # than the whole-hand grasp; ring/pinky keep collision
                # avoidance but are not driven onto a surface.
                _pinch_default = {"index", "middle", "thumb"}
                self.g_contacts = [
                    gui.add_checkbox(
                        lbl, lbl in _pinch_default,
                        hint="Which fingers every constraint above applies "
                             "to (IK only; FK never uses contact). "
                             "Unchecked fingers keep collision avoidance, so "
                             "they stay out of the object and off the table "
                             "without being driven onto either, opposed "
                             "against, or centered on.")
                    for lbl in FINGER_LABELS]

        with gui.add_folder("Collision"):
            self.g_coll_radius = gui.add_slider("sphere radius (m)", 0.001, 0.01, 0.0005, 0.003)
            self.g_coll_sigma = gui.add_slider("log10 sigma", -6, 0, 0.5, -4)
            self.g_cull = gui.add_slider("cull margin (m, 0 off)", 0.0, 0.1, 0.005, 0.0)
            self.g_set_beta = gui.add_slider(
                "ellipsoid-set beta", 100.0, 4000.0, 100.0, ELLIPSOID_SET_BETA,
                disabled=not self.caps.get("ellipsoid_set", False),
                hint="LogSumExp sharpness for a ycb: object (Eq 1.11). The smooth "
                     "min understates by up to ln(K)/beta, so the constraint "
                     "surface sits that far OUTSIDE the object -- 1.4 mm at K=4, "
                     "beta=1000. Raising it tightens that standoff but sharpens "
                     "the gradient at the seams between members, which is the "
                     "smoothness the set formulation exists to buy. Inert for "
                     "every non-ycb object.")

        self._build_ycb_folder(gui)

        with gui.add_folder("Table"):
            # Offset from the scene's own seating, which this app sets to rest
            # the object ON the plane (table_burial = 0, see __init__). Zero
            # default, so every object -- whatever its size, shape or rotation --
            # opens sitting on the table rather than sunk through it. Drag
            # negative to bury it (0.5 * extent reaches the half-buried §1.8
            # geometry HandSolveParams still defaults to headlessly).
            self.g_plane_offset = gui.add_slider("height offset (m)", -0.1, 0.1, 0.002, 0.0)
            # Filled by _refresh_table_readout on every re-place of the slab.
            self.g_table_status = gui.add_markdown("")

        with gui.add_folder("Augmented Lagrangian"):
            self.g_al_mu = gui.add_slider("mu", 0.1, 10.0, 0.1, 1.0)
            self.g_al_rate = gui.add_slider("rate", 1.1, 5.0, 0.1, 2.0)
            self.g_al_iters = gui.add_slider("max iters", 5, 100, 5, 40)

        with gui.add_folder("Display"):
            self.g_show_true_mesh = gui.add_checkbox(
                "true object mesh", True,
                hint="Overlay the object's real geometry behind the analytic "
                     "surface: the scanned mesh inside a ycb: object's ellipsoid "
                     "shells, or the dodecahedron inside the megaminx's "
                     "circumsphere. The analytic surface is what the solver sees, "
                     "so showing both is how the approximation gets judged -- "
                     "where the fingers stop is set by the surface, and the mesh "
                     "says how much object is actually there.")
            self.g_show_contact = gui.add_checkbox("contact spheres", True)
            self.g_show_collision = gui.add_checkbox("collision spheres", True)
            self.g_show_discs = gui.add_checkbox("routing discs", False)
            # The three frame toggles all default ON: which frame anything is
            # expressed in is the first question asked of this scene, and a
            # triad you have to go and switch on answers it too late.
            self.g_show_world = gui.add_checkbox(
                "world frame", True,
                hint="Triad at the world origin. Every position readout in this "
                     "app is in this frame, and it is the frame the wrist pose "
                     "sliders command -- so with the hand at the measured mount "
                     "it doubles as the robot flange frame.")
            self.g_show_obj_frame = gui.add_checkbox(
                "object frame", True,
                hint="Triad at the object's center, oriented by its rotation -- "
                     "the pose the contact and collision factors are written "
                     "against, and what the Object-pose sliders drive. The only "
                     "way to see the orientation of a symmetric primitive, and "
                     "for a ycb: set it is the frame its members sit in.")
            self.g_show_table_frame = gui.add_checkbox(
                "table frame", True,
                hint=f"Triad on the table square's -X/-Y corner, on the top "
                     f"face -- a pure translation of the world frame, so a "
                     f"coordinate read off it is a world coordinate minus the "
                     f"corner. The landmark to measure a real bench against. "
                     f"The square is {TABLE_SPAN:g} x {TABLE_SPAN:g} m "
                     f"whatever object is on it, but the plane is seated UNDER "
                     f"the object, so the frame moves when you switch objects "
                     f"-- the Table folder reports where it is.")
            self.g_show_mount = gui.add_checkbox(
                "mount frames", True,
                hint="Draw the wrist frame and, offset from it by the measured "
                     "mount transform, the robot flange frame the hand bolts to. "
                     "Use with 'Pose at measured robot mount' to check the "
                     "measurement against the CAD assembly by eye.")
            self.g_show_gaps = gui.add_checkbox(
                "contact distance", True,
                hint="Fingertip-to-surface gap/margin overlays in mm: object "
                     "and table contact (green under 15 mm, red over), "
                     "opposition half-space (green = correct side, red = "
                     "violating), and pre-grasp centering (green under 15 mm "
                     "to target).")
            self.g_show_finger_planes = gui.add_checkbox(
                "finger pinch planes", False,
                hint="Per finger, the plane through its metacarpal base, its "
                     "fingertip and the pinch centroid the checked digits "
                     "close on (config.HAND_PINCH_POSES, carried through the "
                     "solved wrist). One colour per finger; the outlined "
                     "triangle is the three defining points. Off by default -- "
                     "five translucent sheets sit right where the grasp is. "
                     "Nothing enforces these planes; they describe the posture "
                     "on screen. Needs a measured pinch pose, so only digit "
                     "sets INCLUDING THE THUMB draw anything.")
            self.g_show_planar_gap = gui.add_checkbox(
                # On by default, but only where the binding can actually
                # measure it: a hard True would tick a DISABLED box on an
                # older .so, and the readout would then permanently report
                # "in-plane distance: UNAVAILABLE" with no way to turn it off.
                "in-plane distance", self.caps["planar_gap"],
                disabled=not self.caps["planar_gap"],
                hint="Eq 11/13: the distance from each fingertip to the object "
                     "measured INSIDE that finger's pulling plane, which is what "
                     "a tendon can actually pull along. Draws the cross-section "
                     "the plane cuts out of the object (exact) plus a line to the "
                     "nearest point on it; the mm label is the C++ factor's own "
                     "first-order value, so a visible mismatch between line and "
                     "label IS the approximation error. '(3D)' means the plane "
                     "missed the object entirely and the number has fallen back "
                     "to the ordinary 3D distance. Spheres, ellipsoids and ycb: "
                     "sets only -- cube/cylinder/capsule have no ellipsoid "
                     "cross-section. On by default wherever the binding "
                     "supports it. Measurement only: no solve uses this yet.")

        # Every value-carrying control, captured as built: this IS the definition
        # of "defaults" that Reset restores, so the two cannot drift.
        self._gui_defaults = [(h, h.value) for h in self._input_handles()]
        self._refresh_warm_start()

        # -- callbacks --
        self.g_fk.on_click(self._fk_solve)
        self.g_ik_step.on_click(self._ik_step)
        self.g_ik_auto.on_click(self._ik_auto)
        self.g_ik_stop.on_click(self._estop)
        self.g_rearm.on_click(self._rearm)
        self.g_warm.on_click(self._toggle_warm_start)
        self.g_reset.on_click(self._reset_defaults)
        self.g_phase0.on_update(lambda _: self._on_phase_toggle("phase0"))
        self.g_phase1.on_update(lambda _: self._on_phase_toggle("phase1"))
        self.g_phase2.on_update(lambda _: self._on_phase_toggle("phase2"))

        self.g_object.on_update(self._on_object_selected)

        # Live FK re-solve on the pose / tension sliders (fast, warm-started).
        for h in ([self.g_tx, self.g_ty, self.g_tz, self.g_roll, self.g_pitch,
                   self.g_yaw, self.g_passive] + self.g_flexors):
            h.on_update(self._live_fk)

        # Object pose: re-places the object and restarts the IK loop (see
        # _object_pose_changed for why it cannot just re-render).
        for h in (self.g_obj_dx, self.g_obj_dy, self.g_obj_dz,
                  self.g_obj_roll, self.g_obj_pitch, self.g_obj_yaw):
            h.on_update(self._object_pose_changed)

        self.g_mount.on_click(self._pose_at_mount)

        # The scan-mesh overlay and the world/object triads are static scene
        # geometry, not per-frame, so they go through _refresh_object rather
        # than the _render_frame path below. (The mount frames are the exception
        # among the frame toggles: they hang off the SOLVED wrist, which moves
        # every iterate, so they render with the hand.)
        for h in (self.g_show_true_mesh, self.g_show_world, self.g_show_obj_frame,
                  self.g_show_table_frame):
            @h.on_update
            def _(_):
                if self._restoring:
                    return
                self._refresh_object()

        # Display toggles re-render the current frame without re-solving.
        for h in (self.g_show_contact, self.g_show_collision, self.g_show_discs,
                  self.g_show_gaps, self.g_show_mount, self.g_show_finger_planes,
                  self.g_show_planar_gap):
            h.on_update(lambda _: (self._sync_params(), self._render_frame()))
        # The contact checkboxes ride along only to keep self.params in sync;
        # like every other solver knob they take effect on the next solve, and
        # the gap lines keep describing the solve that is actually on screen
        # until then. They do change the stepper's constraint set, though -- which
        # surface a finger is driven onto IS that set, so a carried dual is
        # meaningless and the loop has to restart.
        # The two object-contact FORMS additionally settle each other first (they
        # are mutually exclusive), and the per-finger boxes re-check whether the
        # in-plane form is still buildable -- its plane is keyed off the pinch
        # centroid of exactly these digits, so unchecking the thumb can take it
        # away.
        for h in (self.g_obj_contact, self.g_obj_contact_plane):
            h.on_update(lambda _, src=h: self._enforce_object_contact(src))
        for h in self.g_contacts:
            h.on_update(lambda _: self._refresh_planar_contact_gate())
        for h in (self.g_contacts
                  + [self.g_obj_contact, self.g_obj_contact_plane,
                     self.g_tbl_contact]):
            h.on_update(lambda _: (self._sync_params(),
                                   self._invalidate_stepper(),
                                   self._render_frame()))
        # Table toggle / height updates the static slab immediately; opposition
        # half-space rides along since it draws its own static split-plane slab
        # (set_half_space_plane) the same way -- as does its standoff slider,
        # which draws the two boundary planes either side of that split.
        for h in (self.g_table, self.g_plane_offset, self.g_half_space,
                  self.g_half_margin, self.g_half_sides):
            h.on_update(lambda _: (self._sync_params(), self._refresh_object(),
                                   self._invalidate_stepper()))
        # Collision knobs are part of the constraint set too. The sphere-drawing
        # flag follows the toggles, so these re-render as well as invalidate.
        for h in (self.g_collision, self.g_self_collision, self.g_coll_radius,
                  self.g_coll_sigma, self.g_cull, self.g_plane_avoid):
            h.on_update(lambda _: (self._sync_params(),
                                   self._invalidate_stepper(),
                                   self._render_frame()))
        # drop-normal-row / pre-grasp centering / clearance: no static geometry
        # of their own (the centering line only appears once a solve has run,
        # via _render_frame), so just invalidate the stepper -- self.params is
        # refreshed from every handle at the start of the next solve regardless
        # of which widget triggered it.
        for h in (self.g_drop_normal_row, self.g_pregrasp_center, self.g_h_clear,
                  self.g_axis_align, self.g_pregrasp_centroid):
            h.on_update(lambda _: self._invalidate_stepper())
        # AL knobs are baked into the stepper's config at construction, unlike
        # the tensions it re-reads every step, so they need a rebuild too.
        # "settle steps" is read live, but it counts off steps ALREADY taken, so
        # changing it part-way through a run would do nothing without a restart.
        for h in (self.g_al_mu, self.g_al_rate, self.g_sig_pos, self.g_sig_rot,
                  self.g_ik_settle):
            h.on_update(lambda _: self._invalidate_stepper())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Headless self-check of the solver classes (no viser).")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.smoke:
        sys.exit(_smoke())

    import viser
    server = viser.ViserServer(port=args.port)
    app = HandVizApp(server)
    # Which binding got loaded, and what it can do. Printed unconditionally: a
    # capability-gated control that is silently disabled is indistinguishable
    # from one that does not work, and the usual cause is the in-tree .so
    # shadowing the installed build (see binding_path()).
    print(f"crest_sparse: {binding_path()}")
    missing = [k for k, v in app.caps.items() if not v]
    if missing:
        print(f"  capabilities MISSING from this build: {', '.join(missing)}")
    # The landmark's dimensions, printed so they can be copied into a real-robot
    # setup without opening the browser. The corner frame's position is scene
    # state (it follows the object-seated plane), so that one lives in the GUI.
    print(f"table square: {TABLE_SPAN:.3f} x {TABLE_SPAN:.3f} m, "
          f"{TABLE_THICKNESS * 1e3:.0f} mm thick -- top face is the constraint "
          f"plane, frame on its -X/-Y corner")
    print(f"viser hand visualizer running -- open http://localhost:{args.port}")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
