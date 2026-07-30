"""Interactive viser demo for the Section 1.8 phased controller.

Where ``viz_interactive.py`` is a general FK / IK / Planner workbench with the
controller bolted on, this app is built around the controller alone: it steps one
control tick at a time (or free-runs), shows which constraint set is active, and
draws the geometry that constraint set is talking about.

The app has two states:

  * **FK pose** (no controller yet) -- where it opens, and where *Reset scene*
    returns it. The hand starts at the collision-free ``free_space_start_pose``
    that phase 0 servos from, and the base-pose / tension sliders re-solve FK live
    so you can put the hand wherever you want. *manually set Theta_curr* commits
    that posture as the measured state of Eq 1.93-1.95: the base pose T_base, the
    tensions Q, and the tendon lengths L taken from that FK solve.
  * **Control** -- entered by pressing a phase button, which builds the controller
    from the committed Theta_curr and starts ticking. The phases are CONSTRAINT
    SETS over the same single-state graph, not time windows: switching phase keeps
    the converged robot state, so the intended flow is phase 0 to position the
    hand, phase 1 until the fingertips settle on the table in their opposed
    halves, phase 2 to slide onto the ellipsoid proxy, phase 3 to servo onto the
    exact geometry.

Overlays specific to this app: the support plane (as in the other demos), plus
the whole *constraint-goal* layer -- the pre-grasp target frame and the axes
being aligned to it, the support-plane distances, the opposition split, the
object proxy, the witness points -- each behind its own checkbox. Those live in
``_plotting/viser_overlays.py``; the geometry they draw comes from
``solvers.goal_geometry`` and is the same geometry the constraints are written
in. They are drawn in the FK pose state too, so you can see where phase 0 will
take the hand before committing Theta_curr.

Run from the crest-sparse/ root so the import resolves to the installed
extension rather than the in-tree .so:

    python -m python.tests.tendon_hand.viz_controller
    python -m python.tests.tendon_hand.viz_controller --smoke
"""

import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from .scene import get_primitive_specs
from .solvers import (
    HandControllerSolver, HandFKSolver, HandSolveParams, auto_table_origin,
    capabilities, free_space_start_pose, goal_geometry, opposition_axis,
    pregrasp_local_geometry, resolve_scene)
from .viz_interactive import FINGER_LABELS, SDF_DROPDOWN_LABELS, _euler_to_R
from .._plotting.viser_overlays import OVERLAYS


PHASE_LABELS = {0: "0 - pre-grasp",
                1: "1 - support contact",
                2: "2 - object approach",
                3: "3 - object servo"}

# Opening base-pose orientation, as ZYX euler angles about the world axes (the
# same convention the GUI's roll / pitch / yaw sliders use).
#
# The hand's mount puts the palm on the base frame's -x and the thumb-ward axis
# on +z (see ``solvers.pregrasp_local_geometry``'s docstring), so an identity
# base pose opens with the palm facing sideways -- an awkward place to read a
# scene from, and ~90 deg from the palm-down posture phase 0 derives as its
# target. The pitch turns the palm down onto the support surface; the roll then
# turns the thumb onto the +m_hat side of the opposition split, which the pitch
# alone leaves 180 deg away (palm-down is a one-axis condition, so it fixes the
# roll about the palm axis not at all).
#
# Only the START pose: phase 0 still derives ``T_base,pre`` from forward
# kinematics and servos to it, and this just starts that servo somewhere sane.
START_ROLL_RAD = np.pi
START_PITCH_RAD = -1.54

# Shown while no controller exists: the app is posing Theta_curr by FK, not
# enforcing any phase's constraint set.
NO_PHASE_LABEL = "-- none (FK pose)"

# Units for phase_violations()'s families. Phase 0's are not metres, so a single
# hardcoded suffix would mislabel them.
VIOLATION_UNITS = {"pregrasp_rot": "rad", "pregrasp_tension": "N"}


def _log10_default(field):
    """A step-prior slider's default, as log10 of the tuned
    :class:`HandSolveParams` value, snapped to the sliders' 0.5 grid.

    Taken from the params rather than written out, so a retuned sigma reaches
    this app: ``_sync_params`` pushes every slider into params on every tick, so
    a stale hardcoded default does not merely start in the wrong place -- it
    overrides the tuned value permanently.
    """
    value = float(getattr(HandSolveParams(), field))
    return round(np.log10(value) * 2.0) / 2.0


def pregrasp_flexor(params):
    """The flexor tension of the Eq 1.92 pre-grasp posture Q_pre.

    The free-space start pose is computed at this posture, so the visualizer's
    flexor sliders default to it -- at any other tension the lift that was
    verified collision-free is not the pose being shown.
    """
    return float(params.pregrasp_flexor_absolute
                 if params.pregrasp_flexor_absolute is not None
                 else params.passive_tension + params.pregrasp_flexor_offset)


def start_orientation():
    """The opening base-pose rotation: palm down at the support surface, thumb
    rolled onto its opposition half. Shared by the GUI's free-space-start button
    and the smoke test so the headless check starts where the app does."""
    return _euler_to_R(START_ROLL_RAD, START_PITCH_RAD, 0.0)


def default_primitive(caps):
    """The object phase 0 is validated on (``ctrl_5f_phases.py``), falling back to
    the baked sphere on a binding without the analytic ellipsoid surface."""
    return "mid_sphere_ellipsoid" if caps["ellipsoid"] else "big_sphere"


def _absolute_table_origin(params, spec, object_center, height):
    """The support-plane origin at an ABSOLUTE ``height`` along the plane normal
    (with a +Z normal, simply the plane's world z).

    Only the along-normal component of a plane's origin means anything, so the
    in-plane part is taken from the scene's own tangent origin to keep the drawn
    slab centred near the object. Built on :func:`auto_table_origin`, NOT
    ``resolve_table_origin``: the latter returns ``params.plane_origin`` when it
    is set, so a control that publishes its result there would be reading back
    its own output and compounding the height on every call.
    """
    n_hat = np.asarray(params.plane_normal, float)
    n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
    auto = np.asarray(auto_table_origin(params, spec, object_center), float)
    return (auto - (auto @ n_hat) * n_hat) + float(height) * n_hat


def _finite(v):
    """True if ``v`` is absent or wholly finite. Overlay geometry is allowed to be
    missing (a phase may have no such quantity) but never NaN/inf -- viser draws
    those silently as nothing, which reads as 'the constraint is satisfied'."""
    if v is None or isinstance(v, (bool, str, int)):
        return True
    if isinstance(v, dict):
        return all(_finite(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return all(_finite(x) for x in v)
    return bool(np.all(np.isfinite(np.asarray(v, float))))


def _check_geometry(geom, phase):
    """Every overlay quantity present and finite, and the witness reported
    honestly: phases 0-2 instantiate no witness variable, so a solved point there
    would mean the accessor is reading a stale value."""
    bad = [k for k, v in geom.items() if not _finite(v)]
    if bad:
        print(f"  [geom {phase}] NON-FINITE: {', '.join(sorted(bad))}")
        return False
    pts = geom["witness_points"]
    if pts is not None and phase != 3 and any(p is not None for p in pts):
        print(f"  [geom {phase}] BAD: witness points exist outside phase 3")
        return False
    if pts is not None and phase == 3 and all(p is None for p in pts):
        print(f"  [geom {phase}] BAD: phase 3 solved no witness points")
        return False
    return True


def _check_table_absolute():
    """The table-height control must be ABSOLUTE: repeated reads at a fixed
    setting must return the same plane.

    The regression this pins: reading ``resolve_table_origin`` and adding the
    offset, then writing the sum back into ``params.plane_origin``, makes the
    control a per-sync DELTA -- the plane creeps every tick and returning the
    slider to its start does not bring it back.
    """
    p = HandSolveParams()
    p.primitive = default_primitive(capabilities())
    p.table = True
    spec, center, _rot, _pose = resolve_scene(p)
    n_hat = np.asarray(p.plane_normal, float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    height = float(np.asarray(auto_table_origin(p, spec, center), float) @ n_hat)
    first = _absolute_table_origin(p, spec, center, height)
    for _ in range(10):
        # Exactly what the GUI does each sync: resolve, then publish to params.
        p.plane_origin = _absolute_table_origin(p, spec, center, height)
    drift = float(np.linalg.norm(np.asarray(p.plane_origin, float) - first))
    moved = _absolute_table_origin(p, spec, center, height + 0.05)
    commanded = float(np.asarray(moved, float) @ n_hat) - (height + 0.05)

    ok = drift < 1e-12 and abs(commanded) < 1e-12
    print(f"  [table] drift over 10 syncs {drift:.2e} m (expect 0) &  "
          f"height error {commanded:+.2e} m (expect 0) "
          f"[{'ok' if ok else 'BAD'}]")
    return ok


def _check_start_pose():
    """The opening pose satisfies BOTH conditions the pre-grasp orientation is
    built from: the measured palm-facing axis lands on -n_hat (palm down at the
    support surface) and the thumb-ward axis on +m_hat (thumb on its opposition
    half).

    Checking only the palm would pass with the hand rolled 180 deg about it,
    which is what a pitch alone gives -- palm-down is a one-axis condition and
    says nothing about the roll around that axis.
    """
    p = HandSolveParams()
    p.primitive = default_primitive(capabilities())
    spec, center, rotation, _pose = resolve_scene(p)
    n_hat = np.asarray(p.plane_normal, float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    m_hat, _ = opposition_axis(p, spec, rotation)
    m_hat = np.asarray(m_hat, float) / np.linalg.norm(m_hat)

    geom = pregrasp_local_geometry(p)
    R = start_orientation()
    palm = float((R @ geom["a_hat"]) @ -n_hat)
    thumb = float((R @ geom["s_hat"]) @ m_hat)
    ok = palm > 0.9 and thumb > 0.9
    print(f"  [pose ] a_hat . -n_hat = {palm:+.3f}, s_hat . m_hat = {thumb:+.3f} "
          f"(expect both > 0.9) [{'ok' if ok else 'BAD'}]")

    # ...and sits OVER the object. Orientation alone does not put it there: the
    # contact centroid is ~0.14 m out along the hand's own axes, so a rotation
    # swings it far in-plane and a start that is aimed right can still be a
    # quarter of a metre away from what it is aiming at.
    p.collision = p.table = True
    T, info = free_space_start_pose(p, orientation=R, center_on_object=True)
    centroid = T[:3, :3] @ geom["p_bar"] + T[:3, 3]
    lateral = float(np.linalg.norm(
        (centroid - center) - ((centroid - center) @ n_hat) * n_hat))
    over = lateral < 5e-3
    print(f"  [start] centroid lateral offset {lateral * 1000.0:.1f} mm "
          f"(expect < 5) &  lift {info['lift']:.3f} m, table "
          f"{info['table_clearance']:+.3f} object {info['object_clearance']:+.3f} "
          f"[{'ok' if over else 'BAD'}]")
    return ok and over


def _check_base_moves(ticks=4, min_travel=1e-3):
    """Phase 1 must actually MOVE the hand base at the sliders' default step
    prior -- the app's headline failure mode when it does not.

    The regression this pins: ``_sync_params`` writes every step-prior slider
    into params on every tick, so a slider default below the sigma the
    controller was tuned at silently re-imposes it. Under a base prior stiffer
    than the AL penalty ceiling (mu ~ 8e3, i.e. sigma_pos below ~1.1e-2) the
    support equality cannot push the base at all: the fingers flex down onto the
    table while the wrist sits exactly where phase 0 left it. Measured at the old
    hardcoded ``-3`` default: 0.14 mm of base travel over 8 ticks, versus 10 mm
    at the params default -- visually frozen, and no error anywhere.
    """
    p = HandSolveParams()
    p.primitive = default_primitive(capabilities())
    p.collision = True
    p.table = True
    p.flexor_tensions = [pregrasp_flexor(p)] * len(FINGER_LABELS)
    p.wrist_pose, _ = free_space_start_pose(p, orientation=start_orientation(),
                                            center_on_object=True)
    # Exactly what the GUI hands the controller, via the slider defaults.
    p.sigma_wrist_pos_step = 10.0 ** _log10_default("sigma_wrist_pos_step")
    p.sigma_q_step = 10.0 ** _log10_default("sigma_q_step")
    p.sigma_l_step = 10.0 ** _log10_default("sigma_l_step")
    p.phase = 1

    solver = HandControllerSolver(p)
    start = np.asarray(p.wrist_pose, float)[:3, 3].copy()
    for _ in range(ticks):
        solver.step()
    travel = float(np.linalg.norm(
        np.asarray(solver.params.wrist_pose, float)[:3, 3] - start))
    ok = travel > min_travel
    print(f"  [base ] phase-1 base travel {travel * 1000.0:.2f} mm over {ticks} "
          f"ticks at sigma_pos={p.sigma_wrist_pos_step:.3g} "
          f"(expect > {min_travel * 1000.0:.0f}) [{'ok' if ok else 'BAD'}]")
    return ok


def _smoke():
    """Headless self-check: build the controller and run one tick per phase, then
    one tick from a manually posed Theta_curr."""
    if not capabilities()["controller"]:
        print("Smoke test: SKIP -- the installed _crest_sparse has no "
              "TendonHandController (rebuild the extension).")
        return 0

    p = HandSolveParams()
    p.primitive = default_primitive(capabilities())
    p.collision = True
    p.table = True
    p.flexor_tensions = [pregrasp_flexor(p)] * len(FINGER_LABELS)
    # Start clear of the table: phase 0 servos from wherever it starts and cannot
    # dig out of an initial penetration (the inequalities dominate the merit and
    # the inner LM rejects every step).
    p.wrist_pose, _ = free_space_start_pose(p, orientation=start_orientation(),
                                            center_on_object=True)
    solver = HandControllerSolver(p)
    ok = True
    for phase in ([0, 1, 2, 3] if capabilities()["pregrasp"] else [1, 2, 3]):
        solver.set_phase(phase)
        t0 = time.time()
        res = solver.step()
        dt_ms = (time.time() - t0) * 1000.0
        viol = "  ".join(f"{n}={v:.2e}" for n, v in solver.phase_violations())
        n = len(res.frames)
        if n != 1:
            ok = False
        # Every overlay reads this dict, so building it for each phase is the
        # check that the whole debug layer stays drawable as the constraint set
        # changes -- a phase with no witness must yield None, not blow up.
        geom = solver.goal_geometry(res)
        if not _check_geometry(geom, phase):
            ok = False
        print(f"  [phase {phase}] frames={n} (expect 1) "
              f"[{'ok' if n == 1 else 'BAD'}] | {dt_ms:.0f} ms "
              f"iters={res.meta.iterations} | {viol or '(no constraints)'}")

    # The manual-Theta_curr path the GUI's "set Theta_curr" button drives: pose by
    # FK, hand the resulting tendon lengths over as L_curr, tick under the length
    # anchor (the only anchor that consumes them).
    pose = HandFKSolver(p).solve()
    q = replace(p, step_anchor="length", phase=1)
    seeded = HandControllerSolver(q, initial_lengths=pose.tendon_lengths(0))
    res = seeded.step()
    n = len(res.frames)
    if n != 1:
        ok = False
    print(f"  [Theta_curr] frames={n} (expect 1) [{'ok' if n == 1 else 'BAD'}] | "
          f"iters={res.meta.iterations} | seeded L_curr from an FK pose")

    ok = _check_table_absolute() and ok
    ok = _check_start_pose() and ok
    ok = _check_base_moves() and ok

    print("Smoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


class ControllerVizApp:
    _ACTIVE_PHASE_COLOR = "blue"

    def __init__(self, server):
        import viser
        self.viser = viser
        self.server = server
        self.caps = capabilities()

        self.params = HandSolveParams()
        self.params.collision = True
        self.params.table = True          # §1.8 always has a support surface
        self.solver = None                # built when a phase button is pressed
        self.fk_solver = None             # cached: slider drags warm-start it
        self.result = None
        self.fk_result = None
        self.phase = None                 # None => the FK pose state
        self._theta_lengths = None        # L_curr handed to the next controller
        self._busy = False
        self._running = False
        # Set while widgets are written programmatically, so restoring defaults or
        # writing a computed pose does not cascade back through the callbacks.
        self._suspend_live = False
        self.tick_count = 0

        from .._plotting.viser_hand import ViserHandScene
        from .._plotting.viser_overlays import ConstraintOverlays
        self.scene = ViserHandScene(server, FINGER_LABELS)
        self.overlays = ConstraintOverlays(server)
        server.on_client_connect(self._aim_camera)

        self._build_gui()
        # The opening state and the Reset scene state are the same code path.
        self._reset_scene()

    # -- camera / scene ----------------------------------------------------

    def _aim_camera(self, client):
        _spec, center, _rot, _pose = resolve_scene(self.params)
        pos, look = self.scene.grasp_camera(center)
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(float(v) for v in pos)
        client.camera.look_at = tuple(float(v) for v in look)

    def _table_origin(self):
        """The support plane at the ABSOLUTE height the slider commands.

        The slider is the plane's world coordinate along the support normal (its
        z, with the default +Z normal), not an offset from anything -- so reading
        it never depends on where the plane already is. That matters because
        ``_sync_params`` publishes the result into ``params.plane_origin``: an
        implementation that resolved the *current* origin and added the slider on
        top would consume its own output and walk the plane a slider's worth on
        every sync (and there are several per tick).
        """
        spec, center, _rot, _pose = resolve_scene(self.params)
        return _absolute_table_origin(self.params, spec, center,
                                      self.g_table_height.value)

    def _seat_table_height(self):
        """Point the height slider at this object's underside -- the plane the
        scene would choose on its own.

        Called when the object changes and on reset, since the sensible default
        is a property of the object rather than a fixed number, and a plane left
        at the previous object's height would silently be inside the new one.
        """
        spec, center, _rot, _pose = resolve_scene(self.params)
        n_hat = np.asarray(self.params.plane_normal, float)
        n_hat = n_hat / (np.linalg.norm(n_hat) or 1.0)
        auto = np.asarray(auto_table_origin(self.params, spec, center), float)
        was_suspended = self._suspend_live
        self._suspend_live = True
        try:
            self.g_table_height.value = float(auto @ n_hat)
        finally:
            self._suspend_live = was_suspended

    def _refresh_scene(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        self.scene.set_object(spec, center, rotation)
        self.scene.set_table(self._table_origin(), self.params.plane_normal)

    def _render(self):
        if self.result is None:
            return
        gaps = self.result.contact_witness(0)
        names = set(self.result.contact_names())
        gaps = {k: v for k, v in gaps.items() if k in names}
        self.scene.update(self.result.frames[0],
                          tip_radii=self.result.tip_radii,
                          collision_radius=self.params.collision_radius,
                          collision=self.params.collision,
                          gaps=gaps)
        self._render_overlays()

    def _render_overlays(self):
        """Redraw the §1.8 goal overlays for the current result.

        Works in the FK pose state too: the goals are properties of the scene and
        the posture, not of whether a controller is running, so you can see where
        phase 0 will take the hand before committing Theta_curr. Only the pieces
        that ARE controller state -- the achieved base pose, the slewed waypoint,
        the solved witness points -- are missing there, and each overlay skips
        itself when its geometry is absent.
        """
        if self.result is None:
            self.overlays.update(None)
            return
        if self.solver is not None:
            geom = self.solver.goal_geometry(self.result)
        else:
            geom = goal_geometry(self.params, self.result, phase=self.phase,
                                 T_base=self.params.wrist_pose)
        self.overlays.update(geom)

    def _sync_overlays(self):
        """Push the checkboxes into the renderer's toggles.

        The two opposition overlays are additionally gated on the half-space
        switch: with the constraint off there is no split being enforced, and
        drawing one anyway would show a boundary nothing is respecting.
        """
        split_on = self.g_half_space.value and self.caps["controller"]
        for key, box in self.g_overlays.items():
            on = bool(box.value)
            if key in ("split", "split_plane"):
                on = on and split_on
            self.overlays.show[key] = on

    # -- params ------------------------------------------------------------

    def _sync_params(self):
        p = self.params
        p.primitive = self._label_to_key[self.g_object.value]
        p.object_center = None
        p.object_rotation = None

        R = _euler_to_R(self.g_roll.value, self.g_pitch.value, self.g_yaw.value)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.g_tx.value, self.g_ty.value, self.g_tz.value]
        # The sliders command the base pose whenever nothing else owns it: in the
        # FK pose state (no controller) always, and once one exists only in manual
        # mode. In servo mode the controller owns it -- phase 0's target drives it,
        # and every phase feeds the achieved pose back into the step prior -- so
        # writing the sliders each tick would clobber the loop and freeze the servo.
        if self.solver is None or self.g_base_mode.value == "manual":
            p.wrist_pose = T

        p.passive_tension = self.g_passive.value
        p.flexor_tensions = [s.value for s in self.g_flexors]
        p.contact_fingers = [c.value for c in self.g_contacts]

        # NOT read from a widget: the phase buttons own params.phase, because a
        # phase switch has to go through _set_phase() to preserve solver state.
        p.step_anchor = self.g_anchor.value
        p.half_space = self.g_half_space.value
        p.ctrl_al_iters = int(self.g_al_iters.value)
        p.sigma_wrist_pos_step = 10.0 ** self.g_sig_wrist.value
        p.sigma_q_step = 10.0 ** self.g_sig_q.value
        p.sigma_l_step = 10.0 ** self.g_sig_l.value
        p.plane_origin = np.asarray(self._table_origin(), float)

        self.scene.show_discs = self.g_show_discs.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self._sync_overlays()

    # -- Theta_curr: FK posing ---------------------------------------------

    def _rebuild_fk(self):
        """Cache an FK solver so slider drags warm-start (it re-aims the wrist
        prior rather than rebuilding). Rebuilt when the object changes, since the
        result's gap readouts are measured against it."""
        self.fk_solver = HandFKSolver(self.params)

    def _fk_preview(self, _=None):
        """Solve FK at the current sliders and draw it. This is the candidate
        Theta_curr, not yet handed to any controller."""
        if self._busy:
            return
        self._busy = True
        try:
            self._sync_params()
            self.fk_result = self.result = self.fk_solver.solve()
            self._render()
            self._report_pose()
        except Exception as exc:
            self._set_status(f"**Error:** {exc}")
            raise
        finally:
            self._busy = False

    def _live_fk(self, _=None):
        """FK is fast and warm-starts, so re-solve live as sliders move -- but only
        in the pose state; once a controller exists it owns the robot state."""
        if self.solver is None and not self._busy and not self._suspend_live:
            self._fk_preview()

    def _apply_free_space_start(self, _=None, preview=True):
        """Put the hand at the collision-free start pose phase 0 servos from, and
        write it back into the sliders so they keep showing the truth."""
        self._sync_params()
        # Oriented, centred over the object, then lifted. free_space_start_pose
        # re-runs FK at whatever orientation it is given, so the clearance scan
        # accounts for both the rotation and the centring. Centring is what keeps
        # the hand ABOVE the object: the contact centroid is ~0.14 m out along
        # the hand's own axes, so without it the orientation alone decides where
        # the fingers land and the roll throws them ~0.27 m off.
        T, info = free_space_start_pose(self.params,
                                        orientation=start_orientation(),
                                        center_on_object=True)
        self.params.wrist_pose = T
        was_suspended = self._suspend_live
        self._suspend_live = True
        try:
            self.g_tx.value, self.g_ty.value, self.g_tz.value = (
                float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
            self.g_yaw.value = 0.0
            self.g_roll.value = START_ROLL_RAD
            self.g_pitch.value = START_PITCH_RAD
        finally:
            self._suspend_live = was_suspended
        if preview:
            self._fk_preview()
        return info

    def _commit_theta_curr(self, _=None):
        """The measured state of Eq 1.93-1.95, taken from the sliders: FK-solve
        them, keep the resulting tendon lengths as L_curr, and drop the controller
        so the next phase you pick is built from this state."""
        self.solver = None
        self.phase = None
        self.tick_count = 0
        self._running = False
        self.g_auto.value = False
        self._fk_preview()
        self._theta_lengths = self.fk_result.tendon_lengths(0)
        self._refresh_controls()
        t = self.params.wrist_pose[:3, 3]
        self._set_status(
            f"**Theta_curr set**  \n"
            f"T_base t = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m  \n"
            f"Q = passive {self.params.passive_tension:.2f} N, flexors "
            f"{', '.join(f'{v:.2f}' for v in self.params.flexor_tensions)} N  \n"
            f"L_curr seeded from the FK solve  \n"
            f"pick a phase to run it into")

    def _report_pose(self):
        """What the posed hand looks like to the constraint sets that come next:
        where the base is, how far the fingertips are from the object, and how far
        above the support plane they sit."""
        m = self.result.meta
        t = self.params.wrist_pose[:3, 3]
        gaps = self.result.surface_gaps(0)
        n = np.asarray(self.params.plane_normal, float)
        n = n / (np.linalg.norm(n) or 1.0)
        origin = np.asarray(self._table_origin(), float)
        clear = min(
            float((np.asarray(
                self.result.frames[0][name].marginals.rod.states[-1].pose.mean,
                float)[:3, 3] - origin) @ n) - r
            for name, r in zip(self.result.finger_names, self.result.tip_radii))
        lines = [f"**FK pose** &nbsp; no controller &nbsp; iters={m.iterations} "
                 f"&nbsp; err={m.error:.3g}",
                 f"T_base t = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m",
                 f"closest tip-object gap: {min(gaps.values()):+.4f} m",
                 f"min tip-table clearance: {clear:+.4f} m",
                 "",
                 'set "manually set Theta_curr", then pick a phase']
        self._set_status("  \n".join(lines))

    # -- controller --------------------------------------------------------

    def _invalidate(self, _=None):
        """Drop the controller and fall back to the FK pose state.

        Anything that changes the CONSTRAINT SET -- object, contact mask,
        half-space, anchor, AL budget, table height -- invalidates a warm-started
        controller, because its retained state no longer matches the graph. The
        step-prior sigmas are here for a different reason: they are baked into
        the noise model at construction (there is no live setter), so without a
        rebuild dragging one changes params and nothing else, and the slider
        reads as broken. Phase
        switches are deliberately NOT in this list: carrying state across them is
        the whole point of the phased formulation."""
        if self._suspend_live:
            return
        self.solver = None
        self.phase = None
        self.tick_count = 0
        self._theta_lengths = None
        self._running = False
        self._sync_params()
        self._rebuild_fk()
        self._refresh_scene()
        self._refresh_controls()
        self._fk_preview()

    def _tick(self, _=None):
        if self._busy or not self.caps["controller"]:
            return
        if self.phase is None:
            self._set_status("pick a phase to start")
            return
        self._busy = True
        self.g_step.disabled = True
        try:
            self._sync_params()
            if self.solver is None:
                self.solver = HandControllerSolver(
                    self.params, initial_lengths=self._theta_lengths)
                self.tick_count = 0
            else:
                # step() commands the base from the solver's own _base_pose (the
                # Eq 1.93 feedback loop), so a manual re-command has to go through
                # set_theta_curr; assigning params.wrist_pose would not reach it.
                if self.g_base_mode.value == "manual":
                    self.solver.set_theta_curr(wrist_pose=self.params.wrist_pose)
                if self.solver.params.phase != self.params.phase:
                    self.solver.set_phase(self.params.phase)
            t0 = time.time()
            self.result = self.solver.step()
            dt_ms = (time.time() - t0) * 1000.0
            self.tick_count += 1
            self._render()
            self._report(dt_ms)
        except Exception as exc:
            self._set_status(f"**Error:** {exc}")
            raise
        finally:
            self.g_step.disabled = False
            self._busy = False

    def _set_phase(self, phase):
        """Switch the active constraint set, keeping the converged state. Out of
        the pose state this is what builds the controller, from the Theta_curr
        last committed."""
        self.phase = phase
        self.params.phase = phase
        self._refresh_controls()
        if self.solver is not None:
            self.solver.set_phase(phase)
        self._tick()

    def _refresh_controls(self):
        """Reconcile the GUI with the app state: which phase is active (none in
        the pose state) and whether there is anything to step."""
        for ph, btn in self.g_phase_btns.items():
            btn.color = (self._ACTIVE_PHASE_COLOR
                         if ph == self.phase else None)
        self.g_phase.value = (NO_PHASE_LABEL if self.phase is None
                              else PHASE_LABELS[self.phase])
        runnable = self.caps["controller"] and self.phase is not None
        self.g_step.disabled = not runnable
        self.g_auto.disabled = not runnable

    def _set_status(self, text):
        self.g_status.content = text

    def _report(self, dt_ms):
        m = self.result.meta
        lines = [f"**phase {self.phase}** &nbsp; tick {self.tick_count} "
                 f"&nbsp; {dt_ms:.0f} ms",
                 f"iters={m.iterations} &nbsp; err={m.error:.3g}",
                 f"anchor: {self.params.step_anchor}"]
        rows = self.solver.phase_violations()
        if rows:
            lines.append("")
            lines += [f"{name}: {v:.2e} {VIOLATION_UNITS.get(name, 'm')}"
                      for name, v in rows]
        else:
            lines.append("no active equality constraints")
        self._set_status("  \n".join(lines))

    def _autorun(self):
        """Free-run ticks at the requested rate until the toggle clears."""
        while self._running and self.phase is not None:
            self._tick()
            time.sleep(max(0.0, 1.0 / max(self.g_rate.value, 0.1)))

    def _reset_scene(self, _=None):
        """Back to the opening state: default widgets, default params, the
        free-space start pose, no controller and no phase -- just FK posing."""
        self._running = False
        self._suspend_live = True
        try:
            for handle, value in self._defaults:
                handle.value = value
        finally:
            self._suspend_live = False

        self.params = HandSolveParams()
        self.params.collision = True
        self.params.table = True
        self.solver = None
        self.phase = None
        self.tick_count = 0
        self._theta_lengths = None
        self.result = self.fk_result = None

        # Seat the plane under the object BEFORE anything reads it: the height is
        # absolute, so its default is whatever this object's underside is at, and
        # the free-space lift below is measured against it.
        self._seat_table_height()
        self._sync_params()
        self._rebuild_fk()
        info = self._apply_free_space_start(preview=False)
        self._refresh_scene()
        self._refresh_controls()
        self._fk_preview()
        note = ("" if self.caps["controller"] else
                "  \n**Controller unavailable** -- rebuild `_crest_sparse`.")
        self._set_status(
            f"**free-space start**  \n"
            f"lift {info['lift']:.4f} m along the support normal  \n"
            f"table {info['table_clearance']:+.4f} m &nbsp; object "
            f"{info['object_clearance']:+.4f} m  \n"
            f"pose the hand with the sliders, then set Theta_curr{note}")

    # -- GUI ---------------------------------------------------------------

    def _remember(self, handle):
        """Record a widget's constructed default so Reset scene can restore it
        without a second, drift-prone list of defaults."""
        self._defaults.append((handle, handle.value))
        return handle

    def _build_gui(self):
        gui = self.server.gui
        self._defaults = []
        keys = [k for k, v in get_primitive_specs().items()
                if v["type"] != "ellipsoid" or self.caps["ellipsoid"]]
        self._label_to_key = {SDF_DROPDOWN_LABELS.get(k, k): k for k in keys}
        object_default = next(
            lbl for lbl, k in self._label_to_key.items()
            if k == default_primitive(self.caps))

        hint = (None if self.caps["controller"]
                else "requires a rebuilt _crest_sparse with TendonHandController")

        with gui.add_folder("Phase"):
            self.g_phase_btns = {
                ph: gui.add_button(label, disabled=not self.caps["controller"],
                                   hint=hint)
                for ph, label in PHASE_LABELS.items()}
            # Kept in sync with the buttons by _refresh_controls.
            self.g_phase = self._remember(gui.add_dropdown(
                "active", [NO_PHASE_LABEL] + list(PHASE_LABELS.values()),
                initial_value=NO_PHASE_LABEL, disabled=True,
                hint="Pressing a phase builds the controller from the committed "
                     "Theta_curr. Switching phase keeps the converged state."))

        with gui.add_folder("Run"):
            self.g_step = gui.add_button(
                "Step", icon=self.viser.Icon.PLAYER_TRACK_NEXT,
                disabled=True, hint=hint)
            self.g_auto = self._remember(gui.add_checkbox(
                "auto-step", False, disabled=True,
                hint=hint or "Free-run ticks at the rate below."))
            self.g_rate = self._remember(
                gui.add_slider("ticks / s", 0.5, 20.0, 0.5, 4.0))
            self.g_reset = gui.add_button("Reset scene",
                                          icon=self.viser.Icon.REFRESH,
                                          hint="Everything back to defaults: the "
                                               "free-space start pose, no "
                                               "controller, no phase.")
            self.g_status = gui.add_markdown(
                "" if self.caps["controller"]
                else "**Controller unavailable** -- rebuild `_crest_sparse`.")

        with gui.add_folder("Scene"):
            self.g_object = self._remember(gui.add_dropdown(
                "object", list(self._label_to_key),
                initial_value=object_default))
            # ABSOLUTE, not an offset: this is the plane's world coordinate along
            # the support normal (its z, with the default +Z normal). Not
            # _remember()'d -- its default depends on the object, so it is seated
            # by _seat_table_height() on reset and on every object change.
            self.g_table_height = gui.add_slider(
                "table height (m, world)", -0.2, 0.2, 0.001, 0.0,
                hint="World-frame height of the support plane along the surface "
                     "normal. Re-seats to the object's underside when you change "
                     "objects.")
            self.g_half_space = self._remember(gui.add_checkbox(
                "half-space split", True, disabled=not self.caps["controller"],
                hint=hint or ("Phase 1 opposition (Eq 1.92): thumb to one half of "
                              "the table, the other fingers to the other.")))

        with gui.add_folder("Contact fingers"):
            self.g_contacts = [
                self._remember(gui.add_checkbox(
                    lbl, True,
                    hint="Drive this fingertip onto the table and the object. "
                         "Unchecked fingers keep collision avoidance only "
                         "(they stay in D_free), so a small object can be "
                         "pinched with a subset. Rebuilds the controller."))
                for lbl in FINGER_LABELS]

        with gui.add_folder("Step priors (trust region)"):
            # Defaults read from HandSolveParams, NOT written out as numbers: the
            # base sigmas were retuned once already and a hardcoded slider
            # default silently re-imposed the old value on every tick through
            # _sync_params, which is exactly the frozen-base failure below.
            self.g_sig_wrist = self._remember(gui.add_slider(
                "log10 sigma_T,step", -6, 0, 0.5,
                _log10_default("sigma_wrist_pos_step"),
                hint="Eq 1.94: how far the hand base may move in one tick. "
                     "Below ~1e-2 the prior is stiffer than the AL penalty "
                     "ceiling (mu ~ 8e3) and the base FREEZES -- phases 1-3 "
                     "then flex the fingers against a hand that cannot "
                     "reposition. Changing this rebuilds the controller."))
            self.g_sig_q = self._remember(gui.add_slider(
                "log10 sigma_Q,step", -4, 1, 0.5,
                _log10_default("sigma_q_step"),
                hint="Eq 1.95: how far the tendon tensions may move in one "
                     "tick. Changing this rebuilds the controller."))
            self.g_sig_l = self._remember(gui.add_slider(
                "log10 sigma_L,step", -6, 0, 0.5,
                _log10_default("sigma_l_step"),
                hint="Eq 1.13 analogue: how far the tendon lengths may move. "
                     "Changing this rebuilds the controller."))
            self.g_anchor = self._remember(gui.add_dropdown(
                "anchor", ["tension", "length", "both"],
                initial_value="tension",
                disabled=not self.caps["controller"],
                hint=hint or ("What carries the measured state. 'length' is the "
                              "hardware-faithful mode: tendons are inextensible, "
                              "so length is what the motor commands, whereas a "
                              "disturbance contact changes tension without the "
                              "robot having moved.")))

        with gui.add_folder("Theta_curr: hand base pose"):
            # In "servo" mode the controller owns the base pose: phase 0's target
            # drives it and every phase feeds the achieved pose back into the
            # step prior. The sliders would overwrite that every tick, so they
            # are only read in "manual" mode -- or whenever no controller exists.
            self.g_base_mode = self._remember(gui.add_dropdown(
                "control", ("servo", "manual"), initial_value="servo",
                hint="Who owns T_base once a controller is running. The sliders "
                     "always command it in the FK pose state."))
            # +/-0.2 m: the free-space lift alone can exceed 0.1 m.
            self.g_tx = self._remember(gui.add_slider("x (m)", -0.2, 0.2, 0.001, 0.0))
            self.g_ty = self._remember(gui.add_slider("y (m)", -0.2, 0.2, 0.001, 0.0))
            self.g_tz = self._remember(gui.add_slider("z (m)", -0.2, 0.2, 0.001, 0.0))
            # Rolled and pitched by default so the hand opens palm-down at the
            # table with the thumb on its opposition half; remembered at those
            # values so Reset scene keeps them (see START_ROLL_RAD).
            self.g_roll = self._remember(
                gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01,
                               START_ROLL_RAD))
            self.g_pitch = self._remember(
                gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01,
                               START_PITCH_RAD))
            self.g_yaw = self._remember(
                gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, 0.0))
            self.g_set_theta = gui.add_button(
                "manually set Theta_curr", icon=self.viser.Icon.HAND_FINGER,
                hint="Take the base pose and the tensions below as the measured "
                     "state (Eq 1.93): FK-solve them, seed L_curr from that "
                     "solve, and drop the controller so the next phase you pick "
                     "starts from it.")
            self.g_free_start = gui.add_button(
                "reset to free-space start",
                hint="The collision-free pose phase 0 servos from: lifted along "
                     "the support normal until every sphere clears the table and "
                     "the object.")

        with gui.add_folder("Theta_curr: tensions (N)"):
            self.g_passive = self._remember(
                gui.add_slider("passive", 0.0, 3.0, 0.05,
                               HandSolveParams().passive_tension))
            self.g_flexors = [
                self._remember(gui.add_slider(
                    lbl, 0.0, 3.0, 0.05, pregrasp_flexor(HandSolveParams())))
                for lbl in FINGER_LABELS]

        with gui.add_folder("Augmented Lagrangian"):
            self.g_al_iters = self._remember(gui.add_slider(
                "iters / tick", 1, 40, 1, 4,
                hint="Small on purpose: the AL outer loop is amortized across "
                     "ticks, since the constraint set is unchanged between them "
                     "and each tick warm-starts from the last."))

        with gui.add_folder("Display"):
            self.g_show_collision = self._remember(
                gui.add_checkbox("collision spheres", True))
            self.g_show_discs = self._remember(
                gui.add_checkbox("routing discs", False))

        # Constraint-goal overlays, one folder per phase, generated from the
        # renderer's OVERLAYS table so a new overlay needs no edit here. Grouped
        # by the phase each goal BELONGS to, not gated on the active phase: any
        # of them can be left on across a switch, which is how you see what
        # changed -- e.g. watching phase 1's plane distances during phase 2 shows
        # whether the slide is holding support contact.
        self.g_overlays = {}
        for ph, label in PHASE_LABELS.items():
            specs = [s for s in OVERLAYS if s.phase == ph]
            if not specs:
                continue
            with gui.add_folder(f"Goals: phase {label}"):
                for s in specs:
                    self.g_overlays[s.key] = self._remember(
                        gui.add_checkbox(s.label, s.default, hint=s.hint))

        # -- callbacks --
        self.g_step.on_click(self._tick)
        self.g_reset.on_click(self._reset_scene)
        self.g_set_theta.on_click(self._commit_theta_curr)
        self.g_free_start.on_click(self._apply_free_space_start)

        for _ph, _btn in self.g_phase_btns.items():
            _btn.on_click(lambda _, ph=_ph: self._set_phase(ph))

        @self.g_auto.on_update
        def _(_):
            self._running = self.g_auto.value
            if self._running:
                import threading
                threading.Thread(target=self._autorun, daemon=True).start()

        @self.g_object.on_update
        def _(_):
            if self._suspend_live:
                return
            # Re-seat the plane first: the height is absolute, so it has to move
            # to the new object's underside before the free-space lift is
            # measured against it.
            self._seat_table_height()
            self._invalidate()
            self._apply_free_space_start()
            self._aim_all_cameras()

        # Everything that changes the constraint set rebuilds the controller.
        for h in ([self.g_half_space, self.g_anchor, self.g_al_iters,
                   self.g_table_height, self.g_sig_wrist, self.g_sig_q,
                   self.g_sig_l] + self.g_contacts):
            h.on_update(self._invalidate)

        # Theta_curr's own knobs: live FK while no controller owns the state.
        for h in ([self.g_tx, self.g_ty, self.g_tz, self.g_roll, self.g_pitch,
                   self.g_yaw, self.g_passive] + self.g_flexors):
            h.on_update(self._live_fk)

        # Display and overlay toggles only re-render. Overlays deliberately do
        # NOT go through _invalidate: they change nothing the solver reads, and
        # dropping a warm controller to tick a checkbox would lose the state you
        # turned the overlay on to look at.
        for h in ([self.g_show_collision, self.g_show_discs]
                  + list(self.g_overlays.values())):
            h.on_update(lambda _: (self._sync_params(), self._render()))

    def _aim_all_cameras(self):
        for client in self.server.get_clients().values():
            self._aim_camera(client)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true",
                        help="Headless self-check (no viser): one tick per phase.")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if args.smoke:
        return _smoke()

    import viser
    server = viser.ViserServer(port=args.port)
    ControllerVizApp(server)
    print(f"Section 1.8 controller visualizer -- open http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
