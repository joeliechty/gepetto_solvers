"""Interactive viser visualizer for the tendon-hand FK / IK / trajectory-planner
solvers.

Exposes the solver knobs as live web GUI controls -- object picker, wrist start
pose, per-finger flexor tensions, per-finger contact toggles, GP prior stiffness,
collision / table options, AL settings -- with buttons to switch between the FK
solver, the IK solver, and the trajectory planner. Set parameters, hit *Solve*,
and inspect the result; the planner result is scrubbable step by step.

All three solvers are the reusable classes in ``tendon_hand/solvers.py``; the 3D
scene is drawn by ``_plotting/viser_hand.ViserHandScene``. The existing demo
scripts are untouched.

Run (from the ``python/`` directory):
    python -m tests.tendon_hand.viz_interactive
then open the printed http://localhost:8080 URL.

Optional headless self-check of the solver classes (no browser):
    python -m tests.tendon_hand.viz_interactive --smoke
"""

import argparse
import sys

import numpy as np

from .scene import get_primitive_specs, GRASP_FLEXOR_TENSION, TABLE_NORMAL
from .solvers import (
    HandSolveParams, HandFKSolver, HandIKSolver, HandPlannerSolver, SOLVERS,
    NUM_FINGERS, resolve_scene, resolve_table_origin, capabilities)


FINGER_LABELS = ["index", "middle", "ring", "pinky", "thumb"]

# Display-only suffix for the baked-SDF spheres in the object dropdown, so they
# read apart from the analytic ``*_sphere_ellipsoid`` look-alikes. The spec keys
# (and the demo scripts' argparse choices) keep the un-suffixed names; only the
# label the user picks from carries "_sdf".
SDF_DROPDOWN_LABELS = {"sphere": "sphere_sdf", "big_sphere": "big_sphere_sdf"}


def _euler_to_R(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) rotation matrix from radians."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Headless smoke test -- validates the solver classes independently of viser.
# ---------------------------------------------------------------------------

def _smoke():
    print("Smoke-testing the hand solver classes (big_sphere, defaults)...")
    ok = True
    for label, Solver, expect in [
            ("FK", HandFKSolver, 1),
            ("IK", HandIKSolver, 1),
            ("Planner", HandPlannerSolver, None)]:
        params = HandSolveParams()
        if label == "Planner":
            params.K = 6
        res = Solver(params).solve()
        n = len(res.frames)
        exp = (params.K + 1) if expect is None else expect
        gap = res.worst_gap(-1) if label != "FK" else float("nan")
        status = "ok" if n == exp else "BAD"
        if n != exp:
            ok = False
        extra = "" if label == "FK" else f" | terminal worst gap {gap:+.5f} m"
        print(f"  [{label:>7}] frames={n} (expect {exp}) [{status}] | "
              f"iters={res.meta.iterations} err={res.meta.error:.3g}{extra}")
    print("Smoke test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Interactive app.
# ---------------------------------------------------------------------------

class HandVizApp:
    # Solid tint for the active FK/IK/Planner button (inactive ones stay the
    # default subtle style) so the current solver mode is obvious at a glance.
    _ACTIVE_MODE_COLOR = "blue"

    def __init__(self, server):
        import viser  # local import so --smoke needs no viser
        self.viser = viser
        self.server = server
        self.params = HandSolveParams()
        self.mode = "FK"
        self.result = None
        self._solving = False
        # What this installed binding supports, so we can gate controls a stale
        # .so would crash on (ellipsoid objects, the table, cull margin).
        self.caps = capabilities()

        from .._plotting.viser_hand import ViserHandScene
        self.scene = ViserHandScene(server, FINGER_LABELS)

        # Park every (current and future) client's camera on the -X/palmar side so
        # the finger curl reads as a grasp instead of bending backwards. Without
        # this viser opens from the opposite side and the correct solve looks wrong.
        server.on_client_connect(lambda client: self._aim_camera(client))

        self._build_gui()
        # A cached FK solver so wrist/tension tweaks warm-start (rebuilt on object
        # change only).
        self._rebuild_fk()
        self._refresh_object()
        self._solve_and_render()

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

    def _sync_wrist(self):
        R = _euler_to_R(self.g_roll.value, self.g_pitch.value, self.g_yaw.value)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [self.g_tx.value, self.g_ty.value, self.g_tz.value]
        self.params.wrist_pose = T

    def _sync_params(self):
        p = self.params
        p.primitive = self._label_to_key[self.g_object.value]
        # let center/rotation re-derive from the primitive
        p.object_center = None
        p.object_rotation = None
        self._sync_wrist()
        p.passive_tension = self.g_passive.value
        p.flexor_tensions = [s.value for s in self.g_flexors]
        p.contact_fingers = [c.value for c in self.g_contacts]
        p.sigma_wrist_pos = 10.0 ** self.g_sig_pos.value
        p.sigma_wrist_rot = 10.0 ** self.g_sig_rot.value
        # AL
        p.al_mu = self.g_al_mu.value
        p.al_rate = self.g_al_rate.value
        p.al_iters = self.g_al_iters.value
        # planner
        p.K = self.g_K.value
        p.dt = self.g_dt.value
        p.gp_wrist = 10.0 ** self.g_gp_wrist.value
        p.gp_tense = 10.0 ** self.g_gp_tense.value
        p.gp_len = 0.0 if self.g_gp_len.value <= -8 else 10.0 ** self.g_gp_len.value
        p.start_flexor = self.g_start_flexor.value
        # collision
        p.collision = self.g_collision.value
        p.collision_radius = self.g_coll_radius.value
        p.collision_sigma = 10.0 ** self.g_coll_sigma.value
        p.cull_margin = (None if not self.caps["collision_cull"] or self.g_cull.value <= 0
                         else self.g_cull.value)
        # table
        p.table = self.g_table.value and self.caps["table"]
        p.plane_normal = np.array(TABLE_NORMAL, float)
        p.plane_avoidance = self.g_plane_avoid.value
        p.k_touch = int(self.g_k_touch.value) if p.table else None
        p.plane_origin = None  # auto (under object); offset applied below
        # display toggles
        self.scene.show_discs = self.g_show_discs.value
        self.scene.show_contact_spheres = self.g_show_contact.value
        self.scene.show_collision_spheres = self.g_show_collision.value
        self.scene.show_gap_lines = self.g_show_gaps.value

    def _table_origin(self):
        spec, center, _rot, _pose = resolve_scene(self.params)
        origin = resolve_table_origin(self.params, spec, center)
        origin = np.asarray(origin, float) + self.g_plane_offset.value * np.array(
            TABLE_NORMAL, float)
        return origin

    # -- rendering --

    def _refresh_object(self):
        spec, center, rotation, _pose = resolve_scene(self.params)
        self.scene.set_object(spec, center, rotation)
        if self.params.table:
            self.scene.set_table(self._table_origin(), self.params.plane_normal)
        else:
            self.scene.clear_table()

    def _render_frame(self, k):
        if self.result is None:
            return
        k = int(np.clip(k, 0, len(self.result.frames) - 1))
        # Only the fingers this solve drove onto the object get a gap line; a
        # distance readout on a finger nothing asked to touch is just noise.
        gaps = self.result.contact_witness(k)
        names = set(self.result.contact_names())
        gaps = {name: v for name, v in gaps.items() if name in names}
        self.scene.update(self.result.frames[k],
                          tip_radii=self.result.tip_radii,
                          collision_radius=self.params.collision_radius,
                          collision=self.params.collision,
                          gaps=gaps)

    def _set_status(self, text):
        self.g_status.content = text

    # -- solve --

    def _set_mode(self, mode):
        """Switch solver mode: recolor the buttons so only the active one is
        tinted, then re-solve in the new mode."""
        self.mode = mode
        self._refresh_mode_buttons()
        self._solve_and_render()

    def _refresh_mode_buttons(self):
        for m, btn in self.g_mode_btns.items():
            btn.color = self._ACTIVE_MODE_COLOR if m == self.mode else None

    def _set_solving(self, solving):
        """Grey out the Solve button (and the mode buttons) while a solve runs so
        it's clear the app is busy; restore them when it finishes."""
        self.g_solve.disabled = solving
        self.g_solve.label = "Solving..." if solving else "Solve"
        for btn in self.g_mode_btns.values():
            btn.disabled = solving

    def _solve_and_render(self, _=None):
        if self._solving:
            return
        self._solving = True
        self._set_solving(True)
        try:
            self._sync_params()
            self._refresh_object()
            self._set_status(f"Solving ({self.mode})...")
            if self.mode == "FK":
                # Reuse the cached FK solver (shares self.params) so this warm-starts.
                self.result = self.fk_solver.solve()
            else:
                self.result = SOLVERS[self.mode](self.params).solve()
            self._rebuild_step_slider()
            self._render_frame(self._current_step())
            self._report()
        except Exception as exc:  # surface solver errors in the GUI, keep serving
            self._set_status(f"**Error:** {exc}")
            raise
        finally:
            self._set_solving(False)
            self._solving = False

    def _report(self):
        m = self.result.meta
        lines = [f"**{self.mode}** &nbsp; iters={m.iterations} &nbsp; "
                 f"err={m.error:.3g} &nbsp; {m.total_time_ms:.0f} ms",
                 f"frames: {len(self.result.frames)}"]
        if self.mode != "FK":
            names = self.result.contact_names()
            contacting = ("none" if not names
                          else ", ".join(names) if len(names) < len(self.result.finger_names)
                          else "all")
            lines.append(f"contact: {contacting}")
            lines.append(f"terminal worst gap: {self.result.worst_gap(-1):+.5f} m")
        self._set_status("  \n".join(lines))

    def _live_fk(self, _=None):
        """FK is fast and warm-starts, so re-solve live as sliders move."""
        if self.mode == "FK" and not self._solving:
            self._solve_and_render()

    # -- step scrubber (planner) --

    def _current_step(self):
        return int(self.step_slider.value) if getattr(self, "step_slider", None) else 0

    def _rebuild_step_slider(self):
        if getattr(self, "step_slider", None) is not None:
            self.step_slider.remove()
            self.step_slider = None
        n = len(self.result.frames) if self.result else 1
        if n > 1:
            with self.step_folder:
                self.step_slider = self.server.gui.add_slider(
                    "step", min=0, max=n - 1, step=1, initial_value=n - 1)
            self.step_slider.on_update(lambda _: self._render_frame(self.step_slider.value))

    # -- GUI construction --

    def _build_gui(self):
        gui = self.server.gui
        # Ellipsoid objects need the analytic-surface env fields; hide them on a
        # binding that lacks them.
        keys = [k for k, v in get_primitive_specs().items()
                if v["type"] != "ellipsoid" or self.caps["ellipsoid"]]
        # Map the displayed dropdown label back to the real spec key (identity
        # except for the "_sdf"-suffixed baked spheres).
        self._label_to_key = {SDF_DROPDOWN_LABELS.get(k, k): k for k in keys}
        labels = list(self._label_to_key)

        with gui.add_folder("Solver"):
            # Individual mode buttons (not a button-group) so the active one can
            # be tinted a solid color; the group can't highlight its selection.
            self.g_mode_btns = {
                m: gui.add_button(
                    m, color=(self._ACTIVE_MODE_COLOR if m == self.mode else None))
                for m in ("FK", "IK", "Planner")}
            self.g_object = gui.add_dropdown(
                "object", labels,
                initial_value=SDF_DROPDOWN_LABELS["big_sphere"])
            self.g_solve = gui.add_button("Solve", icon=self.viser.Icon.PLAYER_PLAY)
            self.g_status = gui.add_markdown("")

        self.step_folder = gui.add_folder("Trajectory")

        with gui.add_folder("Wrist start pose"):
            self.g_tx = gui.add_slider("x (m)", -0.1, 0.1, 0.001, 0.0)
            self.g_ty = gui.add_slider("y (m)", -0.1, 0.1, 0.001, 0.0)
            self.g_tz = gui.add_slider("z (m)", -0.1, 0.1, 0.001, 0.0)
            self.g_roll = gui.add_slider("roll (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_pitch = gui.add_slider("pitch (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_yaw = gui.add_slider("yaw (rad)", -np.pi, np.pi, 0.01, 0.0)
            self.g_sig_pos = gui.add_slider("log10 sigma_pos", -6, 2, 0.5, -4)
            self.g_sig_rot = gui.add_slider("log10 sigma_rot", -6, 2, 0.5, -3)

        with gui.add_folder("Tensions (N)"):
            self.g_passive = gui.add_slider("passive", 0.0, 3.0, 0.05, 0.5)
            self.g_flexors = [
                gui.add_slider(lbl, 0.0, 3.0, 0.05, GRASP_FLEXOR_TENSION)
                for lbl in FINGER_LABELS]

        with gui.add_folder("Contact fingers"):
            self.g_contacts = [
                gui.add_checkbox(
                    lbl, True,
                    hint="Solve for this fingertip touching the object (IK / "
                         "Planner; FK never uses contact). Unchecked fingers "
                         "keep collision avoidance, so they stay out of the "
                         "object without being driven onto it. Applies on the "
                         "next Solve.")
                for lbl in FINGER_LABELS]

        with gui.add_folder("Planner / GP priors"):
            self.g_K = gui.add_slider("K steps", 2, 30, 1, 10)
            self.g_dt = gui.add_slider("dt (s)", 0.02, 0.5, 0.02, 0.1)
            self.g_gp_wrist = gui.add_slider("log10 gp_wrist", -4, 2, 0.5, -2)
            self.g_gp_tense = gui.add_slider("log10 gp_tense", -4, 2, 0.5, 0)
            self.g_gp_len = gui.add_slider("log10 gp_len (<=-8 off)", -8, 2, 0.5, -8)
            self.g_start_flexor = gui.add_slider("start flexor", 0.0, 3.0, 0.05, 0.5)

        with gui.add_folder("Collision"):
            self.g_collision = gui.add_checkbox("enabled", False)
            self.g_coll_radius = gui.add_slider("sphere radius (m)", 0.001, 0.01, 0.0005, 0.003)
            self.g_coll_sigma = gui.add_slider("log10 sigma", -6, 0, 0.5, -4)
            self.g_cull = gui.add_slider("cull margin (m, 0 off)", 0.0, 0.1, 0.005, 0.0)

        with gui.add_folder("Table"):
            self.g_table = gui.add_checkbox(
                "enabled", False, disabled=not self.caps["table"],
                hint=None if self.caps["table"]
                else "requires a newer _crest_sparse build (plane env fields)")
            self.g_plane_offset = gui.add_slider("height offset (m)", -0.1, 0.1, 0.002, 0.0)
            self.g_plane_avoid = gui.add_checkbox("avoidance", True)
            self.g_k_touch = gui.add_slider("k_touch", 0, 30, 1, 5)

        with gui.add_folder("Augmented Lagrangian"):
            self.g_al_mu = gui.add_slider("mu", 0.1, 10.0, 0.1, 1.0)
            self.g_al_rate = gui.add_slider("rate", 1.1, 5.0, 0.1, 2.0)
            self.g_al_iters = gui.add_slider("max iters", 5, 100, 5, 40)

        with gui.add_folder("Display"):
            self.g_show_contact = gui.add_checkbox("contact spheres", True)
            self.g_show_collision = gui.add_checkbox("collision spheres", True)
            self.g_show_discs = gui.add_checkbox("routing discs", False)
            self.g_show_gaps = gui.add_checkbox(
                "contact distance", True,
                hint="Fingertip-to-object gap in mm; green under 15 mm, red over.")

        # -- callbacks --
        self.g_solve.on_click(self._solve_and_render)

        for _m, _btn in self.g_mode_btns.items():
            _btn.on_click(lambda _, m=_m: self._set_mode(m))

        @self.g_object.on_update
        def _(_):
            self.params.primitive = self._label_to_key[self.g_object.value]
            self.params.object_center = None
            self.params.object_rotation = None
            self._rebuild_fk()      # FK solver carries the object for its result/spec
            self._refresh_object()
            self._aim_all_cameras()  # re-center on the new object's location
            self._solve_and_render()

        # Live FK re-solve on the pose / tension sliders (fast, warm-started).
        for h in ([self.g_tx, self.g_ty, self.g_tz, self.g_roll, self.g_pitch,
                   self.g_yaw, self.g_passive] + self.g_flexors):
            h.on_update(self._live_fk)

        # Display toggles re-render the current frame without re-solving. The
        # contact checkboxes ride along only to keep self.params in sync; like
        # every other solver knob they take effect on the next Solve, and the gap
        # lines keep describing the solve that is actually on screen until then.
        for h in (self.g_show_contact, self.g_show_collision, self.g_show_discs,
                  self.g_show_gaps, *self.g_contacts):
            h.on_update(lambda _: (self._sync_params(),
                                   self._render_frame(self._current_step())))
        # Table toggle / height updates the static slab immediately.
        for h in (self.g_table, self.g_plane_offset):
            h.on_update(lambda _: (self._sync_params(), self._refresh_object()))


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
    HandVizApp(server)
    print(f"viser hand visualizer running -- open http://localhost:{args.port}")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
