"""Drawing the scene: camera aim, the hand frame, the table readout.

A mixin of :class:`~gepetto_solvers.projects.viz.viz_interactive.app.HandVizApp`,
split out of what was one 4284-line class. The methods here use the attributes
that class's ``__init__`` sets up.
"""


import numpy as np

from gepetto_solvers.core.geometry.scene import TABLE_SPAN, TABLE_THICKNESS
from gepetto_solvers.core.solvers import (
    R_to_euler,
    ellipsoid_grasp_wrench_witness,
    ellipsoid_metric_witness,
    finger_plane_witness,
    free_sphere_plane_witness,
    grasp_wrench_witness,
    half_space_witness,
    object_collision_witness,
    planar_gap_witness,
    plane_witness,
    pregrasp_axis_witness,
    pregrasp_center_witness,
    pregrasp_centroid_witness,
    resolve_constraint_plane_origin,
    resolve_scene,
    resolve_table_origin,
    self_collision_witness,
)

from .constants import SELF_PAIR_WINDOW_M


class SceneRenderMixin:
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


    def _mount_pose(self):
        """``T_flange<-wrist`` for the hand being posed, or None if it has none.

        Read off the HAND, because where a hand bolts to the arm is a fact about
        that hand: the tendon hand's is a fit against its Onshape assembly, the
        Allegro hand's is a `tf2_echo` off the running robot, and they are
        different transforms measured different ways. A single shared constant
        would mount whichever hand it was not measured on 130 mm and a quarter
        turn wrong, with nothing on screen to say so.
        """
        pose = getattr(self.hand, "mount_pose", None)
        return pose() if pose is not None else None


    def _pose_at_mount(self, _=None):
        """Drive the wrist sliders to the measured robot mount and re-pose.

        With the wrist at ``T_flange<-wrist``, the viser world frame IS the flange
        frame, so what you see is the hand as it hangs off the arm -- the check
        the measurement actually needs. Turns the mount frames on, since arriving
        here with them off shows nothing new.
        """
        T = self._mount_pose()
        if T is None:
            return                       # the button is disabled for such a hand
        xyz = tuple(float(v) for v in T[:3, 3])
        rpy = R_to_euler(T[:3, :3])
        for handle, value in zip(
                (self.g_tx, self.g_ty, self.g_tz,
                 self.g_roll, self.g_pitch, self.g_yaw),
                xyz + tuple(rpy)):
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
        mount = self._mount_pose()
        if not self.g_show_mount.value or mount is None:
            self.scene.clear_mount_frames()
            return
        T_wrist = (res.wrist_pose(0)
                   if res is not None else self.params.wrist_pose)
        self.scene.set_mount_frames(T_wrist, mount)


    def _table_origin(self):
        """The rendered slab's origin: the TABLE surface, seated from the scene.

        Everything registered against the bench hangs off this -- the slab, its
        corner frame and grid, the calibration target, and ``_corner_viz``, which
        is this app's half of the registration against
        ``lbr_workspace_table_link``. It is therefore deliberately independent of
        the constraint-plane height slider; that moves
        :meth:`_constraint_plane_origin` alone.
        """
        spec, center, _rot, _pose = resolve_scene(self.params)
        return resolve_table_origin(self.params, spec, center)


    def _constraint_plane_origin(self):
        """The origin of the plane the SOLVER sees -- the table raised by the
        Table folder's height slider. Resolved through the solvers module, so the
        picture drawn here is the same plane ``_attach_table`` builds the support
        equality and the avoidance half-space from."""
        spec, center, _rot, _pose = resolve_scene(self.params)
        return resolve_constraint_plane_origin(self.params, spec, center)


    def _refresh_table_readout(self, origin, corner):
        """Publish the landmark's numbers: the square's size and where its corner
        frame currently is, in world coordinates.

        These have to be readable, not inferred. The whole point of the frame is
        to be measured against a real bench, and a triad you can see but whose
        coordinates you cannot read is not a landmark. The table height is quoted
        alongside because the table is seated from the object (see
        ``auto_table_origin``), so it moves when the object changes -- this is
        where you see that it did.

        The CONSTRAINT plane's height is quoted on its own line, because the two
        surfaces are now independent and the whole risk of separating them is
        mistaking one for the other: this says, in world coordinates, exactly
        where the solver's plane sits relative to the bench.
        """
        origin = np.asarray(origin, float).reshape(3)
        corner = np.asarray(corner, float).reshape(3)
        axis = int(np.argmax(np.abs(np.asarray(self.params.plane_normal, float))))
        constraint = np.asarray(self._constraint_plane_origin(), float).reshape(3)
        height = self.params.constraint_plane_height
        self.g_table_status.content = (
            f"square **{TABLE_SPAN:.3f} x {TABLE_SPAN:.3f} m**, "
            f"{TABLE_THICKNESS * 1e3:.0f} mm thick  \n"
            f"table (top face) {'xyz'[axis]} = {origin[axis]:+.4f} m  \n"
            f"constraint plane {'xyz'[axis]} = {constraint[axis]:+.4f} m "
            f"({height * 1e3:+.0f} mm)  \n"
            f"corner frame ({corner[0]:+.4f}, {corner[1]:+.4f}, "
            f"{corner[2]:+.4f}) m")


    def _distance_witnesses(self, res):
        """Every constraint-distance overlay's data for one solved state, as the
        ``scene.update`` keyword arguments that draw them.

        NOT GATED ON THE CONSTRAINTS. Each family is computed when its own
        display box is ticked and the hand supplies what it needs, and never
        because the matching constraint happens to be in the graph. That is the
        whole point of the folder these boxes live in: the distance a collision
        inequality would see is most worth reading while collision is OFF, and
        the arrangement h_grasp measures is worth watching before the equality
        is ever attached.

        Gated on the box rather than computed unconditionally because these are
        not free: the self-collision witness walks every cross-digit sphere pair
        in the hand, and the object clearances run an analytic surface solve per
        sphere. An unticked family costs nothing.

        The DESIGNATION masks still narrow what is reported -- a gap line on a
        finger nothing was ever asked to touch is noise, not information -- but a
        mask is a statement about the task, not about which constraints are
        switched on, so it survives them all being off.
        """
        shown = self.scene.distances
        # The master switch short-circuits the whole group: with it off nothing
        # below would be drawn, so nothing below needs computing.
        if not self.g_show_gaps.value:
            return {}

        out = {}

        if shown.object_contact:
            names = set(res.contact_names())
            out["gaps"] = {name: v for name, v in res.contact_witness(0).items()
                           if name in names}
        if shown.table_contact:
            # The table set when a solve designated one, the object set
            # otherwise: a phase that targets both surfaces designates one list
            # for both, and falling back keeps the plane distance readable while
            # merely posing rather than only after a table solve.
            out["table_gaps"] = plane_witness(
                self.params, res, 0,
                names=(res.table_contact_names() or res.contact_names()))
        if shown.half_space and self.caps["opposition"]:
            out["half_space_gaps"] = half_space_witness(self.params, res, 0)
        if shown.pregrasp:
            # Each returns None where the hand cannot supply what it needs (no
            # opposing digit, no measured pinch pose), so a hand that has only
            # some of the three draws only those.
            out["center_gap"] = pregrasp_center_witness(self.params, res, 0)
            out["axis_align"] = pregrasp_axis_witness(self.params, res, 0)
            out["centroid_gap"] = pregrasp_centroid_witness(self.params, res, 0)
        if shown.object_collision:
            out["object_clearances"] = object_collision_witness(
                self.params, res, 0)
        if shown.table_collision:
            # The same designated set the table gap lines used, so a fingertip
            # is reported by exactly one of the two overlays.
            out["table_clearances"] = free_sphere_plane_witness(
                self.params, res, 0,
                names=(res.table_contact_names() or res.contact_names()))
        if shown.self_collision:
            out["pair_gaps"] = self_collision_witness(
                self.params, res, 0, max_gap=SELF_PAIR_WINDOW_M)
        if shown.ellipsoid_metric:
            # None on an object with no ellipsoid form, where ellipsoid_taubin
            # is inert and the comparison would be one number printed twice.
            out["ellipsoid_metrics"] = ellipsoid_metric_witness(
                self.params, res, 0)
        if shown.grasp_wrench and self.caps["grasp_alignment"]:
            out["grasp_wrench"] = grasp_wrench_witness(res, 0)
            # The origin the moment arms and the residual are measured about --
            # the constraint's own t_obj.
            out["object_center"] = res.object_center
        if (shown.grasp_wrench_ellipsoid
                and self.caps["grasp_alignment_ellipsoid"]):
            out["grasp_wrench_ellipsoid"] = ellipsoid_grasp_wrench_witness(
                self.params, res, 0)
            # Shares t_obj with the family above -- both are measured about the
            # object origin, and setting it twice is setting it to the same
            # thing.
            out["object_center"] = res.object_center
        return out


    def _render_frame(self, live=False):
        if self.result is None:
            # Nothing solved yet, so the commanded wrist pose is all there is.
            self._render_mount()
            self._report_actuation(None)
            self._update_traj()
            return
        # Render whichever solve snapshot the convergence scrubber selects; with
        # no scrubber up this is the result itself, so the gap readouts below
        # describe the intermediate state without knowing about iterates at all.
        # Every result here is a single state, so there is only ever frame 0.
        res = self._iter_view(live)
        self._render_mount(res)
        self._report_actuation(res)
        overlays = self._distance_witnesses(res)
        # Purely a picture of the posture on screen, so unlike the overlays
        # above it is gated on its own display checkbox rather than on a
        # constraint being switched on -- the plane is worth looking at exactly
        # when nothing is enforcing anything about it.
        planes = (finger_plane_witness(res, 0)
                  if (self.g_show_finger_planes is not None
                      and self.g_show_finger_planes.value) else None)
        # Same gating, one step further: the in-plane distance also needs a
        # binding that can evaluate the factor and an object with an analytic
        # ellipsoid form. planar_gap_witness returns None on either, so the
        # overlay simply does not appear rather than the render failing.
        planar = (planar_gap_witness(self.params, res, 0)
                  if (self.g_show_planar_gap is not None
                      and self.g_show_planar_gap.value
                      and self.caps["planar_gap"])
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
                          **overlays,
                          # Visual only. Collision is the sphere set above; the
                          # graph never sees a mesh, so this cannot change a
                          # solve -- and a hand without meshes just draws as a
                          # skeleton.
                          link_meshes=(self._link_meshes()
                                       if self.g_show_meshes.value else None),
                          wrist_pose=res.wrist_pose(0),
                          finger_planes=planes,
                          planar_gaps=planar)
        # Last: the plots describe the state just drawn, and they are the one
        # readout here that spans the WHOLE solve rather than this single frame.
        self._update_traj(live)
