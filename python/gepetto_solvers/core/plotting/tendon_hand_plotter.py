import numpy as np
import pyvista as pv
import vtk

from . import utils
from .cosserat_rod_plotter import CosseratRodMeshManager


TENDON_COLORS = [
    "crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod",
    "deeppink", "darkorange", "teal", "sienna", "slategray",
]


class TendonMeshManager:
    """Manages tendon line meshes and disc meshes for a single finger.

    Extracted from TendonRobotPlotter.update_tendons() and update_discs()
    so that multiple fingers can share a single PlotterBase.
    """

    def __init__(self, tendon_color_offset=0):
        self.tendon_meshes = []
        self.disc_transforms = []
        self.color_offset = tendon_color_offset

    def update_tendons(self, solution, plotter):
        num_tendons = solution.marginals.tendon_config.num_tendons
        num_discs = solution.marginals.tendon_config.num_discs

        if plotter.frame == 0:
            self.tendon_meshes = []
            for i in range(num_tendons):
                active_count = 0
                for ii in range(num_discs):
                    hole = solution.marginals.tendon_config.hole_locations[ii][i]
                    if hole is not None:
                        active_count += 1
                    else:
                        break
                if active_count >= 2:
                    points = np.zeros((active_count, 3))
                    mesh = pv.lines_from_points(points)
                    color_idx = (i + self.color_offset) % len(TENDON_COLORS)
                    plotter.plotter.add_mesh(
                        mesh, line_width=6, color=TENDON_COLORS[color_idx])
                    self.tendon_meshes.append((i, mesh, active_count))
                else:
                    self.tendon_meshes.append((i, None, 0))

        for tendon_idx, mesh, active_count in self.tendon_meshes:
            if mesh is None:
                continue
            points = []
            for ii in range(num_discs):
                hole = solution.marginals.tendon_config.hole_locations[ii][tendon_idx]
                if hole is None:
                    break
                disc_pose_idx = solution.marginals.tendon_config.disc_pose_idx[ii]
                T = solution.marginals.rod.states[disc_pose_idx].pose.mean
                p_world = T[:3, :3] @ hole + T[:3, 3]
                points.append(p_world)

            if len(points) == active_count:
                mesh.points[:] = points

    def update_discs(self, solution, plotter):
        num_discs = solution.marginals.tendon_config.num_discs
        disc_pose_idx = solution.marginals.tendon_config.disc_pose_idx

        if plotter.frame == 0:
            routing_radius = solution.marginals.tendon_config.routing_radius
            disc_radius = 1.3 * routing_radius
            disc_width = 0.3 * routing_radius
            hole_radius = 0.05 * routing_radius
            num_holes_per_disc = 8

            self.disc_transforms = []

            for i in range(num_discs):
                disc_transform = vtk.vtkTransform()
                self.disc_transforms.append(disc_transform)

                if i > 0:
                    mesh = pv.Cylinder(
                        direction=(0, 0, 1), radius=disc_radius,
                        height=disc_width, resolution=8)
                    actor = plotter.plotter.add_mesh(
                        mesh, color='cornflowerblue', opacity=0.2,
                        show_edges=True, line_width=3.0)
                    actor.SetUserTransform(disc_transform)

                for angle in np.linspace(0, 2 * np.pi, num_holes_per_disc, endpoint=False):
                    hole_location = np.array([
                        routing_radius * np.cos(angle),
                        routing_radius * np.sin(angle),
                        0.0,
                    ])
                    mesh = pv.Sphere(radius=hole_radius, center=hole_location)
                    actor = plotter.plotter.add_mesh(
                        mesh, color='black', opacity=0.5, lighting=False)
                    actor.SetUserTransform(disc_transform)

        for ii in range(num_discs):
            T = solution.marginals.rod.states[disc_pose_idx[ii]].pose.mean
            self.disc_transforms[ii].SetMatrix(T.flatten().tolist())

    def update(self, solution, plotter):
        self.update_tendons(solution, plotter)
        self.update_discs(solution, plotter)


class TendonHandPlotter:
    """Multi-finger hand plotter.

    Manages one CosseratRodMeshManager and one TendonMeshManager per finger,
    all sharing a single PlotterBase window.

    Since each finger's base_pose is set in the C++ solver config, all
    solution poses are already in world frame -- no extra transforms needed
    in the plotter.
    """

    ROD_COLORS = [
        'ultramarine', 'royalblue', 'steelblue', 'cadetblue',
        'darkcyan', 'teal',
    ]

    def __init__(self, finger_names,
                 camera_focal_point=None,
                 camera_azimuth=165,
                 camera_elevation=20,
                 camera_distance=0.8,
                 plot_backbone_ellipsoids=True,
                 plot_world_axes=True,
                 world_axes_scale=0.03,
                 plot_wrist_frame=True,
                 wrist_frame_scale=0.04,
                 primitives=None,
                 **kwargs):

        if camera_focal_point is None:
            camera_focal_point = [0, 0.08, 0]

        self.plotter = utils.PlotterBase(
            camera_focal_point=camera_focal_point,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
            **kwargs,
        )

        # Static scene objects (the grasp/collision object). Same spec format
        # as TendonFingerPlotter / get_primitive_specs()'s "plot" lambdas.
        # Added immediately: they never move, so no per-frame update needed.
        for spec in (primitives or []):
            mesh = utils.build_primitive_mesh(spec)
            self.plotter.plotter.add_mesh(
                mesh, color=spec.get("color", "goldenrod"),
                opacity=float(spec.get("opacity", 0.3)), smooth_shading=True)

        self.finger_names = list(finger_names)
        self.plot_world_axes = plot_world_axes
        self.world_axes_scale = world_axes_scale
        self._world_axes_added = False

        # Shared wrist base pose (world frame). The solver anchors every finger
        # to it via a per-finger offset, so drawing it shows where the whole
        # hand is rooted. Driven from outside through set_wrist_pose(); the
        # actors are created lazily on the first update() so a caller that
        # never commands a wrist pose pays nothing.
        self.plot_wrist_frame = plot_wrist_frame
        self.wrist_frame_scale = wrist_frame_scale
        self._wrist_pose = np.eye(4)
        self._wrist_transform = None

        self.rod_managers = {}
        self.tendon_managers = {}

        for i, name in enumerate(self.finger_names):
            self.rod_managers[name] = CosseratRodMeshManager(
                plot_backbone_ellipsoids=plot_backbone_ellipsoids,
                plot_wrenches=False,
                skip_backbone_ellipsoids=4,
                backbone_radius=0.001,
                cartesian_frame_scale=0.01,
                force_scale=0.05,
                moment_scale=0.2,
                rod_color=self.ROD_COLORS[i % len(self.ROD_COLORS)],
                plot_base_plate=False,
            )
            self.tendon_managers[name] = TendonMeshManager(
                tendon_color_offset=i * 6,
            )

    def _add_world_axes(self):
        """Add RGB axes arrows at the world origin (red=x, green=y, blue=z)."""
        s = self.world_axes_scale
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]
        directions = [np.array([1, 0, 0]),
                      np.array([0, 1, 0]),
                      np.array([0, 0, 1])]
        for d, c, label in zip(directions, colors, labels):
            arrow = pv.Arrow(start=(0, 0, 0), direction=d, scale=s,
                             shaft_radius=0.02, tip_radius=0.05)
            self.plotter.plotter.add_mesh(arrow, color=c, lighting=False)
            self.plotter.plotter.add_point_labels(
                [d * s * 1.15], [label], font_size=14, text_color=c,
                shape=None, show_points=False)
        self._world_axes_added = True

    def _add_wrist_frame(self):
        """Add an RGB axes triad driven by a vtkTransform for the wrist pose."""
        self._wrist_transform = vtk.vtkTransform()
        axes = utils.get_axes_frame(length=self.wrist_frame_scale)
        for arrow, color in zip(axes, utils.frame_arrow_colors):
            actor = self.plotter.plotter.add_mesh(
                arrow, color=color, lighting=False)
            actor.SetUserTransform(self._wrist_transform)

    def set_wrist_pose(self, wrist_pose):
        """Command the wrist base pose (4x4 world-frame transform) to draw.

        Call before update(); the triad is re-posed on the next render.
        """
        self._wrist_pose = np.asarray(wrist_pose, dtype=float)

    def update(self, solutions_dict):
        """Update all fingers and render.

        Parameters
        ----------
        solutions_dict : dict
            Maps finger name (str) to TendonRobotSolution.
        """
        for name in self.finger_names:
            if name not in solutions_dict:
                continue
            solution = solutions_dict[name]
            self.rod_managers[name].update(
                solution.marginals.rod, self.plotter)
            self.tendon_managers[name].update(solution, self.plotter)

        if self.plot_world_axes and not self._world_axes_added:
            self._add_world_axes()

        if self.plot_wrist_frame:
            if self._wrist_transform is None:
                self._add_wrist_frame()
            self._wrist_transform.SetMatrix(self._wrist_pose.flatten().tolist())

        first_solution = next(iter(solutions_dict.values()))
        self.plotter.update(first_solution)


class TendonHandMultiViewPlotter:
    """N simultaneous TendonHandPlotter windows, one per camera view.

    Each view gets its own OS window (pyvista render windows cannot share
    actors), tiled in a 2-column grid. ``views`` is a list of
    (azimuth_deg, elevation_deg) pairs; every other constructor kwarg is
    forwarded to each TendonHandPlotter unchanged. ``update()`` fans the same
    solutions out to every window.

    Default views: three azimuths 90 deg apart at working elevation, plus one
    near-top-down — enough to disambiguate finger/object interpenetration.
    """

    DEFAULT_VIEWS = [(165, 20), (255, 20), (345, 20), (165, 70)]

    def __init__(self, finger_names, views=None,
                 window_size=(700, 700), window_gap=40, **kwargs):
        views = list(views) if views is not None else list(self.DEFAULT_VIEWS)

        self.plotters = []
        for i, (azimuth, elevation) in enumerate(views):
            p = TendonHandPlotter(
                finger_names,
                camera_azimuth=azimuth,
                camera_elevation=elevation,
                window_size=window_size,
                **kwargs,
            )
            # Tile 2 columns x N rows so the windows don't stack exactly on
            # top of each other. Position is best-effort (window managers may
            # override); ignore backends that don't expose the render window.
            try:
                col, row = i % 2, i // 2
                p.plotter.plotter.ren_win.SetPosition(
                    col * (window_size[0] + window_gap),
                    row * (window_size[1] + window_gap))
            except AttributeError:
                pass
            self.plotters.append(p)

    def update(self, solutions_dict):
        for p in self.plotters:
            p.update(solutions_dict)
