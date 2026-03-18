import numpy as np
import pyvista as pv
import vtk

from . import utils
from .cosserat_rod_plotter import CosseratRodMeshManager


TENDON_COLORS = [
    "crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod",
    "deeppink", "darkorange", "teal", "sienna", "slategray",
]


class CollisionSphereMeshManager:
    """Manages collision sphere meshes for all fingers."""

    def __init__(self, collision_radius=0.005, sphere_color="limegreen", sphere_opacity=0.3):
        self.collision_radius = collision_radius
        self.sphere_color = sphere_color
        self.sphere_opacity = sphere_opacity
        self.sphere_transforms = {}  # finger_name -> list of vtkTransform

    def update(self, finger_name, solution, plotter):
        """Update collision spheres for a single finger."""
        num_nodes = len(solution.marginals.rod.states)

        if plotter.frame == 0:
            self.sphere_transforms[finger_name] = []
            for _ in range(num_nodes):
                transform = vtk.vtkTransform()
                sphere = pv.Sphere(radius=self.collision_radius)
                actor = plotter.plotter.add_mesh(
                    sphere,
                    color=self.sphere_color,
                    opacity=self.sphere_opacity,
                    lighting=True,
                )
                actor.SetUserTransform(transform)
                self.sphere_transforms[finger_name].append(transform)

        # Update transforms for each node
        transforms = self.sphere_transforms[finger_name]
        for i, state in enumerate(solution.marginals.rod.states):
            pose = state.pose.mean
            # Create a transform that positions the sphere at the node location
            T = np.eye(4)
            T[:3, 3] = pose[:3, 3]
            transforms[i].SetMatrix(T.flatten().tolist())


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
                 plot_collision_spheres=False,
                 collision_sphere_radius=0.005,
                 collision_sphere_color="limegreen",
                 collision_sphere_opacity=0.3,
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

        self.finger_names = list(finger_names)
        self.plot_world_axes = plot_world_axes
        self.world_axes_scale = world_axes_scale
        self._world_axes_added = False

        self.plot_collision_spheres = plot_collision_spheres
        self.collision_sphere_manager = CollisionSphereMeshManager(
            collision_radius=collision_sphere_radius,
            sphere_color=collision_sphere_color,
            sphere_opacity=collision_sphere_opacity,
        ) if plot_collision_spheres else None

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

            if self.collision_sphere_manager is not None:
                self.collision_sphere_manager.update(name, solution, self.plotter)

        if self.plot_world_axes and not self._world_axes_added:
            self._add_world_axes()

        first_solution = next(iter(solutions_dict.values()))
        self.plotter.update(first_solution)
