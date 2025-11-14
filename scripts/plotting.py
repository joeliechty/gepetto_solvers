import numpy as np
from pathlib import Path
import shutil

import pyvista as pv


frame_arrow_colors = ["red", "green", "blue"]


def get_tube_from_points(points, radius):
    spline = pv.Spline(points, n_points=200)
    tube = spline.tube(radius=radius)

    return tube


def get_tube_from_poses(poses, radius):
    points = np.array([T[:3, 3] for T in poses])
    return get_tube_from_points(points, radius)


def transform_ellipsoid(ellipsoid, center, cov, scale=1.0, num_sigma=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    one_sigma = np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    radii = num_sigma * one_sigma

    return (eigvecs @ np.diag(radii) @ ellipsoid.points.T).T + center


def get_arrow(start, vec, shaft_radius=0.003):

    tip_radius = 2 * shaft_radius
    tip_length = 5 * shaft_radius

    scale = np.linalg.norm(vec)

    if scale < 1e-8:
        scale = 1e-8
        vec = vec + scale
    
    arrow = pv.Arrow(
        start=start,
        direction=vec,
        scale='auto',
        shaft_radius=shaft_radius/scale,
        tip_radius=tip_radius/scale,
        tip_length=tip_length/scale,
        tip_resolution=20,
        shaft_resolution=20,
    )

    return arrow


class PlotterBase:
    def __init__(self,
                 save_frames_dir_name=None, 
                 single_plot_mode=False,
                 camera_focal_point=None,
                 camera_azimuth=15,
                 camera_elevation=20,
                 camera_distance=0.6):
        
        self.save_frames_dir_name = save_frames_dir_name
        self.single_plot_mode = single_plot_mode

        if camera_focal_point is None:
            self.camera_focal_point = np.zeros(3)
        else:
            self.camera_focal_point = camera_focal_point
        
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_distance = camera_distance

        if self.save_frames_dir_name:
            self.frames_path = Path("videos") / "frames" / self.save_frames_dir_name
            shutil.rmtree(self.frames_path, ignore_errors=True)
            self.frames_path.mkdir(parents=True, exist_ok=True)

        self.window_size = (2000, 2000)
        self.plotter = pv.Plotter(window_size=self.window_size, off_screen=save_frames_dir_name)
        self.frame = 0
        self.solve_time_ms_history = []

        self.init_scene()

    def init_scene(self):
        p = self.camera_focal_point
        a = np.deg2rad(self.camera_azimuth)
        e = np.deg2rad(self.camera_elevation)
        d = self.camera_distance

        x = p[0] + d * np.cos(e) * np.cos(a)
        y = p[1] + d * np.cos(e) * np.sin(a)
        z = p[2] + d * np.sin(e)

        self.plotter.camera.position = (x, y, z)
        self.plotter.camera.focal_point = self.camera_focal_point

        self.plotter.add_axes()
        # self.plotter.enable_depth_peeling(10)
        self.plotter.enable_anti_aliasing()
    
    def update(self, solution):

        if self.frame == 0:
            show_plot = not self.save_frames_dir_name
            if show_plot:
                interactive_update = not self.single_plot_mode
                self.plotter.show(auto_close=False, interactive_update=interactive_update)

        self.solve_time_ms_history.append(solution.meta.total_time_ms)

        text = (
            f"iter: {solution.meta.iterations:3d}, "
            f"error: {solution.meta.error:3.2e}, "
            f"build: {solution.meta.build_time_ms:3.2f} ms, "
            f"optimize: {solution.meta.optimize_time_ms:3.2f} ms, "
            f"marginalize: {solution.meta.marginalize_time_ms:3.2f} ms, "
            f"extract: {solution.meta.extract_time_ms:3.2f} ms, "
            f"total: {solution.meta.total_time_ms:3.2f} ms, "
            f"avg: {np.mean(self.solve_time_ms_history):3.2f} ms"
        )
        
        self.plotter.add_text(text, position='upper_right', font_size=14, font="courier", name="solve_time")

        self.plotter.render()

        if self.save_frames_dir_name:
            self.plotter.screenshot(self.frames_path / f"{self.frame}.png", window_size=self.window_size)
        
        self.frame += 1


class CosseratRodMeshManager:
    def __init__(self,
                 plot_base_plate=True,
                 plot_tip_plate=False,
                 plot_wrenches=True,
                 plot_internal_wrenches=False,
                 plot_base_wrench=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 backbone_radius=0.005, 
                 moment_scale = 0.2, 
                 force_scale=0.1, 
                 base_plate_size=0.1, 
                 cartesian_frame_scale=0.03):
        
        self.plot_base_plate = plot_base_plate
        self.plot_tip_plate = plot_tip_plate
        self.plot_wrenches = plot_wrenches
        self.plot_internal_wrenches = plot_internal_wrenches
        self.plot_base_wrench = plot_base_wrench
        self.plot_backbone_frames = plot_backbone_frames
        self.plot_backbone_ellipsoids = plot_backbone_ellipsoids

        self.backbone_radius = backbone_radius
        self.moment_scale = moment_scale
        self.force_scale = force_scale
        self.base_plate_size = base_plate_size
        self.cartesian_frame_scale = cartesian_frame_scale

    def get_end_plate(self):
        thick = self.base_plate_size / 10.0
        plate = pv.Cube(
            center=(0, -thick / 2, 0), 
            x_length=self.base_plate_size, 
            y_length=self.base_plate_size, 
            z_length=thick
        )

        return plate

    def update_base_plate(self, solution, plotter):
        if not self.plot_base_plate:
            return
        
        if plotter.frame == 0:
            self.base_plate_ref = self.get_end_plate()
            self.base_plate = self.get_end_plate()
            plotter.plotter.add_mesh(self.base_plate, color="silver", show_edges=True, line_width=2, opacity=0.7)
            return
        
        pose = solution.pose_mean[0]
        self.base_plate.points = (self.base_plate_ref.points @ pose[:3, :3].T) + pose[:3, 3]
    
    def update_tip_plate(self, solution, plotter):
        if not self.plot_tip_plate:
            return
        
        if plotter.frame == 0:
            self.tip_plate_ref = self.get_end_plate()
            self.tip_plate = self.get_end_plate()
            plotter.plotter.add_mesh(self.tip_plate, color="silver", show_edges=True, line_width=2, opacity=0.7)
            return
        
        pose = solution.pose_mean[-1]
        self.tip_plate.points = (self.tip_plate_ref.points @ pose[:3, :3].T) + pose[:3, 3]
    
    def update_rod_tube(self, solution, plotter):
        tube = get_tube_from_poses(solution.pose_mean, radius=self.backbone_radius)
        
        if plotter.frame == 0:
            self.backbone_tube_mesh = tube
            plotter.plotter.add_mesh(self.backbone_tube_mesh, color='ultramarine', opacity = 0.5)
            return
        
        self.backbone_tube_mesh.shallow_copy(tube)

    def update_backbone_ellipsoids(self, solution, plotter):
        if not self.plot_backbone_ellipsoids:
            return

        if plotter.frame == 0:
            self.backbone_ellipsoid_ref = pv.Sphere(radius=1)
            self.backbone_ellipsoid_meshes = [pv.Sphere(radius=1) for _ in range(len(solution.pose_mean))]
            for ellipsoid in self.backbone_ellipsoid_meshes:
                plotter.plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)
        
        for ellipsoid, pose, cov in zip(self.backbone_ellipsoid_meshes, solution.pose_mean, solution.pose_cov):
            R = pose[:3, :3]
            p = pose[:3, 3]
            cov = R @ (cov[3:, 3:] @ R.T)  # World frame

            ellipsoid.points = transform_ellipsoid(self.backbone_ellipsoid_ref, p, cov)


    def update_backbone_frames(self, solution, plotter):
        if not self.plot_backbone_frames:
            return

        frames = []
        shaft_radius = 0.001

        for pose in solution.pose_mean:
            R = pose[:3, :3]
            p = pose[:3, 3]

            frames.append([
                get_arrow(p, self.cartesian_frame_scale * R[:,0], shaft_radius=shaft_radius),
                get_arrow(p, self.cartesian_frame_scale * R[:,1], shaft_radius=shaft_radius),
                get_arrow(p, self.cartesian_frame_scale * R[:,2], shaft_radius=shaft_radius)
            ])

        if plotter.frame == 0:
            self.backbone_frame_meshes = frames
            for frame in self.backbone_frame_meshes:
                for arrow, color in zip(frame, frame_arrow_colors):
                    plotter.plotter.add_mesh(arrow, color=color, lighting=False, opacity=0.4)

            return
        
        for frame_self, frame_new in zip(self.backbone_frame_meshes, frames):
            for mesh_self, mesh_new in zip(frame_self, frame_new):
                mesh_self.shallow_copy(mesh_new)

        return frames
    
    def get_wrench_meshes(self, solution):
        poses = solution.pose_mean
        wrenches = solution.wrench_mean
        covs = solution.wrench_cov

        if not self.plot_internal_wrenches:
            poses = [poses[0], poses[-1]]
            wrenches = [wrenches[0], wrenches[-1]]
            covs = [covs[0], covs[-1]]

        if not self.plot_base_wrench and len(poses) > 1:
            poses = poses[1:]
            wrenches = wrenches[1:]
            covs = covs[1:]

        moment_arrows, moment_ellipsoids = [], []
        force_arrows, force_ellipsoids = [], []

        for pose, wrench, wrench_cov in zip(poses, wrenches, covs):
            p = pose[:3, 3]

            moment_mean, force_mean = wrench[:3], wrench[3:]
            moment_cov, force_cov = wrench_cov[:3, :3], wrench_cov[3:, 3:]

            moment_arrow = get_arrow(p, self.moment_scale * moment_mean)
            force_arrow = get_arrow(p, self.force_scale * force_mean)

            moment_ellipsoid = get_ellipsoid(
                p + self.moment_scale * moment_mean,
                moment_cov,
                self.moment_scale,
            )

            force_ellipsoid = get_ellipsoid(
                p + self.force_scale * force_mean,
                force_cov,
                self.force_scale,
            )

            moment_arrows.append(moment_arrow)
            moment_ellipsoids.append(moment_ellipsoid)
            force_arrows.append(force_arrow)
            force_ellipsoids.append(force_ellipsoid)

        return moment_arrows, moment_ellipsoids, force_arrows, force_ellipsoids

    def update_wrenches(self, solution, plotter):
        if not self.plot_wrenches:
            return 
    
        moment_arrows, moment_ellipsoids, force_arrows, force_ellipsoids = self.get_wrench_meshes(solution)

        if plotter.frame == 0:
            self.moment_arrow_meshes = moment_arrows
            for arrow in self.moment_arrow_meshes:
                plotter.plotter.add_mesh(arrow, color='deeppink', lighting=False)

            self.moment_ellipsoid_meshes = moment_ellipsoids
            for ellipsoid in self.moment_ellipsoid_meshes:
                plotter.plotter.add_mesh(ellipsoid, color="cadmiumlemon", lighting=False, opacity=0.4)

            self.force_arrow_meshes = force_arrows
            for arrow in self.force_arrow_meshes:
                plotter.plotter.add_mesh(arrow, color='darkorchid', lighting=False)

            self.force_ellipsoid_meshes = force_ellipsoids
            for ellipsoid in self.force_ellipsoid_meshes:
                plotter.plotter.add_mesh(ellipsoid, color="cadmiumlemon", lighting=False, opacity=0.4)

            return
        
        for mesh_self, mesh_new in zip(self.moment_arrow_meshes, moment_arrows):
            mesh_self.shallow_copy(mesh_new)
        
        for mesh_self, mesh_new in zip(self.moment_ellipsoid_meshes, moment_ellipsoids):
            mesh_self.shallow_copy(mesh_new)

        for mesh_self, mesh_new in zip(self.force_arrow_meshes, force_arrows):
            mesh_self.shallow_copy(mesh_new)
        
        for mesh_self, mesh_new in zip(self.force_ellipsoid_meshes, force_ellipsoids):
            mesh_self.shallow_copy(mesh_new)

    def update(self, solution, plotter):
        self.update_base_plate(solution, plotter)
        self.update_tip_plate(solution, plotter)
        self.update_rod_tube(solution, plotter)
        self.update_backbone_ellipsoids(solution, plotter)
        self.update_wrenches(solution, plotter)
        self.update_backbone_frames(solution, plotter)


class CosseratRodPlotter:
    def __init__(self,
                 plot_base_plate=True,
                 plot_tip_plate=False,
                 plot_wrenches=True,
                 plot_internal_wrenches=False,
                 plot_base_wrench=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 backbone_radius=0.005, 
                 moment_scale = 0.2, 
                 force_scale=0.1, 
                 base_plate_size=0.1, 
                 cartesian_frame_scale=0.03,
                 **kwargs):
    
        self.plotter = PlotterBase(**kwargs)
        self.mesh_manager = CosseratRodMeshManager(
            plot_base_plate=plot_base_plate,
            plot_tip_plate=plot_tip_plate,
            plot_wrenches=plot_wrenches,
            plot_internal_wrenches=plot_internal_wrenches,
            plot_base_wrench=plot_base_wrench,
            plot_backbone_frames=plot_backbone_frames,
            plot_backbone_ellipsoids=plot_backbone_ellipsoids,
            backbone_radius=backbone_radius,
            moment_scale = moment_scale,
            force_scale=force_scale,
            base_plate_size=base_plate_size,
            cartesian_frame_scale=cartesian_frame_scale
        )
            
    def update(self, solution):
        self.mesh_manager.update(solution.marginals, self.plotter)
        self.plotter.update(solution)


class ParallelRobotPlotter:
    def __init__(self,
                 platform_z_offset=0.0,
                 plot_rod_wrenches=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 **kwargs):

        self.plotter = PlotterBase(**kwargs)

        self.rod_managers = []

        self.moment_scale = 0.2
        self.force_scale = 0.1
        self.platform_z_offset = platform_z_offset

        for _ in range(6):
            self.rod_managers.append(CosseratRodMeshManager(
                plot_base_plate=False,
                plot_wrenches=plot_rod_wrenches,
                plot_base_wrench=plot_base_wrenches,
                plot_backbone_frames=plot_backbone_frames,
                plot_backbone_ellipsoids=plot_backbone_ellipsoids,
                backbone_radius=0.005,
                moment_scale=self.moment_scale, 
                force_scale=self.force_scale, 
                base_plate_size=0.1, 
                cartesian_frame_scale=0.03
                )
            )

        base_plate = pv.Cylinder(direction=(0,0,1), radius=0.35, height=0.02)
        self.plotter.plotter.add_mesh(base_plate, color="silver", show_edges=True, line_width=2, opacity=0.3)
        
  
    def update_platform(self, solution, plotter):

        if plotter.frame == 0:
            self.platform_plate_ref = pv.Cylinder(direction=(0,0,1), radius=0.2, height=0.01)
            self.platform_plate = self.platform_plate_ref.copy()

            self.platform_cylinder_ref = pv.Cylinder(direction=(0,0,1), radius=0.005, height=np.abs(self.platform_z_offset))
            self.platform_cylinder = self.platform_cylinder_ref.copy()

            frame_scale = 0.1
            shaft_radius = 0.005

            # self.platform_frame_ref = [
            #     get_arrow(p, frame_scale * R[:,0], shaft_radius=shaft_radius),
            #     get_arrow(p, frame_scale * R[:,1], shaft_radius=shaft_radius),
            #     get_arrow(p, frame_scale * R[:,2], shaft_radius=shaft_radius)
            # ]

            # self.platform_frame = [arrow.copy() for arrow in self.platform_frame_ref]

            self.platform_ellipsoid_ref = pv.Sphere(radius=1)
            self.platform_ellipsoid = self.platform_ellipsoid_ref.copy()

            plotter.plotter.add_mesh(self.platform_plate, color="silver", show_edges=True, line_width=2, opacity=0.3)
            plotter.plotter.add_mesh(self.platform_cylinder, color="silver")
            plotter.plotter.add_mesh(self.platform_ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)

            # for arrow, color in zip(self.platform_frame, frame_arrow_colors):
            #     plotter.plotter.add_mesh(arrow, color=color)

        p = solution.platform_pose_mean[:3,3]
        R = solution.platform_pose_mean[:3,:3]

        self.platform_plate.points = self.platform_plate_ref.points.copy()
        self.platform_plate.points[:,2] += self.platform_z_offset
        self.platform_plate.points = (self.platform_plate.points @ R.T) + p

        self.platform_cylinder.points = self.platform_cylinder_ref.points.copy()
        self.platform_cylinder.points[:,2] += self.platform_z_offset / 2
        self.platform_cylinder.points = (self.platform_cylinder.points @ R.T) + p

        cov = solution.platform_pose_cov
        cov = R @ (cov[3:, 3:] @ R.T)

        self.platform_ellipsoid.points = transform_ellipsoid(self.platform_ellipsoid_ref, p, cov)

    def update_platform_wrench(self, solution, plotter):
        wrench = solution.platform_wrench_mean
        moment_mean, force_mean = wrench[:3], wrench[3:]
            
        wrench_cov = solution.platform_wrench_cov
        moment_cov, force_cov = wrench_cov[:3, :3], wrench_cov[3:, 3:]

        p = solution.platform_pose_mean[:3,3]

        moment_arrow = get_arrow(p, self.moment_scale * moment_mean)
        force_arrow = get_arrow(p, self.force_scale * force_mean)

        if plotter.frame == 0:
            self.force_arrow = force_arrow
            self.moment_arrow = moment_arrow

            self.force_ellipsoid_ref = pv.Sphere(radius=1)
            self.force_ellipsoid = self.force_ellipsoid_ref.copy()

            self.moment_ellipsoid_ref = pv.Sphere(radius=1)
            self.moment_ellipsoid = self.moment_ellipsoid_ref.copy()

            plotter.plotter.add_mesh(self.force_arrow, color='darkorchid', lighting=False)
            plotter.plotter.add_mesh(self.moment_arrow, color='deeppink', lighting=False)
            plotter.plotter.add_mesh(self.force_ellipsoid, color="cadmiumlemon", lighting=False, opacity=0.4)
            plotter.plotter.add_mesh(self.moment_ellipsoid, color="cadmiumlemon", lighting=False, opacity=0.4)


        self.moment_ellipsoid.points = transform_ellipsoid(
            self.moment_ellipsoid_ref, 
            p + self.moment_scale * moment_mean,
            moment_cov,
            self.moment_scale,
        )

        self.force_ellipsoid.points = transform_ellipsoid(
            self.force_ellipsoid_ref,
            p + self.force_scale * force_mean,
            force_cov,
            self.force_scale,
        )

        # TODO .points
        self.force_arrow.shallow_copy(force_arrow)
        self.moment_arrow.shallow_copy(moment_arrow)

    def update(self, solution):
        for i, manager in enumerate(self.rod_managers):
            manager.update(solution.marginals.rods[i], self.plotter)
        
        self.update_platform(solution.marginals, self.plotter)
        self.update_platform_wrench(solution.marginals, self.plotter)

        self.plotter.update(solution)

# from pathlib import Path
# import shutil

# def get_tendon_disc_meshes(solution):
#     num_discs = solution.tendon_disc_config.num_discs
#     num_tendons = solution.tendon_disc_config.num_tendons
#     routing_radius = solution.tendon_disc_config.routing_radius
#     local_holes = solution.tendon_disc_config.local_holes # num_discs, num_tendons, 3
#     disc_pose_idx = solution.tendon_disc_config.disc_pose_idx

#     disc_radius = 1.3 * routing_radius
#     disc_width = 0.3 * routing_radius
#     tendon_radius = 0.03 * routing_radius

#     discs = []

#     for i in disc_pose_idx:
#         T = solution.backbone_pose_mean[i]
#         cylinder = pv.Cylinder(direction=(0,0,1), radius=disc_radius, height=disc_width, resolution=8)
#         cylinder.points = (T[:3,:3] @ cylinder.points.T + T[:3,3].reshape((3,1))).T
#         discs.append(cylinder)

#     tendons = []
#     for jj in range(num_tendons):
#         # collect all points along this tendon
#         points = []
#         for ii in range(num_discs):
#             T = solution.backbone_pose_mean[disc_pose_idx[ii]]
#             p_world = T[:3, :3] @ local_holes[ii][jj] + T[:3, 3]
#             points.append(p_world)
        
#         line = pv.lines_from_points(points)
#         tendon = line.tube(radius=tendon_radius)
#         tendons.append(tendon)

#     angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
#     x = routing_radius * np.cos(angles)
#     y = routing_radius * np.sin(angles)
#     z = np.zeros_like(x)
#     hole_locations = np.array((x, y, z)).T

#     holes = []
#     hole_radius = 2 * tendon_radius
#     for idx in disc_pose_idx:
#         T = solution.backbone_pose_mean[idx]
#         for loc in hole_locations:
#             loc_world = T[:3,:3] @ loc + T[:3,3]
#             if idx == 0: loc_world[1] += hole_radius 
#             hole = pv.Sphere(radius=hole_radius, center=loc_world)
#             holes.append(hole)
    
#     return tendons, discs, holes


# def get_largest_norm(f_samples, f_gt, f_mean):

#     norms = []

#     if f_samples is not None:
#         norms.append(np.max(np.linalg.norm(f_samples, axis=1)))

#     for f in [f_gt, f_mean]:
#         if f is not None:
#             norms.append(np.linalg.norm(f))

#     max_norm = max(norms) if norms else 1.0  # Fallback value to avoid div-by-zero

#     return max_norm
    

# class TendonRobotPlotter:
#     def __init__(self, 
#                  title, 
#                  save_frames_mode=False,
#                  single_plot_mode=False,
#                  plot_tip_force=False, 
#                  plot_dist_load=False,
#                  plot_backbone_ellipsoids=True,
#                  waypoints=None, 
#                  cylinders=None, 
#                  azimuth=15,
#                  camera_distance=0.6,
#                  focal_point_y=0.12):
        
#         self.save_frames_mode = save_frames_mode
#         self.single_plot_mode = single_plot_mode
#         self.plot_tip_force = plot_tip_force
#         self.plot_dist_load = plot_dist_load
#         self.plot_backbone_ellipsoids = plot_backbone_ellipsoids

#         self.cylinders = cylinders
#         self.waypoints = waypoints

#         self.azimuth = azimuth
#         self.camera_distance = camera_distance
#         self.focal_point_y = focal_point_y

#         if save_frames_mode:
#             dir_name = title.strip().lower().replace(" ", "_")
#             self.frames_path = Path("videos") / "frames" / dir_name
#             shutil.rmtree(self.frames_path, ignore_errors=True)
#             self.frames_path.mkdir(parents=True, exist_ok=True)

#         self.window_size = (2000, 2000)
#         self.plotter = pv.Plotter(window_size=self.window_size, off_screen=save_frames_mode)
#         self.frame = 0
#         self.solve_time_ms_history = []
            
#     def init_scene(self, solution):
#         plate = get_base_plate(solution)
#         self.plotter.add_mesh(plate, color="silver", show_edges=True, line_width=2)
        
#         if self.waypoints is not None:
#             for point in self.waypoints:
#                 mesh = pv.Sphere(0.0015, center=point)
#                 self.plotter.add_mesh(mesh, color="red")
        
#         if self.cylinders is not None:
#             for cylinder in self.cylinders:
#                 mesh = pv.Cylinder(cylinder['center'], cylinder['z'], cylinder['radius'], cylinder['length'])
#                 self.plotter.add_mesh(mesh, smooth_shading=True, color='cadmiumyellow')

#         focal_point = np.array([0.0, self.focal_point_y, 0])
#         elevation = 15

#         az = np.deg2rad(self.azimuth)
#         el = np.deg2rad(elevation)

#         x = focal_point[0] + self.camera_distance * np.cos(el) * np.cos(az)
#         y = focal_point[1] + self.camera_distance * np.cos(el) * np.sin(az)
#         z = focal_point[2] + self.camera_distance * np.sin(el)

#         self.plotter.camera.position = (x, y, z)
#         self.plotter.camera.focal_point = focal_point

#         self.plotter.add_light(pv.Light(position=(1.0, 0.7, 0.5), intensity=0.5, light_type='scene light'))
#         self.plotter.add_light(pv.Light(position=(0.7, -1.0, 0.5), intensity=0.2, light_type='scene light'))
#         self.plotter.add_light(pv.Light(position=(-1.0, -1.0, 0.5), intensity=0.2, light_type='scene light'))

#         # self.plotter.add_axes()
#         self.plotter.enable_depth_peeling(10)
#         self.plotter.enable_anti_aliasing()

#         if not self.save_frames_mode:
#             interactive_update = not self.single_plot_mode
#             # interactive_update=True
#             self.plotter.show(auto_close=False, interactive_update=interactive_update)
    
#     def update(self, solution, p_desired=None, tip_force_gt=None):

#         backbone_radius = 0.1 * solution.tendon_disc_config.routing_radius
#         backbone = get_tube_poses(solution.backbone_pose_mean, radius=backbone_radius)
#         tendons, discs, holes = get_tendon_disc_meshes(solution)
#         backbone_ellipsoids = get_backbone_ellipsoids(solution)

#         if self.plot_tip_force:
#             tip_force_mean_mesh, tip_force_2_sigma_mesh, tip_force_gt_mesh = get_tip_force_meshes(solution, tip_force_gt)
        
#         if self.plot_dist_load:
#             dist_load_meshes = get_dist_load_meshes(solution)

#         if p_desired is not None:
#             p_desired_mesh = pv.Sphere(0.002, p_desired)

#         if self.frame == 0:
#             self.backbone_mesh = backbone
#             self.plotter.add_mesh(self.backbone_mesh, color='ultramarine', opacity = 0.7)

#             self.tendon_meshes = tendons
#             self.disc_meshes = discs
#             tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]

#             for i, disc in enumerate(self.disc_meshes):
#                 if i == 0: continue
#                 disc.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True, inplace=True)
#                 self.plotter.add_mesh(disc, color='cornflowerblue', opacity=0.2, show_edges=True, line_width=3.0)
            
#             for j, tendon in enumerate(self.tendon_meshes):
#                 color = tendon_colors[j]
#                 self.plotter.add_mesh(tendon, color=color)

#             self.hole_meshes = holes
#             for i, hole in enumerate(self.hole_meshes):
#                 opacity = 0.5 if i < 8 else 0.2
#                 self.plotter.add_mesh(hole, color='black', opacity=opacity, lighting=False)
            
#             if self.plot_backbone_ellipsoids:
#                 self.backbone_2_sigma_meshes = backbone_ellipsoids
#                 for ellipsoid in self.backbone_2_sigma_meshes:
#                     self.plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)

#             if self.plot_tip_force:
#                 self.tip_force_mean_mesh = tip_force_mean_mesh
#                 self.plotter.add_mesh(self.tip_force_mean_mesh, color='darkorchid', lighting=False)

#                 self.tip_force_2_sigma_mesh = tip_force_2_sigma_mesh
#                 self.plotter.add_mesh(self.tip_force_2_sigma_mesh, color="cadmiumlemon", lighting=False, opacity=0.4)

#                 if tip_force_gt is not None:
#                     self.tip_force_gt_mesh = tip_force_gt_mesh
#                     self.plotter.add_mesh(self.tip_force_gt_mesh, color='limegreen', lighting=False)
            
#             if self.plot_dist_load:
#                 self.dist_load_meshes = dist_load_meshes
#                 for mesh in self.dist_load_meshes:
#                     self.plotter.add_mesh(mesh, color="darkorchid", lighting=False)

#             if p_desired is not None:
#                 self.p_desired_mesh = p_desired_mesh
#                 self.plotter.add_mesh(self.p_desired_mesh, color="red")

#             self.init_scene(solution)
#         else:
#             self.backbone_mesh.shallow_copy(backbone)

#             for i, (new_disc, disc) in enumerate(zip(discs, self.disc_meshes)):
#                 if i == 0: continue
#                 disc.shallow_copy(new_disc)
            
#             for new_tendon, tendon in zip(tendons, self.tendon_meshes):
#                 tendon.shallow_copy(new_tendon)

#             for new_hole, hole in zip(holes, self.hole_meshes):
#                 hole.shallow_copy(new_hole)

#             if self.plot_backbone_ellipsoids:
#                 for mesh_self, mesh in zip(self.backbone_2_sigma_meshes, backbone_ellipsoids):
#                     mesh_self.shallow_copy(mesh)

#             if self.plot_tip_force:
#                 self.tip_force_mean_mesh.shallow_copy(tip_force_mean_mesh)
#                 self.tip_force_2_sigma_mesh.shallow_copy(tip_force_2_sigma_mesh)
#                 if tip_force_gt is not None:
#                     self.tip_force_gt_mesh.shallow_copy(tip_force_gt_mesh)
            
#             if self.plot_dist_load:
#                 for (mesh_self, mesh) in zip(self.dist_load_meshes, dist_load_meshes):
#                     mesh_self.shallow_copy(mesh)
           
#             if p_desired is not None:
#                 self.p_desired_mesh.shallow_copy(p_desired_mesh)

#         self.solve_time_ms_history.append(solution.total_time_ms)
#         text = f"solve time: {solution.total_time_ms:.2f} ms, average: {np.mean(self.solve_time_ms_history):.2f} ms"
#         self.plotter.add_text(text, position='upper_right', font_size=14, font="courier", name="solve_time")

#         self.plotter.render()

#         if self.save_frames_mode:
#             self.plotter.screenshot(self.frames_path / f"{self.frame}.png", window_size=self.window_size)
        
#         self.frame += 1