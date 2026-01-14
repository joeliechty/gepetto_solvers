import numpy as np

from . import utils
from .cosserat_rod_plotter import CosseratRodMeshManager

class TendonRobotPlotter:
    def __init__(self,
                 plot_rod_wrenches=False,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=False,
                 **kwargs):

        self.plotter = utils.PlotterBase(**kwargs)

        self.rod_manager = CosseratRodMeshManager(
            plot_backbone_ellipsoids=plot_backbone_ellipsoids,
            plot_wrenches=plot_rod_wrenches,
            plot_base_wrench=plot_base_wrenches,
            plot_backbone_frames=plot_backbone_frames,
            backbone_radius=0.005,
            cartesian_frame_scale=0.01,
            force_scale=0.1,
            moment_scale=0.2
        )
  
    # def update_platform(self, solution, plotter):

    #     if plotter.frame == 0:
    #         mesh = pv.Cylinder(direction=(0,0,1), radius=0.2, height=0.01)
    #         mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset])
    #         actor = plotter.plotter.add_mesh(mesh, color="silver", show_edges=True, line_width=2, opacity=0.3)
    #         self.platform_transform = vtk.vtkTransform()
    #         actor.SetUserTransform(self.platform_transform)

    #         mesh = pv.Cylinder(direction=(0,0,1), radius=0.005, height=np.abs(self.platform_z_offset))
    #         mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset / 2])
    #         actor = plotter.plotter.add_mesh(mesh, color="silver")
    #         actor.SetUserTransform(self.platform_transform)

    #         axes = utils.get_axes_frame(length=0.1)
    #         for arrow, color in zip(axes, utils.frame_arrow_colors):
    #             actor = plotter.plotter.add_mesh(arrow, color=color)
    #             actor.SetUserTransform(self.platform_transform)
            
    #         mesh = pv.Sphere(radius=1)
    #         actor = plotter.plotter.add_mesh(mesh, color="deepcadmiumred", lighting=False, opacity=0.2)
    #         self.platform_ellipsoid_transform = vtk.vtkTransform()
    #         actor.SetUserTransform(self.platform_ellipsoid_transform)

    #     pose = solution.platform_pose_mean
    #     self.platform_transform.SetMatrix(pose.flatten().tolist())
        
    #     p = pose[:3,3]
    #     R = pose[:3,:3]
    #     cov = solution.platform_pose_cov
    #     cov = R @ (cov[3:, 3:] @ R.T)
    #     T = utils.get_ellipsoid_transform(p, cov)
    #     self.platform_ellipsoid_transform.SetMatrix(T.flatten().tolist())

    # def update_platform_wrench(self, solution, plotter):
    #     if plotter.frame == 0:
    #         mesh = utils.get_arrow(shaft_scale=0.2)
    #         actor = plotter.plotter.add_mesh(mesh, color='deeppink', lighting=False)
    #         self.platform_moment_arrow_transform = vtk.vtkTransform()
    #         actor.SetUserTransform(self.platform_moment_arrow_transform)

    #         mesh = mesh = pv.Sphere(radius=1)
    #         actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
    #         self.platform_moment_ellipsoid_transform = vtk.vtkTransform()
    #         actor.SetUserTransform(self.platform_moment_ellipsoid_transform)

    #         mesh = utils.get_arrow(shaft_scale=0.2)
    #         actor = plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
    #         self.platform_force_arrow_transform = vtk.vtkTransform()
    #         actor.SetUserTransform(self.platform_force_arrow_transform)

    #         mesh = mesh = pv.Sphere(radius=1)
    #         actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
    #         self.platform_force_ellipsoid_transform = vtk.vtkTransform()
    #         actor.SetUserTransform(self.platform_force_ellipsoid_transform)

    #     # Update vtkTransforms for each actor
    #     p = solution.platform_pose_mean[:3,3]
    #     wrench = solution.platform_wrench_mean
    #     cov = solution.platform_wrench_cov

    #     moment_mean, force_mean = wrench[:3], wrench[3:]
    #     moment_cov, force_cov = cov[:3, :3], cov[3:, 3:]

    #     matrix = utils.get_arrow_transform(p, moment_mean, scale=self.moment_scale)
    #     self.platform_moment_arrow_transform.SetMatrix(matrix.flatten().tolist())

    #     matrix = utils.get_arrow_transform(p, force_mean, scale=self.force_scale)
    #     self.platform_force_arrow_transform.SetMatrix(matrix.flatten().tolist())

    #     matrix = utils.get_ellipsoid_transform(p + force_mean * self.force_scale, force_cov, scale=self.force_scale)
    #     self.platform_force_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

    #     matrix = utils.get_ellipsoid_transform(p + moment_mean * self.moment_scale, moment_cov, scale=self.force_scale)
    #     self.platform_moment_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution):
        self.rod_manager.update(solution.marginals.rod, self.plotter)

        self.plotter.update(solution)


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