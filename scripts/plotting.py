import numpy as np
from pathlib import Path
import shutil

import pyvista as pv


def get_base_plate(solution):
    side_length = 10 * solution.tendon_disc_config.routing_radius
    thick = side_length / 10
    cube = pv.Cube(center=(0, -thick / 2, 0), x_length=side_length, y_length=thick, z_length=side_length)

    return cube


def get_tube_points(points, radius):
    spline = pv.Spline(points, n_points=200)
    tube = spline.tube(radius=radius)

    return tube


def get_tube_poses(poses, radius):
    points = np.array([T[:3, 3] for T in poses])
    return get_tube_points(points, radius)


def get_tendon_disc_meshes(solution):
    num_discs = solution.tendon_disc_config.num_discs
    num_tendons = solution.tendon_disc_config.num_tendons
    routing_radius = solution.tendon_disc_config.routing_radius
    local_holes = solution.tendon_disc_config.local_holes # num_discs, num_tendons, 3
    disc_pose_idx = solution.tendon_disc_config.disc_pose_idx

    disc_radius = 1.1 * routing_radius
    disc_width = 0.12 * routing_radius
    tendon_radius = 0.05 * routing_radius

    discs = []

    for i in disc_pose_idx:
        T = solution.backbone_pose_mean[i]
        cylinder = pv.Cylinder(direction=(0,0,1), radius=disc_radius, height=disc_width)
        cylinder.points = (T[:3,:3] @ cylinder.points.T + T[:3,3].reshape((3,1))).T
        discs.append(cylinder)

    tendons = []

    for j in range(num_tendons):  # iterate over each tendon
        tendon_segments = []  # segments along this tendon
        
        for i in range(num_discs - 1):  # one segment per pair of discs
            T_i = solution.backbone_pose_mean[disc_pose_idx[i]]
            T_ip1 = solution.backbone_pose_mean[disc_pose_idx[i + 1]]

            # Local positions of this tendon on discs i and i+1
            p0_local = local_holes[i][j]
            p1_local = local_holes[i + 1][j]

            p0_world = T_i[:3, :3] @ p0_local + T_i[:3, 3]
            p1_world = T_ip1[:3, :3] @ p1_local + T_ip1[:3, 3]

            # Build a line segment and add to this tendon's segments
            line = pv.lines_from_points([p0_world, p1_world])
            tendon_segments.append(line.tube(radius=tendon_radius))
        
        tendons.append(tendon_segments)

    return tendons, discs


def get_ellipsoid(center, cov, scale, num_sigma=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    one_sigma = np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    radii = num_sigma * one_sigma

    ellipsoid = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
    ellipsoid.points = (eigvecs @ np.diag(radii) @ ellipsoid.points.T).T + center

    return ellipsoid


def get_largest_norm(f_samples, f_gt, f_mean):

    norms = []

    if f_samples is not None:
        norms.append(np.max(np.linalg.norm(f_samples, axis=1)))

    for f in [f_gt, f_mean]:
        if f is not None:
            norms.append(np.linalg.norm(f))

    max_norm = max(norms) if norms else 1.0  # Fallback value to avoid div-by-zero

    return max_norm


def get_arrow(start, vec, shaft_radius=0.001, tip_radius=0.002, tip_length=0.005):
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

    pv.Arrow()
    return arrow


def get_backbone_ellipsoids(solution):
    disc_idx = solution.tendon_disc_config.disc_pose_idx
    poses = [solution.backbone_pose_mean[i] for i in disc_idx]
    covs  = [solution.backbone_pose_cov[i] for i in disc_idx]

    ellipsoids = []

    for pose, cov in zip(poses, covs):
        R = pose[:3, :3]
        p = pose[:3, 3]
        cov = R @ (cov[3:, 3:] @ R.T)  # World frame

        ellipsoid = get_ellipsoid(p, cov, scale=1.0)
        ellipsoids.append(ellipsoid)

    return ellipsoids


def get_tip_force_meshes(solution, tip_force_gt, scale=0.5):
    p_tip = solution.backbone_pose_mean[-1][:3,3]
    
    f_tip_mean = solution.applied_wrench_mean[-1][3:]
    tip_force_mean_mesh = get_arrow(p_tip, scale * f_tip_mean)

    f_tip_cov = solution.applied_wrench_cov[-1][3:,3:]
    center = p_tip + f_tip_mean * scale
    tip_force_2_sigma_mesh = get_ellipsoid(center, f_tip_cov, scale)

    if tip_force_gt is not None:
        f_tip_gt = tip_force_gt
        tip_force_gt_mesh = get_arrow(p_tip, scale * f_tip_gt)
    else:
        tip_force_gt_mesh = None

    return tip_force_mean_mesh, tip_force_2_sigma_mesh, tip_force_gt_mesh


def get_dist_load_meshes(solution, scale=2.0):
    meshes = []

    for pose, wrench in zip(solution.backbone_pose_mean, solution.applied_wrench_mean):
        mesh = get_arrow(pose[:3,3], scale * wrench[3:], shaft_radius=0.0007, tip_radius=0.0015, tip_length=0.003)
        meshes.append(mesh)

    return meshes
    

class TendonRobotPlotter:
    def __init__(self, 
                 title, 
                 save_frames_mode=False, 
                 plot_tip_force=False, 
                 plot_dist_load=False,
                 plot_backbone_ellipsoids=True,
                 waypoints=None, 
                 cylinders=None, 
                 azimuth=20):
        
        self.save_frames_mode = save_frames_mode
        self.plot_tip_force = plot_tip_force
        self.plot_dist_load = plot_dist_load
        self.plot_backbone_ellipsoids = plot_backbone_ellipsoids

        self.cylinders = cylinders
        self.waypoints = waypoints
        self.azimuth = azimuth

        if save_frames_mode:
            dir_name = title.strip().lower().replace(" ", "_")
            self.frames_path = Path("videos") / "frames" / dir_name
            shutil.rmtree(self.frames_path, ignore_errors=True)
            self.frames_path.mkdir(parents=True, exist_ok=True)

        self.window_size = (2000, 2000)
        self.plotter = pv.Plotter(window_size=self.window_size, off_screen=save_frames_mode)
        self.frame = 0
        self.solve_time_ms_history = []
            
    def init_scene(self, solution):
        plate = get_base_plate(solution)
        self.plotter.add_mesh(plate, color="coldgrey", show_edges=True, line_width=1)

        if self.waypoints is not None:
            for point in self.waypoints:
                mesh = pv.Sphere(0.0015, center=point)
                self.plotter.add_mesh(mesh, color="red", smooth_shading=True)
        
        if self.cylinders is not None:
            for cylinder in self.cylinders:
                mesh = pv.Cylinder(cylinder['center'], cylinder['z'], cylinder['radius'], cylinder['length'])
                self.plotter.add_mesh(mesh, color='coral', smooth_shading=True)
        
        light_positions = [
            (0, 5, 5),
            (0, -5, 5),
            (5, 0, -5),
            (-5, 0, -5)
        ]

        for pos in light_positions:
            light = pv.Light(
                position=pos,
                intensity=0.5,
                light_type='scene light'
            )
            self.plotter.add_light(light)

        self.plotter.camera.position = (0.6, 0, 0.2)
        self.plotter.camera.focal_point = (-0.1, 0.12, 0)
        self.plotter.camera.azimuth += self.azimuth

        self.plotter.add_axes()
        self.plotter.enable_depth_peeling(10)
        self.plotter.enable_anti_aliasing()

        if not self.save_frames_mode:
            self.plotter.show(auto_close=False, interactive_update=True)
    
    def update(self, solution, p_desired=None, tip_force_gt=None):

        backbone_radius = solution.tendon_disc_config.routing_radius / 12.0
        backbone = get_tube_poses(solution.backbone_pose_mean, radius=backbone_radius)
        tendons, discs = get_tendon_disc_meshes(solution)
        backbone_ellipsoids = get_backbone_ellipsoids(solution)

        if self.plot_tip_force:
            tip_force_mean_mesh, tip_force_2_sigma_mesh, tip_force_gt_mesh = get_tip_force_meshes(solution, tip_force_gt)
        
        if self.plot_dist_load:
            dist_load_meshes = get_dist_load_meshes(solution)

        if p_desired is not None:
            p_desired_mesh = pv.Sphere(0.002, p_desired)

        if self.frame == 0:
            self.backbone_mesh = backbone
            self.plotter.add_mesh(self.backbone_mesh, color='black', opacity = 0.5, smooth_shading=True)

            self.tendon_meshes = tendons
            self.disc_meshes = discs
            tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]

            for i, disc in enumerate(self.disc_meshes):
                if i == 0: continue
                disc.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True, inplace=True)
                self.plotter.add_mesh(disc, color='steelblue', opacity=0.25, smooth_shading=True)
            
            for j, tendon in enumerate(self.tendon_meshes):
                color = tendon_colors[j % len(tendon_colors)]
                for segment in tendon:
                    self.plotter.add_mesh(segment, color=color, opacity=0.4)

            if self.plot_backbone_ellipsoids:
                self.backbone_2_sigma_meshes = backbone_ellipsoids
                for ellipsoid in self.backbone_2_sigma_meshes:
                    self.plotter.add_mesh(ellipsoid, color="cadmiumorange", opacity=0.3, smooth_shading=True)

            if self.plot_tip_force:
                self.tip_force_mean_mesh = tip_force_mean_mesh
                self.plotter.add_mesh(self.tip_force_mean_mesh, color='blueviolet')

                self.tip_force_2_sigma_mesh = tip_force_2_sigma_mesh
                self.plotter.add_mesh(self.tip_force_2_sigma_mesh, color="gold", opacity=0.3, smooth_shading=True)

                if tip_force_gt is not None:
                    self.tip_force_gt_mesh = tip_force_gt_mesh
                    self.plotter.add_mesh(self.tip_force_gt_mesh, color='forestgreen')
            
            if self.plot_dist_load:
                self.dist_load_meshes = dist_load_meshes
                for mesh in self.dist_load_meshes:
                    self.plotter.add_mesh(mesh, color='blueviolet', opacity=0.5)

            if p_desired is not None:
                self.p_desired_mesh = p_desired_mesh
                self.plotter.add_mesh(self.p_desired_mesh, color="red", smooth_shading=True)

            self.init_scene(solution)
        else:
            self.backbone_mesh.shallow_copy(backbone)

            for i, (new_disc, disc) in enumerate(zip(discs, self.disc_meshes)):
                if i == 0: continue
                disc.shallow_copy(new_disc)
            
            for i, tendon in enumerate(tendons):
                for j, segment in enumerate(tendon):
                    self.tendon_meshes[i][j].shallow_copy(segment)

            if self.plot_backbone_ellipsoids:
                for mesh_self, mesh in zip(self.backbone_2_sigma_meshes, backbone_ellipsoids):
                    mesh_self.shallow_copy(mesh)

            if self.plot_tip_force:
                self.tip_force_mean_mesh.shallow_copy(tip_force_mean_mesh)
                self.tip_force_2_sigma_mesh.shallow_copy(tip_force_2_sigma_mesh)
                if tip_force_gt is not None:
                    self.tip_force_gt_mesh.shallow_copy(tip_force_gt_mesh)
            
            if self.plot_dist_load:
                for (mesh_self, mesh) in zip(self.dist_load_meshes, dist_load_meshes):
                    mesh_self.shallow_copy(mesh)
           
            if p_desired is not None:
                self.p_desired_mesh.shallow_copy(p_desired_mesh)

        self.solve_time_ms_history.append(solution.total_time_ms)
        text = f"solve time: {solution.total_time_ms:.2f} ms, average: {np.mean(self.solve_time_ms_history):.2f} ms"
        self.plotter.add_text(text, position='upper_right', font_size=14, font="courier", name="solve_time")

        self.plotter.render()

        if self.save_frames_mode:
            self.plotter.screenshot(self.frames_path / f"{self.frame}.png", window_size=self.window_size)
        
        self.frame += 1