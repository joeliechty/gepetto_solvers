import numpy as np
from pathlib import Path
import shutil

import pyvista as pv


def get_base_plate(solution):
    side_length = 10 * solution.tendon_disc_config.routing_radius
    thick = side_length / 10
    cube = pv.Cube(center=(0, 0, -thick / 2), x_length=side_length, y_length=side_length, z_length=thick)
    # rotated_points = (base_rotation @ cube.points.T).T
    # cube.points = rotated_points + base_location

    # cube_thick = 0.02
    # cube = pv.Cube(center=(0, 0, 0), x_length=0.3, y_length=0.3, z_length=cube_thick)
    # cube.points[:,2] = cube.points[:,2] - cube_thick / 2.0
    # cube.points = (base_rotation @ cube.points.T + base_location.reshape((3,1))).T

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
    disc_width = 0.1 * routing_radius
    tendon_radius = 0.04 * routing_radius

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


def get_ellipsoid(center, cov, scale, num_sigma=1.0):
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


def get_backbone_ellipsoids(solution, skip=0):
    N = len(solution.backbone_pose_mean)
    selected_indices = list(range(N - 1, -1, -skip))

    ellipsoids = []

    for i in reversed(selected_indices):  # Optional: reverse for chronological order
        pose = solution.backbone_pose_mean[i]
        cov = solution.backbone_pose_cov[i]

        R = pose[:3, :3]
        p = pose[:3, 3]
        cov = R @ (cov[3:, 3:] @ R.T)  # World frame

        ellipsoid = get_ellipsoid(p, cov, scale=1.0, num_sigma=2.80)
        ellipsoids.append(ellipsoid)

    return ellipsoids


class TendonRobotPlotter:
    def __init__(self, title, save_png_mode=False, plot_dist_load=False):
        self.plot_dist_load = plot_dist_load
        self.save_png_mode = save_png_mode

        if save_png_mode:
            dir_name = title.strip().lower().replace(" ", "_")
            self.frames_path = Path("frames") / dir_name
            shutil.rmtree(self.frames_path, ignore_errors=True)
            self.frames_path.mkdir(parents=True, exist_ok=True)

        self.window_size = (1500, 2000)
        self.plotter = pv.Plotter(window_size=self.window_size, off_screen=save_png_mode)
        self.plotter.add_text(title, position='upper_edge', font_size=14, color='black', font="times")
        self.frame = 0

    def init_dist_load(self, solution):
        self.dist_load_scale = 5.0

        self.dist_load_meshes = []
        for pose, wrench in zip(solution.backbone_pose_mean, solution.applied_wrench_mean):
            R = pose[:3,:3]
            p = pose[:3,3]
        
            f = R @ wrench[3:]  # World frame
            mesh = get_arrow(p, self.dist_load_scale * f)

            self.plotter.add_mesh(mesh, color='rebeccapurple', opacity=0.5)
            self.dist_load_meshes.append(mesh)

    def init_tip_force(self, solution, tip_force_gt):
        T_tip = solution.backbone_pose_mean[-1]
        R_tip = T_tip[:3,:3]
        p_tip = T_tip[:3,3]
        self.f_tip_scale = 0.7
        f_tip_mean = R_tip @ solution.applied_wrench_mean[-1][3:]  # World frame
        self.tip_force_mean_mesh = get_arrow(p_tip, self.f_tip_scale * f_tip_mean)
        self.plotter.add_mesh(self.tip_force_mean_mesh, color='rebeccapurple', opacity=0.7)

        f_tip_cov = R_tip @ (solution.applied_wrench_cov[-1][3:,3:] @ R_tip.T)  # World frame
        center = p_tip + f_tip_mean * self.f_tip_scale
        self.f_tip_95_mesh = get_ellipsoid(center, f_tip_cov, self.f_tip_scale, num_sigma=2.80)
        self.plotter.add_mesh(self.f_tip_95_mesh, color="gold", opacity=0.1, smooth_shading=True)

        if tip_force_gt is not None:
            f_tip_gt = R_tip @ tip_force_gt
            self.tip_force_gt_mesh = get_arrow(p_tip, self.f_tip_scale * f_tip_gt)
            self.plotter.add_mesh(self.tip_force_gt_mesh, color='forestgreen', opacity=0.7)

    def update_tip_force(self, solution, tip_force_gt):
        T_tip = solution.backbone_pose_mean[-1]
        R_tip = T_tip[:3,:3]
        p_tip = T_tip[:3,3]
        f_tip_mean = R_tip @ solution.applied_wrench_mean[-1][3:]  # World frame
        arrow = get_arrow(p_tip, self.f_tip_scale * f_tip_mean)
        self.tip_force_mean_mesh.shallow_copy(arrow)

        f_tip_cov = R_tip @ (solution.applied_wrench_cov[-1][3:,3:] @ R_tip.T)  # World frame
        center = p_tip + f_tip_mean * self.f_tip_scale
        ellipsoid = get_ellipsoid(center, f_tip_cov, self.f_tip_scale, num_sigma=2.80)
        self.f_tip_95_mesh.shallow_copy(ellipsoid)

        if tip_force_gt is not None:
            f_tip_gt = R_tip @ tip_force_gt
            arrow = get_arrow(p_tip, self.f_tip_scale * f_tip_gt)
            self.tip_force_gt_mesh.shallow_copy(arrow)
    
    def update_dist_load(self, solution):
        for i, (pose, wrench) in enumerate(zip(solution.backbone_pose_mean, solution.applied_wrench_mean)):
            R = pose[:3,:3]
            p = pose[:3,3]
        
            f = R @ wrench[3:]  # World frame
            mesh = get_arrow(p, self.dist_load_scale * f)
            self.dist_load_meshes[i].shallow_copy(mesh)
            
    def init_scene(self, solution, p_desired, desired_trajectory, tip_force_gt):
        plate = get_base_plate(solution)
        self.plate_actor = self.plotter.add_mesh(plate, color="coldgrey", show_edges=True, line_width=1)

        self.backbone_radius = solution.tendon_disc_config.routing_radius / 12.0
        self.backbone_mesh = get_tube_poses(solution.backbone_pose_mean, radius=self.backbone_radius)
        self.plotter.add_mesh(self.backbone_mesh, color='mediumblue', opacity = 0.8, smooth_shading=True)

        self.tendon_meshes, self.disc_meshes = get_tendon_disc_meshes(solution)

        tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]

        for j, tendon in enumerate(self.tendon_meshes):
            color = tendon_colors[j % len(tendon_colors)]
            for segment in tendon:
                self.plotter.add_mesh(segment, color=color, opacity=0.3)
        
        for i, disc in enumerate(self.disc_meshes):
            if i == 0: continue
            disc.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True, inplace=True)
            self.plotter.add_mesh(disc, color='steelblue', opacity=0.2, smooth_shading=True)

        if self.plot_dist_load:
            self.init_dist_load(solution)
        else:           
            self.init_tip_force(solution, tip_force_gt)

        self.ellipsoid_skip = 4
        self.backbone_cov_meshes = get_backbone_ellipsoids(solution, skip=self.ellipsoid_skip)

        for ellipsoid in self.backbone_cov_meshes:
            self.plotter.add_mesh(ellipsoid, color="crimson", opacity=0.15, smooth_shading=True)

        if desired_trajectory is not None:
            self.trajectory_radius = 0.001
            self.trajectory_mesh = get_tube_points(desired_trajectory, self.trajectory_radius)
            self.plotter.add_mesh(self.trajectory_mesh, color="crimson", opacity=0.2, smooth_shading=True)

        if p_desired is not None:
            self.p_desired_radius = 0.002
            self.p_desired_mesh = pv.Sphere(self.p_desired_radius, p_desired)
            self.plotter.add_mesh(self.p_desired_mesh, color="limegreen", opacity=0.7, smooth_shading=True)

        light_positions = [
            (0, 5, 5),
            (0, -5, 5),
            (5, 0, -5),
            (-5, 0, -5)
        ]

        for pos in light_positions:
            light = pv.Light(
                position=pos,
                intensity=0.4,
                light_type='scene light'
            )
            self.plotter.add_light(light)

        self.plotter.camera.position = (0.6, 0, 0.5)
        self.plotter.camera.focal_point = (0, 0, 0.13)

        self.plotter.add_axes()
        self.plotter.enable_depth_peeling(10)
        self.plotter.enable_anti_aliasing()

        if not self.save_png_mode:
            self.plotter.show(auto_close=False, interactive_update=True)
    
    def update(self, solution, p_desired=None, desired_trajectory=None, tip_force_gt=None):
        if self.frame == 0:
            self.init_scene(solution, p_desired, desired_trajectory, tip_force_gt)
        else:
            self.update_meshes(solution, p_desired, tip_force_gt)
        
        self.plotter.render()

        if self.save_png_mode:
            self.plotter.screenshot(self.frames_path / f"{self.frame}.png", window_size=self.window_size)
        
        self.frame = self.frame + 1

    def update_meshes(self, solution, p_desired, tip_force_gt):
        tube = get_tube_poses(solution.backbone_pose_mean, radius=self.backbone_radius)
        self.backbone_mesh.shallow_copy(tube)

        tendons, discs = get_tendon_disc_meshes(solution)

        for i, (new_disc, disc) in enumerate(zip(discs, self.disc_meshes)):
            if i == 0: continue
            disc.shallow_copy(new_disc)
        
        for i, tendon in enumerate(tendons):
            for j, segment in enumerate(tendon):
                self.tendon_meshes[i][j].shallow_copy(segment)

        ellipsoids = get_backbone_ellipsoids(solution, skip=self.ellipsoid_skip)
        for ellipsoid_plot, ellipsoid in zip(self.backbone_cov_meshes, ellipsoids):
            ellipsoid_plot.shallow_copy(ellipsoid)
        
        if self.plot_dist_load:
            self.update_dist_load(solution)
        else:
            self.update_tip_force(solution, tip_force_gt)

        if p_desired is not None:
            mesh = pv.Sphere(self.p_desired_radius, p_desired)
            self.p_desired_mesh.shallow_copy(mesh)

        self.plotter.camera.azimuth += 0.3
