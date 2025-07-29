import numpy as np
from scipy.interpolate import interp1d

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


def get_tube(T_list, radius):
    points = np.array([T[:3, 3] for T in T_list])
    spline = pv.Spline(points, n_points=200)
    tube = spline.tube(radius=radius)

    return tube


def get_tendon_disc_meshes(solution):
    num_discs = solution.tendon_disc_config.num_discs
    num_tendons = solution.tendon_disc_config.num_tendons
    routing_radius = solution.tendon_disc_config.routing_radius
    local_holes = solution.tendon_disc_config.local_holes # num_discs, num_tendons, 3
    disc_pose_idx = solution.tendon_disc_config.disc_pose_idx

    disc_radius = 1.1 * routing_radius
    disc_width = 0.1 * routing_radius
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


class TendonRobotPlotter:
    def __init__(self, title):
        self.plotter = pv.Plotter(lighting='three lights')
        self.plotter.add_text(title, position='upper_edge', font_size=14, color='black', font="times")
        self.is_first_plot = True

    def init_scene(self, solution, tip_force_gt):
        plate = get_base_plate(solution)
        self.plate_actor = self.plotter.add_mesh(plate, color="dimgrey", show_edges=True, line_width=1)

        self.backbone_mesh = get_tube(solution.backbone_pose_mean, radius=0.0015)
        self.plotter.add_mesh(self.backbone_mesh, color='blue', silhouette=True, opacity=0.5, smooth_shading=True)

        self.tendon_meshes, self.disc_meshes = get_tendon_disc_meshes(solution)

        tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]

        for j, tendon in enumerate(self.tendon_meshes):
            color = tendon_colors[j % len(tendon_colors)]
            for segment in tendon:
                self.plotter.add_mesh(segment, color=color, opacity=0.7, smooth_shading=True, silhouette=True)
        
        for disc in self.disc_meshes:
            disc.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True, inplace=True)
            self.plotter.add_mesh(disc, color='steelblue', opacity=0.2, silhouette=True, smooth_shading=True)

        # self.tip_pose_radius = 0.1 * solution.tendon_disc_config.routing_radius
        # self.tip_pose_meshes = []
        # for tip_pose_sample in solution.tip_pose_samples:
        #     sphere = pv.Sphere(self.tip_pose_radius, tip_pose_sample[:3,3])
        #     self.plotter.add_mesh(sphere, color='red', opacity=0.3, smooth_shading=True)
        #     self.tip_pose_meshes.append(sphere)

        T_tip = solution.backbone_pose_mean[-1]
        R_tip = T_tip[:3,:3]
        p_tip = T_tip[:3,3]
        self.f_tip_scale = 0.5
        f_tip_mean = R_tip @ solution.tip_wrench_mean[3:]  # World frame
        self.tip_force_mean_mesh = get_arrow(p_tip, self.f_tip_scale * f_tip_mean)
        self.plotter.add_mesh(self.tip_force_mean_mesh, color='blue', opacity=0.5)

        f_tip_cov = R_tip @ (solution.tip_wrench_cov[3:,3:] @ R_tip.T)  # World frame
        center = p_tip + f_tip_mean * self.f_tip_scale
        self.f_tip_95_mesh = get_ellipsoid(center, f_tip_cov, self.f_tip_scale, num_sigma=2.80)
        self.plotter.add_mesh(self.f_tip_95_mesh, color="orange", opacity=0.2, smooth_shading=True)

        if tip_force_gt is not None:
            f_tip_gt = R_tip @ tip_force_gt
            self.tip_force_gt_mesh = get_arrow(p_tip, self.f_tip_scale * f_tip_gt)
            self.plotter.add_mesh(self.tip_force_gt_mesh, color='green', opacity=0.5)

        self.plotter.add_axes()
        # self.plotter.enable_depth_peeling(10)
        self.plotter.enable_anti_aliasing()

        light = pv.Light(light_type='scenelight')
        light.position = (5, 5, 10)           # Light source location
        light.intensity = 0.2
        self.plotter.add_light(light)

        light = pv.Light(light_type='scenelight')
        light.position = (5, -5, 10)           # Light source location
        light.intensity = 0.2
        self.plotter.add_light(light)

        light = pv.Light(light_type='scenelight')
        light.position = (-5, 0, 10)           # Light source location
        light.intensity = 0.2
        self.plotter.add_light(light)

        self.plotter.view_isometric()
        self.plotter.show(auto_close=False, interactive_update=True, full_screen=False)

    def update(self, solution, tip_force_gt=None):
        if self.is_first_plot:
            self.init_scene(solution, tip_force_gt)
            self.is_first_plot = False
        else:
            self.update_meshes(solution, tip_force_gt)

    def update_meshes(self, solution, tip_force_gt):
        tube = get_tube(solution.backbone_pose_mean, radius=0.0015)
        self.backbone_mesh.points[:] = tube.points

        tendons, discs = get_tendon_disc_meshes(solution)

        for new_disc, self_disc in zip(discs, self.disc_meshes):
            self_disc.points[:] = new_disc.points
        
        for i, tendon in enumerate(tendons):
            for j, segment in enumerate(tendon):
                self.tendon_meshes[i][j].points[:] = tendons[i][j].points

        # for i, tip_pose_sample in enumerate(solution.tip_pose_samples):
        #     sphere = pv.Sphere(self.tip_pose_radius, tip_pose_sample[:3,3])
        #     self.tip_pose_meshes[i].points[:] = sphere.points
        
        T_tip = solution.backbone_pose_mean[-1]
        R_tip = T_tip[:3,:3]
        p_tip = T_tip[:3,3]
        f_tip_mean = R_tip @ solution.tip_wrench_mean[3:]  # World frame
        arrow = get_arrow(p_tip, self.f_tip_scale * f_tip_mean)
        self.tip_force_mean_mesh.points[:] = arrow.points

        f_tip_cov = R_tip @ (solution.tip_wrench_cov[3:,3:] @ R_tip.T)  # World frame
        center = p_tip + f_tip_mean * self.f_tip_scale
        ellipsoid = get_ellipsoid(center, f_tip_cov, self.f_tip_scale, num_sigma=2.80)
        self.f_tip_95_mesh.points[:] = ellipsoid.points

        if tip_force_gt is not None:
            f_tip_gt = R_tip @ tip_force_gt
            arrow = get_arrow(p_tip, self.f_tip_scale * f_tip_gt)
            self.tip_force_gt_mesh.points[:] = arrow.points

        self.plotter.camera.azimuth += 0.5
        self.plotter.render()













def plot_robot(solution, title=''):


    
    # if p_goal is not None:
    #     goal = pv.Sphere(radius=0.003, center=p_goal)
    #     plotter.add_mesh(goal, color='green', opacity=0.7, smooth_shading=True, specular=1.0, specular_power=10, lighting="light_kit")

    # if f_dist is not None:
    #     f_dist = f_dist.reshape(3,-1).T
    #     f_dist_max = np.linalg.norm(f_dist, axis=1).max()
    #     f_dist_scale = 0.075 / f_dist_max
    #     for ii in range(len(f_dist)):
    #         R_i = T_mean[ii,:3,:3]
    #         p_i = T_mean[ii,:3,3]
    #         f_i = R_i @ f_dist[ii]  # World frame
    #         arrow = get_arrow(p_i, f_dist_scale * f_i)
    #         plotter.add_mesh(arrow, color='green', opacity=0.5, lighting="light_kit")
    
    plotter.add_title(title, font_size=12, font='times')
    plotter.add_axes()
    # plotter.enable_depth_peeling(10)
    plotter.enable_anti_aliasing()
    light = pv.Light(light_type='headlight')
    light.intensity = 0.8
    plotter.add_light(light)
    plotter.view_isometric()
    
    plotter.show()