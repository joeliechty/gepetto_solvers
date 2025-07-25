from copy import deepcopy
from multiprocessing import Process

import numpy as np
from scipy.interpolate import interp1d

import pyvista as pv

# import se3_math
# import tendon_robot


class RobotPlotter():
    def __init__(self):
        self.process_list = []

    def add_plot(self, T_mean, tendon_info=None, title='', T_samples=None, f_tip_samples=None, f_tip_gt=None, f_tip_mean=None, f_tip_cov=None, f_dist=None, p_goal=None):
        process = Process(
            target=plot_robot,
            args=(T_mean,),
            kwargs={
                'title': title,
                'tendon_info': tendon_info,
                'T_samples': T_samples,
                'f_tip_samples': f_tip_samples,
                'f_tip_gt': f_tip_gt,
                'f_tip_mean': f_tip_mean,
                'f_tip_cov': f_tip_cov,
                'f_dist': f_dist,
                'p_goal': p_goal}
        )

        self.process_list.append(process)
    
    def show(self):
        [process.start() for process in self.process_list]
        [process.join() for process in self.process_list]


def get_base_plate():

    cube_thick = 0.008
    cube = pv.Cube(center=(0, 0, -cube_thick / 2), x_length=0.1, y_length=0.1, z_length=cube_thick)
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
    eigvals, eigvecs = np.linalg.eigh(cov.numpy())
    one_sigma = np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    radii = num_sigma * one_sigma

    ellipsoid = pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)
    ellipsoid.points = (eigvecs @ np.diag(radii) @ ellipsoid.points.T).T + center.numpy()

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


def plot_robot(solution, title=''):
    plotter = pv.Plotter(lighting='three lights')

    T_tip = solution.backbone_pose_mean[-1]
    R_tip = T_tip[:3,:3]
    p_tip = T_tip[:3,3]

    plate = get_base_plate()
    plotter.add_mesh(plate, color="dimgrey", show_edges=True, line_width=1, lighting="light_kit")

    tube = get_tube(solution.backbone_pose_mean, radius=0.0015)
    plotter.add_mesh(tube, color='blue', silhouette=True, specular=0.8, specular_power=10, opacity=0.5, smooth_shading=True, lighting="light_kit")

    tendons, discs = get_tendon_disc_meshes(solution)

    tendon_colors = ["crimson", "forestgreen", "royalblue", "mediumorchid", "goldenrod", "deeppink"]

    for j, tendon in enumerate(tendons):
        color = tendon_colors[j % len(tendon_colors)]
        for segment in tendon:
            plotter.add_mesh(segment, color=color, specular=0.8, specular_power=10, opacity=0.7, smooth_shading=True, silhouette=True, lighting="light_kit")
            
    for disc in discs:
        edges = disc.extract_feature_edges(boundary_edges=True, non_manifold_edges=False, feature_edges=False, manifold_edges=False)
        plotter.add_mesh(edges, color='k', line_width=1)
        plotter.add_mesh(disc, color='lightsteelblue', specular=0.8, specular_power=10, opacity=0.4, smooth_shading=True, split_sharp_edges=True, silhouette=True, lighting="light_kit")

    # f_tip_max = get_largest_norm(f_tip_samples, f_tip_gt, f_tip_mean)
    # f_tip_scale = 0.05 / f_tip_max
    f_tip_scale = 1.0
    f_tip_mean = R_tip @ solution.tip_wrench_mean[3:]  # World frame
    arrow = get_arrow(p_tip, f_tip_scale * f_tip_mean)
    plotter.add_mesh(arrow, color='blue', opacity=0.5)
    
    # if T_samples is not None:
    #     for ii in range(len(T_samples)):
    #         T_i = T_samples[ii]
    #         T_i = se3_math.batch_inverse_transform(T_i)
            
    #         tube = get_tube(T_i, radius=0.00025)
    #         plotter.add_mesh(tube, color='red', specular=0.8, specular_power=10, opacity=0.05, smooth_shading=True, lighting="light_kit")

    #         if f_tip_samples is not None:
    #             f_i = R_tip @ f_tip_samples[ii]  # World frame
    #             arrow = get_arrow(T_i[-1][:3,3], f_tip_scale * f_i)
    #             plotter.add_mesh(arrow, color='red', opacity=0.5, lighting="light_kit")

    # if f_tip_mean is not None:


    # if f_tip_gt is not None:
    #     f_tip_gt = R_tip @ f_tip_gt  # World frame
    #     arrow = get_arrow(p_tip, f_tip_scale * f_tip_gt)
    #     plotter.add_mesh(arrow, color='green', opacity=0.5, lighting="light_kit")
    
    # if f_tip_cov is not None:
    #     f_tip_cov = R_tip @ (f_tip_cov @ R_tip.T)  # World frame
    #     center = p_tip + f_tip_mean * f_tip_scale
    #     ellipsoid_50 = get_ellipsoid(center, f_tip_cov, f_tip_scale, num_sigma=1.18)
    #     ellipsoid_95 = get_ellipsoid(center, f_tip_cov, f_tip_scale, num_sigma=2.80)

    #     plotter.add_mesh(ellipsoid_50, color="#FF6600", opacity=0.3, smooth_shading=True, specular=1.0, specular_power=10, lighting="light_kit")
    #     plotter.add_mesh(ellipsoid_95, color="#FF6600", opacity=0.2, smooth_shading=True, specular=1.0, specular_power=10, lighting="light_kit")

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
    plotter.enable_depth_peeling(10)
    plotter.enable_anti_aliasing()
    light = pv.Light(light_type='headlight')
    light.intensity = 0.8
    plotter.add_light(light)
    plotter.view_isometric()
    
    plotter.show()