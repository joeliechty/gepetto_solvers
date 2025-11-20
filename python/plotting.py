import numpy as np
from pathlib import Path
import shutil

import vtk
import pyvista as pv
import matplotlib.pyplot as plt

frame_arrow_colors = ["red", "green", "blue"]


def get_tube_from_points(points, radius):
    spline = pv.Spline(points, n_points=200)
    tube = spline.tube(radius=radius)

    return tube


def get_tube_from_poses(poses, radius):
    points = np.array([T[:3, 3] for T in poses])
    return get_tube_from_points(points, radius)


def get_ellipsoid_transform(center, cov, scale=1.0, num_sigma=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    one_sigma = np.sqrt(np.maximum(eigvals, 1e-12)) * scale
    radii = num_sigma * one_sigma

    A = eigvecs @ np.diag(radii)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3,  3] = center

    return T


def get_arrow(length=1.0, direction=None):
    if direction is None:
        direction = np.array([1, 0, 0])

    shaft_scale = length / 25

    arrow = pv.Arrow(
        start=np.zeros(3),
        direction=direction,
        scale=length,
        tip_resolution=20,
        shaft_resolution=20,
        shaft_radius=shaft_scale / length,
        tip_radius=2 * shaft_scale / length,
        tip_length=2 * shaft_scale / length
    )

    return arrow


def get_axes_frame(length=1.0):
    return [
        get_arrow(length=length, direction=np.eye(3)[:,0]),
        get_arrow(length=length, direction=np.eye(3)[:,1]),
        get_arrow(length=length, direction=np.eye(3)[:,2])
    ]


def get_arrow_transform(p, vec, scale=1.0):
    length = np.linalg.norm(vec) * scale
    if length < 1e-12:
        dir = np.array([1.0, 0.0, 0.0])
    else:
        dir = vec / np.linalg.norm(vec)

    x_axis = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_axis, dir)
    c = np.dot(x_axis, dir)
    if np.linalg.norm(v) < 1e-12:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    # Scale along x for magnitude, then rotate x vector to the target vector 
    T = np.eye(4)
    T[:3, :3] = R @ np.diag([length, 1.0, 1.0])
    T[:3, 3] = p

    return T


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

        self.window_size = (4000, 4000)
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
                 cartesian_frame_scale=0.01):
        
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
            actor = plotter.plotter.add_mesh(self.get_end_plate(), color="silver", show_edges=True, line_width=2, opacity=0.7)
            self.base_plate_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.base_plate_transform)
        
        pose = solution.pose_mean[0]
        self.base_plate_transform.SetMatrix(pose.flatten().tolist())
    
    def update_tip_plate(self, solution, plotter):
        if not self.plot_tip_plate:
            return
        
        if plotter.frame == 0:
            actor = plotter.plotter.add_mesh(self.get_end_plate(), color="silver", show_edges=True, line_width=2, opacity=0.7)
            self.tip_plate_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.tip_plate_transform)
        
        pose = solution.pose_mean[-1]
        self.tip_plate_transform.SetMatrix(pose.flatten().tolist())
    
    def update_rod_tube(self, solution, plotter):
        tube = get_tube_from_poses(solution.pose_mean, radius=self.backbone_radius)
        
        if plotter.frame == 0:
            self.backbone_tube_mesh = tube
            plotter.plotter.add_mesh(self.backbone_tube_mesh, color='ultramarine', opacity = 0.5)
            return
        
        # Not really a lightweight way to update this?
        self.backbone_tube_mesh.shallow_copy(tube)

    def update_backbone_ellipsoids(self, solution, plotter):
        if not self.plot_backbone_ellipsoids:
            return

        if plotter.frame == 0:
            self.backbone_ellipsoid_transforms = []
            for _ in range(len(solution.pose_mean)):
                transform = vtk.vtkTransform()
                ellipsoid = pv.Sphere(radius=1)
                actor = plotter.plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)
                actor.SetUserTransform(transform)
                self.backbone_ellipsoid_transforms.append(transform)

        for transform, pose, cov in zip(self.backbone_ellipsoid_transforms, solution.pose_mean, solution.pose_cov):
            R = pose[:3, :3]
            p = pose[:3, 3]
            cov = R @ (cov[3:, 3:] @ R.T)  # World frame

            matrix = get_ellipsoid_transform(p, cov)
            transform.SetMatrix(matrix.flatten().tolist())


    def update_backbone_frames(self, solution, plotter):
        if not self.plot_backbone_frames:
            return

        if plotter.frame == 0:
            self.backbone_frame_transforms = []
            for _ in solution.pose_mean:
                axes = get_axes_frame(length=self.cartesian_frame_scale)
                transform = vtk.vtkTransform()
                for arrow, color in zip(axes, frame_arrow_colors):
                    actor = plotter.plotter.add_mesh(arrow, color=color)
                    actor.SetUserTransform(transform)
                self.backbone_frame_transforms.append(transform)

        for transform, pose in zip(self.backbone_frame_transforms, solution.pose_mean):
            transform.SetMatrix(pose.flatten().tolist())

    def update_wrenches(self, solution, plotter):
        if not self.plot_wrenches:
            return 
    
        if plotter.frame == 0:
            self.moment_arrow_transforms = []
            self.moment_ellipsoid_transforms = []
            self.force_arrow_transforms = []
            self.force_ellipsoid_transforms = []

            for _ in range(len(solution.pose_cov)):
                mesh = get_arrow(shaft_scale=0.002)
                transform = vtk.vtkTransform()
                actor = plotter.plotter.add_mesh(mesh, color='deeppink', lighting=False)
                actor.SetUserTransform(transform)
                self.moment_arrow_transforms.append(transform)

                mesh = get_arrow(shaft_scale=0.002)
                transform = vtk.vtkTransform()
                actor = plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
                actor.SetUserTransform(transform)
                self.force_arrow_transforms.append(transform)

                mesh = pv.Sphere(radius=1)
                transform = vtk.vtkTransform()
                actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
                actor.SetUserTransform(transform)
                self.moment_ellipsoid_transforms.append(transform)

                mesh = pv.Sphere(radius=1)
                transform = vtk.vtkTransform()
                actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
                actor.SetUserTransform(transform)
                self.force_ellipsoid_transforms.append(transform)

        # Update vtkTransforms for each actor
        poses = solution.pose_mean
        wrenches = solution.wrench_mean
        covs = solution.wrench_cov

        for ii in range(len(poses)):
            p, w, cov = poses[ii][:3, 3], wrenches[ii], covs[ii]
            
            moment_mean, force_mean = w[:3], w[3:]
            moment_cov, force_cov = cov[:3, :3], cov[3:, 3:]

            matrix = get_arrow_transform(p, moment_mean, scale=self.moment_scale)
            self.moment_arrow_transforms[ii].SetMatrix(matrix.flatten().tolist())

            matrix = get_arrow_transform(p, force_mean, scale=self.force_scale)
            self.force_arrow_transforms[ii].SetMatrix(matrix.flatten().tolist())

            matrix = get_ellipsoid_transform(p + force_mean * self.force_scale, force_cov, scale=self.force_scale)
            self.force_ellipsoid_transforms[ii].SetMatrix(matrix.flatten().tolist())

            matrix = get_ellipsoid_transform(p + moment_mean * self.moment_scale, moment_cov, scale=self.force_scale)
            self.moment_ellipsoid_transforms[ii].SetMatrix(matrix.flatten().tolist())

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
            mesh = pv.Cylinder(direction=(0,0,1), radius=0.2, height=0.01)
            mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset])
            actor = plotter.plotter.add_mesh(mesh, color="silver", show_edges=True, line_width=2, opacity=0.3)
            self.platform_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_transform)

            mesh = pv.Cylinder(direction=(0,0,1), radius=0.005, height=np.abs(self.platform_z_offset))
            mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset / 2])
            actor = plotter.plotter.add_mesh(mesh, color="silver")
            actor.SetUserTransform(self.platform_transform)

            axes = get_axes_frame(length=0.1)
            for arrow, color in zip(axes, frame_arrow_colors):
                actor = plotter.plotter.add_mesh(arrow, color=color)
                actor.SetUserTransform(self.platform_transform)
            
            mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="deepcadmiumred", lighting=False, opacity=0.2)
            self.platform_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_ellipsoid_transform)

        pose = solution.platform_pose_mean
        self.platform_transform.SetMatrix(pose.flatten().tolist())
        
        p = pose[:3,3]
        R = pose[:3,:3]
        cov = solution.platform_pose_cov
        cov = R @ (cov[3:, 3:] @ R.T)
        T = get_ellipsoid_transform(p, cov)
        self.platform_ellipsoid_transform.SetMatrix(T.flatten().tolist())

    def update_platform_wrench(self, solution, plotter):
        if plotter.frame == 0:
            mesh = get_arrow(shaft_scale=0.002)
            actor = plotter.plotter.add_mesh(mesh, color='deeppink', lighting=False)
            self.platform_moment_arrow_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_moment_arrow_transform)

            mesh = mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            self.platform_moment_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_moment_ellipsoid_transform)

            mesh = get_arrow(shaft_scale=0.002)
            actor = plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
            self.platform_force_arrow_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_force_arrow_transform)

            mesh = mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            self.platform_force_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_force_ellipsoid_transform)

        # Update vtkTransforms for each actor
        p = solution.platform_pose_mean[:3,3]
        wrench = solution.platform_wrench_mean
        cov = solution.platform_wrench_cov

        moment_mean, force_mean = wrench[:3], wrench[3:]
        moment_cov, force_cov = cov[:3, :3], cov[3:, 3:]

        matrix = get_arrow_transform(p, moment_mean, scale=self.moment_scale)
        self.platform_moment_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = get_arrow_transform(p, force_mean, scale=self.force_scale)
        self.platform_force_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = get_ellipsoid_transform(p + force_mean * self.force_scale, force_cov, scale=self.force_scale)
        self.platform_force_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

        matrix = get_ellipsoid_transform(p + moment_mean * self.moment_scale, moment_cov, scale=self.force_scale)
        self.platform_moment_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution):
        for i, manager in enumerate(self.rod_managers):
            manager.update(solution.marginals.rods[i], self.plotter)
        
        self.update_platform(solution.marginals, self.plotter)
        self.update_platform_wrench(solution.marginals, self.plotter)

        self.plotter.update(solution)


class CosseratShellPlotter:
    def __init__(self, **kwargs):
        self.cartesian_frame_scale = 0.04

        self.plotter = PlotterBase(**kwargs)

        yz_length = 0.05
        x_length = 0.7
        base_plate = pv.Cube(center=(x_length / 2, -yz_length / 2, 0), x_length=x_length, y_length=yz_length, z_length=yz_length)
        self.plotter.plotter.add_mesh(base_plate, color="silver", show_edges=True, line_width=2)
        
        tip_plate = base_plate = pv.Cube(center=(x_length / 2, yz_length / 2, 0), x_length=x_length, y_length=yz_length, z_length=yz_length)
        actor = self.plotter.plotter.add_mesh(tip_plate, color="silver", show_edges=True, line_width=2)
        self.tip_plate_transform = vtk.vtkTransform()
        actor.SetUserTransform(self.tip_plate_transform)

    def update_frames(self, solution, plotter):
        if plotter.frame == 0:
            self.frame_transforms = []
            for pose_col in solution.pose_mean:
                transforms_col = []
                for pose in pose_col:
                    axes = get_axes_frame(length=self.cartesian_frame_scale)
                    transform = vtk.vtkTransform()
                    for arrow, color in zip(axes, frame_arrow_colors):
                        actor = plotter.plotter.add_mesh(arrow, color=color)
                        actor.SetUserTransform(transform)
                    transforms_col.append(transform)
                self.frame_transforms.append(transforms_col)

        for transform_col, pose_col in zip(self.frame_transforms, solution.pose_mean):
            for transform, pose in zip(transform_col, pose_col):
                transform.SetMatrix(pose.flatten().tolist())
    
    def update_mesh(self, solution, plotter):
        points = []
        for pose_col in solution.pose_mean:
            points_col = [pose[:3,3] for pose in pose_col]
            points.append(points_col)

        pts = np.array(points).transpose(1, 0, 2)
        nx, ny, _ = pts.shape
        flat_points = pts.reshape(-1, 3)

        if plotter.frame == 0:
            def idx(i, j):
                return i * ny + j

            faces = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    p00 = idx(i, j)
                    p10 = idx(i+1, j)
                    p01 = idx(i, j+1)
                    p11 = idx(i+1, j+1)

                    faces += [
                        3, p00, p10, p11,   # triangle 1
                        3, p00, p11, p01    # triangle 2
                    ]

            self.edge_mesh = pv.PolyData(flat_points, faces=np.array(faces))
            self.face_mesh = pv.PolyData(flat_points, faces=np.array(faces))

            plotter.plotter.add_mesh(self.face_mesh, color="slateblue", opacity=0.15)
            plotter.plotter.add_mesh(self.edge_mesh, color="black", style="wireframe", line_width=1.5)

        self.face_mesh.points = flat_points
        self.edge_mesh.points = flat_points

    def update_stress_plots(self, solution):
        nx = len(solution.stress_mean)
        ny = len(solution.stress_mean[0])

        bending = np.zeros((nx, ny))
        torsion = np.zeros_like(bending)
        shear = np.zeros_like(bending)
        tensile = np.zeros_like(bending)

        for i in range(nx):
            for j in range(ny):
                sx = solution.stress_mean[i][j][0]
                sy = solution.stress_mean[i][j][1]

                bending[i, j] = np.linalg.norm([sx[1], sx[2], sy[0], sy[2]])
                torsion[i, j] = np.linalg.norm([sx[0], sy[1]])
                shear[i, j] = np.linalg.norm([sx[4], sx[5], sy[3], sy[5]])
                tensile[i, j] = np.linalg.norm([sx[3], sy[4]])

        plt.figure(figsize=(15,15))
        plt.subplot(4,1,1)
        plt.imshow(np.abs(bending), origin='lower', cmap='Oranges', vmin=0)
        plt.colorbar(label="Bending Stress")

        plt.subplot(4,1,2)
        plt.imshow(np.abs(torsion), origin='lower', cmap='Oranges', vmin=0)
        plt.colorbar(label="Torsion Stress")

        plt.subplot(4,1,3)
        plt.imshow(np.abs(shear), origin='lower', cmap='Oranges', vmin=0)
        plt.colorbar(label="Shear Stress")

        plt.subplot(4,1,4)
        plt.imshow(np.abs(tensile), origin='lower', cmap='Oranges', vmin=0)
        plt.colorbar(label="Tensile Stress")
        
        # plt.show()
        plt.tight_layout()
        plt.savefig(f"videos/frames/{self.plotter.save_frames_dir_name}/plt_{self.plotter.frame}.png")
        plt.close()

    def update_tip_plate(self, solution):
        self.tip_plate_transform.SetMatrix(solution.pose_mean[0][-1].flatten().tolist())

    def update_ellipsoids(self, solution, plotter):
        if plotter.frame == 0:
            self.ellipsoid_transforms = []
            for _ in range(len(solution.pose_mean)):
                transforms_col = []
                for _ in range(len(solution.pose_mean[0])):
                    transform = vtk.vtkTransform()
                    ellipsoid = pv.Sphere(radius=1)
                    actor = plotter.plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)
                    actor.SetUserTransform(transform)
                    transforms_col.append(transform)
                self.ellipsoid_transforms.append(transforms_col)

        for transform_col, pose_col, cov_col in zip(self.ellipsoid_transforms, solution.pose_mean, solution.pose_cov):
            for transform, pose, cov in zip (transform_col, pose_col, cov_col):
                R = pose[:3, :3]
                p = pose[:3, 3]
                cov = R @ (cov[3:, 3:] @ R.T)  # World frame

                matrix = get_ellipsoid_transform(p, cov)
                transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution):
        self.update_frames(solution.marginals, self.plotter)
        self.update_mesh(solution.marginals, self.plotter)
        self.update_tip_plate(solution.marginals)
        self.update_ellipsoids(solution.marginals, self.plotter)

        self.plotter.update(solution)
        self.update_stress_plots(solution.marginals)


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