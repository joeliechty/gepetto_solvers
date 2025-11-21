import pyvista as pv
import vtk

from . import plotting_tools


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
        tube = plotting_tools.get_tube_from_poses(solution.pose_mean, radius=self.backbone_radius)
        
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

            matrix = plotting_tools.get_ellipsoid_transform(p, cov)
            transform.SetMatrix(matrix.flatten().tolist())


    def update_backbone_frames(self, solution, plotter):
        if not self.plot_backbone_frames:
            return

        if plotter.frame == 0:
            self.backbone_frame_transforms = []
            for _ in solution.pose_mean:
                axes = plotting_tools.get_axes_frame(length=self.cartesian_frame_scale)
                transform = vtk.vtkTransform()
                for arrow, color in zip(axes, plotting_tools.frame_arrow_colors):
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

            shaft_scale=0.05
            
            for _ in range(len(solution.pose_cov)):
                mesh = plotting_tools.get_arrow(shaft_scale=shaft_scale)
                transform = vtk.vtkTransform()
                actor = plotter.plotter.add_mesh(mesh, color='deeppink', lighting=False)
                actor.SetUserTransform(transform)
                self.moment_arrow_transforms.append(transform)

                mesh = plotting_tools.get_arrow(shaft_scale=shaft_scale)
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

            matrix = plotting_tools.get_arrow_transform(p, moment_mean, scale=self.moment_scale)
            self.moment_arrow_transforms[ii].SetMatrix(matrix.flatten().tolist())

            matrix = plotting_tools.get_arrow_transform(p, force_mean, scale=self.force_scale)
            self.force_arrow_transforms[ii].SetMatrix(matrix.flatten().tolist())

            matrix = plotting_tools.get_ellipsoid_transform(p + force_mean * self.force_scale, force_cov, scale=self.force_scale)
            self.force_ellipsoid_transforms[ii].SetMatrix(matrix.flatten().tolist())

            matrix = plotting_tools.get_ellipsoid_transform(p + moment_mean * self.moment_scale, moment_cov, scale=self.force_scale)
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
    
        self.plotter = plotting_tools.PlotterBase(**kwargs)
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