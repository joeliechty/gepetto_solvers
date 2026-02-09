import pyvista as pv
import vtk
import numpy as np

from . import utils
from .._plotting.cosserat_rod_plotter import CosseratRodMeshManager


class ParallelRobotPlotter:
    def __init__(self,
                 platform_z_offset=0.0,
                 plot_rod_wrenches=True,
                 plot_platform_wrench=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 **kwargs):

        self.plotter = utils.PlotterBase(**kwargs)

        self.rod_managers = []

        self.moment_scale = 0.2
        self.force_scale = 0.1
        self.platform_z_offset = platform_z_offset
        self.plot_platform_wrench = plot_platform_wrench

        rod_colors = ['rebeccapurple', 'deepcadmiumred', 'cadmiumorange', 'lightcadmiumyellow', 'seagreen', 'royalblue']
        for i in range(6):
            self.rod_managers.append(CosseratRodMeshManager(
                plot_base_plate=False,
                plot_wrenches=plot_rod_wrenches,
                plot_base_wrench=plot_base_wrenches,
                plot_backbone_frames=plot_backbone_frames,
                plot_backbone_ellipsoids=plot_backbone_ellipsoids,
                backbone_radius=0.005,
                moment_scale=self.moment_scale, 
                force_scale=self.force_scale, 
                cartesian_frame_scale=0.025,
                rod_opacity=0.5,
                rod_color=rod_colors[i]
                )
            )

        base_plate = pv.Cylinder(direction=(0,0,1), radius=0.35, height=0.02)
        self.plotter.plotter.add_mesh(base_plate, color="silver", show_edges=True, line_width=2, opacity=0.3)
        
  
    def update_platform(self, solution, plotter):

        if plotter.frame == 0:
            mesh = pv.Cylinder(direction=(0,0,1), radius=0.15, height=0.01)
            mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset])
            actor = plotter.plotter.add_mesh(mesh, color="silver", show_edges=True, line_width=2, opacity=0.3)
            self.platform_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_transform)

            mesh = pv.Cylinder(direction=(0,0,1), radius=0.005, height=np.abs(self.platform_z_offset))
            mesh.points = mesh.points + np.array([0, 0, self.platform_z_offset / 2])
            actor = plotter.plotter.add_mesh(mesh, color="silver")
            actor.SetUserTransform(self.platform_transform)

            axes = utils.get_axes_frame(length=0.1)
            for arrow, color in zip(axes, utils.frame_arrow_colors):
                actor = plotter.plotter.add_mesh(arrow, color=color)
                actor.SetUserTransform(self.platform_transform)
            
            mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="deepcadmiumred", lighting=False, opacity=0.2)
            self.platform_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_ellipsoid_transform)

        pose = solution.platform_pose.mean
        self.platform_transform.SetMatrix(pose.flatten().tolist())
        
        p = pose[:3,3]
        R = pose[:3,:3]
        cov = solution.platform_pose.cov
        cov = R @ (cov[3:, 3:] @ R.T)
        T = utils.get_ellipsoid_transform(p, cov)
        self.platform_ellipsoid_transform.SetMatrix(T.flatten().tolist())

    def update_platform_wrench(self, solution, plotter):
        if not self.plot_platform_wrench:
            return
        
        if plotter.frame == 0:
            mesh = utils.get_arrow(shaft_scale=0.2)
            actor = plotter.plotter.add_mesh(mesh, color='deeppink', lighting=False)
            self.platform_moment_arrow_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_moment_arrow_transform)

            mesh = mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            self.platform_moment_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_moment_ellipsoid_transform)

            mesh = utils.get_arrow(shaft_scale=0.2)
            actor = plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
            self.platform_force_arrow_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_force_arrow_transform)

            mesh = mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            self.platform_force_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.platform_force_ellipsoid_transform)

        # Update vtkTransforms for each actor
        p = solution.platform_pose.mean[:3,3]
        wrench = solution.platform_wrench.mean
        cov = solution.platform_wrench.cov

        moment_mean, force_mean = wrench[:3], wrench[3:]
        moment_cov, force_cov = cov[:3, :3], cov[3:, 3:]

        matrix = utils.get_arrow_transform(p, moment_mean, scale=self.moment_scale)
        self.platform_moment_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = utils.get_arrow_transform(p, force_mean, scale=self.force_scale)
        self.platform_force_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = utils.get_ellipsoid_transform(p + force_mean * self.force_scale, force_cov, scale=self.force_scale)
        self.platform_force_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

        matrix = utils.get_ellipsoid_transform(p + moment_mean * self.moment_scale, moment_cov, scale=self.force_scale)
        self.platform_moment_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution):
        for i, manager in enumerate(self.rod_managers):
            manager.update(solution.marginals.rods[i], self.plotter)
        
        self.update_platform(solution.marginals, self.plotter)
        self.update_platform_wrench(solution.marginals, self.plotter)

        self.plotter.update(solution)

