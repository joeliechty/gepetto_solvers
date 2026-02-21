import pyvista as pv
import vtk
import numpy as np

from . import utils
from .._plotting.cosserat_rod_plotter import CosseratRodMeshManager


class MultiRobotPlotter:
    def __init__(self,
                 plot_rod_wrenches=False,
                 plot_tip_force=True,
                 plot_base_wrenches=False,
                 plot_backbone_frames=False,
                 plot_backbone_ellipsoids=True,
                 **kwargs):

        self.plotter = utils.PlotterBase(**kwargs)

        self.moment_scale = 0.2
        self.force_scale = 0.5
        self.plot_tip_force = plot_tip_force

        rod_names = ['main', 'helper', 'end_effector']
        rod_colors = ['royalblue', 'rebeccapurple', 'royalblue']
        plot_base_plate = [True, True, False]

        self.rod_managers = {}
        for name, color, do_base in zip(rod_names, rod_colors, plot_base_plate):
            self.rod_managers[name] = CosseratRodMeshManager(
                plot_base_plate=do_base,
                plot_wrenches=plot_rod_wrenches,
                plot_base_wrench=plot_base_wrenches,
                plot_backbone_frames=plot_backbone_frames,
                plot_backbone_ellipsoids=plot_backbone_ellipsoids,
                backbone_radius=0.005,
                moment_scale=self.moment_scale, 
                force_scale=self.force_scale, 
                cartesian_frame_scale=0.025,
                rod_opacity=0.5,
                rod_color=color
                )
  
    def update_tip_force(self, solution, plotter, tip_force_gt):
        if not self.plot_tip_force:
            return
        
        if plotter.frame == 0:
            shaft_scale = 0.05
            mesh = utils.get_arrow(shaft_scale=shaft_scale)
            actor = plotter.plotter.add_mesh(mesh, color='darkorchid', lighting=False)
            self.tip_force_arrow_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.tip_force_arrow_transform)

            mesh = mesh = pv.Sphere(radius=1)
            actor = plotter.plotter.add_mesh(mesh, color="cadmiumlemon", lighting=False, opacity=0.4)
            self.tip_force_ellipsoid_transform = vtk.vtkTransform()
            actor.SetUserTransform(self.tip_force_ellipsoid_transform)

            if tip_force_gt is not None:
                mesh = utils.get_arrow(shaft_scale=shaft_scale)
                self.tip_force_gt_transform = vtk.vtkTransform()
                actor = self.plotter.plotter.add_mesh(mesh, color='green', lighting=False)
                actor.SetUserTransform(self.tip_force_gt_transform)
            
        # Update vtkTransforms for each actor
        p = solution.states[-1].pose.mean[:3,3]
        f = solution.states[-1].wrench.mean[3:]
        cov = solution.states[-1].wrench.cov[3:, 3:]

        matrix = utils.get_arrow_transform(p, f, scale=self.force_scale)
        self.tip_force_arrow_transform.SetMatrix(matrix.flatten().tolist())

        matrix = utils.get_ellipsoid_transform(p + f * self.force_scale, cov, scale=self.force_scale)
        self.tip_force_ellipsoid_transform.SetMatrix(matrix.flatten().tolist())

        if tip_force_gt is not None:
            matrix = utils.get_arrow_transform(p, tip_force_gt, scale=self.force_scale)
            self.tip_force_gt_transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution, tip_force_gt=None):
        self.rod_managers['main'].update(solution.marginals.main_rod, self.plotter)
        self.rod_managers['helper'].update(solution.marginals.helper_rod, self.plotter)
        self.rod_managers['end_effector'].update(solution.marginals.end_effector_rod, self.plotter)

        self.update_tip_force(solution.marginals.end_effector_rod, self.plotter, tip_force_gt)
        
        self.plotter.update(solution)

