import numpy as np
import pyvista as pv
import vtk
import matplotlib.pyplot as plt 

from . import utils


class CosseratShellPlotter:
    def __init__(self, **kwargs):
        self.cartesian_frame_scale = 0.04

        self.plotter = utils.PlotterBase(**kwargs)

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
            for states_col in solution.states:
                transforms_col = []
                for state_col in states_col:
                    axes = utils.get_axes_frame(length=self.cartesian_frame_scale)
                    transform = vtk.vtkTransform()
                    for arrow, color in zip(axes, utils.frame_arrow_colors):
                        actor = plotter.plotter.add_mesh(arrow, color=color)
                        actor.SetUserTransform(transform)
                    transforms_col.append(transform)
                self.frame_transforms.append(transforms_col)

        for transform_col, state_col in zip(self.frame_transforms, solution.states):
            for transform, state in zip(transform_col, state_col):
                transform.SetMatrix(state.pose.mean.flatten().tolist())
    
    def update_mesh(self, solution, plotter):
        points = []
        for state_col in solution.states:
            points_col = [state.pose.mean[:3,3] for state in state_col]
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
        self.tip_plate_transform.SetMatrix(solution.states[0][-1].pose.mean.flatten().tolist())

    def update_ellipsoids(self, solution, plotter):
        if plotter.frame == 0:
            self.ellipsoid_transforms = []
            for _ in range(len(solution.states)):
                transforms_col = []
                for _ in range(len(solution.states[0])):
                    transform = vtk.vtkTransform()
                    ellipsoid = pv.Sphere(radius=1)
                    actor = plotter.plotter.add_mesh(ellipsoid, color="deepcadmiumred", lighting=False, opacity=0.2)
                    actor.SetUserTransform(transform)
                    transforms_col.append(transform)
                self.ellipsoid_transforms.append(transforms_col)

        for transform_col, state_col in zip(self.ellipsoid_transforms, solution.states):
            for transform, state in zip(transform_col, state_col):
                pose = state.pose.mean
                cov = state.pose.cov
                R = pose[:3, :3]
                p = pose[:3, 3]
                cov = R @ (cov[3:, 3:] @ R.T)  # World frame

                matrix = utils.get_ellipsoid_transform(p, cov)
                transform.SetMatrix(matrix.flatten().tolist())

    def update(self, solution):
        self.update_frames(solution.marginals, self.plotter)
        self.update_mesh(solution.marginals, self.plotter)
        self.update_tip_plate(solution.marginals)
        self.update_ellipsoids(solution.marginals, self.plotter)

        self.plotter.update(solution)
        # self.update_stress_plots(solution.marginals)

