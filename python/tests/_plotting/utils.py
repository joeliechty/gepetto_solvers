import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt

import numpy as np
import pyvista as pv


frame_arrow_colors = ["red", "green", "blue"]


def setup_plt(width=3.0, height=5.0, grid=False):

    os.makedirs("figures", exist_ok=True)

    plt.rcParams.update({
        "figure.figsize": (width, height),
        "font.family": "STIXGeneral",
        "font.size": 8,               
        "xtick.labelsize": 7,        
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,  # embed fonts in PDF
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",  # math text compatible with Times
        "mathtext.rm": "stix",
        "lines.markersize": 4
    })


def build_primitive_mesh(spec):
    """Build a pyvista mesh from an object-primitive spec dict.

    ``spec`` has a "type" key ("sphere", "cylinder", or "cube"/"box") plus the
    geometry fields used by the ``get_primitive_specs()`` "plot" lambdas:
    sphere -> center/radius; cylinder -> center/direction/radius/height;
    cube/box -> center/extents (full side lengths). Shared by the finger and
    hand plotters so both render the same scene objects.
    """
    ptype = spec.get("type", "sphere")
    center = np.asarray(spec["center"], dtype=float)

    if ptype == "sphere":
        return pv.Sphere(radius=float(spec["radius"]), center=center)
    if ptype == "cylinder":
        # Axis defaults to Y to match _objects/make_cylinder.py.
        direction = spec.get("direction", (0.0, 1.0, 0.0))
        return pv.Cylinder(
            center=center, direction=direction,
            radius=float(spec["radius"]), height=float(spec["height"]))
    if ptype == "capsule":
        # Cylinder of length "height" with hemispherical caps, matching
        # _objects/make_capsule.py. Composed rather than pv.Capsule so it works
        # across pyvista versions. The tube is uncapped and each cap is a true
        # hemisphere (pole aimed outward along the axis), so the translucent
        # surface has no interior faces showing through.
        direction = np.asarray(spec.get("direction", (0.0, 1.0, 0.0)), dtype=float)
        direction = direction / np.linalg.norm(direction)
        radius, height = float(spec["radius"]), float(spec["height"])
        cylinder = pv.Cylinder(center=center, direction=direction,
                               radius=radius, height=height, capping=False)
        caps = [pv.Sphere(radius=radius,
                          center=center + s * direction * (height / 2.0),
                          direction=s * direction, end_phi=90)
                for s in (+1.0, -1.0)]
        return cylinder.merge(caps)
    if ptype in ("cube", "box"):
        ex, ey, ez = spec["extents"]
        return pv.Cube(center=center,
                       x_length=float(ex), y_length=float(ey),
                       z_length=float(ez))
    raise ValueError(f"Unknown primitive type: {ptype!r}")


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


def get_arrow(length=1.0, direction=None, shaft_scale=1.0):
    if direction is None:
        direction = np.array([1, 0, 0])

    shaft_prescale = 0.05

    arrow = pv.Arrow(
        start=np.zeros(3),
        direction=direction,
        scale=length,
        tip_resolution=20,
        shaft_resolution=20,
        shaft_radius=shaft_prescale *  shaft_scale,
        tip_radius=2 * shaft_prescale * shaft_scale,
        tip_length=2 * shaft_prescale * shaft_scale
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
                 frames_base_dir=None,
                 single_plot_mode=False,
                 plot_rviz_coords=False,
                 camera_focal_point=None,
                 camera_azimuth=15,
                 camera_elevation=20,
                 camera_distance=0.6,
                 window_size=(1200, 1200)):

        self.save_frames_dir_name = save_frames_dir_name
        # Base directory the per-run frames subdirectory is created under. Defaults
        # to videos/frames for backward compatibility; callers (e.g. the tendon
        # finger tests) can redirect frames into a figures/<experiment> tree.
        self.frames_base_dir = Path(frames_base_dir) if frames_base_dir else Path("videos") / "frames"
        self.single_plot_mode = single_plot_mode
        self.plot_rviz_coords = plot_rviz_coords

        if camera_focal_point is None:
            self.camera_focal_point = np.zeros(3)
        else:
            self.camera_focal_point = camera_focal_point

        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.camera_distance = camera_distance

        if self.save_frames_dir_name:
            self.frames_path = self.frames_base_dir / self.save_frames_dir_name
            shutil.rmtree(self.frames_path, ignore_errors=True)
            self.frames_path.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size
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

        if self.plot_rviz_coords:
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

    def save_video(self, fps=10, name=None, output_dir=None, frame_range=None):
        """Assemble the saved PNG frames into an animated GIF.

        Frames must have been captured during update() (save_frames_dir_name set).
        The GIF is written as <output_dir>/<name>.gif, defaulting to a file named
        after the frames subdirectory, placed alongside that subdirectory.

        ``frame_range`` optionally restricts the GIF to frames whose integer file
        stem satisfies ``start <= idx < end`` -- lets one shared frames/ dir (the
        monotonic frame counter is never reset) be sliced into several GIFs.

        Returns the GIF path, or None if no frames were captured.
        """
        from PIL import Image

        if not self.save_frames_dir_name:
            raise RuntimeError("save_video requires save_frames_dir_name to be set")

        frame_files = sorted(
            self.frames_path.glob("*.png"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        if frame_range is not None:
            start, end = frame_range
            frame_files = [f for f in frame_files
                           if f.stem.isdigit() and start <= int(f.stem) < end]
        if not frame_files:
            print(f"save_video: no frames found in {self.frames_path}")
            return None

        out_dir = Path(output_dir) if output_dir else self.frames_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = out_dir / f"{name or self.save_frames_dir_name}.gif"

        frames = [Image.open(f).convert("RGB") for f in frame_files]
        duration_ms = int(round(1000.0 / fps))
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=duration_ms, loop=0, optimize=True)
        print(f"Saved {len(frames)}-frame animation to {gif_path}")
        return gif_path
