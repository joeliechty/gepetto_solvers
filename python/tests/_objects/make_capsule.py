"""Generate ``capsule.vdb`` — a capsule (cylinder with hemispherical end caps)
level-set for the grasp tests.

A capsule is the ``radius``-offset of a line segment, so its analytic SDF is
simply ``distance_to_segment - radius`` everywhere (interior included). Like
``make_cylinder.py`` the segment is aligned with the local Y-axis; the grasp
tests rotate it 90 deg about X so the axis stands up along world +Z, giving the
four fingers a rod to wrap around while the thumb opposes from the other side.

Default size (radius 0.04, cylinder length 0.07 -> total length 0.15) is chosen
for the full five-finger anatomical-hand grasp: placed at the flexed-fingertip
locus (see ``five_finger_hand_grasp_test.GRASP_SPHERE_CENTER``) every fingertip
reaches the surface. The band settings match ``make_cylinder.py`` so the
witness-point contact solver gets a gradient far from the surface.

Run (from the ``python/`` directory):
    python -m tests._objects.make_capsule
"""

try:
    import openvdb as vdb
except ImportError:
    import pyopenvdb as vdb
import numpy as np


def create_capsule_sdf(radius=0.04, height=0.07, voxel_size=0.001,
                       band_halfwidth=0.06, filename="capsule.vdb"):
    # ``height`` is the length of the cylindrical section (the segment the caps
    # are offset from); the total capsule length is ``height + 2 * radius``.
    # band_halfwidth controls how far from the surface SDF values are stored;
    # outside it the sampler returns the constant background (10.0) with zero
    # gradient (matches make_cylinder.py). The capsule is aligned with the
    # Y-axis.
    grid = vdb.FloatGrid(10.0)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    half_height = height / 2.0

    accessor = grid.getAccessor()

    margin = int(band_halfwidth / voxel_size)
    rx = int(radius / voxel_size) + margin
    ry = int((half_height + radius) / voxel_size) + margin
    rz = int(radius / voxel_size) + margin

    print(f"Generating SDF voxels (radius={radius}, height={height})...")
    for i in range(-rx, rx + 1):
        for j in range(-ry, ry + 1):
            for k in range(-rz, rz + 1):
                # Convert voxel indices to world coordinates
                x = i * voxel_size
                y = j * voxel_size
                z = k * voxel_size

                # Capsule SDF: distance to the Y-axis segment
                # [-half_height, half_height] minus the radius. Clamp Y onto the
                # segment, then take the full 3D distance to that closest point.
                dy = y - np.clip(y, -half_height, half_height)
                dist = np.sqrt(x**2 + dy**2 + z**2)

                sdf_val = dist - radius

                # Store a wide band so the solver gets a gradient far from the surface.
                if abs(sdf_val) < band_halfwidth:
                    accessor.setValueOn((i, j, k), sdf_val)

    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")


if __name__ == "__main__":
    create_capsule_sdf()
