try:
    import openvdb as vdb
except ImportError:
    import pyopenvdb as vdb
import numpy as np


def create_cube_sdf(half_extents=(0.025, 0.02, 0.025), voxel_size=0.001,
                    band_halfwidth=0.06):
    # band_halfwidth controls how far from the surface SDF values are stored.
    # Outside this band the sampler returns the constant background (10.0) with
    # zero gradient, so the witness-point contact solver gets no pull. A wide
    # band gives a usable gradient signal far from the surface to guide the
    # solver in toward contact (matches make_sphere.py).
    grid = vdb.FloatGrid(10.0)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    accessor = grid.getAccessor()

    hx, hy, hz = half_extents

    margin = int(band_halfwidth / voxel_size)
    rx = int(hx / voxel_size) + margin
    ry = int(hy / voxel_size) + margin
    rz = int(hz / voxel_size) + margin

    print("Generating SDF voxels...")
    for i in range(-rx, rx + 1):
        for j in range(-ry, ry + 1):
            for k in range(-rz, rz + 1):
                x = i * voxel_size
                y = j * voxel_size
                z = k * voxel_size

                # Analytic SDF for an axis-aligned box centered at the origin.
                dx = abs(x) - hx
                dy = abs(y) - hy
                dz = abs(z) - hz

                out_dist = np.sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2 + max(dz, 0) ** 2)
                in_dist = min(max(dx, max(dy, dz)), 0.0)

                sdf_val = out_dist + in_dist

                if abs(sdf_val) < band_halfwidth:
                    accessor.setValueOn((i, j, k), sdf_val)

    filename = "cube.vdb"
    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")


if __name__ == "__main__":
    create_cube_sdf()
