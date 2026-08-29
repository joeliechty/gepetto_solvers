try:
    import openvdb as vdb
except ImportError:
    import pyopenvdb as vdb
import os

import numpy as np

from gepetto_solvers.core.objects import OBJECTS_DIR


def create_cube_sdf(half_extents=(0.025, 0.02, 0.025), voxel_size=0.001,
                    band_halfwidth=0.06, edge_radius=0.005):
    # band_halfwidth controls how far from the surface SDF values are stored.
    # Outside this band the sampler returns the constant background (10.0) with
    # zero gradient, so the witness-point contact solver gets no pull. A wide
    # band gives a usable gradient signal far from the surface to guide the
    # solver in toward contact (matches make_sphere.py).
    #
    # edge_radius rounds the box's 12 edges / 8 corners by that radius so a
    # gradient-based contact solver doesn't get stuck on the C0 crease where two
    # faces meet (the analytic SDF is non-smooth there). We do this exactly, not
    # with an OpenVDB LevelSetFilter (the Python binding exposes no filter), by
    # the standard rounded-box construction: shrink the box bounds by
    # edge_radius, then offset the level set back out by edge_radius. The flat
    # faces (and hence the outer dimensions) are unchanged; only within
    # edge_radius of an edge does the surface become a smooth fillet.
    grid = vdb.FloatGrid(10.0)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    accessor = grid.getAccessor()

    hx, hy, hz = half_extents
    # Shrunken bounds for the rounded-box construction (offset back out below).
    bx, by, bz = hx - edge_radius, hy - edge_radius, hz - edge_radius

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

                # Rounded-box SDF: box of shrunken half-extents (bx, by, bz),
                # then subtract edge_radius so the surface bulges back out to the
                # original half-extents with edge_radius fillets on the edges.
                dx = abs(x) - bx
                dy = abs(y) - by
                dz = abs(z) - bz

                out_dist = np.sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2 + max(dz, 0) ** 2)
                in_dist = min(max(dx, max(dy, dz)), 0.0)

                sdf_val = out_dist + in_dist - edge_radius

                if abs(sdf_val) < band_halfwidth:
                    accessor.setValueOn((i, j, k), sdf_val)

    # Written into the package's objects/ directory, where the readers look.

    # These used to write a bare relative name, i.e. into the caller's CWD.

    filename = os.path.join(OBJECTS_DIR, "cube.vdb")
    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")


if __name__ == "__main__":
    create_cube_sdf()
