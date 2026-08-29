try:
    import openvdb as vdb
except ImportError:
    import pyopenvdb as vdb
import os

import numpy as np

from gepetto_solvers.core.objects import OBJECTS_DIR


def create_cylinder_sdf(radius=0.025, height=0.04, voxel_size=0.001,
                        band_halfwidth=0.06, edge_radius=0.005):
    # band_halfwidth controls how far from the surface SDF values are stored.
    # Outside this band the sampler returns the constant background (10.0) with
    # zero gradient, so the witness-point contact solver gets no pull. A wide
    # band gives a usable gradient signal far from the surface to guide the
    # solver in toward contact (matches make_sphere.py). The cylinder is aligned
    # with the Y-axis.
    #
    # edge_radius rounds the two circular rims (where each flat cap meets the
    # curved side) by that radius so a gradient-based contact solver doesn't get
    # stuck on the C0 crease there. We do this exactly, not with an OpenVDB
    # LevelSetFilter (the Python binding exposes no filter), by the same
    # shrink-then-offset trick as the rounded box: shrink the radius and
    # half-height by edge_radius, then offset the level set back out by
    # edge_radius. The curved side and flat caps (and hence the outer
    # dimensions) are unchanged; only within edge_radius of a rim does the
    # surface become a smooth fillet.
    grid = vdb.FloatGrid(10.0)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    half_height = height / 2.0

    accessor = grid.getAccessor()

    margin = int(band_halfwidth / voxel_size)
    rx = int(radius / voxel_size) + margin
    ry = int(half_height / voxel_size) + margin
    rz = int(radius / voxel_size) + margin

    print("Generating SDF voxels...")
    for i in range(-rx, rx + 1):
        for j in range(-ry, ry + 1):
            for k in range(-rz, rz + 1):
                # Convert voxel indices to world coordinates
                x = i * voxel_size
                y = j * voxel_size
                z = k * voxel_size

                # Rounded cylinder SDF (Y-axis). Distance from center axis in the
                # XZ plane, measured against the shrunken bounds
                # [radius - edge_radius, half_height - edge_radius], then offset
                # back out by edge_radius so the rims become edge_radius fillets.
                dist_xz = np.sqrt(x**2 + z**2)

                # Distance vector to the shrunken bounds.
                dx = abs(dist_xz) - (radius - edge_radius)
                dy = abs(y) - (half_height - edge_radius)

                # Exterior distance (if outside) + Interior distance (if inside)
                out_dist = np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                in_dist = min(max(dx, dy), 0.0)

                sdf_val = out_dist + in_dist - edge_radius

                # Store a wide band so the solver gets a gradient far from the surface.
                if abs(sdf_val) < band_halfwidth:
                    accessor.setValueOn((i, j, k), sdf_val)

    # Written into the package's objects/ directory, where the readers look.

    # These used to write a bare relative name, i.e. into the caller's CWD.

    filename = os.path.join(OBJECTS_DIR, "cylinder.vdb")
    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")


if __name__ == "__main__":
    create_cylinder_sdf()
