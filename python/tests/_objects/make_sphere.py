try:
    import openvdb as vdb
except ImportError:
    import pyopenvdb as vdb
import numpy as np


def create_sphere_sdf(radius=0.025, voxel_size=0.001, band_halfwidth=0.06):
    # band_halfwidth controls how far from the surface SDF values are stored.
    # Outside this band the sampler returns the constant background (10.0) with
    # zero gradient, so the witness-point contact solver gets no pull. A wide
    # band gives a usable gradient signal far from the surface to guide the
    # solver in toward contact.
    grid = vdb.FloatGrid(10.0)
    grid.gridClass = vdb.GridClass.LEVEL_SET
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    accessor = grid.getAccessor()

    margin = int(band_halfwidth / voxel_size)
    r_vox = int(radius / voxel_size) + margin

    print("Generating SDF voxels...")
    for i in range(-r_vox, r_vox + 1):
        for j in range(-r_vox, r_vox + 1):
            for k in range(-r_vox, r_vox + 1):
                x = i * voxel_size
                y = j * voxel_size
                z = k * voxel_size
                sdf_val = np.sqrt(x**2 + y**2 + z**2) - radius
                if abs(sdf_val) < band_halfwidth:
                    accessor.setValueOn((i, j, k), sdf_val)

    filename = "sphere.vdb"
    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")


if __name__ == "__main__":
    create_sphere_sdf()
