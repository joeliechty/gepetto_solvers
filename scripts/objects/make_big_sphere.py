"""Generate ``big_sphere.vdb`` — a larger sphere level-set for the full-hand
grasp test.

Same SDF generator as ``make_sphere.py`` (radius 0.025), but baked at a bigger
radius. Radius 0.05 m is chosen so that, placed at the five-finger anatomical
hand's flexed-fingertip locus (flexor ~2 N; center ~[-0.022, 0.089, -0.016] with
an identity wrist), every fingertip — thumb included — lands within ~3 mm of the
surface, i.e. a graspable sphere. Kept as a separate asset so the original
``sphere.vdb`` (used by the single-/two-finger tests) is unchanged.

Run (from the ``python/`` directory):
    python -m tests._objects.make_big_sphere
"""

try:
    import openvdb as vdb
except ImportError:
    import pyopenvdb as vdb
import os

import numpy as np

from gepetto_solvers.core.objects import OBJECTS_DIR


def create_big_sphere_sdf(radius=0.05, voxel_size=0.001, band_halfwidth=0.06,
                          filename=None):
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

    print(f"Generating SDF voxels (radius={radius})...")
    for i in range(-r_vox, r_vox + 1):
        for j in range(-r_vox, r_vox + 1):
            for k in range(-r_vox, r_vox + 1):
                x = i * voxel_size
                y = j * voxel_size
                z = k * voxel_size
                sdf_val = np.sqrt(x**2 + y**2 + z**2) - radius
                if abs(sdf_val) < band_halfwidth:
                    accessor.setValueOn((i, j, k), sdf_val)

    # Default to the package objects/ dir, where the readers look. Resolved
    # here rather than in the signature so the default is not computed at
    # import time (ruff B008).
    if filename is None:
        filename = os.path.join(OBJECTS_DIR, "big_sphere.vdb")
    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")


if __name__ == "__main__":
    create_big_sphere_sdf()
