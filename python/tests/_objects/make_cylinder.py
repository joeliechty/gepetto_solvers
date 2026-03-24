import pyopenvdb as vdb
import numpy as np

def create_cylinder_sdf(radius=0.01, height=0.08, voxel_size=0.002):
    # Create an empty floating-point grid. The background value represents 
    # the distance in empty space (positive means outside the object).
    grid = vdb.FloatGrid(10.0) 
    grid.gridClass = vdb.GridClass.LEVEL_SET
    
    # Set the voxel size (2mm resolution)
    grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)

    # Define the bounding box of our cylinder
    half_height = height / 2.0
    
    # Create an accessor to write values to the grid efficiently
    accessor = grid.getAccessor()

    # Iterate over a bounding box slightly larger than the cylinder
    margin = int(0.02 / voxel_size) # 2cm margin
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
                
                # Mathematical SDF for a cylinder aligned with the Y-axis
                # Distance from center axis in XZ plane
                dist_xz = np.sqrt(x**2 + z**2)
                
                # Distance vector to the bounds [radius, half_height]
                dx = abs(dist_xz) - radius
                dy = abs(y) - half_height
                
                # Exterior distance (if outside) + Interior distance (if inside)
                out_dist = np.sqrt(max(dx, 0)**2 + max(dy, 0)**2)
                in_dist = min(max(dx, dy), 0.0)
                
                sdf_val = out_dist + in_dist
                
                # Only store values near the surface to keep the grid sparse
                if abs(sdf_val) < 0.03: 
                    accessor.setValueOn((i, j, k), sdf_val)

    # Save to file
    filename = "cylinder.vdb"
    vdb.write(filename, grids=[grid])
    print(f"Saved {filename}!")

if __name__ == "__main__":
    create_cylinder_sdf()