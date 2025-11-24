import open3d as o3d
import numpy as np

# Load the full point cloud
pcd = o3d.io.read_point_cloud("time1.pcd")
print(f"Original number of points: {len(pcd.points)}")

# Downsample: keep 1 point every 10
points = np.asarray(pcd.points)#[::10]
if pcd.has_colors():
    colors = np.asarray(pcd.colors)#[::10]
else:
    colors = None

if pcd.has_normals():
    normals = np.asarray(pcd.normals)#[::10]
else:
    normals = None

# Create new point cloud
pcd_down = o3d.geometry.PointCloud()
pcd_down.points = o3d.utility.Vector3dVector(points)

if colors is not None:
    pcd_down.colors = o3d.utility.Vector3dVector(colors)

if normals is not None:
    pcd_down.normals = o3d.utility.Vector3dVector(normals)

print(f"Downsampled number of points: {len(pcd_down.points)}")

# Visualize
o3d.visualization.draw_geometries([pcd_down])
