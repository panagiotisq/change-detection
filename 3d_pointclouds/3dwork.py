import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib

# ========== CONFIGURATION ==========
FILE1 = "time1.pcd" # C:/Users/panag/Desktop/labwork/change_detection/3d/
FILE2 = "time2.pcd"
VOXEL_SIZE = 0.2        # Size of voxel cell (in same unit as your point cloud)
THRESHOLD = 1.5         # Distance threshold to detect change
DBSCAN_EPS = 1.5       # DBSCAN max distance
DBSCAN_MIN_SAMPLES = 4   # DBSCAN minimum points per cluster
# ===================================

# --- Load point clouds ---
pcd1 = o3d.io.read_point_cloud(FILE1)
pcd2 = o3d.io.read_point_cloud(FILE2)
points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)

# --- Compute nearest neighbor distances ---
tree1 = KDTree(points1)
tree2 = KDTree(points2)
d1, _ = tree2.query(points1, k=1)
d2, _ = tree1.query(points2, k=1)



'''-----------------------changes-----------------------'''

# --- Step 1: Filter significantly changed points ---
changed_points1 = points1[d1 > THRESHOLD]
changed_points2 = points2[d2 > THRESHOLD]

# Combine changed points from both timestamps
changed_points = np.vstack((changed_points1, changed_points2))

if len(changed_points) == 0:
    print("No significant point-level changes found.")
    exit()


# --- Step 2: DBSCAN clustering on changed points ---
db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(changed_points)
labels = db.labels_

# --- Step 3: Separate clusters into time1 and time2 groups ---
mask1 = np.isin(changed_points, changed_points1).all(axis=1)
mask2 = np.isin(changed_points, changed_points2).all(axis=1)

labels1 = labels[mask1]
labels2 = labels[mask2]

# --- Build color maps ---
red_cmap = matplotlib.colormaps.get_cmap("Reds")     # shades of red
green_cmap = matplotlib.colormaps.get_cmap("Greens") # shades of green

colors = np.zeros((len(changed_points), 3))  # initialize RGB

# Assign colors for points from time1 (reds)
unique_labels1 = [l for l in set(labels1) if l != -1]
for i, l in enumerate(unique_labels1):
    cluster_mask = (labels == l) & mask1
    colors[cluster_mask] = red_cmap(i / max(1, len(unique_labels1)))[:3]

# Assign colors for points from time2 (greens)
unique_labels2 = [l for l in set(labels2) if l != -1]
for i, l in enumerate(unique_labels2):
    cluster_mask = (labels == l) & mask2
    colors[cluster_mask] = green_cmap(i / max(1, len(unique_labels2)))[:3]

# Noise points (label == -1) → black
colors[labels == -1] = (0, 0, 0)

# --- Step 4: Filter out noise ---
valid_mask = labels != -1
filtered_points = changed_points[valid_mask]
filtered_colors = colors[valid_mask]

# --- Step 5: Output summary ---
n_clusters_total = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters1 = len(set(labels1)) - (1 if -1 in labels1 else 0)
n_clusters2 = len(set(labels2)) - (1 if -1 in labels2 else 0)
n_noise = np.sum(labels == -1)

print(f"Detected {n_clusters_total} total change clusters.")
print(f"  • {n_clusters1} clusters from time1 (disappeared → red hues)")
print(f"  • {n_clusters2} clusters from time2 (appeared → green hues)")
print(f"  • {n_noise} noise points")

# ---- unchanged points ----
unchanged_points1 = points1[d1 <= THRESHOLD]
unchanged_points2 = points2[d2 <= THRESHOLD]
unchanged_points = np.vstack((unchanged_points1, unchanged_points2))

gray_color = np.array([[0.6, 0.6, 0.6]])  # light gray
unchanged_colors = np.repeat(gray_color, len(unchanged_points), axis=0)

# --- Combine unchanged + clustered changed ---
combined_points = np.vstack((unchanged_points, filtered_points))
combined_colors = np.vstack((unchanged_colors, filtered_colors))

# --- Step 6: Visualize ---
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(combined_points)
final_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

if len(combined_points) == 0:
    print("⚠️ No points to visualize (check THRESHOLD and DBSCAN parameters).")
else:
    o3d.visualization.draw_geometries([final_pcd])
