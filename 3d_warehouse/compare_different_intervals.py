"""
compare_different_intervals.py

Loads PRE-REGISTERED .pcd files (from step1_register_clouds.py).
Performs only the change detection coloring and visualization.
"""

import numpy as np
import open3d as o3d
import os
import sys

# -------------------------
# 1) Utilities
# -------------------------
def numpy_to_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def downsample(pcd, voxel):
    return pcd.voxel_down_sample(voxel_size=voxel)

# -------------------------
# 2) Change detection coloring
# -------------------------
def mark_colors_unique_common(points_list, threshold=0.2):
    # Convert numpy points to Open3D clouds if they aren't already
    pcds = []
    for p in points_list:
        if isinstance(p, o3d.geometry.PointCloud):
            pcds.append(p)
        else:
            pcds.append(numpy_to_pcd(p))

    kdtree_list = [o3d.geometry.KDTreeFlann(pcd) for pcd in pcds]
    colors = []

    for i, pcd in enumerate(pcds):
        pts = np.asarray(pcd.points)
        color = np.zeros_like(pts)
        for j, point in enumerate(pts):
            found = [False] * len(pcds)
            for k, kdtree in enumerate(kdtree_list):
                _, idx, _ = kdtree.search_radius_vector_3d(point, threshold)
                if len(idx) > 0:
                    found[k] = True
            
            # Logic for coloring
            if all(found):
                color[j] = [0.5, 0.5, 0.5] # Grey (Common/Static)
            elif found[i] and not any(f for n, f in enumerate(found) if n != i):
                # Unique to this instance
                if i == 0: color[j] = [1, 0, 0] # Red
                if i == 1: color[j] = [0, 1, 0] # Green
                if i == 2: color[j] = [0, 0, 1] # Blue
            else:
                color[j] = [0.7, 0.7, 0.7] # Partial overlap
        
        colors.append(color)

    for pcd, c in zip(pcds, colors):
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcds

# -------------------------
# 3) MAIN
# -------------------------
def main():
    print("Loading REGISTERED clouds from disk...")
    
    # Check if files exist
    files = ["registered_instance_1.pcd", "registered_instance_2.pcd", "registered_instance_3.pcd"]
    for f in files:
        if not os.path.exists(f):
            print(f"Error: File '{f}' not found.")
            print("Please run 'step1_register_clouds.py' first to generate these files.")
            sys.exit(1)

    # 1. Load the pre-registered clouds
    pcd1_reg = o3d.io.read_point_cloud("registered_instance_1.pcd")
    pcd2_reg = o3d.io.read_point_cloud("registered_instance_2.pcd")
    pcd3_reg = o3d.io.read_point_cloud("registered_instance_3.pcd")

    print(f"Loaded: {len(pcd1_reg.points)}, {len(pcd2_reg.points)}, {len(pcd3_reg.points)} points.")

    # -------------------------
    # Visualization (Downsampling for clean display)
    # -------------------------
    visualization_voxel = 0.05
    print(f"Downsampling for visualization (voxel={visualization_voxel})...")
    
    pcd1_vis = downsample(pcd1_reg, visualization_voxel)
    pcd2_vis = downsample(pcd2_reg, visualization_voxel)
    pcd3_vis = downsample(pcd3_reg, visualization_voxel)

    # Convert to numpy for KDTree processing
    pts1_vis = np.asarray(pcd1_vis.points)
    pts2_vis = np.asarray(pcd2_vis.points)
    pts3_vis = np.asarray(pcd3_vis.points)

    print("Running change detection (KDTree radius = 0.4 m)...")
    # pcd_out_list contains the three colored point clouds
    pcd_out_list = mark_colors_unique_common([pcd1_vis, pcd2_vis, pcd3_vis], threshold=0.4)

    # --- SAVE THE RESULT ---
    print("\nMerging and saving the colored point cloud...")
    
    pcd_combined_colored = o3d.geometry.PointCloud()
    for pcd in pcd_out_list:
        pcd_combined_colored += pcd 
    
    output_filename = "registered_colored_output.pcd"
    if o3d.io.write_point_cloud(output_filename, pcd_combined_colored, write_ascii=True):
        print(f"Successfully saved combined colored point cloud to: {output_filename}")
    else:
        print(f"ERROR: Could not save the point cloud to {output_filename}")

    # -------------------------
    # Visualization
    # -------------------------
    print("Starting visualization...")
    o3d.visualization.draw_geometries(
        [pcd_combined_colored],
        window_name="Registered Change Detection (Baseline)",
        width=1280, height=900
    )

if __name__ == "__main__":
    main()