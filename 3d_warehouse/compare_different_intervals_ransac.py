"""
compare_different_intervals_ransac.py

Loads PRE-REGISTERED .pcd files (from step1_register_clouds.py).
Performs Geometric Separation (RANSAC) to isolate floor/walls.
Visualizes change detection on the remaining objects.
"""

import numpy as np
import open3d as o3d
import os
import sys
import copy

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
# 2) Geometric Segmentation
# -------------------------
def geometric_separate_static(pcd, dist_thresh=0.05, ransac_n=3, num_iter=1000, loops=3):
    """
    Iteratively removes the largest planes (floor/walls).
    Returns tuple: (dynamic_pcd, static_pcd)
    """
    dynamic = copy.deepcopy(pcd)
    static_points = []
    
    # Try to find 'loops' number of planes (e.g. 1 floor + 2 walls)
    for i in range(loops):
        if len(dynamic.points) < 100:
            break
            
        # RANSAC plane segmentation
        plane_model, inliers = dynamic.segment_plane(distance_threshold=dist_thresh,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iter)
        
        # If too few points are part of the plane, stop early
        if len(inliers) < 100: 
            break

        # Extract static (plane) points
        inlier_cloud = dynamic.select_by_index(inliers)
        static_points.append(np.asarray(inlier_cloud.points))
        
        # Keep dynamic (non-plane) points for next iteration
        dynamic = dynamic.select_by_index(inliers, invert=True)
        
    # Combine all static parts into one point cloud
    if static_points:
        all_static = np.vstack(static_points)
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(all_static)
        # Color static background dark grey
        static_pcd.paint_uniform_color([0.2, 0.2, 0.2]) 
    else:
        static_pcd = o3d.geometry.PointCloud()

    return dynamic, static_pcd

# -------------------------
# 3) Change detection coloring
# -------------------------
def mark_colors_unique_common(points_list, threshold=0.2):
    pcds = [numpy_to_pcd(p) for p in points_list]
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
            
            if all(found):
                color[j] = [0.5, 0.5, 0.5] # Grey (Common object)
            elif found[i] and not any(f for n, f in enumerate(found) if n != i):
                if i == 0: color[j] = [1, 0, 0]       # Red (Only in 1)
                elif i == 1: color[j] = [0, 1, 0]     # Green (Only in 2)
                elif i == 2: color[j] = [0, 0, 1]     # Blue (Only in 3)
            else:
                color[j] = [0.7, 0.7, 0.7] # Mixed/Noise
        colors.append(color)

    for pcd, c in zip(pcds, colors):
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcds

# -------------------------
# 4) MAIN
# -------------------------
def main():
    print("Loading REGISTERED clouds from disk...")
    
    files = ["registered_instance_1.pcd", "registered_instance_2.pcd", "registered_instance_3.pcd"]
    for f in files:
        if not os.path.exists(f):
            print(f"Error: File '{f}' not found.")
            print("Please run 'step1_register_clouds.py' first.")
            sys.exit(1)

    pcd1_reg = o3d.io.read_point_cloud("registered_instance_1.pcd")
    pcd2_reg = o3d.io.read_point_cloud("registered_instance_2.pcd")
    pcd3_reg = o3d.io.read_point_cloud("registered_instance_3.pcd")
    
    print(f"Loaded {len(pcd1_reg.points)} points from Inst 1.")

    # -------------------------
    # GEOMETRIC SEPARATION (Remove floors/walls)
    # -------------------------
    print("\nSeparating Static Environment (Floor/Walls) from Dynamic Objects...")
    # Adjust dist_thresh if needed (0.15 is roughly 15cm tolerance for wall flatness)
    
    dyn1, stat1 = geometric_separate_static(pcd1_reg, dist_thresh=0.15, loops=3)
    dyn2, stat2 = geometric_separate_static(pcd2_reg, dist_thresh=0.15, loops=3)
    dyn3, stat3 = geometric_separate_static(pcd3_reg, dist_thresh=0.15, loops=3)
    
    # -------------------------
    # CHANGE DETECTION (On Dynamic Objects Only)
    # -------------------------
    visualization_voxel = 0.05
    
    # Downsample only the dynamic parts for change detection
    dyn1_vis = downsample(dyn1, visualization_voxel)
    dyn2_vis = downsample(dyn2, visualization_voxel)
    dyn3_vis = downsample(dyn3, visualization_voxel)

    pts1_dyn = np.asarray(dyn1_vis.points)
    pts2_dyn = np.asarray(dyn2_vis.points)
    pts3_dyn = np.asarray(dyn3_vis.points)

    print(f"Running change detection on Objects (KDTree radius = 0.4 m)...")
    colored_dynamic_list = mark_colors_unique_common([pts1_dyn, pts2_dyn, pts3_dyn], threshold=0.4)

    # -------------------------
    # MERGE & SAVE
    # -------------------------
    print("\nMerging Static Background (Grey) with Colored Objects...")
    
    pcd_combined_final = o3d.geometry.PointCloud()
    
    # Add the colored dynamic objects
    for pcd in colored_dynamic_list:
        pcd_combined_final += pcd

    # Add the grey static backgrounds (downsampled slightly for file size)
    pcd_combined_final += downsample(stat1, visualization_voxel)
    pcd_combined_final += downsample(stat2, visualization_voxel)
    pcd_combined_final += downsample(stat3, visualization_voxel)

    output_filename = "registered_semantic_filtered_output_Ransac.pcd"
    if o3d.io.write_point_cloud(output_filename, pcd_combined_final, write_ascii=True):
        print(f"Successfully saved combined point cloud to: {output_filename}")
    else:
        print(f"ERROR: Could not save to {output_filename}")

    # -------------------------
    # VISUALIZATION
    # -------------------------
    print("Starting visualization...")
    o3d.visualization.draw_geometries(
        [pcd_combined_final],
        window_name="Geometric Segmented Change Detection",
        width=1280, height=900
    )

if __name__ == "__main__":
    main()