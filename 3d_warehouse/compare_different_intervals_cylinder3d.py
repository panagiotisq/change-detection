import torch
import numpy as np
import open3d as o3d
import os
import sys
import copy
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Check Imports
try:
    import mmcv
    from mmdet3d.apis import init_model, inference_detector
    import mmdet3d
    from mmengine.dataset import Compose
except ImportError as e:
    print(f"\nImport Error: {e}")
    sys.exit(1)

# -------------------------
# 1) CONFIGURATION
# -------------------------

# A. AUTO-LOCATE CONFIG FILE
package_root = os.path.dirname(mmdet3d.__file__)
possible_paths = [
    os.path.join(package_root, '.mim', 'configs', 'cylinder3d', 'cylinder3d_4xb4-3x_semantickitti.py'),
    os.path.join(package_root, 'configs', 'cylinder3d', 'cylinder3d_4xb4-3x_semantickitti.py'),
]

CONFIG_FILE = None
for p in possible_paths:
    if os.path.exists(p):
        CONFIG_FILE = p
        print(f"Found Config: {CONFIG_FILE}")
        break

if CONFIG_FILE is None:
    print("Error: Could not automatically find the Cylinder3D config file.")
    sys.exit(1)

# B. CHECKPOINT
CHECKPOINT_FILE = 'cylinder3d_4xb4_3x_semantickitti_20230318_191107-822a8c31.pth'

if not os.path.exists(CHECKPOINT_FILE):
    print(f"Error: Checkpoint file '{CHECKPOINT_FILE}' not found.")
    sys.exit(1)


# MMDetection3D Training Labels (0-18)
# ONLY remove clear ground classes. 
# We remove: 8 (Road), 9 (Parking), 10 (Sidewalk), 11 (Other-ground), 16 (Terrain)
# We KEEP:  13 (Fence), 14 (Vegetation), etc. so they can be checked for changes.
STATIC_LABELS = [8, 9, 10, 11, 12, 16]

# -------------------------
# 2) UTILITIES
# -------------------------
def numpy_to_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def downsample(pcd, voxel):
    return pcd.voxel_down_sample(voxel_size=voxel)

# -------------------------
# 3) AI FILTERING (Cylinder3D) - TUPLE FIX
# -------------------------
def filter_static_cylinder3d(pcd, model):
    print(f"   -> Inference on {len(pcd.points)} points...")
    points = np.asarray(pcd.points)
    
    # Fake Intensity column
    points_4d = np.hstack([points, np.zeros((len(points), 1))]).astype(np.float32)
    
    temp_bin = "temp_scan.bin"
    points_4d.tofile(temp_bin)
    
    # Run Inference
    try:
        result = inference_detector(model, temp_bin)
    except Exception as e:
        print(f"      CRITICAL INFERENCE FAILURE: {e}")
        if os.path.exists(temp_bin): os.remove(temp_bin)
        return o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    # Cleanup
    if os.path.exists(temp_bin): os.remove(temp_bin)

    # Parse Labels
    labels = None
    try:
        if isinstance(result, tuple):
            data_sample = result[0]
        elif isinstance(result, list):
            data_sample = result[0]
        else:
            data_sample = result

        if hasattr(data_sample, 'pred_pts_seg'):
            labels = data_sample.pred_pts_seg.pts_semantic_mask.cpu().numpy()
        elif hasattr(data_sample, 'pts_semantic_mask'):
            labels = data_sample.pts_semantic_mask.cpu().numpy()
            
        # --- NEW DEBUG PRINT ---
        if labels is not None:
            unique_classes = np.unique(labels)
            print(f"      [DEBUG] Detected classes: {unique_classes}")
        # -----------------------
            
    except Exception as e:
        print(f"      Error parsing result: {e}")
        return pcd, o3d.geometry.PointCloud()

    if labels is None:
        print("      Failed to extract labels (Tuple fix didn't work?)")
        # Print debug again just in case
        print(f"      Tuple contents: {result}")
        return pcd, o3d.geometry.PointCloud()

    # Filter
    is_dynamic = ~np.isin(labels, STATIC_LABELS)
    
    dynamic_pts = points[is_dynamic]
    static_pts = points[~is_dynamic]
    
    print(f"      Found {len(static_pts)} static points (Floor/Wall)")
    print(f"      Found {len(dynamic_pts)} dynamic points (Objects)")
    
    dyn_pcd = o3d.geometry.PointCloud()
    dyn_pcd.points = o3d.utility.Vector3dVector(dynamic_pts)
    
    stat_pcd = o3d.geometry.PointCloud()
    stat_pcd.points = o3d.utility.Vector3dVector(static_pts)
    stat_pcd.paint_uniform_color([0.2, 0.2, 0.2]) # Dark Grey

    return dyn_pcd, stat_pcd


# -------------------------
# 4) Change Detection
# -------------------------
def detect_changes(pcd_list, threshold=0.4, min_height=0.20):
    # --- CONFIGURATION TOGGLE ---
    USE_HEIGHT_FILTER = True  # Set to True to enable the floor guardrail, False to disable it
    # ----------------------------

    if any(len(p.points) == 0 for p in pcd_list):
        return []

    kdtree_list = [o3d.geometry.KDTreeFlann(p) for p in pcd_list]
    
    # Auto-detect floor z (Calculated regardless, used only if filter is ON)
    all_pts = np.vstack([np.asarray(p.points) for p in pcd_list])
    if len(all_pts) > 0:
        floor_z = np.min(all_pts[:, 2])
        if USE_HEIGHT_FILTER:
            print(f"   -> Auto-detected floor level at Z = {floor_z:.2f}m")
            print(f"   -> [ACTIVE] Ignoring changes below Z = {floor_z + min_height:.2f}m")
        else:
            print(f"   -> [INACTIVE] Floor height detected at Z = {floor_z:.2f}m but filter is DISABLED.")
    else:
        floor_z = 0.0

    for i, pcd in enumerate(pcd_list):
        pts = np.asarray(pcd.points)
        colors = np.zeros_like(pts)
        for j, point in enumerate(pts):
            
            # Floor Guardrail (Controlled by flag)
            if USE_HEIGHT_FILTER and point[2] < (floor_z + min_height):
                colors[j] = [0.2, 0.2, 0.2] # Dark Grey (Floor)
                continue

            found = [False] * len(pcd_list)
            for k, tree in enumerate(kdtree_list):
                _, idx, _ = tree.search_radius_vector_3d(point, threshold)
                if len(idx) > 0: found[k] = True
            
            if all(found):
                colors[j] = [0.6, 0.6, 0.6] # Light Grey (Static Object)
            elif found[i] and not any(f for n, f in enumerate(found) if n != i):
                if i == 0: colors[j] = [1, 0, 0] # Red
                if i == 1: colors[j] = [0, 1, 0] # Green
                if i == 2: colors[j] = [0, 0, 1] # Blue
            else:
                colors[j] = [0.6, 0.6, 0.6] # Partial overlap
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd_list

# -------------------------
# MAIN
# -------------------------
def main():
    print("Initializing Cylinder3D Model...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    
    print("Patching config for inference (DEEP SCRUB)...")
    
    # 1. Define the Safe Pipeline (No Labels)
    safe_pipeline = [
        dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='Pack3DDetInputs', keys=['points'])
    ]

    # 2. Patch Global Pipeline
    model.cfg.test_pipeline = safe_pipeline
    model.cfg.train_pipeline = safe_pipeline # Just in case
    
    # 3. Patch Dataloader Pipeline (This is where your error was hiding!)
    if hasattr(model.cfg, 'test_dataloader') and hasattr(model.cfg.test_dataloader, 'dataset'):
        model.cfg.test_dataloader.dataset.pipeline = safe_pipeline
        # Fix box_type while we are here
        if not hasattr(model.cfg.test_dataloader.dataset, 'box_type_3d'):
             model.cfg.test_dataloader.dataset.box_type_3d = 'LiDAR'

    # 4. Force Rebuild Internal Pipeline
    model.test_pipeline = Compose(safe_pipeline)
    
    print("Config patched. Loading REGISTERED clouds...")
    
    if not os.path.exists("registered_instance_1.pcd"):
        print("Error: Registered files not found. Run step1_register_clouds.py first!")
        sys.exit(1)
        
    p1 = o3d.io.read_point_cloud("registered_instance_1.pcd")
    p2 = o3d.io.read_point_cloud("registered_instance_2.pcd")
    p3 = o3d.io.read_point_cloud("registered_instance_3.pcd")

    process_voxel = 0.05

    print("\n--- Filtering Instance 1 ---")
    d1, s1 = filter_static_cylinder3d(downsample(p1, process_voxel), model)
    
    print("\n--- Filtering Instance 2 ---")
    d2, s2 = filter_static_cylinder3d(downsample(p2, process_voxel), model)
    
    print("\n--- Filtering Instance 3 ---")
    d3, s3 = filter_static_cylinder3d(downsample(p3, process_voxel), model)

    print("\nDetecting changes on Objects...")
    colored = detect_changes([d1, d2, d3], threshold=0.4)
    
    print("Merging scenes...")
    final_scene = o3d.geometry.PointCloud()
    for c in colored: final_scene += c
    final_scene += s1 + s2 + s3
    
    output_filename = "cylinder3d_final_result.pcd"
    if len(final_scene.points) > 0:
        print(f"Saving final colored cloud to {output_filename}...")
        o3d.io.write_point_cloud(output_filename, final_scene, write_ascii=True)
        print("Save successful.")
        print("Visualizing...")
        o3d.visualization.draw_geometries([final_scene], window_name="Cylinder3D Change Detection")
    else:
        print("Error: Final scene is empty.")

if __name__ == "__main__":
    main()