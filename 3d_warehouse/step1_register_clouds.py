"""
step1_register_clouds.py

Isolates the working registration logic (Global RANSAC + Multiscale GICP).
Reads ROS2 bags, aligns them to Instance 1, and saves them as .pcd files.
"""

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy as np
import open3d as o3d
import copy
import os

# -------------------------
# 1) Read PointCloud2 from ROS2 bag
# -------------------------
def read_first_cloud(bag_path, topic_name):
    print(f"Reading {bag_path}...")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic_name:
                msg = reader.deserialize(rawdata, connection.msgtype)
                point_step = msg.point_step
                dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                if point_step > 12:
                    dtype_list.append(('padding', 'u1', point_step - 12))
                cloud_arr = np.frombuffer(msg.data, dtype=dtype_list)
                pts = np.vstack((cloud_arr['x'], cloud_arr['y'], cloud_arr['z'])).T
                pts = pts[~np.isnan(pts).any(axis=1)]
                return pts
    return None

# -------------------------
# 2) Registration Utilities
# -------------------------
def numpy_to_pcd(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def downsample(pcd, voxel):
    return pcd.voxel_down_sample(voxel_size=voxel)

def estimate_normals(pcd, radius_normal, max_nn=30):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn))

def compute_fpfh(pcd_down, voxel_size):
    radius_feature = voxel_size * 5.0
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def global_ransac_init(source, target, voxel_size):
    print(f"  -> Running Global RANSAC (voxel={voxel_size})...")
    src_down = downsample(source, voxel_size)
    tgt_down = downsample(target, voxel_size)
    estimate_normals(src_down, voxel_size * 2.0)
    estimate_normals(tgt_down, voxel_size * 2.0)
    fpfh_src = compute_fpfh(src_down, voxel_size)
    fpfh_tgt = compute_fpfh(tgt_down, voxel_size)

    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, fpfh_src, fpfh_tgt, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(40000, 1000)
    )
    return result

def multiscale_gicp_refine(source, target, voxel_list, max_correspondence_factor=0.5):
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)
    current_trans = np.eye(4)

    for v in voxel_list:
        print(f"  -> GICP Refinement at scale: {v}m")
        src_down = downsample(src, v)
        tgt_down = downsample(tgt, v)
        
        estimate_normals(src_down, radius_normal=v * 2.0)
        estimate_normals(tgt_down, radius_normal=v * 2.0)

        max_corr_dist = max(v * max_correspondence_factor, 0.01)

        try:
            result_gicp = o3d.pipelines.registration.registration_generalized_icp(
                src_down, tgt_down, max_corr_dist, current_trans,
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            current_trans = result_gicp.transformation
        except Exception as e:
            print(f"     GICP Failed at {v}m, using Point-to-Point fallback.")
            result_icp = o3d.pipelines.registration.registration_icp(
                src_down, tgt_down, max_corr_dist, current_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            current_trans = result_icp.transformation

    src.transform(current_trans)
    return src

# -------------------------
# MAIN
# -------------------------
def main():
    # PATHS
    bag1 = r'C:\Users\panag\didymos_dataset\warehouse\warehouse_3d_capture\same_area_scans_different_intervals\Instance1'
    bag2 = r'C:\Users\panag\didymos_dataset\warehouse\warehouse_3d_capture\same_area_scans_different_intervals\Instance2'
    bag3 = r'C:\Users\panag\didymos_dataset\warehouse\warehouse_3d_capture\same_area_scans_different_intervals\Instance3'
    topic = '/map'

    # 1. Load Raw Data
    print("\n--- 1. Loading Point Clouds ---")
    p1 = numpy_to_pcd(read_first_cloud(bag1, topic))
    p2 = numpy_to_pcd(read_first_cloud(bag2, topic))
    p3 = numpy_to_pcd(read_first_cloud(bag3, topic))

    # 2. Registration Config
    registration_voxel = 1.0
    voxel_list = [1.0, 0.5, 0.2, 0.1]

    # 3. Register 2 -> 1
    print("\n--- 2. Registering Instance 2 to Instance 1 ---")
    ransac12 = global_ransac_init(p2, p1, voxel_size=registration_voxel)
    p2.transform(ransac12.transformation) # Apply coarse alignment
    p2_reg = multiscale_gicp_refine(p2, p1, voxel_list) # Apply fine alignment

    # 4. Register 3 -> 1
    print("\n--- 3. Registering Instance 3 to Instance 1 ---")
    ransac13 = global_ransac_init(p3, p1, voxel_size=registration_voxel)
    p3.transform(ransac13.transformation) # Apply coarse alignment
    p3_reg = multiscale_gicp_refine(p3, p1, voxel_list) # Apply fine alignment
    
    # 5. Save Results
    print("\n--- 4. Saving Registered Files ---")
    o3d.io.write_point_cloud("registered_instance_1.pcd", p1)
    print("Saved: registered_instance_1.pcd")
    
    o3d.io.write_point_cloud("registered_instance_2.pcd", p2_reg)
    print("Saved: registered_instance_2.pcd")
    
    o3d.io.write_point_cloud("registered_instance_3.pcd", p3_reg)
    print("Saved: registered_instance_3.pcd")
    
    print("\nDone! Use these files in your Cylinder3D script.")

if __name__ == "__main__":
    main()
    