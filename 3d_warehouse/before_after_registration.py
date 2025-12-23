import open3d as o3d
import os
import sys
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import numpy as np
import open3d as o3d
import copy
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

def load_registered_pcd(path, color):
    if not os.path.isfile(path):
        print(f"ERROR: Registered file not found: {path}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(path)

    if len(pcd.points) == 0:
        print(f"ERROR: Empty registered point cloud: {path}")
        sys.exit(1)

    pcd.paint_uniform_color(color)
    return pcd


def main():

    # --------------------------------------------------
    # PATHS (RAW DATA)
    # --------------------------------------------------

    bag1 = r'C:\Users\panag\didymos_dataset\warehouse\warehouse_3d_capture\same_area_scans_different_intervals\Instance1'
    bag2 = r'C:\Users\panag\didymos_dataset\warehouse\warehouse_3d_capture\same_area_scans_different_intervals\Instance2'
    bag3 = r'C:\Users\panag\didymos_dataset\warehouse\warehouse_3d_capture\same_area_scans_different_intervals\Instance3'
    topic = '/map'

    # --------------------------------------------------
    # PATHS (REGISTERED FILES)
    # --------------------------------------------------

    reg1_path = r"registered_instance_1.pcd"
    reg2_path = r"registered_instance_2.pcd"
    reg3_path = r"registered_instance_3.pcd"

    # --------------------------------------------------
    # COLORS (CONSISTENT)
    # --------------------------------------------------

    colors = {
        "p1": [1.0, 0.0, 0.0],  # Red
        "p2": [0.0, 1.0, 0.0],  # Green
        "p3": [0.0, 0.0, 1.0],  # Blue
    }

    # --------------------------------------------------
    # LOAD UNREGISTERED (RAW)
    # --------------------------------------------------

    print("\n--- Loading UNREGISTERED point clouds ---")

    p1_raw = numpy_to_pcd(read_first_cloud(bag1, topic))
    p2_raw = numpy_to_pcd(read_first_cloud(bag2, topic))
    p3_raw = numpy_to_pcd(read_first_cloud(bag3, topic))

    p1_raw.paint_uniform_color(colors["p1"])
    p2_raw.paint_uniform_color(colors["p2"])
    p3_raw.paint_uniform_color(colors["p3"])

    # --------------------------------------------------
    # VISUALIZE BEFORE
    # --------------------------------------------------

    o3d.visualization.draw_geometries(
        [p1_raw, p2_raw, p3_raw],
        window_name="BEFORE Registration (Raw / Unregistered)",
        width=1280,
        height=800
    )

    # --------------------------------------------------
    # LOAD REGISTERED
    # --------------------------------------------------

    print("\n--- Loading REGISTERED point clouds ---")

    p1_reg = load_registered_pcd(reg1_path, colors["p1"])
    p2_reg = load_registered_pcd(reg2_path, colors["p2"])
    p3_reg = load_registered_pcd(reg3_path, colors["p3"])

    # --------------------------------------------------
    # VISUALIZE AFTER
    # --------------------------------------------------

    o3d.visualization.draw_geometries(
        [p1_reg, p2_reg, p3_reg],
        window_name="AFTER Registration (Aligned)",
        width=1280,
        height=800
    )


if __name__ == "__main__":
    main()
