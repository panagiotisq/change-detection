import open3d as o3d
import os
import numpy as np
import copy
import time

def get_camera_parameters(vis):
    """Helper to get current camera parameters (intrinsics + extrinsics)"""
    ctr = vis.get_view_control()
    return ctr.convert_to_pinhole_camera_parameters()

def set_camera_parameters(vis, params):
    """Helper to set camera parameters"""
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(params)

def main():
    # --- CONFIGURATION ---
    files_to_view = [
        ("results/registered_colored_output.pcd", "1. Geometric approach"),
        ("results/registered_semantic_filtered_output_Ransac.pcd", "2. Semantic Filter (RANSAC)"),
        #("results/registered_SAM_filtered_output.pcd", "3. SAM Filtered (Image-based)"),
        ("results/cylinder3d_final_result.pcd", "3. Cylinder3D  --machine learning approach "),
        ("results/cylinder3d_final_result_GOOD_RESULT.pcd", "4. Cylinder3D (BEST Result) --machine learning approach")
    ]
    
    # --- WINDOW SETTINGS ---
    W_WIDTH = 640
    W_HEIGHT = 480
    PAD_X = 0   # Horizontal spacing
    PAD_Y = 50  # Vertical spacing (includes title bar)
    ROW_LIMIT = 1200 # Wrap to next row after this many pixels width
    # -----------------------

    visualizers = []
    
    # 1. Setup Windows
    screen_x = 0
    screen_y = 50
    
    print(f"Initializing synchronized windows ({W_WIDTH}x{W_HEIGHT})...")
    print("NOTE: Rotate ANY window, and the others will follow.")
    
    for filename, title in files_to_view:
        if not os.path.exists(filename):
            print(f"Skipping {filename} (Not found)")
            continue

        pcd = o3d.io.read_point_cloud(filename)
        if not pcd.has_points():
            print(f"Skipping {filename} (Empty)")
            continue

        # Create Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=W_WIDTH, height=W_HEIGHT, 
                          left=screen_x, top=screen_y)
        vis.add_geometry(pcd)
        
        # Style
        opt = vis.get_render_option()
        opt.background_color = [0, 0, 0]
        opt.point_size = 2.0
        
        visualizers.append(vis)
        
        # Calculate position for next window
        screen_x += W_WIDTH + PAD_X
        
        # Grid Logic: Wrap to next row if we run out of horizontal space
        if screen_x > ROW_LIMIT: 
            screen_x = 0
            screen_y += W_HEIGHT + PAD_Y

    if not visualizers:
        print("No valid point clouds found to visualize.")
        return

    # 2. Synchronization Loop
    last_extrinsics = [np.eye(4) for _ in visualizers]
    
    keep_running = True
    while keep_running:
        
        leader_idx = -1
        
        # A. Poll events for all windows and find the "Leader"
        for i, vis in enumerate(visualizers):
            if not vis.poll_events():
                keep_running = False
                break
            
            vis.update_renderer()
            
            # Check if camera has moved
            current_cam = get_camera_parameters(vis)
            current_extrinsic = current_cam.extrinsic
            
            # If changed, this window is the leader
            if not np.allclose(current_extrinsic, last_extrinsics[i], atol=1e-6):
                leader_idx = i
                
            last_extrinsics[i] = current_extrinsic

        if not keep_running:
            break

        # B. Sync everyone to the Leader
        if leader_idx != -1:
            leader_cam = get_camera_parameters(visualizers[leader_idx])
            
            for i, vis in enumerate(visualizers):
                if i == leader_idx: continue
                set_camera_parameters(vis, leader_cam)
                last_extrinsics[i] = leader_cam.extrinsic

        time.sleep(0.01)

    for vis in visualizers:
        vis.destroy_window()

if __name__ == "__main__":
    main()