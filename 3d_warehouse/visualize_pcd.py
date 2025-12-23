import open3d as o3d
import os
import multiprocessing
import time

def worker_visualize(filename, window_title, position_offset):
    """
    Worker function to run in a separate process.
    Opens one Open3D window.
    """
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return

    print(f"Loading {filename}...")
    pcd = o3d.io.read_point_cloud(filename)
    
    if not pcd.has_points():
        print(f"Warning: {filename} is empty.")
        return

    # Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=800, height=600, left=position_offset, top=100)
    vis.add_geometry(pcd)
    
    # Optional: Set a nice view or background color here
    opt = vis.get_render_option()
    opt.background_color = [0, 0, 0] # Black background
    
    vis.run()
    vis.destroy_window()

def main():
    # List of files you want to open
    # Format: (Filename, Title)
    files_to_view = [
        ("results/registered_colored_output.pcd", "1. GICP Registration (Raw)"),
        ("results/registered_semantic_filtered_output_Ransac.pcd", "2. Semantic Filter (RANSAC)"), 
        ("results/registered_SAM_filtered_output.pcd", "3. SAM Filtered (Image-based)"),
        ("results/cylinder3d_final_result.pcd", "4. Cylinder3D"),
        ("results/cylinder3d_final_result_GOOD_RESULT.pcd", "5. Cylinder3D fine-tuned (Best Result)")
    ]
    #files_to_view=[
     #   ("registered_instance_1.pcd","1st scan")
    #]

    processes = []
    screen_offset = 0

    print("Launching windows...")
    
    for filename, title in files_to_view:
        # We start a new process for each window
        p = multiprocessing.Process(target=worker_visualize, args=(filename, title, screen_offset))
        p.start()
        processes.append(p)
        
        # Shift the next window slightly to the right so they don't perfectly overlap
        screen_offset += 100 

    # Wait for all windows to be closed before exiting the script
    for p in processes:
        p.join()

    print("All windows closed.")

if __name__ == "__main__":
    # On Windows, multiprocessing requires this protection
    multiprocessing.freeze_support()
    main()