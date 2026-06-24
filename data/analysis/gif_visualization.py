import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_trajectory_comparison(data_source, file_name):
    """
    Visualizes raw and processed trajectories side-by-side. 
    Left: Raw data (noisy). Right: Processed data (smooth + downsampled).
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_file = project_root / "data" / data_source / "logs" / file_name
    raw_df = pd.read_csv(raw_file, skiprows=1, header=None)
    raw_full_data = raw_df.iloc[:, 14:30].values
    track_data = raw_df.iloc[:, 57:73].values
    
    save_folder = project_root / "data" / data_source / "visualizations"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / f"{file_name[:-4]}_side_by_side.gif"

    def get_pose(matrix_flat):
        # Restore 4x4 matrix to extract position (mm) and rotation
        T = matrix_flat.reshape(4, 4)
        return T[:3, 3], T[:3, :3]

    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d') # Left: Raw
    ax2 = fig.add_subplot(122, projection='3d') # Right: Processed

    # Initialization for Left (Raw)
    line_raw, = ax1.plot([], [], [], 'r-', alpha=0.6, label='Reference')
    quivers_raw = [ax1.quiver(0,0,0, 0,0,0, color=c) for c in ['r', 'g', 'b']]
    ax1.set_title("Reference Trajectory")

    # Initialization for Right (Processed)
    line_proc, = ax2.plot([], [], [], 'b-', linewidth=2, label='Tracking')
    quivers_proc = [ax2.quiver(0,0,0, 0,0,0, color=c) for c in ['r', 'g', 'b']]
    ax2.set_title("Tracking Trajectory")

    # Unified Scaling Logic to keep XYZ proportions equal
    all_p_raw = np.array([get_pose(r)[0] for r in raw_full_data[::10]]) 
    max_range = np.array([all_p_raw[:,0].max()-all_p_raw[:,0].min(), 
                          all_p_raw[:,1].max()-all_p_raw[:,1].min(), 
                          all_p_raw[:,2].max()-all_p_raw[:,2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = all_p_raw.mean(axis=0)

    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range - 5, mid_x + max_range + 5)
        ax.set_ylim(mid_y - max_range - 5, mid_y + max_range + 5)
        ax.set_zlim(mid_z - max_range - 5, mid_z + max_range + 5)
        ax.set_box_aspect((1, 1, 1)) # Uniform aspect ratio
        ax.legend()

    def update(frame):
        # Update Left (Raw) - show history and current pose
        p_raw_slice = np.array([get_pose(raw_full_data[i])[0] for i in range(0, frame + 1, 5)])
        line_raw.set_data(p_raw_slice[:, 0], p_raw_slice[:, 1])
        line_raw.set_3d_properties(p_raw_slice[:, 2])
        
        pos_r, rot_r = get_pose(raw_full_data[frame])
        nonlocal quivers_raw
        for i in range(3):
            quivers_raw[i].remove()
            quivers_raw[i] = ax1.quiver(pos_r[0], pos_r[1], pos_r[2], rot_r[0,i], rot_r[1,i], rot_r[2,i], 
                                       color=['r','g','b'][i], length=15, normalize=True)

        # Update Right (Processed) - show history and current pose
        p_proc_slice = np.array([get_pose(track_data[i])[0] for i in range(frame + 1)])
        line_proc.set_data(p_proc_slice[:, 0], p_proc_slice[:, 1])
        line_proc.set_3d_properties(p_proc_slice[:, 2])
        
        pos_p, rot_p = get_pose(track_data[frame])
        nonlocal quivers_proc
        for i in range(3):
            quivers_proc[i].remove()
            quivers_proc[i] = ax2.quiver(pos_p[0], pos_p[1], pos_p[2], rot_p[0,i], rot_p[1,i], rot_p[2,i], 
                                        color=['r','g','b'][i], length=15, normalize=True)

        return line_raw, line_proc, *quivers_raw, *quivers_proc

    # Animation frames based on the shorter downsampled array to maintain 1:1 time scale
    ani = FuncAnimation(fig, update, frames=len(track_data), interval=50)
    ani.save(save_path, writer='pillow')
    plt.close()


if __name__ == "__main__":
    data_source = "260408"  # Source data folder
    file_name = "policy_new_ignore_h_Log_3.0_O.csv"  # "250_Log_3.0_O.csv"  # 
    animate_trajectory_comparison(data_source, file_name)
    