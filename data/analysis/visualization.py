import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from data.analysis.utils import rotvec_to_angle_axes


def plot_y_t_motor(data_source: str, log_name: str, y_name: str):
    """
    Plot y-t curve for motor data and save figure.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
        y_name (str): One of ["l_motor", "vel_motor", "torque_motor", "target_vel_motor"].
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / (y_name + "_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)

    t = df.iloc[:, 1]

    if y_name == "target_vel_motor":
        y = df.iloc[:, 112:121]
        # convert to mm/s
        y.iloc[:, 0:3] = -y.iloc[:, 0:3] * np.pi / 7500
        y.iloc[:, 3:6] = -y.iloc[:, 3:6] * np.pi / 2500
        y.iloc[:, 6:9] = -y.iloc[:, 6:9] * np.pi / 7500
        ylabel = "Target motor velocity (mm/s)"
        ytitle = "Target Motor Velocity"
    elif y_name == "l_motor":
        y = df.iloc[:, 30:39]
        ylabel = "Motor rotations (mm)"
        ytitle = "Motor Rotations"
    elif y_name == "vel_motor":
        y = df.iloc[:, 39:48]
        ylabel = "Motor velocity (mm/s)"
        ytitle = "Motor Velocity"
    elif y_name == "torque_motor":
        y = df.iloc[:, 48:57]
        ylabel = "Motor torque (Nm)"
        ytitle = "Motor Torque"
    else:
        raise ValueError(f"Invalid y_name: {y_name}. Must be one of "
                         '["target_vel_motor", "l_motor", "vel_motor", "torque_motor"]')
    
    if not np.issubdtype(y.values.dtype, np.number) or not np.issubdtype(t.values.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, y_name={y_name}")
        return False
    if not np.isfinite(y.values).all() or not np.isfinite(t.values).all():
        print(f"Error: NaN or inf in data, file {log_name}, y_name={y_name}")
        return False


    # Plot 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(f"{ytitle} vs Time", fontsize=14)

    axes = axes.flatten()
    for i in range(y.shape[1]):
        axes[i].plot(t, y.iloc[:, i], label=f"Motor {i+1}")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True)

    for ax in axes[6:]:
        ax.set_xlabel("Time (s)")
    
    for ax in axes[0:-1:3]:
        ax.set_ylabel(ylabel)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close()

    return True


def plot_cmp_vel_motor(data_source: str, log_name: str):
    """
    Plot y-t curve for motor data and save figure.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / ("cmp_vel_motor_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)

    t = df.iloc[1:, 1]

    y_target = df.iloc[:-1, 112:121]
    # convert to mm/s
    y_target.iloc[:, 0:3] = -y_target.iloc[:, 0:3] * np.pi / 7500
    y_target.iloc[:, 3:6] = -y_target.iloc[:, 3:6] * np.pi / 2500
    y_target.iloc[:, 6:9] = -y_target.iloc[:, 6:9] * np.pi / 7500
    y_actual = df.iloc[1:, 39:48]

    if not np.issubdtype(y_target.values.dtype, np.number) or not np.issubdtype(
        y_actual.values.dtype, np.number) or not np.issubdtype(t.values.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, motor velocity")
        return False
    if not (np.isfinite(y_target.values).all() and np.isfinite(y_actual.values).all() and np.isfinite(t.values).all()):
        print(f"Error: NaN or inf in data, file {log_name}, motor velocity")
        return False

    # Plot 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle("Motor Velocity vs Time", fontsize=14)

    axes = axes.flatten()
    for i in range(y_target.shape[1]):
        axes[i].plot(t, y_target.iloc[:, i], label=f"Motor {i+1} Target")
        axes[i].plot(t, y_actual.iloc[:, i], label=f"Motor {i+1} Actual")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True)

    for ax in axes[6:]:
        ax.set_xlabel("Time (s)")
    
    for ax in axes[0:-1:3]:
        ax.set_ylabel("Motor velocity (mm/s)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def plot_cmp_ee_path(data_source: str, log_name: str):
    """
    Plot end-effector path and save figure.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / ("cmp_trans_ee_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)
    t = df.iloc[1:, 1].to_numpy()

    trans_ee = df.iloc[1:, 57:73].to_numpy()
    trans_ee_ref = df.iloc[:-1, 14:30].to_numpy()

    if not np.issubdtype(t.dtype, np.number) or not np.issubdtype(
        trans_ee.dtype, np.number) or not np.issubdtype(trans_ee_ref.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector translation")
        return False
    if not (np.isfinite(t).all() and np.isfinite(trans_ee).all() and np.isfinite(trans_ee_ref).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector translation")
        return False

    translations = []
    for row in trans_ee:
        trans = row.reshape(4, 4)
        translations.append(trans[:3, 3])
    translations = np.array(translations)  # Nx3

    translations_ref = []
    for row in trans_ee_ref:
        trans = row.reshape(4, 4)
        translations_ref.append(trans[:3, 3])
    translations_ref = np.array(translations_ref)  # Nx3

    # Plot 3D trajectory vs time
    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111, projection="3d")
    ax.plot(translations_ref[:, 0], translations_ref[:, 1], translations_ref[:, 2], 
            label="Reference Trajectory")
    ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 
            label="Actual Trajectory")
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title("End-Effector Trajectory", fontsize=14)

    # Range equalization
    all_points = np.vstack([translations, translations_ref])
    x_min, x_max = all_points[:,0].min(), all_points[:,0].max()
    y_min, y_max = all_points[:,1].min(), all_points[:,1].max()
    z_min, z_max = all_points[:,2].min(), all_points[:,2].max()

    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2.0
    mid_x = (x_max+x_min)/2
    mid_y = (y_max+y_min)/2
    mid_z = (z_max+z_min)/2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1,1,1])

    # plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(6, 5))
    fig.suptitle("End-Effector Translation", fontsize=14)

    axes = axes.flatten()
    for i in range(3):
        axes[i].plot(t, translations_ref[:, i], label="Refrence Positions", color='blue')
        axes[i].plot(t, translations[:, i], label="Actual Positions", color='orange')
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel(f"translation {i} (mm)")

    axes[0].set_ylim(mid_x - max_range, mid_x + max_range)
    axes[1].set_ylim(mid_y - max_range, mid_y + max_range)
    axes[2].set_ylim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = save_folder / ("cmp_positions_" + log_name.replace(".csv", ".png"))
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def plot_cmp_ee_error(data_source: str, log_name: str):
    """
    Plot Euclidean distance and geodesic error of end-effector trajectories.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / ("cmp_error_ee_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)

    t = df.iloc[1:, 1].to_numpy()
    trans_ee = df.iloc[1:, 57:73].to_numpy()
    trans_ee_ref = df.iloc[:-1, 14:30].to_numpy()

    if not np.issubdtype(t.dtype, np.number) or not np.issubdtype(
        trans_ee.dtype, np.number) or not np.issubdtype(trans_ee_ref.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector translation")
        return None, None
    if not (np.isfinite(t).all() and np.isfinite(trans_ee).all() and np.isfinite(trans_ee_ref).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector translation")
        return None, None

    # Compute translation and rotation errors
    trans_errors = []
    rot_errors = []

    for row_act, row_ref in zip(trans_ee, trans_ee_ref):
        T_act = row_act.reshape(4, 4)
        T_ref = row_ref.reshape(4, 4)

        # Translation error (Euclidean distance)
        p_act = T_act[:3, 3]
        p_ref = T_ref[:3, 3]
        trans_err = np.linalg.norm(p_act - p_ref)

        # Rotation error (geodesic distance)
        R_act = T_act[:3, :3]
        R_ref = T_ref[:3, :3]
        R_rel = R_ref.T @ R_act
        cos_theta = (np.trace(R_rel) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # avoid numerical issues
        rot_err = np.arccos(cos_theta)  # radians

        trans_errors.append(trans_err)
        rot_errors.append(rot_err)

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    rot_errors = rot_errors * (180.0 / np.pi)  # convert to degrees

    # Plot errors vs time
    fig, ax1 = plt.subplots(figsize=(6,4))

    ax1.plot(t, trans_errors, label="Position Error (mm)")
    ax1.set_xlabel("Time (s)", fontsize=8)
    ax1.set_ylabel("Position Error (mm)", fontsize=8)
    ax1.tick_params(labelsize=8)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t, rot_errors, color="orange", label="Orientation Error (deg)")
    ax2.set_ylabel("Orientation Error (deg)", fontsize=8)
    ax2.tick_params(labelsize=8)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.title("End-Effector Error vs Time", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    avg_ee_pos_error = np.mean(trans_errors)
    avg_ee_geodesic_error = np.mean(rot_errors)

    return avg_ee_pos_error, avg_ee_geodesic_error


def plot_cmp_ee_quaternion(data_source: str, log_name: str):
    """
    Plot rotation (quaternion) for end-effector and save figure.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / ("cmp_quat_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)

    t = df.iloc[1:, 1].to_numpy()

    # convert to quaternions
    rotation_matrix = np.eye(3)[None, :, :].repeat(df.shape[0] - 1, axis=0)
    rotation_matrix[:, 0, :] = df.iloc[1:, 57:60].to_numpy()
    rotation_matrix[:, 1, :] = df.iloc[1:, 61:64].to_numpy()
    rotation_matrix[:, 2, :] = df.iloc[1:, 65:68].to_numpy()

    if not np.issubdtype(t.dtype, np.number) or not np.issubdtype(
        rotation_matrix.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector quaternion")
        return False
    if not (np.isfinite(t).all() and np.isfinite(rotation_matrix).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector quaternion")
        return False

    try:
        rot = R.from_matrix(rotation_matrix)
    except ValueError as e:
        print(f"Error converting rotation matrix to quaternion in file {log_name}: {e}")
        return False
    q_xyzw = rot.as_quat()  # quarternion in (x, y, z, w) format
    q_wxyz = np.concatenate([q_xyzw[:, 3:], q_xyzw[:, :3]], axis=1)  # convert to (w, x, y, z) format

    rotation_matrix_ref = np.eye(3)[None, :, :].repeat(df.shape[0] - 1, axis=0)
    rotation_matrix_ref[:, 0, :] = df.iloc[:-1, 14:17].to_numpy()
    rotation_matrix_ref[:, 1, :] = df.iloc[:-1, 18:21].to_numpy()
    rotation_matrix_ref[:, 2, :] = df.iloc[:-1, 22:25].to_numpy()

    if not np.issubdtype(rotation_matrix_ref.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector quaternion")
        return False
    if not (np.isfinite(rotation_matrix_ref).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector quaternion")
        return False

    try:
        rot_ref = R.from_matrix(rotation_matrix_ref)
    except ValueError as e:
        print(f"Error converting reference rotation matrix to quaternion in file {log_name}: {e}")
        return False
    q_xyzw_ref = rot_ref.as_quat()  # quarternion in (x, y, z, w) format
    q_wxyz_ref = np.concatenate([q_xyzw_ref[:, 3:], q_xyzw_ref[:, :3]], axis=1)  # convert to (w, x, y, z) format

    # Plot 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True)
    fig.suptitle("End-Effector Rotation in Quaternion vs Time", fontsize=14)

    axes = axes.flatten()
    for i in range(q_wxyz.shape[1]):
        axes[i].plot(t, q_wxyz_ref[:, i], label="Target")
        axes[i].plot(t, q_wxyz[:, i], label="Actual")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True)

    for ax in axes[2:]:
        ax.set_xlabel("Time (s)")

    axes[0].set_ylabel("w")
    axes[1].set_ylabel("x")
    axes[2].set_ylabel("y")
    axes[3].set_ylabel("z")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def plot_cmp_ee_rotvec(data_source: str, log_name: str):
    """
    Plot rotation (rotvec) for end-effector and save figure.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / ("cmp_rotvec_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)

    t = df.iloc[1:, 1].to_numpy()

    # convert to quaternions
    rotation_matrix = np.eye(3)[None, :, :].repeat(df.shape[0] - 1, axis=0)
    rotation_matrix[:, 0, :] = df.iloc[1:, 57:60].to_numpy()
    rotation_matrix[:, 1, :] = df.iloc[1:, 61:64].to_numpy()
    rotation_matrix[:, 2, :] = df.iloc[1:, 65:68].to_numpy()

    if not np.issubdtype(t.dtype, np.number) or not np.issubdtype(
        rotation_matrix.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector rotvec")
        return False
    if not (np.isfinite(t).all() and np.isfinite(rotation_matrix).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector rotvec")
        return False
    
    try:
        rot = R.from_matrix(rotation_matrix)
    except ValueError as e:
        print(f"Error converting rotation matrix to rotvec in file {log_name}: {e}")
        return False
    rotvec = rot.as_rotvec()  # axis-angle

    rotation_matrix_ref = np.eye(3)[None, :, :].repeat(df.shape[0] - 1, axis=0)
    rotation_matrix_ref[:, 0, :] = df.iloc[:-1, 14:17].to_numpy()
    rotation_matrix_ref[:, 1, :] = df.iloc[:-1, 18:21].to_numpy()
    rotation_matrix_ref[:, 2, :] = df.iloc[:-1, 22:25].to_numpy()
    if not np.issubdtype(rotation_matrix_ref.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector rotvec")
        return False
    if not (np.isfinite(rotation_matrix_ref).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector rotvec")
        return False
    
    try:
        rot_ref = R.from_matrix(rotation_matrix_ref)
    except ValueError as e:
        print(f"Error converting reference rotation matrix to rotvec in file {log_name}: {e}")
        return False
    rotvec_ref = rot_ref.as_rotvec()  # axis-angle

    # Plot 3x1 subplots
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True, sharey=True)
    fig.suptitle("End-Effector Rotation in RotVec vs Time", fontsize=14)

    axes = axes.flatten()
    for i in range(rotvec.shape[1]):
        axes[i].plot(t, rotvec_ref[:, i], label="Target")
        axes[i].plot(t, rotvec[:, i], label="Actual")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel(f"rotvec {i} (rad)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def plot_cmp_ee_angle_axes(data_source: str, log_name: str):
    """
    Plot rotation (angle-axis) for end-effector and save figure.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        log_name (str): Log file name (e.g., "Log_1.0_B_0.0_1.0_0.0.csv").
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    file_path = project_root / "data" / data_source / "logs" / log_name
    save_folder = project_root / "data" / data_source / "figs"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / ("cmp_angle_axes_" + log_name.replace(".csv", ".png"))

    # Read data
    df = pd.read_csv(file_path, skiprows=1, header=None)

    t = df.iloc[1:, 1].to_numpy()

    # convert to quaternions
    rotation_matrix = np.eye(3)[None, :, :].repeat(df.shape[0] - 1, axis=0)
    rotation_matrix[:, 0, :] = df.iloc[1:, 57:60].to_numpy()
    rotation_matrix[:, 1, :] = df.iloc[1:, 61:64].to_numpy()
    rotation_matrix[:, 2, :] = df.iloc[1:, 65:68].to_numpy()
    if not np.issubdtype(t.dtype, np.number) or not np.issubdtype(
        rotation_matrix.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector angle-axis")
        return False
    if not (np.isfinite(t).all() and np.isfinite(rotation_matrix).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector angle-axis")
        return False
    
    try:
        rot = R.from_matrix(rotation_matrix)
    except ValueError as e:
        print(f"Error converting rotation matrix to angle-axis in file {log_name}: {e}")
        return False
    rotvec = rot.as_rotvec()  # axis-angle
    angle_axes = rotvec_to_angle_axes(rotvec)

    rotation_matrix_ref = np.eye(3)[None, :, :].repeat(df.shape[0] - 1, axis=0)
    rotation_matrix_ref[:, 0, :] = df.iloc[:-1, 14:17].to_numpy()
    rotation_matrix_ref[:, 1, :] = df.iloc[:-1, 18:21].to_numpy()
    rotation_matrix_ref[:, 2, :] = df.iloc[:-1, 22:25].to_numpy()
    if not np.issubdtype(rotation_matrix_ref.dtype, np.number):
        print(f"Error: Non-numeric data type in data, file {log_name}, end-effector angle-axis")
        return False
    if not (np.isfinite(rotation_matrix_ref).all()):
        print(f"Error: NaN or inf in data, file {log_name}, end-effector angle-axis")
        return False
    
    try:
        rot_ref = R.from_matrix(rotation_matrix_ref)
    except ValueError as e:
        print(f"Error converting reference rotation matrix to angle-axis in file {log_name}: {e}")
        return False
    rotvec_ref = rot_ref.as_rotvec()  # axis-angle
    angle_axes_ref = rotvec_to_angle_axes(rotvec_ref)

    # Plot 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharex=True, sharey=True)
    fig.suptitle("End-Effector Rotation in Angle-Axes vs Time", fontsize=14)

    axes = axes.flatten()
    axes[0].plot(t, angle_axes_ref[:, 0], label="Target")
    axes[0].plot(t, angle_axes[:, 0], label="Actual")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True)
    axes[0].set_ylabel("Angle (rad)")
    for i in range(1, angle_axes.shape[1]):
        axes[i].plot(t, angle_axes_ref[:, i], label="Target")
        axes[i].plot(t, angle_axes[:, i], label="Actual")
        axes[i].legend(loc="upper right", fontsize=8)
        axes[i].grid(True)
        axes[i].set_ylabel(f"Axis {i} (rad)")

    for ax in axes[2:]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close()
    return True


def visualize_results(data_source, log_name):
    y_name = "l_motor"
    plot_y_t_motor(data_source, log_name, y_name)
    y_name = "vel_motor"
    plot_y_t_motor(data_source, log_name, y_name)
    y_name = "torque_motor"
    plot_y_t_motor(data_source, log_name, y_name)

    plot_cmp_vel_motor(data_source, log_name)

    plot_cmp_ee_path(data_source, log_name)
    avg_ee_pos_error, avg_ee_geodesic_error = plot_cmp_ee_error(data_source, log_name)
    plot_cmp_ee_rotvec(data_source, log_name)

    return avg_ee_pos_error, avg_ee_geodesic_error
    

if __name__ == "__main__":
    data_source = "260205_test"  # "251223_random"   # "origin_tracking_251117" # 

    log_name  = f"Log_5.0_T.csv"

    y_name = "l_motor"
    plot_y_t_motor(data_source, log_name, y_name)
    y_name = "vel_motor"
    plot_y_t_motor(data_source, log_name, y_name)
    y_name = "torque_motor"
    plot_y_t_motor(data_source, log_name, y_name)

    plot_cmp_vel_motor(data_source, log_name)

    plot_cmp_ee_path(data_source, log_name)
    avg_ee_pos_error, avg_ee_geodesic_error = plot_cmp_ee_error(data_source, log_name)
    
    # plot_cmp_ee_quaternion(data_source, log_name)
    plot_cmp_ee_rotvec(data_source, log_name)
    # plot_cmp_ee_angle_axes(data_source, log_name)

    if avg_ee_pos_error is not None and avg_ee_geodesic_error is not None:
        print(f"Avg EE Position Error: {avg_ee_pos_error:.2f} mm")
        print(f"Avg EE Geodesic Error: {avg_ee_geodesic_error:.2f} deg")