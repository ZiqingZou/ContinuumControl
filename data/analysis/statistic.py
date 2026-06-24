import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from data.analysis.utils import rotvec_to_angle_axes


def statistical_trans_ee(logs_path: Path, save_folder: Path) -> dict:
    """
    Compute statistics for the end-effector translation data.

    Args:
        logs_path (Path): Path to the logs directory.
        save_folder (Path): Path to the folder to save scatter plots.

    Returns:
        dict: A dictionary containing mean, std, min, and max position statistics.
    """
    t_total = []
    p_total = []
    delta_p_total = []
    q_wxyz_total = []
    rotvec_total = []
    angle_axes_total = []
    delta_rotvec_total = []
    R_delta_rotvec_total = []
    delta_rot_angle_total = []
    for log_file in os.listdir(logs_path):
        if not log_file.endswith(".csv"):
            continue
        file_path = logs_path / log_file
        df = pd.read_csv(file_path, skiprows=1, header=None)

        t =  df.iloc[:, 1:2].to_numpy()  # time
        p = df.iloc[:, 60:70:4].to_numpy()  # position
        delta_p = p[1:, :] - p[:-1, :]  # delta position

        rotation_matrix = np.eye(3)[None, :, :].repeat(df.shape[0], axis=0)
        rotation_matrix[:, 0, :] = df.iloc[:, 57:60].to_numpy()
        rotation_matrix[:, 1, :] = df.iloc[:, 61:64].to_numpy()
        rotation_matrix[:, 2, :] = df.iloc[:, 65:68].to_numpy()
        rot = R.from_matrix(rotation_matrix)
        q_xyzw = rot.as_quat()  # quarternion in (x, y, z, w) format
        q_wxyz = np.concatenate([q_xyzw[:, 3:], q_xyzw[:, :3]], axis=1)  # convert to (w, x, y, z) format
        rotvec = rot.as_rotvec()
        delta_rotvec = rotvec[1:, :] - rotvec[:-1, :]
        angle_axes = rotvec_to_angle_axes(rotvec)

        # compute rotvec (rotvec)
        R_delta = R.from_quat(q_xyzw[1:, :]) * R.from_quat(q_xyzw[:-1, :]).inv()
        R_delta_rotvec = R_delta.as_rotvec()
        delta_rot_angle = np.linalg.norm(R_delta_rotvec, axis=1)

        t_total.append(t)
        p_total.append(p)
        delta_p_total.append(delta_p)
        q_wxyz_total.append(q_wxyz)
        rotvec_total.append(rotvec)
        angle_axes_total.append(angle_axes)
        delta_rotvec_total.append(delta_rotvec)
        R_delta_rotvec_total.append(R_delta_rotvec)
        delta_rot_angle_total.append(delta_rot_angle)
    t_total = np.vstack(t_total)
    p_total = np.vstack(p_total)
    delta_p_total = np.vstack(delta_p_total)
    q_wxyz_total = np.vstack(q_wxyz_total)
    rotvec_total = np.vstack(rotvec_total)
    angle_axes_total = np.vstack(angle_axes_total)
    delta_rotvec_total = np.vstack(delta_rotvec_total)
    R_delta_rotvec_total = np.vstack(R_delta_rotvec_total)
    delta_rot_angle_total = np.concatenate(delta_rot_angle_total)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(delta_rot_angle_total.shape[0]), delta_rot_angle_total,
                s=0.1, alpha=0.1, label="delta_rot_angle")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of delta_rot_angle")
    plt.xlabel("Data Index")
    plt.ylabel("Rotation Angle (rad)")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_delta_rot_angle.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(p_total.shape[1]):
        plt.scatter(i + np.arange(p_total.shape[0]) / p_total.shape[0], p_total[:, i],
                    s=0.1, alpha=0.1, label=f"pos {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of position")
    plt.xlabel("Pos Index")
    plt.ylabel("Position (mm)")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_position.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(delta_p_total.shape[1]):
        plt.scatter(i + np.arange(delta_p_total.shape[0]) / delta_p_total.shape[0], delta_p_total[:, i],
                    s=0.1, alpha=0.1, label=f"delta_pos {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of delta_position")
    plt.xlabel("Delta Pos Index")
    plt.ylabel("Delta Position (mm)")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_delta_position.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(R_delta_rotvec_total.shape[1]):
        plt.scatter(i + np.arange(R_delta_rotvec_total.shape[0]) / R_delta_rotvec_total.shape[0], R_delta_rotvec_total[:, i], 
                    s=0.1, alpha=0.1, label=f"R_delta_rotvec {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of diff_rotvec")
    plt.xlabel("Diff Rotvec Index")
    plt.ylabel("Diff Rotvec (rad)")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_diff_rotvec.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(delta_rotvec_total.shape[1]):
        plt.scatter(i + np.arange(delta_rotvec_total.shape[0]) / delta_rotvec_total.shape[0], delta_rotvec_total[:, i], 
                    s=0.1, alpha=0.1, label=f"delta_rotvec {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of delta_rotvec")
    plt.xlabel("Delta Rotvec Index")
    plt.ylabel("Delta Rotvec (rad)")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_delta_rotvec.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(q_wxyz_total.shape[1]):
        plt.scatter(i + np.arange(q_wxyz_total.shape[0]) / q_wxyz_total.shape[0], q_wxyz_total[:, i], 
                    s=0.1, alpha=0.1, label=f"quat {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of quaternion (w, x, y, z)")
    plt.xlabel("Quat Index")
    plt.ylabel("Quaternion Component")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_quaternion.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(rotvec_total.shape[1]):
        plt.scatter(i + np.arange(rotvec_total.shape[0]) / rotvec_total.shape[0], rotvec_total[:, i], 
                    s=0.1, alpha=0.1, label=f"rotvec {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of rotation vector")
    plt.xlabel("Rotvec Index")
    plt.ylabel("Rotation Vector Component")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_rotvec.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(angle_axes_total.shape[1]):
        plt.scatter(i + np.arange(angle_axes_total.shape[0]) / angle_axes_total.shape[0], angle_axes_total[:, i], 
                    s=0.1, alpha=0.1, label=f"angle_axis {i}")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title(f"Scatter Plot of angle-axes")
    plt.xlabel("Angle-Axes Index")
    plt.ylabel("Angle-Axes Component")
    plt.grid(True)
    plt.savefig(save_folder / f"scatter_angle_axes.png")
    plt.close()

    position_stats = {
        "mean_pos": np.mean(p_total, axis=0),
        "mean_abs_pos": np.mean(np.abs(p_total), axis=0),
        "std_pos": np.std(p_total, axis=0),
        "min_pos": np.min(p_total, axis=0),
        "max_pos": np.max(p_total, axis=0),
    }

    delta_position_stats = {
        "mean_delta_pos": np.mean(delta_p_total, axis=0),
        "mean_abs_delta_pos": np.mean(np.abs(delta_p_total), axis=0),
        "std_delta_pos": np.std(delta_p_total, axis=0),
        "min_delta_pos": np.min(delta_p_total, axis=0),
        "max_delta_pos": np.max(delta_p_total, axis=0),
    }

    quaternion_stats = {
        "mean_quat": np.mean(q_wxyz_total, axis=0),
        "mean_abs_quat": np.mean(np.abs(q_wxyz_total), axis=0),
        "std_quat": np.std(q_wxyz_total, axis=0),
        "min_quat": np.min(q_wxyz_total, axis=0),
        "max_quat": np.max(q_wxyz_total, axis=0),
    }

    rotvec_stats = {
        "mean_rotvec": np.mean(rotvec_total, axis=0),
        "std_rotvec": np.std(rotvec_total, axis=0),
        "min_rotvec": np.min(rotvec_total, axis=0),
        "max_rotvec": np.max(rotvec_total, axis=0),

    }

    angle_axes_stats = {
        "mean_angle_axes": np.mean(angle_axes_total, axis=0),
        "std_angle_axes": np.std(angle_axes_total, axis=0),
        "min_angle_axes": np.min(angle_axes_total, axis=0),
        "max_angle_axes": np.max(angle_axes_total, axis=0),
    }

    R_delta_rotvec_stats = {
        "mean_R_delta_rotvec": np.mean(R_delta_rotvec_total, axis=0),
        "mean_abs_R_delta_rotvec": np.mean(np.abs(R_delta_rotvec_total), axis=0),
        "std_R_delta_rotvec": np.std(R_delta_rotvec_total, axis=0),
        "min_R_delta_rotvec": np.min(R_delta_rotvec_total, axis=0),
        "max_R_delta_rotvec": np.max(R_delta_rotvec_total, axis=0),
    }

    delta_rotvec_stats = {
        "mean_delta_rotvec": np.mean(delta_rotvec_total, axis=0),
        "std_delta_rotvec": np.std(delta_rotvec_total, axis=0),
        "min_delta_rotvec": np.min(delta_rotvec_total, axis=0),
        "max_delta_rotvec": np.max(delta_rotvec_total, axis=0),
    }

    delta_rot_angle_stats = {
        "mean_delta_rot_angle": np.mean(delta_rot_angle_total, axis=0),
        "std_delta_rot_angle": np.std(delta_rot_angle_total, axis=0),
        "min_delta_rot_angle": np.min(delta_rot_angle_total, axis=0),
        "max_delta_rot_angle": np.max(delta_rot_angle_total, axis=0),
    }

    stats = {**position_stats, **delta_position_stats,
             **quaternion_stats, **rotvec_stats, **angle_axes_stats,
             **R_delta_rotvec_stats, **delta_rotvec_stats, **delta_rot_angle_stats}
    return stats


def statistical_y(data_source: str, y_name: str):
    """
    Compute statistics for a specific variable (y_name) in the dataset.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        y_name (str): One of ["trans_ee", "l_motor", "vel_motor", "torque_motor", "target_vel_motor"].
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    logs_path = project_root / "data" / data_source / "logs"
    save_folder = project_root / "data" / data_source / "stats"
    os.makedirs(save_folder, exist_ok=True)
    save_path = save_folder / f"stats_{y_name}.txt"

    stats = None

    if y_name == "target_vel_motor":
        la = 112
        lb = 121
    elif y_name == "l_motor":
        la = 30
        lb = 39
    elif y_name == "vel_motor":
        la = 39
        lb = 48
    elif y_name == "torque_motor":
        la = 48
        lb = 57
    elif y_name == "trans_ee":
        stats = statistical_trans_ee(logs_path, save_folder)
    else:
        raise ValueError(f"Invalid y_name: {y_name}. Must be one of "
                        '["trans_ee", "l_motor", "vel_motor", "torque_motor", "target_vel_motor"]')

    # Read data
    if stats is None:
        t_total = []
        y_total = []
        delta_y_total = []
        for log_file in os.listdir(logs_path):
            if not log_file.endswith(".csv"):
                continue
            file_path = logs_path / log_file
            df = pd.read_csv(file_path, skiprows=1, header=None)
            t =  df.iloc[:, 1:2].to_numpy()
            y = df.iloc[:, la:lb].to_numpy()
            delta_y = y[1:, :] - y[:-1, :]
            t_total.append(t)
            y_total.append(y)
            delta_y_total.append(delta_y)
        t_total = np.vstack(t_total)
        y_total = np.vstack(y_total)
        delta_y_total = np.vstack(delta_y_total)
        if y_name == "target_vel_motor":
            # convert to mm/s
            y_total[:, 0:3] = -y_total[:, 0:3] * np.pi / 7500
            y_total[:, 3:6] = -y_total[:, 3:6] * np.pi / 2500
            y_total[:, 6:9] = -y_total[:, 6:9] * np.pi / 7500
            delta_y_total[:, 0:3] = -delta_y_total[:, 0:3] * np.pi / 7500
            delta_y_total[:, 3:6] = -delta_y_total[:, 3:6] * np.pi / 2500
            delta_y_total[:, 6:9] = -delta_y_total[:, 6:9] * np.pi / 7500

        # Compute statistics
        stats = {
            "mean": np.mean(y_total, axis=0),
            "mean_abs": np.mean(np.abs(y_total), axis=0),
            "std": np.std(y_total, axis=0),
            "min": np.min(y_total, axis=0),
            "max": np.max(y_total, axis=0),
            "mean_delta": np.mean(delta_y_total, axis=0),
            "mean_abs_delta": np.mean(np.abs(delta_y_total), axis=0),
            "std_delta": np.std(delta_y_total, axis=0),
            "min_delta": np.min(delta_y_total, axis=0),
            "max_delta": np.max(delta_y_total, axis=0),
        }

        # Scatter plot
        plt.figure(figsize=(8, 6))
        for i in range(3):
            plt.scatter(i + np.arange(y_total.shape[0])/y_total.shape[0],
                         y_total[:, i * 3] + y_total[:, i * 3 + 1] + y_total[:, i * 3 + 2],
                        s=0.1, alpha=0.1, label=f"motor {i}")
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.title(f"Scatter Plot of Joint Sum {y_name}")
        plt.xlabel("Joint Index")
        plt.ylabel(y_name)
        plt.grid(True)
        plt.savefig(save_folder / f"scatter_joint_sum_{y_name}.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        for i in range(delta_y_total.shape[1]):
            plt.scatter(i + np.arange(delta_y_total.shape[0])/delta_y_total.shape[0], delta_y_total[:, i], 
                        s=0.1, alpha=0.1, label=f"delta_motor {i}")
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.title(f"Scatter Plot of delta_{y_name}")
        plt.xlabel("Delta Motor Index")
        plt.ylabel(f"delta_{y_name}")
        plt.grid(True)
        plt.savefig(save_folder / f"scatter_delta_{y_name}.png")
        plt.close()

        if "target_vel_motor" in y_name:
            plt.figure(figsize=(8, 6))
            for i in range(3):
                plt.scatter(i + np.arange(y_total.shape[0])/y_total.shape[0], y_total[:, i], 
                            s=0.1, alpha=0.1, label=f"motor {i}")
            plt.legend(loc='upper right', fontsize='small', ncol=2)
            plt.title(f"Scatter Plot of {y_name}")
            plt.xlabel("Motor Index")
            plt.ylabel(y_name)
            plt.grid(True)
            plt.savefig(save_folder / f"scatter_{y_name}.png")
            plt.close()

    # Save statistics to file
    with open(save_path, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    data_source = "total"  # "noise_tracking"   # "origin_tracking_251117" # 
    y_name = "l_motor"
    statistical_y(data_source, y_name)
    y_name = "vel_motor"
    statistical_y(data_source, y_name)
    y_name = "torque_motor"
    statistical_y(data_source, y_name)
    y_name = "target_vel_motor"
    statistical_y(data_source, y_name)
    y_name = "trans_ee"
    statistical_y(data_source, y_name)
