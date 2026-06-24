import os
import h5py
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from config.parser import Config


def process_y(data_source: str, save_folder: str, ref_horizon: int, history_window: int, train_only: bool = False):
    """
    Pre-process dataset file by file.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
        save_folder (str): Folder name to save processed data (e.g., "origin_tracking_100").
        ref_horizon (int): Reference horizon for each step.
        history_window (int): History window size concatenated before step 1. Step -history_window to step -1.
        downsample_rate (int): Downsample rate for the data. 1 means no downsampling.
        Data in reality part: step 0 to (length - 2), length_new = length - 1.
        Final processed data: step -history_window to (length - 2), 
            length_out = history_window + length - 1.
    """
    # Find paths
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    logs_path = project_root / "data" / data_source / "logs"
    save_folder_train = project_root / "data" / save_folder / "processed_logs_train"
    shutil.rmtree(save_folder_train, ignore_errors=True)
    os.makedirs(save_folder_train, exist_ok=True)
    if not train_only:
        save_folder_eval = project_root / "data" / save_folder / "processed_logs_eval"
        shutil.rmtree(save_folder_eval, ignore_errors=True)
        os.makedirs(save_folder_eval, exist_ok=True)

    trainset_length = 0
    evalset_length = 0
    sucessful_files = 0

    for log_file in os.listdir(logs_path):
        if not log_file.endswith(".csv"):
            continue
        file_path = logs_path / log_file
        df = pd.read_csv(file_path, skiprows=1, header=None)

        u = df.iloc[:, 112:121].to_numpy()  # [length, ctl_dim=9]
        l = df.iloc[:, 30:39].to_numpy()  # [length, 9]
        v = df.iloc[:, 39:48].to_numpy()  # [length, 9]
        torque = df.iloc[:, 48:57].to_numpy()  # [length, 9]
        rotation_matrix = np.eye(3)[None, :, :].repeat(df.shape[0], axis=0)
        rotation_matrix[:, 0, :] = df.iloc[:, 57:60].to_numpy()
        rotation_matrix[:, 1, :] = df.iloc[:, 61:64].to_numpy()
        rotation_matrix[:, 2, :] = df.iloc[:, 65:68].to_numpy()
        pos = df.iloc[:, 60:70:4].to_numpy()  # [length, 3]
        ref_rotation_matrix = np.eye(3)[None, :, :].repeat(df.shape[0], axis=0)
        ref_rotation_matrix[:, 0, :] = df.iloc[:, 14:17].to_numpy()
        ref_rotation_matrix[:, 1, :] = df.iloc[:, 18:21].to_numpy()
        ref_rotation_matrix[:, 2, :] = df.iloc[:, 22:25].to_numpy()
        ref_pos = df.iloc[:, 17:27:4].to_numpy()  # [length, 3]

        # convert to mm/s
        u[:, 0:3] = -u[:, 0:3] * np.pi / 7500
        u[:, 3:6] = -u[:, 3:6] * np.pi / 2500
        u[:, 6:9] = -u[:, 6:9] * np.pi / 7500
        u_t = u[:-1, :]  # [length_new, 9]
        u_t_minus_1 = np.concatenate([np.zeros((1, 9)), u[:-2, :]], axis=0)  # [length_new, 9]
        u_t = np.concatenate([np.zeros((history_window, 9)), u_t], axis=0)  # [length_out, 9]
        u_t_minus_1 = np.concatenate([np.zeros((history_window, 9)), u_t_minus_1], axis=0)  # [length_out, 9]

        l_t = l[:-1, :]  # [length_new, 9]
        l_t_plus_1 = l[1:, :]  # [length_new, 9]
        l_t = np.concatenate([
            np.repeat(l[0:1, :], history_window, axis=0), l_t], axis=0)  # [length_out, 9]
        l_t_plus_1 = np.concatenate([
            np.repeat(l[0:1, :], history_window, axis=0), l_t_plus_1], axis=0)  # [length_out, 9]

        v_t = v[:-1, :]  # [length_new, 9]
        v_t_plus_1 = v[1:, :]  # [length_new, 9]
        v_t = np.concatenate([np.zeros((history_window, 9)), v_t], axis=0)  # [length_out, 9]
        v_t_plus_1 = np.concatenate([
            np.zeros((history_window - 1, 9)), v[0:1, :], v_t_plus_1], axis=0)  # [length_out, 9]

        torque_t = torque[:-1, :]  # [length_new, 9]
        torque_t_plus_1 = torque[1:, :]  # [length_new, 9]
        torque_t = np.concatenate([
            np.zeros((history_window, 9)), torque_t], axis=0)  # [length_out, 9]
        torque_t_plus_1 = np.concatenate([
            np.zeros((history_window - 1, 9)), torque[0:1, :], torque_t_plus_1], axis=0)  # [length_out, 9]

        try: 
            rot = R.from_matrix(rotation_matrix) 
        except Exception as e: 
            print(f"{e}") 
            print(log_file)
            break
        rotvec = rot.as_rotvec()  # [length, 3]
        rotvec_t = rotvec[:-1, :]  # [length_new, 3]
        rotvec_t_plus_1 = rotvec[1:, :]  # [length_new, 3]
        rotvec_t = np.concatenate([
            np.repeat(rotvec[0:1, :], history_window, axis=0), rotvec_t], axis=0)  # [length_out, 3]
        rotvec_t_plus_1 = np.concatenate([
            np.repeat(rotvec[0:1, :], history_window, axis=0), rotvec_t_plus_1], axis=0)  # [length_out, 3]

        pos_t = pos[:-1, :]  # [length_new, 3]
        pos_t_plus_1 = pos[1:, :]  # [length_new, 3]
        pos_t = np.concatenate(
            [np.repeat(pos[0:1, :], history_window, axis=0), pos_t], axis=0)  # [length_out, 3]
        pos_t_plus_1 = np.concatenate([
            np.repeat(pos[0:1, :], history_window, axis=0), pos_t_plus_1], axis=0)  # [length_out, 3]

        try: 
            ref_rot = R.from_matrix(ref_rotation_matrix) 
        except Exception as e: 
            print(f"{e}") 
            break
        ref_rotvec = ref_rot.as_rotvec()  # [length, 3]
        ref_rotvec_minus = rotvec[0:1, :]
        ref_pos_minus = pos[0:1, :]

        length = ref_pos.shape[0]
        parts = []
        for i in range(ref_horizon):
            # ref_{t+i}
            if history_window > i:
                pre_part_rotvec = np.repeat(ref_rotvec_minus, history_window - i, axis=0)  # [history_window - i, 3]
                mid_part_rotvec = ref_rotvec[:-1, :]  # [length_new, 3]
                post_part_rotvec = np.repeat(ref_rotvec[-1:, :], i, axis=0)  # [i, 3]
                part_rotvec = np.concatenate([
                    pre_part_rotvec, mid_part_rotvec, post_part_rotvec], axis=0)  # [length_out, 3]
                parts.append(part_rotvec)
                pre_part_pos = np.repeat(ref_pos_minus, history_window - i, axis=0)  # [history_window - i, 3]
                mid_part_pos = ref_pos[:-1, :]  # [length_new, 3]
                post_part_pos = np.repeat(ref_pos[-1:, :], i, axis=0)  # [i, 3]
                part_pos = np.concatenate([
                    pre_part_pos, mid_part_pos, post_part_pos], axis=0)  # [length_out, 3]
                parts.append(part_pos)
            else:
                pre_part_rotvec = ref_rotvec[i - history_window : -1, :]  # [length_out - i, 3]
                post_part_rotvec = np.repeat(ref_rotvec[-1:, :], i, axis=0)  # [i, 3]
                part_rotvec = np.concatenate([pre_part_rotvec, post_part_rotvec], axis=0)  # [length_out, 3]
                parts.append(part_rotvec)
                pre_part_pos = ref_pos[i - history_window : -1, :]  # [length_out - i, 3]
                post_part_pos = np.repeat(ref_pos[-1:, :], i, axis=0)  # [i, 3]
                part_pos = np.concatenate([pre_part_pos, post_part_pos], axis=0)  # [length_out, 3]
                parts.append(part_pos)
        ref_t = np.concatenate(parts, axis=1)  # [length_out, 6 * ref_horizon]

        parts = []
        for i in range(ref_horizon):
            # ref_{t+i}
            if history_window > i:
                pre_part_rotvec = np.repeat(ref_rotvec_minus, history_window - i, axis=0)  # [history_window - i, 3]
                mid_part_rotvec = rotvec[1:, :]  # [length_new, 3]
                post_part_rotvec = np.repeat(rotvec[-1:, :], i, axis=0)  # [i, 3]
                part_rotvec = np.concatenate([
                    pre_part_rotvec, mid_part_rotvec, post_part_rotvec], axis=0)  # [length_out, 3]
                parts.append(part_rotvec)
                pre_part_pos = np.repeat(ref_pos_minus, history_window - i, axis=0)  # [history_window - i, 3]
                mid_part_pos = pos[1:, :]  # [length_new, 3]
                post_part_pos = np.repeat(pos[-1:, :], i, axis=0)  # [i, 3]
                part_pos = np.concatenate([
                    pre_part_pos, mid_part_pos, post_part_pos], axis=0)  # [length_out, 3]
                parts.append(part_pos)
            else:
                pre_part_rotvec = rotvec[i - history_window + 1:, :]  # [length_out - i, 3]
                post_part_rotvec = np.repeat(rotvec[-1:, :], i, axis=0)  # [i, 3]
                part_rotvec = np.concatenate([pre_part_rotvec, post_part_rotvec], axis=0)  # [length_out, 3]
                parts.append(part_rotvec)
                pre_part_pos = pos[i - history_window + 1:, :]  # [length_out - i, 3]
                post_part_pos = np.repeat(pos[-1:, :], i, axis=0)  # [i, 3]
                part_pos = np.concatenate([pre_part_pos, post_part_pos], axis=0)  # [length_out, 3]
                parts.append(part_pos)
        ref_t_obs = np.concatenate(parts, axis=1)  # [length_out, 6 * ref_horizon]

        processed_data = np.concatenate(
            [u_t_minus_1, rotvec_t, pos_t, l_t, v_t, torque_t, u_t, ref_t, ref_t_obs,
             rotvec_t_plus_1, pos_t_plus_1, l_t_plus_1, v_t_plus_1, torque_t_plus_1],
            axis=1)  # [length_new, output_dim=84 + 6 * ref_horizon]
        print(processed_data.shape)

        if (random.random() < 0.8) or train_only:
            save_path = save_folder_train / log_file
            trainset_length += processed_data.shape[0]
        else:
            save_path = save_folder_eval / log_file
            evalset_length += processed_data.shape[0]

        h5_path = save_path.with_suffix('.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('u_t_minus_1', data=u_t_minus_1)
            f.create_dataset('rotvec_t', data=rotvec_t)
            f.create_dataset('pos_t', data=pos_t)
            f.create_dataset('l_t', data=l_t)
            f.create_dataset('v_t', data=v_t)
            f.create_dataset('torque_t', data=torque_t)
            f.create_dataset('u_t', data=u_t)
            f.create_dataset('ref_t', data=ref_t)
            f.create_dataset('ref_t_obs', data=ref_t_obs)
            f.create_dataset('rotvec_t_plus_1', data=rotvec_t_plus_1)
            f.create_dataset('pos_t_plus_1', data=pos_t_plus_1)
            f.create_dataset('l_t_plus_1', data=l_t_plus_1)
            f.create_dataset('v_t_plus_1', data=v_t_plus_1)
            f.create_dataset('torque_t_plus_1', data=torque_t_plus_1)
            f.attrs['ref_horizon'] = ref_horizon
            f.attrs['history_window'] = history_window
            f.attrs['source_csv'] = log_file
        sucessful_files += 1
            
    print(f"Total training set length: {trainset_length}")
    print(f"Total evaluation set length: {evalset_length}")
    print(f"Total sucessful processed files: {sucessful_files}")
    # print(f"Total steps in source folder: {trainset_length + evalset_length - 49 * sucessful_files}")
    return sucessful_files


if __name__ == "__main__":
    data_source = "260612"
    save_folder = "260612"
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "models_config.yaml"
    config = Config.load(config_path)
    ref_horizon = 50  # config.policy.ref_horizon
    history_window = 50  # max(
        # config.dynamics.net_params.mlp.history_window, config.policy.net_params.mlp.history_window)

    process_y(data_source, save_folder, ref_horizon, history_window)
