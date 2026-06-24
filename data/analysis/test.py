import os
from pathlib import Path
import numpy as np
import pandas as pd


def numpy_matrix_geodesic_error(R1, R2):
    """
    Compute the geodesic error (rotation angle in radians) between two rotation matrices R1 and R2.
    R1, R2: (Batch, 3, 3) numpy arrays representing rotation matrices.
    """
    # R_rel = R1^T @ R2
    R_rel = np.matmul(R1.transpose(0, 2, 1), R2)
    
    # trace of R_rel
    trace = np.trace(R_rel, axis1=-2, axis2=-1)
    
    arg = (trace - 1.0) / 2.0
    arg = np.clip(arg, -1.0, 1.0)  # clip to valid range for arccos
    
    angle = np.arccos(arg)
    return angle # (Batch,)


def test_logs_fp(data_source: str):
    """
    Compute statistics for testing data.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    logs_path = project_root / "data" / data_source / "logs"
    save_path = project_root / "data" / data_source / "test_metrics.csv"

    file_name = []
    step_num = []
    pos_error = []  # mean position error in mm
    rot_error = []  # mean geodesic error in rad
    x_error = []    # mean x position error in mm
    y_error = []    # mean y position error in mm
    z_error = []    # mean z position error in mm
    xy_error = []   # mean xy position error in mm
    
    file_list = [
                 "policy_new_mlp_hout_50g_Log_2.0_B.csv",
                 "policy_new_mlp_hout_50g_Log_2.0_O.csv",
                 "policy_new_mlp_hout_50g_Log_2.0_R.csv",
                 "policy_new_mlp_hout_50g_Log_2.0_T.csv",
                 "policy_new_mlp_hout_50g_Log_2.0_straight_line.csv",
                 "policy_new_mlp_hout_50g_Log_3.0_B.csv",
                 "policy_new_mlp_hout_50g_Log_3.0_O.csv",
                 "policy_new_mlp_hout_50g_Log_3.0_R.csv",
                 "policy_new_mlp_hout_50g_Log_3.0_T.csv",
                 "policy_new_mlp_hout_50g_Log_3.0_straight_line.csv",
                 "policy_new_mlp_hout_50g_Log_5.0_B.csv",
                 "policy_new_mlp_hout_50g_Log_5.0_O.csv",
                 "policy_new_mlp_hout_50g_Log_5.0_R.csv",
                 "policy_new_mlp_hout_50g_Log_5.0_T.csv",
                 "policy_new_mlp_hout_50g_Log_5.0_straight_line.csv",]
    
    for log_file in sorted(os.listdir(logs_path)):
        if not log_file.endswith(".csv"):
            continue
        if log_file not in file_list:
            continue
        print(f"Processing {log_file}...")
        file_path = logs_path / log_file
        df = pd.read_csv(file_path, skiprows=1, header=None)

        steps = df.shape[0] - 1

        t =  df.iloc[1:, 1:2].to_numpy()  # time
        p = df.iloc[1:, 60:70:4].to_numpy()  # position
        # rotation matrix
        rotation_matrix = np.eye(3)[None, :, :].repeat(steps, axis=0)
        rotation_matrix[:, 0, :] = df.iloc[1:, 57:60].to_numpy()
        rotation_matrix[:, 1, :] = df.iloc[1:, 61:64].to_numpy()
        rotation_matrix[:, 2, :] = df.iloc[1:, 65:68].to_numpy()

        # reference
        p_ref = df.iloc[:-1, 17:27:4].to_numpy()
        rotation_matrix_ref = np.eye(3)[None, :, :].repeat(steps, axis=0)
        rotation_matrix_ref[:, 0, :] = df.iloc[:-1, 14:17].to_numpy()
        rotation_matrix_ref[:, 1, :] = df.iloc[:-1, 18:21].to_numpy()
        rotation_matrix_ref[:, 2, :] = df.iloc[:-1, 22:25].to_numpy()

        # errors
        # pos_error = np.linalg.norm(p - p_ref, axis=1)
        # rot_error = numpy_matrix_geodesic_error(rotation_matrix, rotation_matrix_ref)
        pos_err = np.linalg.norm(p - p_ref, axis=1).mean()
        rot_err = numpy_matrix_geodesic_error(rotation_matrix, rotation_matrix_ref).mean()
        x_err = np.abs(p[:, 0] - p_ref[:, 0]).mean()
        y_err = np.abs(p[:, 1] - p_ref[:, 1]).mean()
        z_err = np.abs(p[:, 2] - p_ref[:, 2]).mean()
        xy_err = np.linalg.norm(p[:, :2] - p_ref[:, :2], axis=1).mean()

        # ref speed
        speed_ref = np.linalg.norm(np.diff(p_ref, axis=0) / np.diff(t, axis=0), axis=1).mean()
        print(f"Reference speed (mm/s): {speed_ref:.4f}")

        # save pos
        save_pos_path = project_root / "data" / data_source / (log_file.replace(".csv", "_pos_metrics.csv"))
        df_pos = pd.DataFrame({
            'x': p[:, 0],
            'y': p[:, 1],
            'z': p[:, 2],
            'x_ref': p_ref[:, 0],
            'y_ref': p_ref[:, 1],
            'z_ref': p_ref[:, 2]
        })
        df_pos.to_csv(save_pos_path, index=False)

        # # save errors
        # save_error_path = project_root / "data" / data_source / (log_file.replace(".csv", "_error.csv"))
        # df_pos = pd.DataFrame({
        #     'step': list(range(1, len(rot_error) + 1)),
        #     'rot_error': rot_error,
        #     'pos_error': pos_error,
        # })
        # df_pos.to_csv(save_error_path, index=False)

        file_name.append(log_file)
        step_num.append(steps)
        pos_error.append(pos_err)
        rot_error.append(rot_err)
        x_error.append(x_err)
        y_error.append(y_err)
        z_error.append(z_err)
        xy_error.append(xy_err)
    
    avg_pos_error = np.sum(np.array(pos_error) * np.array(step_num)) / np.sum(step_num)  # weighted average by step number
    avg_rot_error = np.sum(np.array(rot_error) * np.array(step_num)) / np.sum(step_num)
    avg_x_error = np.sum(np.array(x_error) * np.array(step_num)) / np.sum(step_num)
    avg_y_error = np.sum(np.array(y_error) * np.array(step_num)) / np.sum(step_num)
    avg_z_error = np.sum(np.array(z_error) * np.array(step_num)) / np.sum(step_num)
    avg_xy_error = np.sum(np.array(xy_error) * np.array(step_num)) / np.sum(step_num)

    print(f"Average position error (mm): {avg_pos_error:.2f}")
    print(f"Average rotation error (degree): {np.degrees(avg_rot_error):.2f}")

    # print(f"Average position error (mm): {avg_pos_error:.4f}")
    # print(f"Average rotation error (rad): {avg_rot_error:.4f}")
    # print(f"Average x error (mm): {avg_x_error:.4f}")
    # print(f"Average y error (mm): {avg_y_error:.4f}")
    # print(f"Average z error (mm): {avg_z_error:.4f}")
    # print(f"Average xy error (mm): {avg_xy_error:.4f}")
    
    # df = pd.DataFrame({
    #     'file_name': file_name,
    #     'step_num': step_num,
    #     'pos_error_mm': pos_error,
    #     'rot_error_rad': rot_error,
    #     'x_error_mm': x_error,
    #     'y_error_mm': y_error,
    #     'z_error_mm': z_error,
    #     'xy_error_mm': xy_error
    # })
    # df.to_csv(save_path, index=False)


def test_errors_fp(data_source: str):
    """
    Compute statistics for testing data.

    Args:
        data_source (str): Data source folder name (e.g., "origin_tracking").
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    logs_path = project_root / "data" / data_source
    
    file_list = [
                 "250_Log_2.0_B_test.csv",
                 "250_Log_2.0_O_test.csv",
                 "250_Log_2.0_R_test.csv",
                 "250_Log_2.0_T_test.csv",
                 "250_Log_2.0_straight_line_test.csv",
                 "250_Log_3.0_B_test.csv",
                 "250_Log_3.0_O_test.csv",
                 "250_Log_3.0_R_test.csv",
                 "250_Log_3.0_T_test.csv",
                 "250_Log_3.0_straight_line_test.csv",
                 "250_Log_5.0_B_test.csv",
                 "250_Log_5.0_O_test.csv",
                 "250_Log_5.0_R_test.csv",
                 "250_Log_5.0_T_test.csv",
                 "250_Log_5.0_straight_line_test.csv",
                 ]
    
    step_num = []
    pos_error = []  # mean position error in mm
    rot_error = []  # mean geodesic error in rad
    
    for log_file in sorted(os.listdir(logs_path)):
        if not log_file.endswith(".csv"):
            continue
        if log_file not in file_list:
            continue
        print(f"Processing {log_file}...")
        file_path = logs_path / log_file
        df = pd.read_csv(file_path, skiprows=1, header=None)

        steps = df.shape[0]
        rot_err = df.iloc[:, 1].to_numpy().mean()
        pos_err = df.iloc[:, 2].to_numpy().mean()

        # # special case
        # rot_err = df.iloc[:, 1].to_numpy()
        # pos_err = df.iloc[:, 2].to_numpy()
        # rot_err = rot_err[150:200].mean()
        # pos_err = pos_err[150:200].mean()
 
        step_num.append(steps)
        rot_error.append(rot_err)
        pos_error.append(pos_err)
    
    avg_pos_error = np.sum(np.array(pos_error) * np.array(step_num)) / np.sum(step_num)  # weighted average by step number
    avg_rot_error = np.sum(np.array(rot_error) * np.array(step_num)) / np.sum(step_num)
    print(f"Average position error (mm): {avg_pos_error:.2f}")
    print(f"Average rotation error (degree): {np.degrees(avg_rot_error):.2f}")


if __name__ == "__main__":
    data_source = "policy_test_on_model/policy_new_test_detach_175"  #"260416"  # 
    # test_logs_fp(data_source)
    test_errors_fp(data_source)