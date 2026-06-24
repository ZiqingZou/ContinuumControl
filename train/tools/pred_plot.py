import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def pred_plot(sampled_pred, sampled_tgt, save_path, obs_type, sampled_origin_obs=None,
              sampled_ctl=None, sampled_data_ctl=None, sampled_history=None):
    sampled_pred = np.stack(sampled_pred, axis=0)
    sampled_tgt = np.stack(sampled_tgt, axis=0)
    if sampled_origin_obs is not None:
        sampled_origin_obs = np.stack(sampled_origin_obs, axis=0)

    t = np.arange(0, sampled_tgt.shape[0])
    t = t * 0.02  # assuming 50Hz

    if sampled_history is not None:
        t_history = np.arange(0, sampled_history.shape[0])
        t_history = - t_history[::-1] * 0.02  # assuming 50Hz

    pos_only = sampled_tgt.shape[-1] == 3
    if not pos_only:
        # rotvec plot
        fig, axes = plt.subplots(3, 1, figsize=(6, 5))
        fig.suptitle("End-Effector Rotation in RotVec", fontsize=14)

        axes = axes.flatten()
        for i in range(3):
            if sampled_history is not None and "T" in obs_type:
                axes[i].plot(t_history, sampled_history[:, i], label="History", color='blue', alpha=0.3)
            axes[i].plot(t[:2], sampled_tgt[:2, i], label="Target", color='blue', linestyle='--')
            axes[i].plot(t[1:], sampled_tgt[1:, i], label="Target", color='blue')
            axes[i].plot(t, sampled_pred[:, i], label="Predicted", color='orange')
            if sampled_origin_obs is not None and "T" in obs_type:
                axes[i].plot(t[:-1], sampled_origin_obs[:, i], label="Origin Obs", color='blue', alpha=0.3)
            axes[i].legend(loc="upper right", fontsize=8)
            axes[i].grid(True)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel(f"rotvec {i} (rad)")

        all_points = np.vstack([sampled_tgt[:, :3], sampled_pred[:, :3]])
        rot_x_min, rot_x_max = all_points[:,0].min(), all_points[:,0].max()
        rot_y_min, rot_y_max = all_points[:,1].min(), all_points[:,1].max()
        rot_z_min, rot_z_max = all_points[:,2].min(), all_points[:,2].max()
        rot_max_range = max(rot_x_max - rot_x_min, rot_y_max - rot_y_min, rot_z_max - rot_z_min) / 2.0
        rot_mid_x = (rot_x_min + rot_x_max) / 2
        rot_mid_y = (rot_y_min + rot_y_max) / 2
        rot_mid_z = (rot_z_min + rot_z_max) / 2

        axes[0].set_ylim(rot_mid_x - rot_max_range, rot_mid_x + rot_max_range)
        axes[1].set_ylim(rot_mid_y - rot_max_range, rot_mid_y + rot_max_range)
        axes[2].set_ylim(rot_mid_z - rot_max_range, rot_mid_z + rot_max_range)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path / "rotvec.png", dpi=300)
        plt.close()

        # ee position plot
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(sampled_tgt[:2, 3], sampled_tgt[:2, 4], sampled_tgt[:2, 5], 
                label="Target Trajectory", color='blue', linestyle='--')
        ax.plot(sampled_tgt[1:, 3], sampled_tgt[1:, 4], sampled_tgt[1:, 5], 
                label="Target Trajectory", color='blue')
        ax.plot(sampled_pred[:, 3], sampled_pred[:, 4], sampled_pred[:, 5], 
                label="Predicted Trajectory", color='orange')
        if sampled_history is not None and "T" in obs_type:
            ax.plot(sampled_history[:, 3], sampled_history[:, 4], sampled_history[:, 5], 
                    label="History Trajectory", color='blue', alpha=0.3)
        if sampled_origin_obs is not None and "T" in obs_type:
            ax.plot(sampled_origin_obs[:, 3], sampled_origin_obs[:, 4], sampled_origin_obs[:, 5], 
                    label="Origin Obs Trajectory", color='blue', alpha=0.3)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Z (mm)", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title("End-Effector Trajectory", fontsize=14)

        # Range equalization
        all_points = np.vstack([sampled_tgt[:, 3:6], sampled_pred[:, 3:6]])
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
        plt.savefig(save_path / "ee_trajectory.png", dpi=300)
        plt.close()

        fig, axes = plt.subplots(3, 1, figsize=(6, 5))
        fig.suptitle("End-Effector Translation", fontsize=14)

        axes = axes.flatten()
        for i in range(3):
            if sampled_history is not None and "T" in obs_type:
                axes[i].plot(t_history, sampled_history[:, 3 + i], label="History", color='blue', alpha=0.3)
            if sampled_origin_obs is not None and "T" in obs_type:
                axes[i].plot(t[:-1], sampled_origin_obs[:, 3 + i], label="Origin Obs", color='blue', alpha=0.3)
            axes[i].plot(t[:2], sampled_tgt[:2, 3 + i], label="Target", color='blue', linestyle='--')
            axes[i].plot(t[1:], sampled_tgt[1:, 3 + i], label="Target", color='blue')
            axes[i].plot(t, sampled_pred[:, 3 + i], label="Predicted", color='orange')
            axes[i].legend(loc="upper right", fontsize=8)
            axes[i].grid(True)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel(f"translation {i} (mm)")

        axes[0].set_ylim(mid_x - max_range, mid_x + max_range)
        axes[1].set_ylim(mid_y - max_range, mid_y + max_range)
        axes[2].set_ylim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path / "translation.png", dpi=300)
        plt.close()

    else:
        # ee position plot
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(sampled_tgt[:2, 0], sampled_tgt[:2, 1], sampled_tgt[:2, 2], 
                label="Target Trajectory", color='blue', linestyle='--')
        ax.plot(sampled_tgt[1:, 0], sampled_tgt[1:, 1], sampled_tgt[1:, 2], 
                label="Target Trajectory", color='blue')
        ax.plot(sampled_pred[:, 3], sampled_pred[:, 4], sampled_pred[:, 5], 
                label="Predicted Trajectory", color='orange')
        if sampled_history is not None and "T" in obs_type:
            ax.plot(sampled_history[:, 3], sampled_history[:, 4], sampled_history[:, 5], 
                    label="History Trajectory", color='blue', alpha=0.3)
        if sampled_origin_obs is not None and "T" in obs_type:
            ax.plot(sampled_origin_obs[:, 3], sampled_origin_obs[:, 4], sampled_origin_obs[:, 5], 
                    label="Origin Obs Trajectory", color='blue', alpha=0.3)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Z (mm)", fontsize=8)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title("End-Effector Trajectory", fontsize=14)

        # Range equalization
        all_points = np.vstack([sampled_tgt[:, :3], sampled_pred[:, 3:6]])
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
        plt.savefig(save_path / "ee_trajectory.png", dpi=300)
        plt.close()

        fig, axes = plt.subplots(3, 1, figsize=(6, 5))
        fig.suptitle("End-Effector Translation", fontsize=14)

        axes = axes.flatten()
        for i in range(3):
            if sampled_history is not None and "T" in obs_type:
                axes[i].plot(t_history, sampled_history[:, 3 + i], label="History", color='blue', alpha=0.3)
            if sampled_origin_obs is not None and "T" in obs_type:
                axes[i].plot(t[:-1], sampled_origin_obs[:, 3 + i], label="Origin Obs", color='blue', alpha=0.3)
            axes[i].plot(t[:2], sampled_tgt[:2, i], label="Target", color='blue', linestyle='--')
            axes[i].plot(t[1:], sampled_tgt[1:, i], label="Target", color='blue')
            axes[i].plot(t, sampled_pred[:, 3 + i], label="Predicted", color='orange')
            axes[i].legend(loc="upper right", fontsize=8)
            axes[i].grid(True)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel(f"translation {i} (mm)")

        axes[0].set_ylim(mid_x - max_range, mid_x + max_range)
        axes[1].set_ylim(mid_y - max_range, mid_y + max_range)
        axes[2].set_ylim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path / "translation.png", dpi=300)
        plt.close()

    if sampled_ctl is None:
        index = 6
        if "T" in obs_type:
            history_index = 6
        else:
            history_index = 0

        if "l" in obs_type:
            # Plot l subplots
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            fig.suptitle("Motor Length vs Time", fontsize=14)

            axes = axes.flatten()
            for i in range(9):
                if sampled_history is not None:
                    axes[i].plot(t_history, sampled_history[
                        :, history_index + i], label=f"Motor {i+1} History", color='blue', alpha=0.5)
                axes[i].plot(t, sampled_tgt[:, index + i], label=f"Motor {i+1} Target", color='blue')
                axes[i].plot(t, sampled_pred[:, index + i], label=f"Motor {i+1} Predicted", color='orange')
                axes[i].legend(loc="upper right", fontsize=8)
                axes[i].set_xlabel("Time (s)")
                axes[i].set_ylabel("Motor length (mm)")
                axes[i].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(save_path / "motor_length.png", dpi=300)
            plt.close()
            index += 9
            history_index += 9
            
        if "v" in obs_type:
            # Plot v subplots
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            fig.suptitle("Motor Velocity vs Time", fontsize=14)

            axes = axes.flatten()
            for i in range(9):
                if sampled_history is not None:
                    axes[i].plot(t_history, sampled_history[
                        :, history_index + i], label=f"Motor {i+1} History", color='blue', alpha=0.5)
                axes[i].plot(t, sampled_tgt[:, index + i], label=f"Motor {i+1} Target", color='blue')
                axes[i].plot(t, sampled_pred[:, index + i], label=f"Motor {i+1} Predicted", color='orange')
                axes[i].legend(loc="upper right", fontsize=8)
                axes[i].set_xlabel("Time (s)")
                axes[i].set_ylabel("Motor velocity (mm/s)")
                axes[i].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(save_path / "motor_velocity.png", dpi=300)
            plt.close()
            index += 9
            history_index += 9

        if "q" in obs_type:
            # Plot q subplots
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            fig.suptitle("Motor Torque vs Time", fontsize=14)

            axes = axes.flatten()
            for i in range(9):
                if sampled_history is not None:
                    axes[i].plot(t_history, sampled_history[
                        :, history_index + i], label=f"Motor {i+1} History", color='blue', alpha=0.5)
                axes[i].plot(t, sampled_tgt[:, index + i], label=f"Motor {i+1} Target", color='blue')
                axes[i].plot(t, sampled_pred[:, index + i], label=f"Motor {i+1} Predicted", color='orange')
                axes[i].legend(loc="upper right", fontsize=8)
                axes[i].set_xlabel("Time (s)")
                axes[i].set_ylabel("Motor torque (Nm)")
                axes[i].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.98])
            plt.savefig(save_path / "motor_torque.png", dpi=300)
            plt.close()
            # index += 9
            # history_index += 9

    else:
        sampled_ctl = np.stack(sampled_ctl, axis=0)
        if sampled_data_ctl is not None:
            sampled_data_ctl = np.stack(sampled_data_ctl, axis=0)
        t = np.arange(0, sampled_ctl.shape[0])
        t = t * 0.02  # assuming 50Hz

        # Plot u subplots
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        fig.suptitle("Control vs Time", fontsize=14)

        axes = axes.flatten()
        for i in range(9):
            axes[i].plot(t, sampled_ctl[:, i], label=f"Motor {i+1} Target", color='blue')
            if sampled_data_ctl is not None:
                axes[i].plot(t, sampled_data_ctl[:, i], label=f"Motor {i+1} Data", linestyle='dashed')
            axes[i].legend(loc="upper right", fontsize=8)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Motor velocity (mm/s)")
            axes[i].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path / "control.png", dpi=300)
        plt.close()

        
if __name__ == "__main__":

    from config.parser import Config
    from train.trainer import DynamicsTrainer

    # Load configurations
    config_path = Path(__file__).resolve().parent.parent / "config"
    cfg_model = Config.load(config_path / "models_config.yaml")
    cfg_training = Config.load(config_path / "training_config.yaml")

    save_path = Path(__file__).resolve().parent / "checkpoints" / cfg_training.dynamics.load_name

    trainer = DynamicsTrainer(cfg_model, cfg_training)
    loader = trainer.val_loader

    for batch in loader:
        batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}

        sampled_pred, sampled_tgt = trainer.evaluate(batch)
        pred_plot(sampled_pred, sampled_tgt, save_path, cfg_model.dynamics.get("obs_type"))
