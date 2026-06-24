import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def test_plot(ctl, obs_real, obs_predict, rot_error, pos_error, l_error, v_error, save_folder, stem):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # to numpy
    ctl = np.array(ctl)
    obs_real = np.array(obs_real)
    obs_predict = np.array(obs_predict)

    step_obs = np.arange(len(obs_real)) / 50.0
    step_ctl = np.arange(len(ctl)) / 50.0

    # ---------------- 1：rotvec ----------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # ctl
    axes[0].plot(step_ctl, ctl, label="ctl")
    axes[0].legend()
    axes[0].set_ylabel("ctl")
    axes[0].set_title("Control input")

    # rotvec x,y,z
    axes[1].plot(step_obs, obs_real[:, 0], label="rotvec_x (target)")
    axes[1].plot(step_obs, obs_predict[:, 0], label="rotvec_x (predict)")
    axes[1].legend()

    axes[2].plot(step_obs, obs_real[:, 1], label="rotvec_y (target)")
    axes[2].plot(step_obs, obs_predict[:, 1], label="rotvec_y (predict)")
    axes[2].legend()

    axes[3].plot(step_obs, obs_real[:, 2], label="rotvec_z (target)")
    axes[3].plot(step_obs, obs_predict[:, 2], label="rotvec_z (predict)")
    axes[3].legend()
    axes[3].set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig(save_folder / f"rotvec_{stem}.png")
    plt.close(fig)

    # ---------------- 2：position ----------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # ctl
    axes[0].plot(step_ctl, ctl, label="ctl")
    axes[0].legend()
    axes[0].set_title("Control input")

    # pos x,y,z
    axes[1].plot(step_obs, obs_real[:, 3], label="pos_x (target)")
    axes[1].plot(step_obs, obs_predict[:, 3], label="pos_x (predict)")
    axes[1].legend()

    axes[2].plot(step_obs, obs_real[:, 4], label="pos_y (target)")
    axes[2].plot(step_obs, obs_predict[:, 4], label="pos_y (predict)")
    axes[2].legend()

    axes[3].plot(step_obs, obs_real[:, 5], label="pos_z (target)")
    axes[3].plot(step_obs, obs_predict[:, 5], label="pos_z (predict)")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(save_folder / f"position_{stem}.png")
    plt.close(fig)

    # ---------------- 3：error ----------------
    if l_error is None or v_error is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(step_ctl, rot_error, label="rot_error")
        axes[0].legend()

        axes[1].plot(step_ctl, pos_error, label="pos_error")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_folder / f"error_{stem}.png")
        plt.close(fig)
    else:
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(step_ctl, rot_error, label="rot_error")
        axes[0].legend()

        axes[1].plot(step_ctl, pos_error, label="pos_error")
        axes[1].legend()

        axes[2].plot(step_ctl, l_error, label="l_error")
        axes[2].legend()

        axes[3].plot(step_ctl, v_error, label="v_error")
        axes[3].legend()

        plt.tight_layout()
        plt.savefig(save_folder / f"error_{stem}.png")
        plt.close(fig)
