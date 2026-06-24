import os
import io
import csv
import socket
import struct
import time
import wandb
from pathlib import Path

import torch

from config.parser import Config
from train.tools.pred_plot import pred_plot
from train.tools.export_model import trace_model
from data.analysis.pre_process import process_y
from data.analysis.visualization import visualize_results
from train.trainer import DynamicsTrainer, PolicyTrainer


wandb_run = True  # Set to True to enable W&B logging


def initialize_wandb(cfg_training: Config):
    print("Initializing W&B...")
    if wandb_run:
        wandb.init(
            project="online_continuum_0112",
            name=cfg_training.online.get("name"),
            config=cfg_training.to_dict(),
        )
        wandb.define_metric("dynamics/*", step_metric="dynamics/iteration")
        wandb.define_metric("policy/*", step_metric="policy/iteration")
        wandb.define_metric("epoch/*", step_metric="epoch/epoch")
        wandb.define_metric("test/*", step_metric="test/epoch")


def recv_all(sock, length):
    """
    Helper function to receive a specific number of bytes from a socket.
    """
    data = b'' 
    while len(data) < length: 
        packet = sock.recv(length - len(data)) 
        if not packet: 
            return None 
        data += packet 
    return data


def train_model(dynamics_trainer, policy_trainer, epoch, min_avg_dynamics_loss, 
                round_time_per_epoch, update_dynamics):
    start_time = time.time()
    if update_dynamics:
        avg_dynamics_loss = []
        for round_idx in range(round_time_per_epoch):
            batch_loss = []
            for batch in dynamics_trainer.train_loader:
                batch = {k: v.to(dynamics_trainer.device, non_blocking=True) for k, v in batch.items()}
                (iter_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl,
                sampled_history) = dynamics_trainer.update(batch, teacher_forcing=False)
                iter_info = {k.replace("train", "dynamics"): v for k, v in iter_info.items()}
                iter_info["dynamics/epoch"] = epoch
                batch_loss.append(iter_info["dynamics/loss"])
                for k, v in iter_info.items():
                    print(f"{k}: {v}")
                if wandb_run:
                    wandb.log(iter_info)
                if sampled_pred and sampled_tgt:
                    pred_plot(sampled_pred, sampled_tgt, dynamics_trainer.save_folder, dynamics_trainer.obs_type, 
                            sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)                                
            avg_dynamics_loss.append(sum(batch_loss) / len(batch_loss))

        avg_dynamics_loss = sum(avg_dynamics_loss) / len(avg_dynamics_loss)
        print(f"Epoch {epoch} average loss: {avg_dynamics_loss}\n")

    save_path = dynamics_trainer.save_folder / f"dynamics_latest.pth"
    dynamics_trainer.dynamics.save(save_path)
    print(f"\nLateast dynamics model saved to {save_path} !\n")

    avg_policy_loss = []
    for round_idx in range(round_time_per_epoch):
        batch_loss = []
        for batch in policy_trainer.train_loader:
            batch = {k: v.to(policy_trainer.device, non_blocking=True) for k, v in batch.items()}
            (iter_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl, 
             sampled_history)= policy_trainer.update(batch, teacher_forcing=False)
            iter_info = {k.replace("train", "policy"): v for k, v in iter_info.items()}
            iter_info["policy/epoch"] = epoch
            batch_loss.append(iter_info["policy/loss"])
            for k, v in iter_info.items():
                print(f"{k}: {v}")
            if wandb_run:
                wandb.log(iter_info)
            if sampled_pred and sampled_tgt:
                pred_plot(sampled_pred, sampled_tgt, policy_trainer.save_folder, policy_trainer.obs_type, 
                          sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)       
        avg_policy_loss.append(sum(batch_loss) / len(batch_loss))

    avg_policy_loss = sum(avg_policy_loss) / len(avg_policy_loss)
    print(f"Epoch {epoch} average loss: {avg_policy_loss}\n")

    save_path = policy_trainer.save_folder / f"policy_latest.pth"
    policy_trainer.policy.save(save_path)
    print(f"\nLateast policy model saved to {save_path} !\n")

    if update_dynamics:
        if avg_dynamics_loss <= min_avg_dynamics_loss:
            min_avg_dynamics_loss = avg_dynamics_loss
            save_path = dynamics_trainer.save_folder / f"dynamics_best.pth"
            dynamics_trainer.dynamics.save(save_path)
            print(f"\nNew best dynamics model saved to {save_path} !\n")
            save_path = policy_trainer.save_folder / f"policy_best.pth"
            policy_trainer.policy.save(save_path)
            print(f"\nCorresponding policy model saved to {save_path} !\n")

    training_time = time.time() - start_time
    print(f"Epoch {epoch} time taken: {training_time} seconds")

    if wandb_run:
        wandb.log({"epoch/dynamics_avg_loss": 0, # avg_dynamics_loss,  
                   "epoch/policy_avg_loss": avg_policy_loss,
                   "epoch/training_time": training_time, 
                   "epoch/epoch": epoch})

    return min_avg_dynamics_loss


def update_data_source(models_config: Config, training_name: str):
    data_source = training_name
    save_folder = "online_learning"
    ref_horizon = models_config.policy.ref_horizon
    history_window = max(
        models_config.dynamics.net_params.mlp.history_window, models_config.policy.net_params.mlp.history_window)

    sucessful_files = process_y(data_source, save_folder, ref_horizon, history_window, train_only=True)
    return sucessful_files


def analysis_run(epoch: int, training_name: str):
    log_dir = Path(__file__).resolve().parent.parent / "data" / training_name / "logs" 
    csv_files = [f.name for f in log_dir.iterdir() if f.suffix == ".csv"]

    avg_ee_pos_error = []
    avg_ee_geodesic_error = []
    for file_name in csv_files:
        ee_pos_error, ee_geodesic_error = visualize_results(data_source=training_name, log_name=file_name)
        if ee_pos_error is None or ee_geodesic_error is None:
            continue
        avg_ee_pos_error.append(ee_pos_error)
        avg_ee_geodesic_error.append(ee_geodesic_error)

    if len(avg_ee_pos_error) == 0 or len(avg_ee_geodesic_error) == 0:
        print("No valid data for analysis this epoch.")
        new_dir = log_dir.parent / f"epoch_{epoch}" 
        log_dir.rename(new_dir)
        return None, None
    avg_ee_pos_error = sum(avg_ee_pos_error) / len(avg_ee_pos_error)
    avg_ee_geodesic_error = sum(avg_ee_geodesic_error) / len(avg_ee_geodesic_error)

    new_dir = log_dir.parent / f"epoch_{epoch}" 
    log_dir.rename(new_dir)
    return avg_ee_pos_error, avg_ee_geodesic_error


def online_train():
    config_path = Path(__file__).resolve().parent.parent / "config"
    models_config = Config.load(config_path / "models_config.yaml")
    training_config = Config.load(config_path / "training_config.yaml")

    dynamics_trainer = DynamicsTrainer(models_config, training_config, online=True)
    policy_trainer = PolicyTrainer(models_config, training_config, online=True, dynamics=dynamics_trainer.dynamics)

    initialize_wandb(training_config)
    min_avg_dynamics_loss = float("inf")
    log_dir = Path(__file__).resolve().parent.parent / "data" / training_config.online.get("name") / "logs" 
    test_dir = Path(__file__).resolve().parent.parent / "data" / (training_config.online.get("name") + "_test") / "logs" 

    # socket setup for communication with control computer
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.bind(("0.0.0.0", training_config.online.get("socket_port"))) 
    server.listen(1)
    print(f"Server listening on port {training_config.online.get('socket_port')}...")
    conn, addr = server.accept()
    print("Connected by", addr)

    # Training loop
    print(f"Start online training!")
    total_epochs = training_config.online.get("epochs")
    rollout_iters = training_config.online.get("rollouts_per_epoch")
    round_time_per_epoch = training_config.online.get("round_time_per_epoch")
    test_iters = training_config.online.get("test_traj_num")
    last_epoch_failure = False
    skip_initial_test = training_config.online.get("skip_initial_test")
    test_after_epoch = training_config.online.get("test_after_epoch")
    update_dynamics=training_config.online.get("update_dynamics")
    for epoch in range(total_epochs):
        if not last_epoch_failure:
            print(f"\nEpoch {epoch + 1}/{total_epochs} start !")
            conn.sendall(b"NEW")

            data = recv_all(conn, 5).decode("utf-8")
            if data != "READY":
                raise ValueError("Control computer not ready to receive traced models.")

            # send latest models to control computer
            dynamics_traced, policy_traced = trace_model(dynamics_trainer.dynamics, policy_trainer.policy)
            buffer = io.BytesIO() 
            torch.jit.save(dynamics_traced, buffer) 
            dynamics_bytes = buffer.getvalue() 
            buffer = io.BytesIO() 
            torch.jit.save(policy_traced, buffer) 
            policy_bytes = buffer.getvalue()

            conn.sendall(struct.pack("!I", len(dynamics_bytes)))
            conn.sendall(dynamics_bytes)
            conn.sendall(struct.pack("!I", len(policy_bytes)))
            conn.sendall(policy_bytes)

            data = recv_all(conn, 5).decode("utf-8")
            if data != "READY":
                raise ValueError("Control computer not received traced models.")

            # test policy before rollout
            if (epoch % test_after_epoch == 0) and (not (skip_initial_test and epoch == 0)):
                print(f"Testing current policy...")
                test_dir.mkdir(parents=True, exist_ok=True)
                for test_idx in range(test_iters):
                    test_file = test_dir / f"test_{test_idx}.csv"
                    dynamics_trainer.dynamics.set_zero_h(batch_size=1)

                    conn.sendall(f"START TEST {test_idx}".encode())
                    # recieve state data
                    data = recv_all(conn, 4).decode("utf-8")
                    if data == "DONE":
                        conn.sendall(b"LOG")
                        
                        # recieve .csv log file from control computer and save to test_file
                        raw_size = recv_all(conn, 4)
                        filesize = struct.unpack("!I", raw_size)[0]
                        print(f"Expecting file of size {filesize} bytes")

                        received = 0 
                        with open(test_file, "wb") as f: 
                            while received < filesize: 
                                chunk = conn.recv(4096) 
                                if not chunk: 
                                    break 
                                f.write(chunk) 
                                received += len(chunk) 
                        print(f"File received, total {received} bytes")

                        if received != filesize:
                            raise ValueError("File size mismatch!")
                    
                    else: 
                        raise ValueError("Unknown command received from control computer.")
                        
                        
                # analyze test results
                avg_ee_pos_error, avg_ee_geodesic_error = analysis_run(epoch, training_config.online.get("name") + "_test")
                if wandb_run and avg_ee_pos_error is not None and avg_ee_geodesic_error is not None:
                    wandb.log({"test/avg_ee_pos_error": avg_ee_pos_error, "test/avg_ee_geodesic_error": avg_ee_geodesic_error,
                            "test/epoch": epoch})

        # rollout policy to collect data and add to training set
        print(f"Collecting data with current policy...")
        log_dir.mkdir(parents=True, exist_ok=True)
        for rollout_idx in range(rollout_iters):
            rollout_file = log_dir / f"rollout_{rollout_idx}.csv"
            dynamics_trainer.dynamics.set_zero_h(batch_size=1)

            conn.sendall(f"START ROLLOUT {rollout_idx}".encode())
            print(f"Sent START ROLLOUT {rollout_idx} command.")

            # recieve state data
            data = recv_all(conn, 4).decode("utf-8")
            if data == "DONE":
                conn.sendall(b"LOG")
                
                # recieve .csv log file from control computer and save to rollout_file
                raw_size = recv_all(conn, 4)
                filesize = struct.unpack("!I", raw_size)[0]
                print(f"Expecting file of size {filesize} bytes")

                received = 0 
                with open(rollout_file, "wb") as f: 
                    while received < filesize: 
                        chunk = conn.recv(4096) 
                        if not chunk: 
                            break 
                        f.write(chunk) 
                        received += len(chunk) 
                print(f"File received, total {received} bytes")

                if received != filesize:
                    raise ValueError("File size mismatch!")
            
            else: 
                raise ValueError("Unknown command received from control computer.")

        # process new data
        sucessful_files = update_data_source(models_config, training_config.online.get("name"))
        if sucessful_files == 0:
            print("No new valid data collected this epoch. Re-trying.")
            last_epoch_failure = True
            epoch -= 1
            continue
        last_epoch_failure = False
        dynamics_trainer.train_loader.dataset.reload()
        policy_trainer.train_loader.dataset.reload()

        # analyze rollout results
        avg_ee_pos_error, avg_ee_geodesic_error = analysis_run(epoch, training_config.online.get("name"))
        if wandb_run:
            wandb.log({"epoch/avg_ee_pos_error": avg_ee_pos_error, "epoch/avg_ee_geodesic_error": avg_ee_geodesic_error,
                    "epoch/epoch": epoch})

        # train on new data
        min_avg_dynamics_loss = train_model(
            dynamics_trainer, policy_trainer, epoch, min_avg_dynamics_loss, 
            round_time_per_epoch, update_dynamics)

        conn.sendall(b"END EPOCH")

    conn.sendall(b"END")
    conn.close()
    server.close()
    if wandb_run:
        wandb.finish()
    print(f"Training completed.")


if __name__ == "__main__":
    online_train()
