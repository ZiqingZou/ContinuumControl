import os
import time
import wandb
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

import torch

from config.parser import Config
from train.trainer import DynamicsTrainer, PolicyTrainer
from train.tools.pred_plot import pred_plot
from train.tools.test_plot import test_plot
from train.tools.loss_fn import geodesic_loss


wandb_run = True  # Set to True to enable W&B logging


def train_model(cfg_model: Config, cfg_training: Config, model_type: str):
    validation_list = cfg_training.get(f"{model_type}.validation_list")

    # Initialize W&B
    print("Initializing W&B...")
    if wandb_run:
        wandb.init(
            project="continuum_policy_new",  # "continuum_total_wo_t",
            name=cfg_training.dynamics.get(
                "name") if model_type == "dynamics" else cfg_training.policy.get("name"),
            config=cfg_training.dynamics.to_dict(
                ) if model_type == "dynamics" else cfg_training.policy.to_dict(),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("eval/epoch")
        wandb.define_metric("eval/*", step_metric="eval/step")

        if validation_list is not None:
            for name in validation_list:
                wandb.define_metric(f"{name}/epoch")
                wandb.define_metric(f"{name}/*", step_metric=f"{name}/step")


    trainer = DynamicsTrainer(
        cfg_model, cfg_training) if model_type == "dynamics" else PolicyTrainer(cfg_model, cfg_training)
    min_avg_eval_loss = float("inf")
    train_step = 0
    eval_step = 0
    val_steps = [0] * len(validation_list) if validation_list is not None else []

    # Training loop
    print(f"Start training {model_type} model !")
    total_epochs = cfg_training.dynamics.get(
        "epochs") if model_type == "dynamics" else cfg_training.policy.get("epochs")

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch}/{total_epochs - 1} start !")
        start_time = time.time()
        # iterate over training data
        for batch in trainer.train_loader:
            batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
            (iter_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl, 
             sampled_history) = trainer.update(batch, teacher_forcing=False)
            iter_info["train/epoch"] = epoch
            for k, v in iter_info.items():
                print(f"{k}: {v}")
            if wandb_run:
                iter_info["train/step"] = train_step
                wandb.log(iter_info)
            train_step += 1
            if sampled_pred and sampled_tgt:
                pred_plot(sampled_pred, sampled_tgt, trainer.save_folder, trainer.obs_type, 
                          sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)
                                   
        # evaluate at the end of each epoch
        print(f"Start evaluating {model_type} model !")
        batch_loss = []
        for batch in trainer.eval_loader:
            batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
            (batch_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl, 
             sampled_history) = trainer.evaluate(batch, type="eval")
            batch_loss.append(batch_info["eval/loss"])
            batch_info["eval/epoch"] = epoch
            for k, v in batch_info.items():
                print(f"{k}: {v}")
            if wandb_run:
                batch_info["eval/step"] = eval_step
                wandb.log(batch_info)
            eval_step += 1
            if sampled_pred and sampled_tgt:
                pred_plot(sampled_pred, sampled_tgt, trainer.save_folder, trainer.obs_type, 
                          sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)

        avg_eval_loss = sum(batch_loss) / len(batch_loss)
        print(f"Epoch {epoch} average evaluation loss: {avg_eval_loss}\n")

        min_avg_eval_loss = min(min_avg_eval_loss, avg_eval_loss)
        save_path = trainer.save_folder / f"{model_type}_latest.pth"
        if model_type == "dynamics":
            trainer.dynamics.save(save_path)
        else:
            trainer.policy.save(save_path)
            if trainer.hidden_dynamics is not None:
                trainer.hidden_dynamics.save(trainer.save_folder / f"hidden_dynamics_latest.pth")
        print(f"\nNeweast model saved to {save_path} !\n")
        if avg_eval_loss == min_avg_eval_loss:
            save_path = trainer.save_folder / f"{model_type}_best.pth"
            if model_type == "dynamics":
                trainer.dynamics.save(save_path)
            else:
                trainer.policy.save(save_path)
                if trainer.hidden_dynamics is not None:
                    trainer.hidden_dynamics.save(trainer.save_folder / f"hidden_dynamics_best.pth")
            print(f"\nNew best model saved to {save_path} !\n")

        if validation_list is not None:
            for val_idx, valset_name in enumerate(validation_list):
                # validate at the end of each epoch
                print(f"Start validating {model_type} model on {valset_name} !")
                batch_loss = []
                for batch in trainer.val_loader[val_idx]:
                    batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
                    (batch_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl, 
                    sampled_history)= trainer.evaluate(batch, type="val")
                    batch_info = {k.replace("val/", f"{valset_name}/"): v for k, v in batch_info.items()}
                    batch_loss.append(batch_info[f"{valset_name}/loss"])
                    batch_info[f"{valset_name}/epoch"] = epoch
                    for k, v in batch_info.items():
                        print(f"{k}: {v}")
                    if wandb_run:
                        batch_info[f"{valset_name}/step"] = val_steps[val_idx]
                        wandb.log(batch_info)
                    val_steps[val_idx] += 1
                    if sampled_pred and sampled_tgt:
                        pred_plot(sampled_pred, sampled_tgt, trainer.save_folder, trainer.obs_type, 
                                sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)

                avg_val_loss = sum(batch_loss) / len(batch_loss)
                print(f"Epoch {epoch} average validation loss: {avg_val_loss}\n")

    end_time = time.time()
    print(f"Epoch {epoch} time taken: {(end_time - start_time) / 60} minutes")

    if wandb_run:
        wandb.finish()

    print(f"Training {model_type} completed.")


def evaluate_model(cfg_model: Config, cfg_training: Config, model_type: str):
    trainer = DynamicsTrainer(
        cfg_model, cfg_training) if model_type == "dynamics" else PolicyTrainer(cfg_model, cfg_training)

    print(f"Start evaluating {model_type} model in training set!")
    batch_loss = []
    # iterate over training data
    for batch in trainer.train_loader:
        batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
        (batch_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl, 
         sampled_history) = trainer.evaluate(batch, type="train")
        batch_loss.append(batch_info["train/loss"])
        for k, v in batch_info.items():
            print(f"{k}: {v}")
        if sampled_pred and sampled_tgt:
            pred_plot(sampled_pred, sampled_tgt, trainer.save_folder, trainer.obs_type, 
                      sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)
    avg_train_loss = sum(batch_loss) / len(batch_loss)
    print(f"Average training set loss: {avg_train_loss}\n")
                                
    print(f"Start evaluating {model_type} model in evaluation set!")
    batch_loss = []
    for batch in trainer.eval_loader:
        batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
        (batch_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl,
          sampled_history) = trainer.evaluate(batch, type="eval")
        batch_loss.append(batch_info["eval/loss"])
        for k, v in batch_info.items():
            print(f"{k}: {v}")
        if sampled_pred and sampled_tgt:
            pred_plot(sampled_pred, sampled_tgt, trainer.save_folder, trainer.obs_type, 
                      sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history)
    avg_eval_loss = sum(batch_loss) / len(batch_loss)
    print(f"Average evaluation set loss: {avg_eval_loss}\n")

    print(f"Evaluation of {model_type} completed.")


def test_model(cfg_model: Config, cfg_training: Config, model_type: str):
    if model_type == "dynamics":
        trainer = DynamicsTrainer(cfg_model, cfg_training, test=True, test_shuffle=True)
    else:
        raise NotImplementedError("Policy model test not implemented yet.")
        # trainer = PolicyTrainer(cfg_model, cfg_training)

    print(f"Start testing {model_type} model in test set!")
    test_list = cfg_training.dynamics.get("test_list")
    for loader, test_name in zip(trainer.test_loader, test_list):
        print(f"Testing {test_name} start.")
        dataset_size = 0
        dataset_avg_rot_error = [0.0] * trainer.bptt_steps  # geodesic distance in rad from step 0 to self.bptt_step
        dataset_avg_pos_error = [0.0] * trainer.bptt_steps  # mm
        dataset_avg_l_error = [0.0] * trainer.bptt_steps  # mm
        dataset_avg_v_error = [0.0] * trainer.bptt_steps  # mm/s
        # iterate over training data
        for batch in loader:
            batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
            (batch_size, batch_avg_rot_error, batch_avg_pos_error, 
            batch_avg_l_error, batch_avg_v_error) = trainer.test(batch)
            dataset_size += batch_size
            for i, b in enumerate(batch_avg_rot_error):
                dataset_avg_rot_error[i] += b * batch_size
            for i, b in enumerate(batch_avg_pos_error):
                dataset_avg_pos_error[i] += b * batch_size
            for i, b in enumerate(batch_avg_l_error):
                dataset_avg_l_error[i] += b * batch_size
            for i, b in enumerate(batch_avg_v_error):
                dataset_avg_v_error[i] += b * batch_size

        dataset_avg_rot_error = [x / dataset_size for x in dataset_avg_rot_error]
        dataset_avg_pos_error = [x / dataset_size for x in dataset_avg_pos_error]
        dataset_avg_l_error = [x / dataset_size for x in dataset_avg_l_error]
        dataset_avg_v_error = [x / dataset_size for x in dataset_avg_v_error]
        
        df = pd.DataFrame({
            'step': range(trainer.bptt_steps),
            'avg_rot_error': dataset_avg_rot_error,
            'avg_pos_error': dataset_avg_pos_error,
            'avg_l_error': dataset_avg_l_error,
            'avg_v_error': dataset_avg_v_error
            })

        df.to_csv(f'{trainer.save_folder}/{test_name}.csv', index=False)


@ torch.inference_mode()
def closed_loop_dynamics_test(cfg_model: Config, cfg_training: Config):
    trainer = DynamicsTrainer(cfg_model, cfg_training, test=True)
    trainer.dynamics.eval()
    start_recurrent_step = 7500
    back_to_supervised_step = 15000
    history_max_len = None

    p = Path(trainer.test_loader[0].dataset.paths)
    files = [f for f in p.iterdir() if f.suffix in {'.h5', '.hdf5'}]
    for fpath in files:
        if "Log_1.0_random_1.0_0.0_0.0.h5" not in fpath.name:
            continue
        rot_error = []
        pos_error = []
        l_error = []
        v_error = []
        rot_error_1 = []
        pos_error_1 = []
        l_error_1 = []
        v_error_1 = []
        rot_error_2 = []
        pos_error_2 = []
        l_error_2 = []
        v_error_2 = []
        rot_error_3 = []
        pos_error_3 = []
        l_error_3 = []
        v_error_3 = []

        # rnn
        h = torch.zeros(1, trainer.dynamics.rnn_depth, trainer.dynamics.hidden_dim).to(trainer.device)

        # lstm
        # c = trainer.dynamics._dynamics.c0.clone().to(trainer.device)
        # h = (h, c)

        # mlp
        # h = torch.zeros(1, trainer.dynamics._dynamics.history_window, 
        #                 trainer.dynamics._dynamics.his_dim_per_step).to(trainer.device)

        step = 0
        ctl = []
        data = trainer.test_loader[0].dataset.get_data(fpath, step, history_max_len)

        obs_real = [data["input_seq"][0, :trainer.dynamics.obs_dim].cpu().numpy()]
        obs_predict = [data["input_seq"][0, :trainer.dynamics.obs_dim].cpu().numpy()]  # align init

        print(f"\nStart testing {fpath}!")

        while data is not None:
            data = {k: v.to(trainer.device, non_blocking=True) for k, v in data.items()}
            ctl.append(data["input_seq"][0, trainer.dynamics.obs_dim:].cpu().numpy())
            
            if step < start_recurrent_step:
                # if history_max_len == 0:
                prediction, h = trainer.dynamics.forward(data["input_seq"], h)
                # else:
                #     prediction = trainer.dynamics.recur_forward(
                #         data["input_seq"], data["initial_history"], data["history_mask"])
                rot_error_1.append(geodesic_loss(
                    prediction[:, 0:3], data["target_seq"][:, 0:3]).item())
                pos_error_1.append(torch.norm(
                    prediction[:, 3:6] - data["target_seq"][:, 3:6], dim=-1).item())
                l_error_1.append(torch.norm(
                    prediction[:, 6:15] - data["target_seq"][:, 6:15], dim=-1).item())
                v_error_1.append(torch.norm(
                    prediction[:, 15:24] - data["target_seq"][:, 15:24], dim=-1).item())
            elif step < back_to_supervised_step:
                ctl_t = data["input_seq"][:, trainer.dynamics.obs_dim:]
                # if history_max_len == 0:
                prediction, h = trainer.dynamics.forward(torch.cat([prediction, ctl_t], dim=-1), h)
                # else:
                #     prediction = trainer.dynamics.recur_forward(
                #         torch.cat([prediction, ctl_t], dim=-1),
                #         data["initial_history"], data["history_mask"])
                rot_error_2.append(geodesic_loss(
                    prediction[:, 0:3], data["target_seq"][:, 0:3]).item())
                pos_error_2.append(torch.norm(
                    prediction[:, 3:6] - data["target_seq"][:, 3:6], dim=-1).item())
                l_error_2.append(torch.norm(
                    prediction[:, 6:15] - data["target_seq"][:, 6:15], dim=-1).item())
                v_error_2.append(torch.norm(
                    prediction[:, 15:24] - data["target_seq"][:, 15:24], dim=-1).item())
            else:
                prediction, h = trainer.dynamics.forward(data["input_seq"], h)
                rot_error_3.append(geodesic_loss(
                    prediction[:, 0:3], data["target_seq"][:, 0:3]).item())
                pos_error_3.append(torch.norm(
                    prediction[:, 3:6] - data["target_seq"][:, 3:6], dim=-1).item())
                l_error_3.append(torch.norm(
                    prediction[:, 6:15] - data["target_seq"][:, 6:15], dim=-1).item())
                v_error_3.append(torch.norm(
                    prediction[:, 15:24] - data["target_seq"][:, 15:24], dim=-1).item())
                
            rot_error.append(geodesic_loss(
                prediction[:, 0:3], data["target_seq"][:, 0:3]).item())
            pos_error.append(torch.norm(
                prediction[:, 3:6] - data["target_seq"][:, 3:6], dim=-1).item())
            l_error.append(torch.norm(
                prediction[:, 6:15] - data["target_seq"][:, 6:15], dim=-1).item())
            v_error.append(torch.norm(
                prediction[:, 15:24] - data["target_seq"][:, 15:24], dim=-1).item())

            obs_predict.append(prediction[0, :trainer.dynamics.obs_dim].cpu().numpy())
            obs_real.append(data["target_seq"][0, :trainer.dynamics.obs_dim].cpu().numpy())
            
            data = trainer.test_loader[0].dataset.get_data(fpath, step, history_max_len)
            step += 1

        save_data = {
            'step': list(range(1, len(rot_error) + 1)),
            'pos_x_real': [obs[3] for obs in obs_real[1:]], # skip step 0 which is aligned
            'pos_y_real': [obs[4] for obs in obs_real[1:]],
            'pos_z_real': [obs[5] for obs in obs_real[1:]],
            'pos_x_pred': [obs[3] for obs in obs_predict[1:]],
            'pos_y_pred': [obs[4] for obs in obs_predict[1:]],
            'pos_z_pred': [obs[5] for obs in obs_predict[1:]],
            'rot_error': rot_error,
            'pos_error': pos_error,
        }
        df = pd.DataFrame(save_data)
        csv_path = Path(trainer.save_folder) / f"{fpath.stem}_test.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

        test_plot(ctl, obs_real, obs_predict, rot_error, pos_error, l_error, v_error, 
                  trainer.save_folder, fpath.stem)

        avg_rot_error_1 = sum(rot_error_1) / len(rot_error_1) * 180 / np.pi
        print(f"Average rotvec error: {avg_rot_error_1} degree")
        avg_pos_error_1 = sum(pos_error_1) / len(pos_error_1)
        print(f"Average position error: {avg_pos_error_1} mm")
        avg_l_error_1 = sum(l_error_1) / len(l_error_1)
        print(f"Average length error: {avg_l_error_1} mm")
        avg_v_error_1 = sum(v_error_1) / len(v_error_1)
        print(f"Average velocity error: {avg_v_error_1} mm/s \n")

        avg_rot_error_2 = sum(rot_error_2) / len(rot_error_2) * 180 / np.pi
        print(f"Average rotvec error (2): {avg_rot_error_2} degree")
        avg_pos_error_2 = sum(pos_error_2) / len(pos_error_2)
        print(f"Average position error (2): {avg_pos_error_2} mm")
        avg_l_error_2 = sum(l_error_2) / len(l_error_2)
        print(f"Average length error (2): {avg_l_error_2} mm")
        avg_v_error_2 = sum(v_error_2) / len(v_error_2)
        print(f"Average velocity error (2): {avg_v_error_2} mm/s \n")

        avg_rot_error_3 = sum(rot_error_3) / len(rot_error_3) * 180 / np.pi
        print(f"Average rotvec error: {avg_rot_error_3} degree")
        avg_pos_error_3 = sum(pos_error_3) / len(pos_error_3)
        print(f"Average position error: {avg_pos_error_3} mm")
        avg_l_error_3 = sum(l_error_3) / len(l_error_3)
        print(f"Average length error: {avg_l_error_3} mm")
        avg_v_error_3 = sum(v_error_3) / len(v_error_3)
        print(f"Average velocity error: {avg_v_error_3} mm/s")
        

@ torch.inference_mode()
def closed_loop_policy_test(cfg_model: Config, cfg_training: Config):
    trainer = PolicyTrainer(cfg_model, cfg_training, test=True)
    trainer.dynamics.eval()
    trainer.policy.eval()
    if trainer.hidden_dynamics is not None:
        trainer.hidden_dynamics.eval()
    start_recurrent_step = 0
    start_recurrent_h_step = 0  # < start_recurrent_step
    data_ctl = False

    p = Path(trainer.test_loader[0].dataset.paths)
    files = [f for f in p.iterdir() if f.suffix in {'.h5', '.hdf5'}]
    for fpath in files:
        rot_error = []
        pos_error = []
        step = 0
        data = trainer.test_loader[0].dataset.get_data(fpath, step, history_max_len=50)
        ctl = []
        ref = []
        obs_predict = []
        ctl_t = torch.zeros(1, trainer.policy.ctl_dim).to(trainer.device)
        prediction = data["input_seq"][
            :, -trainer.dynamics.input_dim:-trainer.policy.ctl_dim].to(trainer.device)  # align init
        h = torch.zeros(
            1, trainer.dynamics.rnn_depth, trainer.dynamics.hidden_dim).to(trainer.device) # for rnn dynamics only
        h_ = torch.zeros(
            1, trainer.dynamics.rnn_depth, trainer.dynamics.hidden_dim).to(trainer.device) # for rnn dynamics only
        if trainer.hidden_dynamics is not None:
            policy_h = torch.zeros(
                1, trainer.dynamics.rnn_depth, trainer.dynamics.hidden_dim).to(trainer.device) # for hidden dynamics
            policy_h_ = torch.zeros(
                1, trainer.dynamics.rnn_depth, trainer.dynamics.hidden_dim).to(trainer.device) # for hidden dynamics
        if trainer.policy.net_type == "mlp":
            history = data["initial_history"].to(trainer.device)
        print(f"\nStart testing {fpath}!")

        while (data is not None) and (step < 20000):
            data = {k: v.to(trainer.device, non_blocking=True) for k, v in data.items()}
            ref.append(data["input_seq"][0, :-trainer.dynamics.input_dim].cpu().numpy())

            if step < start_recurrent_step:
                ctl_t = data["input_seq"][:, -trainer.policy.ctl_dim:]
                if step < start_recurrent_h_step:
                    prediction, h_ = trainer.dynamics.forward(data["input_seq"][:, -trainer.dynamics.input_dim:], h_)
                    if trainer.hidden_dynamics is not None:
                        _, policy_h_ = trainer.hidden_dynamics.forward(
                            data["input_seq"][:, -trainer.dynamics.input_dim:], policy_h_)
                else:
                    prediction, h = trainer.dynamics.forward(data["input_seq"][:, -trainer.dynamics.input_dim:], h)
                    if trainer.hidden_dynamics is not None:
                        _, policy_h = trainer.hidden_dynamics.forward(
                            data["input_seq"][:, -trainer.dynamics.input_dim:], policy_h)
                if trainer.policy.net_type == "mlp":
                    history = data["initial_history"]
            else:
                if data_ctl:
                    ctl_t = data["input_seq"][:, -trainer.policy.ctl_dim:]
                else:
                    ref_t = data["input_seq"][:, :-trainer.dynamics.input_dim]
                    if trainer.policy.ref_dim == 3:
                        ref_t = ref_t.reshape(-1, 6)[:, 3:6].reshape(-1, 3 * trainer.policy.ref_horizon)
                    if trainer.policy.net_type == "mlp":
                        ctl_t = trainer.policy.recur_forward(torch.cat([ref_t, prediction, ctl_t], dim=-1), history)  # r-o-u
                        history = torch.cat([history[:, 1:, :], torch.cat([prediction, ctl_t], dim=-1).unsqueeze(1)], dim=1)
                    else:
                        if trainer.hidden_dynamics is not None:
                            ctl_t = trainer.policy.recur_forward(torch.cat([ref_t, prediction, ctl_t], dim=-1), policy_h)  # r-o-u
                        else:
                            ctl_t = trainer.policy.recur_forward(torch.cat([ref_t, prediction, ctl_t], dim=-1), h)  # r-o-u
                prediction, h = trainer.dynamics(torch.cat([prediction, ctl_t], dim=-1), h)
                if trainer.hidden_dynamics is not None:
                    _, policy_h = trainer.hidden_dynamics(torch.cat([prediction, ctl_t], dim=-1), policy_h)

            ctl.append(ctl_t[0].cpu().numpy())
            obs_predict.append(prediction[0, :trainer.dynamics.obs_dim].cpu().numpy())

            rot_error.append(geodesic_loss(
                prediction[:, 0:3], data["input_seq"][:, 0:3]).item())
            pos_error.append(torch.norm(
                prediction[:, 3:6] - data["input_seq"][:, 3:6], dim=-1).item())
            
            step += 1   
            data = trainer.test_loader[0].dataset.get_data(fpath, step, history_max_len=50)

        save_data = {
            'step': list(range(1, len(rot_error) + 1)),
            'rot_error': rot_error,
            'pos_error': pos_error,
            'x': [obs[3] for obs in obs_predict],
            'y': [obs[4] for obs in obs_predict],
            'z': [obs[5] for obs in obs_predict],
            'x_ref': [obs[3] for obs in ref],
            'y_ref': [obs[4] for obs in ref],
            'z_ref': [obs[5] for obs in ref],
            'rot_x': [obs[0] for obs in obs_predict],
            'rot_y': [obs[1] for obs in obs_predict],
            'rot_z': [obs[2] for obs in obs_predict],
            'u_1': [u[0] for u in ctl],
            'u_2': [u[1] for u in ctl],
            'u_3': [u[2] for u in ctl],
            'u_4': [u[3] for u in ctl],
            'u_5': [u[4] for u in ctl],
            'u_6': [u[5] for u in ctl],
            'u_7': [u[6] for u in ctl],
            'u_8': [u[7] for u in ctl],
            'u_9': [u[8] for u in ctl],
        }
        df = pd.DataFrame(save_data)
        save_folder = Path(trainer.save_folder).parent.parent.parent / "data" / "policy_test_on_model" / Path(
            trainer.save_folder).name
        os.makedirs(save_folder, exist_ok=True)
        df.to_csv(save_folder / f"{fpath.stem}_test.csv", index=False)
        print(f"Results saved to: {save_folder / f'{fpath.stem}_test.csv'}")

        test_plot(ctl, ref, obs_predict, rot_error, pos_error, None, None, save_folder, fpath.stem)


def offline_train():
    config_path = Path(__file__).resolve().parent.parent / "config"
    config_model = Config.load(config_path / "models_config.yaml")
    config_training = Config.load(config_path / "training_config.yaml")

    # train_model(config_model, config_training, "dynamics")

    # train_model(config_model, config_training, "policy")

    # test_model(config_model, config_training, "dynamics")

    # closed_loop_dynamics_test(config_model, config_training)

    closed_loop_policy_test(config_model, config_training)


if __name__ == "__main__":
    offline_train()
