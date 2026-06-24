import os
import time
import copy
import random
import itertools
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dynamics import Dynamics
from models.policy import Policy
import train.tools.transforms3d as t3d
from train.tools.loss_fn import LogMSELoss, geodesic_loss
from train.data_loader import get_dataloader


class DynamicsTrainer:
    def __init__(
        self, 
        cfg_model: dict, 
        cfg_training: dict, 
        online: bool = False, 
        test: bool = False, 
        test_shuffle: bool = False,
    ):
        self.cfg_model = cfg_model
        self.cfg_training = cfg_training
        self.obs_type = cfg_model.dynamics.get("obs_type")
        self.device = torch.device(self.cfg_training.dynamics.get("device"))
        self.fine_tune = self.cfg_training.dynamics.get("fine_tune")

        # Loss computation
        self.bptt_steps = self.cfg_training.dynamics.get("bptt_steps")
        self.discount = self.cfg_training.dynamics.get("discount")
        loss_k_rotvec = self.cfg_training.dynamics.get("loss_weight.k_rotvec")
        loss_k_pos = self.cfg_training.dynamics.get("loss_weight.k_pos")
        loss_k_output_smooth = self.cfg_training.dynamics.get("loss_weight.k_output_smooth")
        loss_k_hidden_smooth = self.cfg_training.dynamics.get("loss_weight.k_hidden_smooth")
        if loss_k_output_smooth < 0.0001:
            loss_k_output_smooth = 0.0
            print("Output smooth loss weight is set to 0.")
        if loss_k_hidden_smooth < 0.0001:
            loss_k_hidden_smooth = 0.0
            print("Hidden smooth loss weight is set to 0.")
        self.loss_k_list = [loss_k_output_smooth, loss_k_hidden_smooth, loss_k_rotvec, loss_k_pos]
        self.loss_k_name_list = ["output_speed_smooth", "hidden_value_smooth", "rotvec", "pos"]
        self.loss_k_size_list = [3, 3]  # dim_size for rotvec and pos
        self.loss_fn = nn.MSELoss() # nn.SmoothL1Loss(beta=1.0)
        if self.discount == 1.0:
            self.scale = 1.0
        else:
            scale = (1 - self.discount ** self.bptt_steps) / (1 - self.discount)
            self.scale = self.bptt_steps / scale

        # Getdata loader and update loss weights
        if online:
            self.train_loader = self.get_loader(train_only=True)
        elif test:
            self.test_loader = self.get_loader(test_only=True, test_shuffle=test_shuffle)
        else:
            self.train_loader, self.eval_loader, self.val_loader = self.get_loader()

        # Initialize model
        print("Initializing model...")
        script_path = Path(__file__).resolve()
        train_root = script_path.parent
        self.save_folder = train_root / "checkpoints" / self.cfg_training.dynamics.get("name")
        os.makedirs(self.save_folder, exist_ok=True)

        self.dynamics = self.get_model()

        # Training setup
        weight_decay = self.cfg_training.dynamics.get("weight_decay")
        self.optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=self.cfg_training.dynamics.get("learning_rate"),
            weight_decay=weight_decay,
        )
        self.max_grad_norm = self.cfg_training.dynamics.get("max_grad_norm")

        self.iteration = 0
    
    def get_loader(self, train_only: bool = False, test_only: bool = False, test_shuffle: bool = False):
        # Getdata loader
        print("Getting data loader...")
        data_source = self.cfg_training.dynamics.get("data_source")
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        # if test_only:
        #     test_data_folder = project_root / "data" / data_source / "processed_logs_test"
        if test_only:
            test_list = self.cfg_training.dynamics.get("test_list")
            test_data_folder = []
            if test_list is not None:
                for testset_name in test_list:
                    test_data_folder.append(
                        project_root / "data" / data_source / f"processed_logs_{testset_name}")
        else:
            train_data_folder = project_root / "data" / data_source / "processed_logs_train"
            if not train_only:
                eval_data_folder = project_root / "data" / data_source / "processed_logs_eval"

                validation_list = self.cfg_training.dynamics.get("validation_list")
                val_data_folder = []
                if validation_list is not None:
                    for valset_name in validation_list:
                        val_data_folder.append(
                            project_root / "data" / data_source / f"processed_logs_{valset_name}")

        input_key_list = []
        output_key_list = ["rotvec_t_plus_1", "pos_t_plus_1"]
        if "T" in self.obs_type:
            input_key_list += ["rotvec_t", "pos_t"]
        if "l" in self.obs_type:
            input_key_list += ["l_t"]
            output_key_list += ["l_t_plus_1"]
            self.loss_k_list.append(self.cfg_training.dynamics.get("loss_weight.k_l"))
            self.loss_k_name_list.append("l")
            self.loss_k_size_list.append(9)
        if "v" in self.obs_type:
            input_key_list += ["v_t"]
            output_key_list += ["v_t_plus_1"]
            self.loss_k_list.append(self.cfg_training.dynamics.get("loss_weight.k_v"))
            self.loss_k_name_list.append("v")
            self.loss_k_size_list.append(9)
        if "q" in self.obs_type:
            input_key_list += ["torque_t"]
            output_key_list += ["torque_t_plus_1"]
            self.loss_k_list.append(self.cfg_training.dynamics.get("loss_weight.k_torque"))
            self.loss_k_name_list.append("torque")
            self.loss_k_size_list.append(9)
        input_key_list += ["u_t"]

        if test_only:
            test_loader = []
            for folder in test_data_folder:
                test_loader.append(get_dataloader(
                    paths=folder,
                    input_key_list=list(input_key_list),
                    output_key_list=output_key_list,
                    history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                    bptt_steps=self.bptt_steps,
                    min_regressive_steps=self.bptt_steps,
                    same_len_history=self.cfg_model.dynamics.get("net_type") == "mlp",
                    batch_size=self.cfg_training.dynamics.get("batch_size"),
                    num_workers=self.cfg_training.dynamics.get("num_workers"),
                    history_max_len=self.cfg_model.dynamics.get(
                        "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
                    shuffle=test_shuffle,
                ))
                print(f"Test set length: {len(test_loader[-1].dataset)}")
            return test_loader

        else:
            train_loader = get_dataloader(
                paths=train_data_folder,
                input_key_list=list(input_key_list),
                output_key_list=output_key_list,
                history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                bptt_steps=self.bptt_steps,
                min_regressive_steps=self.bptt_steps,
                same_len_history=self.cfg_model.dynamics.get("net_type") == "mlp",
                batch_size=self.cfg_training.dynamics.get("batch_size"),
                num_workers=self.cfg_training.dynamics.get("num_workers"),
                history_max_len=self.cfg_model.dynamics.get(
                    "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
            )
            print(f"Train set length: {len(train_loader.dataset)}")

            if train_only:
                return train_loader
            else:
                eval_loader = get_dataloader(
                    paths=eval_data_folder,
                    input_key_list=list(input_key_list),
                    output_key_list=output_key_list,
                    history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                    bptt_steps=self.bptt_steps,
                    min_regressive_steps=self.bptt_steps,
                    same_len_history=self.cfg_model.dynamics.get("net_type") == "mlp",
                    batch_size=self.cfg_training.dynamics.get("batch_size"),
                    num_workers=self.cfg_training.dynamics.get("num_workers"),
                    history_max_len=self.cfg_model.dynamics.get(
                        "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
                )
                print(f"Evaluation set length: {len(eval_loader.dataset)}")

                val_loader = []
                for folder in val_data_folder:
                    val_loader.append(get_dataloader(
                        paths=folder,
                        input_key_list=input_key_list,
                        output_key_list=output_key_list,
                        history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                        bptt_steps=self.bptt_steps,
                        min_regressive_steps=self.bptt_steps,
                        same_len_history=self.cfg_model.dynamics.get("net_type") == "mlp",
                        batch_size=self.cfg_training.dynamics.get("batch_size"),
                        num_workers=self.cfg_training.dynamics.get("num_workers"),
                        history_max_len=self.cfg_model.dynamics.get(
                            "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
                    ))
                    print(f"Validation set length: {len(val_loader[-1].dataset)}")
                return train_loader, eval_loader, val_loader
        
    
    def get_model(self):
        load_name = self.cfg_training.dynamics.get("load_name")
        if load_name is not None:
            load_pth = self.save_folder.parent / load_name / \
                  f"dynamics_{self.cfg_training.dynamics.get('load_type')}.pth"
        else:
            load_pth = None
        dynamics = Dynamics(self.cfg_model, self.device, load_pth)
        print(f"Dynamics model total parameters: {dynamics.total_params}")
        print(f"Dynamics model trainable parameters: {dynamics.trainable_params}")
        return dynamics
    
    def compute_loss(self, predictions, targets, last_predictions, 
                     last_last_predictions, last_last_predictions_mask, h, last_h=None):
        loss = []
        smooth_mse = []

        # output smooth loss
        if last_last_predictions_mask is not None:
            smooth_mse.append(self.loss_fn(
                (predictions - last_predictions) * last_last_predictions_mask,
                (last_predictions - last_last_predictions) * last_last_predictions_mask))
        else:
            smooth_mse.append(self.loss_fn(
                predictions - last_predictions,
                last_predictions - last_last_predictions))
        if self.loss_k_list[0] < 0.0001:
            loss.append(torch.zeros((), device=self.device))
        else:
            loss.append(self.loss_k_list[0] * smooth_mse[-1])

        # hidden smooth loss
        if last_h is not None:
            smooth_mse.append(self.loss_fn(h, last_h))
        else:
            smooth_mse.append(torch.zeros((), device=self.device))
        if self.loss_k_list[1] < 0.0001:
            loss.append(torch.zeros((), device=self.device))
        else:
            loss.append(self.loss_k_list[1] * smooth_mse[-1])

        # obs loss
        pred_splits = torch.split(predictions, self.loss_k_size_list, dim=-1)
        target_splits = torch.split(targets, self.loss_k_size_list, dim=-1)
        for k, p, t in zip(self.loss_k_list[2:], pred_splits, target_splits):
            loss.append(k * self.loss_fn(p, t))

        return loss, smooth_mse
    
    def update(self, batch, teacher_forcing=False):
        # update model parameters based on a single batch
        self.optimizer.zero_grad(set_to_none=True)
        self.dynamics.train()
        if self.fine_tune:
            for name, param in self.dynamics.named_parameters():
                if "rnn_net" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True 
        else:
            for param in self.dynamics.parameters():
                param.requires_grad = True
        
        # forward pass
        sampled_pred = []
        sampled_tgt = []
        record = random.random() < 0.1
        total_loss = [torch.zeros((), device=self.device) for _ in self.loss_k_list]
        total_smooth_mse = [torch.zeros((), device=self.device) for _ in range(2)]
        sampled_history = None
        last_h = None
        for step in range(self.bptt_steps):
            if step == 0:
                inputs = batch["input_seq"][:, step, :]
                initial_history = batch["initial_history"]
                history_mask = batch["history_mask"]
                last_predictions = inputs[:, :self.dynamics.obs_dim]
                last_last_predictions = initial_history[:, -1, :self.dynamics.obs_dim]
                last_last_predictions_mask = history_mask[:, -1].unsqueeze(-1)
                if record:
                    sampled_pred.append(inputs[0, :self.dynamics.obs_dim].detach().cpu().numpy())
                    sampled_tgt.append(inputs[0, :self.dynamics.obs_dim].detach().cpu().numpy())
                    sampled_history = torch.cat([
                        initial_history[0, :, :self.dynamics.obs_dim][history_mask[0]], 
                        inputs[0, :self.dynamics.obs_dim].unsqueeze(0)], dim=0).detach().cpu().numpy()
            else:
                if teacher_forcing:
                    inputs = batch["input_seq"][:, step, :]
                else:
                    inputs = torch.cat(
                        [predictions, batch["input_seq"][:, step, -self.dynamics.ctl_dim:]], dim=-1)
                initial_history = None
                history_mask = None
            predictions = self.dynamics.recur_forward(inputs, initial_history, history_mask)
            targets = batch["target_seq"][:, step, :]
            if record:
                sampled_pred.append(predictions[0].detach().cpu().numpy())
                sampled_tgt.append(targets[0].detach().cpu().numpy())
            
            dynamics_h = self.dynamics.h[0] if isinstance(self.dynamics.h, tuple) else self.dynamics.h
            step_loss, step_smooth_mse = self.compute_loss(
                predictions, targets, last_predictions, last_last_predictions, 
                last_last_predictions_mask, dynamics_h, last_h)
        
            step_loss = [s * (self.discount ** step) for s in step_loss]
            step_smooth_mse = [s * (self.discount ** step) for s in step_smooth_mse]
            total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
            total_smooth_mse = [t_l + s_l for t_l, s_l in zip(total_smooth_mse, step_smooth_mse)]

            last_last_predictions = last_predictions
            last_last_predictions_mask = None
            last_predictions = predictions
            last_h = dynamics_h

        total_loss = [t_l / self.bptt_steps for t_l in total_loss]
        total_smooth_mse = [t_l / self.bptt_steps for t_l in total_smooth_mse]
        loss = sum(total_loss)
        
        # update model parameters
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.max_grad_norm)
        self.optimizer.step()

        iter_info = {"train/iteration": self.iteration,
                     "train/loss": loss.detach().cpu().item()}
        for i, l in enumerate(total_loss):
            iter_info[f"train/loss_{self.loss_k_name_list[i]}"] = l.detach().cpu().item()
        for i, l in enumerate(total_smooth_mse):
            iter_info[f"train/rmse_{self.loss_k_name_list[i]}"] = (
                l.detach().cpu().item() * self.scale) ** 0.5
        for i, l in enumerate(total_loss[2:]):
            iter_info[f"train/rmse_{self.loss_k_name_list[i+2]}"] = (
                l.detach().cpu().item() / self.loss_k_list[i+2] * self.scale) ** 0.5

        self.iteration += 1

        return iter_info, sampled_pred, sampled_tgt, None, None, None, sampled_history
    
    @torch.inference_mode()
    def evaluate(self, batch, type="eval"):
        self.dynamics.eval()

        # forward pass
        sampled_pred = []
        sampled_tgt = []
        record = random.random() < 0.1
        total_loss = [torch.zeros((), device=self.device) for _ in range(len(self.loss_k_list))]
        total_smooth_mse = [torch.zeros((), device=self.device) for _ in range(2)]
        sampled_history = None
        last_h = None
        for step in range(self.bptt_steps):
            if step == 0:
                inputs = batch["input_seq"][:, step, :]
                initial_history = batch["initial_history"]
                history_mask = batch["history_mask"]
                last_predictions = inputs[:, :self.dynamics.obs_dim]
                last_last_predictions = initial_history[:, -1, :self.dynamics.obs_dim]
                last_last_predictions_mask = history_mask[:, -1].unsqueeze(-1)
                if record:
                    sampled_pred.append(inputs[0, :self.dynamics.obs_dim].detach().cpu().numpy())
                    sampled_tgt.append(inputs[0, :self.dynamics.obs_dim].detach().cpu().numpy())
                    sampled_history = torch.cat([
                        initial_history[0, :, :self.dynamics.obs_dim][history_mask[0]], 
                        inputs[0, :self.dynamics.obs_dim].unsqueeze(0)], dim=0).detach().cpu().numpy()
            else:
                inputs = torch.cat(
                        [predictions, batch["input_seq"][:, step, -self.dynamics.ctl_dim:]], dim=-1)
                initial_history = None
                history_mask = None
            # start = time.time()
            predictions = self.dynamics.recur_forward(inputs, initial_history, history_mask)
            # inference_time = time.time() - start
            # print(f"Inference time for step {step}: {inference_time} seconds")
            targets = batch["target_seq"][:, step, :]
            if record:
                sampled_pred.append(predictions[0].detach().cpu().numpy())
                sampled_tgt.append(targets[0].detach().cpu().numpy())
                if random.random() < 0.01:
                    print(f"Step {step} prediction - target: {sampled_pred[-1] - sampled_tgt[-1]}")

            dynamics_h = self.dynamics.h[0] if isinstance(self.dynamics.h, tuple) else self.dynamics.h
            step_loss, step_smooth_mse = self.compute_loss(
                predictions, targets, last_predictions, last_last_predictions, 
                last_last_predictions_mask, dynamics_h, last_h)
            step_loss = [s * (self.discount ** step) for s in step_loss]
            step_smooth_mse = [s * (self.discount ** step) for s in step_smooth_mse]
            total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
            total_smooth_mse = [t_l + s_l for t_l, s_l in zip(total_smooth_mse, step_smooth_mse)]

            last_last_predictions = last_predictions
            last_last_predictions_mask = None
            last_predictions = predictions
            last_h = dynamics_h

        total_loss = [t_l / self.bptt_steps for t_l in total_loss]
        total_smooth_mse = [t_l / self.bptt_steps for t_l in total_smooth_mse]
        loss = sum(total_loss)

        batch_info = {f"{type}/loss": loss.detach().cpu().item()}
        # for i, l in enumerate(total_loss):
        #     batch_info[f"{type}/loss_k{self.loss_k_name_list[i]}"] = l.detach().cpu().item()
        #     batch_info[f"{type}/rmse_{self.loss_k_name_list[i]}"] = (
        #         l.detach().cpu().item() / self.loss_k_list[i] * self.scale) ** 0.5
        for i, l in enumerate(total_loss):
            batch_info[f"{type}/loss_{self.loss_k_name_list[i]}"] = l.detach().cpu().item()
        for i, l in enumerate(total_smooth_mse):
            batch_info[f"{type}/rmse_{self.loss_k_name_list[i]}"] = (
                l.detach().cpu().item() * self.scale) ** 0.5
        for i, l in enumerate(total_loss[2:]):
            batch_info[f"{type}/rmse_{self.loss_k_name_list[i+2]}"] = (
                l.detach().cpu().item() / self.loss_k_list[i+2] * self.scale) ** 0.5

        return batch_info, sampled_pred, sampled_tgt, None, None, None, sampled_history
    
    @torch.inference_mode()
    def test(self, batch):
        self.dynamics.eval()

        batch_size = batch["input_seq"].size(0)
        batch_avg_rot_error = []  # geodesic distance in rad from step 0 to self.bptt_step
        batch_avg_pos_error = []  # mm
        batch_avg_l_error = []  # mm
        batch_avg_v_error = []   # mm/s

        # forward pass
        for step in range(self.bptt_steps):
            if step == 0:
                inputs = batch["input_seq"][:, step, :]
                initial_history = batch["initial_history"]
                history_mask = batch["history_mask"]
            else:
                inputs = torch.cat(
                        [predictions, batch["input_seq"][:, step, -self.dynamics.ctl_dim:]], dim=-1)
                initial_history = None
                history_mask = None
            predictions = self.dynamics.recur_forward(inputs, initial_history, history_mask)
            targets = batch["target_seq"][:, step, :]

            batch_avg_rot_error.append(geodesic_loss(
                predictions[:, 0:3], targets[:, 0:3]).mean().item())
            batch_avg_pos_error.append(torch.norm(
                predictions[:, 3:6] - targets[:, 3:6], dim=-1).mean().item())
            batch_avg_l_error.append(torch.norm(
                predictions[:, 6:15] - targets[:, 6:15], dim=-1).mean().item())
            batch_avg_v_error.append(torch.norm(
                predictions[:, 15:24] - targets[:, 15:24], dim=-1).mean().item())

        return batch_size, batch_avg_rot_error, batch_avg_pos_error, batch_avg_l_error, batch_avg_v_error
    

class PolicyTrainer:
    def __init__(
        self, 
        cfg_model: dict,
        cfg_training: dict, 
        online: bool = False, 
        dynamics: Dynamics = None,
        test: bool = False
    ):
        self.cfg_model = cfg_model
        self.cfg_training = cfg_training
        self.obs_type = cfg_model.policy.get("obs_type")
        self.device = torch.device(self.cfg_training.policy.get("device"))
        self.obs_ref = self.cfg_training.policy.get("obs_ref")
        self.fine_tune = self.cfg_training.policy.get("fine_tune")
        self.bptt_steps = self.cfg_training.policy.get("bptt_steps")
        self.min_regressive_steps = self.cfg_training.policy.get("min_regressive_steps")

        # Initialize model
        print("Initializing model...")
        script_path = Path(__file__).resolve()
        train_root = script_path.parent
        self.save_folder = train_root / "checkpoints" / self.cfg_training.policy.get("name")                     
        os.makedirs(self.save_folder, exist_ok=True)
        self.hidden_dynamics = None
        if dynamics is not None:
            self.dynamics = dynamics
            self.policy = self.get_model(policy_only=True)
            if self.cfg_model.policy.get("hidden_dynamics"):
                self.hidden_dynamics = copy.deepcopy(self.dynamics)
        else:
            if self.cfg_model.policy.get("hidden_dynamics"):
                self.dynamics, self.policy, self.hidden_dynamics = self.get_model()
            else:
                self.dynamics, self.policy = self.get_model()
        self.ref_t_dim = self.policy.ref_dim * self.policy.ref_horizon
        self.ref_pos_only = "pos_only" in self.policy.input_type

        # Getdata loader and update loss weights
        if self.obs_ref:
            self.sine_freq_max = self.cfg_training.policy.get("sine_freq_max")
            self.step_rate = self.cfg_training.policy.get("step_rate")
            self.bias_max_pos = self.cfg_training.policy.get("bias_max_pos")
            self.sine_amp_max_pos = self.cfg_training.policy.get("sine_amp_max_pos")
            self.step_max_pos = self.cfg_training.policy.get("step_max_pos")
            self.bias_max_rot = self.cfg_training.policy.get("bias_max_rot")
            self.sine_amp_max_rot = self.cfg_training.policy.get("sine_amp_max_rot")
            self.step_max_rot = self.cfg_training.policy.get("step_max_rot")
        else:
            self.closer_scale = self.cfg_training.policy.get("closer_scale")
        if online:
            self.train_loader = self.get_loader(train_only=True)
        elif test:
            self.test_loader = self.get_loader(test_only=True)
        else:
            self.train_loader, self.eval_loader, self.val_loader = self.get_loader()

        self.truncation_steps = self.cfg_training.policy.get("truncation_steps")
        self.discount = self.cfg_training.policy.get("discount")

        # Loss computation
        self.loss_k_smooth = self.cfg_training.policy.get("loss_weight.k_smooth")
        self.loss_k_rotvec = self.cfg_training.policy.get("loss_weight.k_rotvec")
        self.loss_k_pos = self.cfg_training.policy.get("loss_weight.k_pos")
        self.loss_k_imitation = self.cfg_training.policy.get("loss_weight.k_imitation")
        self.loss_k_list = []
        self.loss_k_name_list = []
        if self.loss_k_smooth < 0.0001:
            print("Smooth loss weight is set to 0.")
        else:
            self.loss_k_list.append(self.loss_k_smooth)
            self.loss_k_name_list.append("smooth")
        if self.loss_k_rotvec < 0.0001:
            print("Rotvec loss weight is set to 0.")
        else:
            self.loss_k_list.append(self.loss_k_rotvec)
            self.loss_k_name_list.append("rotvec")
        if self.loss_k_pos < 0.0001:
            print("Pos loss weight is set to 0.")
        else:
            self.loss_k_list.append(self.loss_k_pos)
            self.loss_k_name_list.append("pos")
        if self.loss_k_imitation < 0.0001:
            print("Imitation loss weight is set to 0.")
        else:
            self.loss_k_list.append(self.loss_k_imitation)
            self.loss_k_name_list.append("imitation")

        self.loss_fn = nn.MSELoss()  # nn.SmoothL1Loss(beta=1.0)
        self.loss_fn_rotvec = nn.MSELoss()  # LogMSELoss(epsilon=0.5)
        self.loss_fn_pos = nn.MSELoss()  # LogMSELoss(epsilon=0.1, contraction=1e-6)
        if self.discount == 1.0:
            self.scale = 1.0
        else:
            scale = (1 - self.discount ** self.bptt_steps) / (1 - self.discount)
            self.scale = self.bptt_steps / scale

        # Training setup
        weight_decay = self.cfg_training.policy.get("weight_decay")
        if self.hidden_dynamics is not None:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.hidden_dynamics.parameters()),
                lr=self.cfg_training.policy.get("learning_rate"),
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=self.cfg_training.policy.get("learning_rate"),
                weight_decay=weight_decay,
            )
        self.scaler = torch.amp.GradScaler(self.device)
        self.max_grad_norm = self.cfg_training.policy.get("max_grad_norm")

        self.iteration = 0
    
    def get_loader(self, train_only=False, test_only: bool = False):
        # Getdata loader
        print("Getting data loader...")
        data_source = self.cfg_training.policy.get("data_source")
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        if test_only:
            test_list = self.cfg_training.policy.get("test_list")
            test_data_folder = []
            if test_list is not None:
                for testset_name in test_list:
                    test_data_folder.append(
                        project_root / "data" / data_source / f"processed_logs_{testset_name}")
        else:
            train_data_folder = project_root / "data" / data_source / "processed_logs_train"
            if not train_only:
                eval_data_folder = project_root / "data" / data_source / "processed_logs_eval"

                validation_list = self.cfg_training.policy.get("validation_list")
                val_data_folder = []
                if validation_list is not None:
                    for valset_name in validation_list:
                        val_data_folder.append(
                            project_root / "data" / data_source / f"processed_logs_{valset_name}")

        if self.obs_ref: # use real obs as ref
            input_key_list = ["ref_t_obs"]
        else:
            input_key_list = ["ref_t"]

        if "T" in self.obs_type:
            input_key_list += ["rotvec_t", "pos_t"]
        if "l" in self.obs_type:
            input_key_list += ["l_t"]
        if "v" in self.obs_type:
            input_key_list += ["v_t"]
        if "q" in self.obs_type:
            input_key_list += ["torque_t"]

        input_key_list += ["u_t"]

        output_key_list = ["u_t_minus_1"]  # make sure output_list is not empty

        if test_only:
            test_loader = []
            for folder in test_data_folder:
                test_loader.append(get_dataloader(
                    paths=folder,
                    input_key_list=list(input_key_list),
                    output_key_list=output_key_list,
                    history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                    bptt_steps=self.bptt_steps,
                    min_regressive_steps=self.min_regressive_steps,
                    same_len_history=self.cfg_model.policy.get("net_type") == "mlp",
                    batch_size=self.cfg_training.policy.get("batch_size"),
                    num_workers=self.cfg_training.policy.get("num_workers"),
                    ref_horizon=self.cfg_model.policy.get("ref_horizon"),
                    ref_pos_only=self.ref_pos_only,
                    history_max_len=self.cfg_model.dynamics.get(
                        "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
                    shuffle=False,
                ))
                print(f"Test set length: {len(test_loader[-1].dataset)}")
            return test_loader
        
        train_loader = get_dataloader(
            paths=train_data_folder,
            input_key_list=list(input_key_list),
            output_key_list=output_key_list,
            history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
            bptt_steps=self.bptt_steps,
            min_regressive_steps=self.min_regressive_steps,
            same_len_history=self.cfg_model.policy.get("net_type") == "mlp",
            batch_size=self.cfg_training.policy.get("batch_size"),
            num_workers=self.cfg_training.policy.get("num_workers"),
            ref_horizon=self.cfg_model.policy.get("ref_horizon"),
            ref_pos_only=self.ref_pos_only,
            history_max_len=self.cfg_model.dynamics.get(
                "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
        )
        print(f"Train set length: {len(train_loader.dataset)}")

        if not train_only:
            eval_loader = get_dataloader(
                paths=eval_data_folder,
                input_key_list=list(input_key_list),
                output_key_list=output_key_list,
                history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                bptt_steps=self.bptt_steps,
                min_regressive_steps=self.min_regressive_steps,
                same_len_history=self.cfg_model.policy.get("net_type") == "mlp",
                batch_size=self.cfg_training.policy.get("batch_size"),
                num_workers=self.cfg_training.policy.get("num_workers"),
                ref_horizon=self.cfg_model.policy.get("ref_horizon"),
                ref_pos_only=self.ref_pos_only,
                history_max_len=self.cfg_model.dynamics.get(
                    "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
            )
            print(f"Evaluation set length: {len(eval_loader.dataset)}")

            val_loader = []
            for folder in val_data_folder:
                val_loader.append(get_dataloader(
                    paths=folder,
                    input_key_list=list(input_key_list),
                    output_key_list=output_key_list,
                    history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                    bptt_steps=self.bptt_steps,
                    min_regressive_steps=self.min_regressive_steps,
                    same_len_history=self.cfg_model.policy.get("net_type") == "mlp",
                    batch_size=self.cfg_training.policy.get("batch_size"),
                    num_workers=self.cfg_training.policy.get("num_workers"),
                    ref_horizon=self.cfg_model.policy.get("ref_horizon"),
                    ref_pos_only=self.ref_pos_only,
                    history_max_len=self.cfg_model.dynamics.get(
                        "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
                ))
                print(f"Validation set length: {len(val_loader[-1].dataset)}")
            return train_loader, eval_loader, val_loader
        return train_loader

    def get_model(self, policy_only: bool = False):
        load_name = self.cfg_training.policy.get("load_name")
        if load_name is not None:
            load_path = self.save_folder.parent / load_name / \
                f"policy_{self.cfg_training.policy.get('policy_load_type')}.pth"
        else:
            load_path = None
        policy = Policy(self.cfg_model, self.device, load_path)
        print(f"Policy model total parameters: {policy.total_params}")
        print(f"Policy model trainable parameters: {policy.trainable_params}")
        if policy_only:
            return policy

        dynamics_name = self.cfg_training.policy.get("dynamics_name")
        dynamics_pth = self.save_folder.parent / dynamics_name / \
            f"dynamics_{self.cfg_training.policy.get('dynamics_load_type')}.pth"
        dynamics = Dynamics(self.cfg_model, self.device, dynamics_pth)

        if self.cfg_model.policy.get("hidden_dynamics"):
            if load_name is not None:
                hidden_dynamics_pth = self.save_folder.parent / load_name / \
                    f"hidden_dynamics_{self.cfg_training.policy.get('policy_load_type')}.pth"
                hidden_dynamics = Dynamics(self.cfg_model, self.device, hidden_dynamics_pth)
            else:
                hidden_dynamics = copy.deepcopy(dynamics)
            return dynamics, policy, hidden_dynamics
        
        return dynamics, policy
    
    def geodesic_mse_loss(self, rotvec_pred, rotvec_ref): 
        q_pred = t3d.axis_angle_to_quaternion(rotvec_pred) 
        q_ref = t3d.axis_angle_to_quaternion(rotvec_ref) 
        q_ref_conj = t3d.quaternion_invert(q_ref)
        q_delta = t3d.quaternion_multiply(q_pred, q_ref_conj)
        w = torch.clamp(q_delta[..., 0].abs(), 0.0, 1.0 - 1e-5)  # shape (B,)
        theta = 2.0 * torch.arccos(w)  # shape (B,)
        return theta ** 2  # shape (B,)
    
    def compute_loss(self, ref_t, predictions, ctl, last_ctl=None, target_ctl=None):
        loss = []
        if self.loss_k_smooth >= 0.0001:
            if last_ctl is not None:
                loss.append(self.loss_k_smooth * self.loss_fn(
                    ctl, last_ctl))  # no detach to ensure both a2b and b2a
            else:
                loss.append(torch.zeros((), device=self.device))

        if self.ref_pos_only:
            if self.loss_k_pos >= 0.0001:
                loss.append(self.loss_k_pos * self.loss_fn_pos(
                    predictions[:, 3:6], 
                    (ref_t[:, :self.policy.ref_dim]).detach()))
        else:
            if self.loss_k_rotvec >= 0.0001:
                loss.append(self.loss_k_rotvec * self.loss_fn_rotvec(
                    predictions[:, :3], (ref_t[:, :3]).detach()))
            if self.loss_k_pos >= 0.0001:
                loss.append(self.loss_k_pos * self.loss_fn_pos(
                    predictions[:, 3:self.policy.ref_dim], 
                    (ref_t[:, 3:self.policy.ref_dim]).detach()))
        
        if self.loss_k_imitation >= 0.0001:
            if target_ctl is not None:
                loss.append(self.loss_k_imitation * self.loss_fn(
                    ctl, target_ctl.detach()))
            else:
                raise ValueError("Target control is required for imitation loss.")

        return loss
    
    @ torch.no_grad()
    def compute_delta(self, obs, ref_t_origin):
        """ compute delta between ref and interped ref based on the first ref and obs"""
        if self.closer_scale < 0.0001:
            k_dis = torch.zeros(obs.size(0), 1, device=self.device)
        else:
            k_dis = self.closer_scale + (1 - self.closer_scale) * torch.rand(
                obs.size(0), 1, device=self.device)  # make sure little error tracking

        if not self.ref_pos_only:
            q_obs = t3d.axis_angle_to_quaternion(obs[:, :3])
            q_ref = t3d.axis_angle_to_quaternion(ref_t_origin[:, :3])
            q_ref_conj = t3d.quaternion_invert(q_ref)
            q_delta = t3d.quaternion_multiply(q_obs, q_ref_conj)
            rotvec_delta = t3d.quaternion_to_axis_angle(q_delta)
            delta_ref_rot = rotvec_delta * k_dis  # rotvec
            delta_ref_rot = t3d.axis_angle_to_quaternion(delta_ref_rot) # delta quaternion

            pos_delta = obs[:, 3:self.policy.ref_dim] - ref_t_origin[:, 3:self.policy.ref_dim]
        else:
            delta_ref_rot = None
            pos_delta = obs[:, :self.policy.ref_dim] - ref_t_origin[:, :self.policy.ref_dim]

        delta_ref_pos = pos_delta * k_dis
        
        new_ref_t = self.interp_ref_t(ref_t_origin, delta_ref_rot, delta_ref_pos)

        return delta_ref_rot, delta_ref_pos, new_ref_t
    
    @ torch.no_grad()
    def interp_ref_t(self, ref_t, delta_ref_rot, delta_ref_pos):
        """ interpolate ref towards obs by delta_ref """
        if delta_ref_rot is not None:
            def split_ref(ref_t):
                ret_t_view = ref_t.view(
                    ref_t.shape[0], self.policy.ref_horizon, self.policy.ref_dim)
                ref_t_rot = ret_t_view[..., :3]
                ref_t_pos = ret_t_view[..., 3:]
                return ref_t_rot, ref_t_pos
            
            def integrate_ref(ref_t_rot, ref_t_pos):
                new_ref = torch.cat([ref_t_rot, ref_t_pos], dim=-1).view(
                    ref_t.shape[0], self.ref_t_dim)
                return new_ref

            ref_t_rot, ref_t_pos = split_ref(ref_t)
            if delta_ref_rot is None:
                new_ref_rot = ref_t_rot
            else:
                q_ref = t3d.axis_angle_to_quaternion(ref_t_rot)  # (B, H, 4)
                q_new_ref = t3d.quaternion_multiply(delta_ref_rot.unsqueeze(1), q_ref)  # (B, H, 4)
                new_ref_rot = t3d.quaternion_to_axis_angle(q_new_ref)

            new_ref_pos = ref_t_pos + delta_ref_pos.unsqueeze(1)
            new_ref = integrate_ref(new_ref_rot, new_ref_pos)
        else:
            new_ref_pos = ref_t.view(
                ref_t.shape[0], self.policy.ref_horizon, self.policy.ref_dim) + delta_ref_pos.unsqueeze(1)
            new_ref = new_ref_pos.view(ref_t.shape[0], self.ref_t_dim)

        return new_ref

    @ torch.no_grad()
    def interp_ref(self, ref, delta_ref_rot, delta_ref_pos):
        """ interpolate ref towards obs by delta_ref """
        if delta_ref_rot is not None:
            q_ref = t3d.axis_angle_to_quaternion(ref[:, :3])
            q_new_ref = t3d.quaternion_multiply(delta_ref_rot, q_ref)
            new_ref_rot = t3d.quaternion_to_axis_angle(q_new_ref)
            new_ref_pos = ref[:, 3:self.policy.ref_dim] + delta_ref_pos
            return new_ref_rot, new_ref_pos
        else:
            new_ref_pos = ref[:, 3:self.policy.ref_dim] + delta_ref_pos
            return new_ref_pos, 

    @ torch.no_grad()
    def init_noise(self, ref_t_origin):
        bs = ref_t_origin.size(0)
        horizon = self.policy.ref_horizon
        t = torch.arange(horizon, device=self.device).view(1, horizon, 1)

        if self.ref_pos_only:
            ref_t_origin = ref_t_origin.view(bs, horizon, 3)

            bias = (torch.rand(bs, 1, 3, device=self.device) * 2 - 1) * self.bias_max_pos
            freq = torch.rand(bs, 1, 3, device=self.device) * self.sine_freq_max
            amp = torch.rand(bs, 1, 3, device=self.device) * self.sine_amp_max_pos
            phi = torch.rand(bs, 1, 3, device=self.device) * 2 * np.pi
            
            sine_noise = amp * torch.sin(2 * np.pi * freq * t + phi)  # [bs, horizon, 3]
            
            step_mask = (torch.rand(bs, horizon, 3, device=self.device) < self.step_rate).float()
            step_values = (torch.rand(bs, horizon, 3, device=self.device) * 2 - 1) * self.step_max_pos
            step_noise = torch.cumsum(step_mask * step_values, dim=1)
            
            total_noise = bias + sine_noise + step_noise

            mask = (torch.rand(1, 1, 3, device=self.device) < 0.5).float()
            ref_t_initial = ref_t_origin[:, 0:1, :]
            ref_base = mask * ref_t_origin + (1 - mask) * ref_t_initial
            ref_t = ref_base + total_noise
            ref_t = ref_t.reshape(bs, horizon * 3)
            
            noise_info = {
                "bias": bias, "freq": freq, "amp": amp, "phi": phi,
                "current_step_offset": step_noise[:, -1:, :], "mask": mask,
                "ref_t_initial": ref_t_initial,
                "t_next": horizon
            }
        else:
            ref_t_origin = ref_t_origin.view(bs, horizon, 6)

            # pos noise
            bias_pos = (torch.rand(bs, 1, 3, device=self.device) * 2 - 1) * self.bias_max_pos
            freq_pos = torch.rand(bs, 1, 3, device=self.device) * self.sine_freq_max
            amp_pos = torch.rand(bs, 1, 3, device=self.device) * self.sine_amp_max_pos
            phi_pos = torch.rand(bs, 1, 3, device=self.device) * 2 * np.pi
            
            sine_noise_pos = amp_pos * torch.sin(2 * np.pi * freq_pos * t + phi_pos)  # [bs, horizon, 3]
            
            step_mask_pos = (torch.rand(bs, horizon, 3, device=self.device) < self.step_rate).float()
            step_values_pos = (torch.rand(
                bs, horizon, 3, device=self.device) * 2 - 1) * self.step_max_pos
            step_noise_pos = torch.cumsum(step_mask_pos * step_values_pos, dim=1)
            
            total_noise_pos = bias_pos + sine_noise_pos + step_noise_pos

            mask_pos = (torch.rand(1, 1, 3, device=self.device) < 0.5).float()
            ref_t_origin_pos = ref_t_origin[:, :, 3:6]
            ref_t_initial_pos = ref_t_origin_pos[:, 0:1, :]
            ref_base_pos = mask_pos * ref_t_origin_pos + (1 - mask_pos) * ref_t_initial_pos

            ref_t_pos = ref_base_pos + total_noise_pos

            # rot noise
            bias_rot = (torch.rand(bs, 1, 2, device=self.device) * 2 - 1) * self.bias_max_rot
            freq_rot = torch.rand(bs, 1, 2, device=self.device) * self.sine_freq_max
            amp_rot = torch.rand(bs, 1, 2, device=self.device) * self.sine_amp_max_rot
            phi_rot = torch.rand(bs, 1, 2, device=self.device) * 2 * np.pi
            
            sine_noise_rot = amp_rot * torch.sin(2 * np.pi * freq_rot * t + phi_rot)  # [bs, horizon, 2]
            
            step_mask_rot = (torch.rand(bs, horizon, 2, device=self.device) < self.step_rate).float()
            step_values_rot = (torch.rand(
                bs, horizon, 2, device=self.device) * 2 - 1) * self.step_max_rot
            step_noise_rot = torch.cumsum(step_mask_rot * step_values_rot, dim=1)
            
            total_noise_rot = bias_rot + sine_noise_rot + step_noise_rot

            mask_rot = (torch.rand(1, 1, 2, device=self.device) < 0.5).float()
            ref_t_origin_rot = ref_t_origin[:, :, :2]
            ref_t_initial_rot = ref_t_origin_rot[:, 0:1, :]
            ref_base_rot = mask_rot * ref_t_origin_rot + (1 - mask_rot) * ref_t_initial_rot

            ref_t_rot = torch.zeros(bs, horizon, 3, device=self.device)
            ref_t_rot[:, :, :2] = ref_base_rot + total_noise_rot

            ref_t = torch.cat([ref_t_rot, ref_t_pos], dim=2).reshape(bs, horizon * 6)
            noise_info = {
                "bias_pos": bias_pos, "freq_pos": freq_pos, "amp_pos": amp_pos, "phi_pos": phi_pos,
                "current_step_offset_pos": step_noise_pos[:, -1:, :], "mask_pos": mask_pos,
                "ref_t_initial_pos": ref_t_initial_pos,
                "bias_rot": bias_rot, "freq_rot": freq_rot, "amp_rot": amp_rot, "phi_rot": phi_rot,
                "current_step_offset_rot": step_noise_rot[:, -1:, :], "mask_rot": mask_rot,
                "ref_t_initial_rot": ref_t_initial_rot,
                "t_next": horizon
            }
        return ref_t, noise_info

    @ torch.no_grad()
    def update_noise(self, ref_origin, noise_info):
        bs = ref_origin.shape[0]
        t = noise_info["t_next"]
        noise_info["t_next"] += 1

        if self.ref_pos_only:
            bias = noise_info["bias"] # [bs, 1, 3]
            
            sine_next = noise_info["amp"] * torch.sin(
                2 * np.pi * noise_info["freq"] * t + noise_info["phi"])
            
            new_step_mask = (torch.rand(bs, 1, 3, device=self.device) < self.step_rate).float()
            new_step_val = (torch.rand(bs, 1, 3, device=self.device) * 2 - 1) * self.step_max
            noise_info["current_step_offset"] += (new_step_mask * new_step_val)
            
            next_frame_noise = bias + sine_next + noise_info["current_step_offset"]
            
            ref_origin = ref_origin.view(bs, 1, 3)
            ref_base = noise_info["mask"] * ref_origin + (
                1 - noise_info["mask"]) * noise_info["ref_t_initial"]
            new_ref_pos = ref_base + next_frame_noise
            new_ref_pos = new_ref_pos.reshape(bs, 3)
            return new_ref_pos
        else:
            bias_pos = noise_info["bias_pos"] # [bs, 1, 3]
            bias_rot = noise_info["bias_rot"] # [bs, 1, 3]

            sine_next_pos = noise_info["amp_pos"] * torch.sin(
                2 * np.pi * noise_info["freq_pos"] * t + noise_info["phi_pos"])
            sine_next_rot = noise_info["amp_rot"] * torch.sin(
                2 * np.pi * noise_info["freq_rot"] * t + noise_info["phi_rot"])

            new_step_mask_pos = (torch.rand(bs, 1, 3, device=self.device) < self.step_rate).float()
            new_step_val_pos = (torch.rand(bs, 1, 3, device=self.device) * 2 - 1) * self.step_max_pos
            noise_info["current_step_offset_pos"] += (new_step_mask_pos * new_step_val_pos)

            new_step_mask_rot = (torch.rand(bs, 1, 2, device=self.device) < self.step_rate).float()
            new_step_val_rot = (torch.rand(bs, 1, 2, device=self.device) * 2 - 1) * self.step_max_rot
            noise_info["current_step_offset_rot"] += (new_step_mask_rot * new_step_val_rot)
            
            next_frame_noise_pos = bias_pos + sine_next_pos + noise_info["current_step_offset_pos"]
            next_frame_noise_rot = bias_rot + sine_next_rot + noise_info["current_step_offset_rot"]

            ref_origin = ref_origin.view(bs, 1, 6)
            ref_base_pos = noise_info["mask_pos"] * ref_origin[
                :, :, 3:6] + (1 - noise_info["mask_pos"]) * noise_info["ref_t_initial_pos"]
            new_ref_pos = ref_base_pos + next_frame_noise_pos
            new_ref_pos = new_ref_pos.reshape(bs, 3)

            ref_base_rot = noise_info["mask_rot"] * ref_origin[
                :, :, :2] + (1 - noise_info["mask_rot"]) * noise_info["ref_t_initial_rot"]
            new_ref_rot = ref_base_rot + next_frame_noise_rot
            new_ref_rot = new_ref_rot.reshape(bs, 2)
            new_ref_rot_z = torch.zeros(bs, 1, device=self.device)
            return torch.cat([new_ref_rot, new_ref_rot_z, new_ref_pos], dim=1)
    
    def update(self, batch, teacher_forcing=False): 
        # update model parameters based on a single batch
        self.optimizer.zero_grad(set_to_none=True)
        self.dynamics.eval()
        for param in self.dynamics.parameters():
            param.requires_grad = False

        if self.hidden_dynamics is not None:
            self.hidden_dynamics.train()
            if self.fine_tune:
                for name, param in self.hidden_dynamics.named_parameters():
                    param.requires_grad = False
            else:
                for param in self.hidden_dynamics.parameters():
                    param.requires_grad = True

        self.policy.train()
        if self.fine_tune:
            for name, param in self.policy.named_parameters():
                if "rnn_net" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True 
        else:
            for param in self.policy.parameters():
                param.requires_grad = True
        
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
            # forward pass
            sampled_pred = []
            sampled_tgt = []
            sampled_origin_obs = []
            sampled_ctl = []
            sampled_data_ctl = []
            record = random.random() < 0.05
            total_loss = [torch.zeros((), device=self.device) for _ in range(len(self.loss_k_list))]
            sampled_history = None
            last_ctl = None
            for step in range(self.bptt_steps):
                if step == 0:
                    ref_t_origin = batch["input_seq"][:, step, :self.ref_t_dim]
                    data_inputs = batch["input_seq"][:, step, :]  # ref_t + obs_t + u_t
                    obs = data_inputs[:, self.ref_t_dim:-self.dynamics.ctl_dim]  # obs_t
                    initial_history = batch["initial_history"][:, :, -self.dynamics.input_dim:]  # obs_t + u_t
                    history_mask = batch["history_mask"]
                    self.dynamics.set_initial_h(initial_history, history_mask)
                    if self.hidden_dynamics is not None:
                        self.hidden_dynamics.set_initial_h(initial_history, history_mask)
                    if not self.obs_ref:
                        delta_ref_rot, delta_ref_pos, ref_t = self.compute_delta(obs, ref_t_origin)
                    else:
                        ref_t, noise_info = self.init_noise(ref_t_origin)
                    if "u" in self.policy.input_type:
                        last_ctl = batch["target_seq"][:, step, :self.policy.ctl_dim]

                    if record:
                        if self.ref_pos_only:
                            sampled_pred.append(obs[0, :6].detach().cpu().numpy())  # ee only
                            sampled_tgt.append(obs[0, 3:6].detach().cpu().numpy())  # ee only
                        else:
                            sampled_pred.append(obs[0, :self.policy.ref_dim].detach().cpu().numpy())  # ee only
                            sampled_tgt.append(obs[0, :self.policy.ref_dim].detach().cpu().numpy())  # ee only
                        sampled_history = torch.cat([
                            batch["initial_history"][0, :, :self.dynamics.obs_dim][history_mask[0]], 
                            obs[0].unsqueeze(0)], dim=0).detach().cpu().numpy()
                else:
                    ref_origin = batch["input_seq"][
                        :, step, self.ref_t_dim-self.policy.ref_dim:self.ref_t_dim]
                    if not self.obs_ref:
                        ref_t = torch.cat([ref_t[:, self.policy.ref_dim:], *self.interp_ref(
                            ref_origin, delta_ref_rot, delta_ref_pos)], dim=-1)
                    else:
                        new_ref = self.update_noise(ref_origin, noise_info)
                        ref_t = torch.cat(
                            [ref_t[:, self.policy.ref_dim:], new_ref], dim=-1)
                        
                    if step % self.truncation_steps == 0:
                        if isinstance(self.dynamics.h, tuple):
                            self.dynamics.h = tuple(t.detach() for t in self.dynamics.h)
                            if self.hidden_dynamics is not None:
                                self.hidden_dynamics.h = tuple(t.detach() for t in self.hidden_dynamics.h)
                        else:
                            self.dynamics.h = self.dynamics.h.detach()
                            if self.hidden_dynamics is not None:
                                self.hidden_dynamics.h = self.hidden_dynamics.h.detach()
                        obs = obs.detach()
                        last_ctl = last_ctl.detach()

                data_ctl = batch["input_seq"][:, step, -self.dynamics.ctl_dim:]
                if "ro" in self.policy.input_type:
                    inputs = torch.cat([ref_t, obs], dim=-1)  # ref_t + obs_t
                else:
                    inputs = ref_t
                if "u" in self.policy.input_type:
                    inputs = torch.cat([inputs, last_ctl], dim=-1)  # ref_t + obs_t + u_t_minus_1
                if step % self.policy.output_step == 0:
                    if self.policy.net_type == "mlp":
                        multi_step_ctl = self.policy.recur_forward(inputs, initial_history)
                    else:
                        if self.hidden_dynamics is not None:
                            if isinstance(self.hidden_dynamics.h, tuple):
                                multi_step_ctl = self.policy.recur_forward(inputs, self.hidden_dynamics.h[0])
                            else:
                                multi_step_ctl = self.policy.recur_forward(inputs, self.hidden_dynamics.h)
                        else:
                            if isinstance(self.dynamics.h, tuple):
                                multi_step_ctl = self.policy.recur_forward(inputs, self.dynamics.h[0])
                            else:
                                multi_step_ctl = self.policy.recur_forward(inputs, self.dynamics.h)
                ctl = multi_step_ctl[:, (step % self.policy.output_step) * self.policy.ctl_dim:
                                    (step % self.policy.output_step + 1) * self.policy.ctl_dim]  # u_t, h of obs_net
                dynamics_inputs = torch.cat([obs, ctl], dim=-1)  # obs_t + u_t
                if self.policy.net_type == "mlp":
                    initial_history = torch.cat(
                        [initial_history[:, 1:, :], dynamics_inputs.unsqueeze(1)], dim=1)
                obs = self.dynamics.recur_forward(dynamics_inputs, None, None)  # obs_t_plus_1_pred
                if self.hidden_dynamics is not None:
                    self.hidden_dynamics.recur_forward(dynamics_inputs, None, None)  # update hidden dynamics h

                if record:
                    if self.ref_pos_only:
                        sampled_pred.append(obs[0, :6].detach().cpu().numpy())
                    else:
                        sampled_pred.append(obs[0, :self.policy.ref_dim].detach().cpu().numpy())
                    sampled_tgt.append(inputs[0, :self.policy.ref_dim].detach().cpu().numpy())
                    sampled_origin_obs.append(batch["input_seq"][
                        0, step, self.ref_t_dim:-self.dynamics.ctl_dim].detach().cpu().numpy())
                    sampled_ctl.append(ctl[0].detach().cpu().numpy())
                    sampled_data_ctl.append(data_ctl[0].detach().cpu().numpy())
                    
                step_loss = self.compute_loss(ref_t, obs, ctl, last_ctl, data_ctl)
                step_loss = [s * (self.discount ** step) for s in step_loss]
                total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
                last_ctl = ctl

            total_loss = [t_l / self.bptt_steps for t_l in total_loss]
            loss = sum(total_loss)
        
        # update model parameters
        self.scaler.scale(loss).backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        iter_info = {"train/iteration": self.iteration,
                     "train/loss": loss.detach().cpu().item()}
        for i, l in enumerate(total_loss):
            iter_info[f"train/loss_{self.loss_k_name_list[i]}"] = l.detach().cpu().item()
            if self.loss_k_name_list[i] in ["contraction_rot", "contraction_pos"]:
                iter_info[f"train/me_{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / self.loss_k_list[i] * self.scale)
            elif self.loss_k_name_list[i] == "kl_divergence":
                iter_info[f"train/{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / (self.loss_k_list[i] + 1e-5)) * self.scale
            elif l > 0:
                iter_info[f"train/rmse_{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / self.loss_k_list[i] * self.scale) ** 0.5

        self.iteration += 1

        return (iter_info, sampled_pred, sampled_tgt, sampled_origin_obs, 
                sampled_ctl, sampled_data_ctl, sampled_history)
    
    @torch.inference_mode()
    def evaluate(self, batch, type="eval"):
        self.dynamics.eval()
        if self.hidden_dynamics is not None:
            self.hidden_dynamics.eval()
        self.policy.eval()

        # forward pass
        sampled_pred = []
        sampled_tgt = []
        sampled_origin_obs = []
        sampled_ctl = []
        sampled_data_ctl = []
        record = random.random() < 0.1
        total_loss = [torch.zeros((), device=self.device) for _ in range(len(self.loss_k_list))]
        sampled_history = None
        last_ctl = None
        for step in range(self.bptt_steps):
            if step == 0:
                ref_t_origin = batch["input_seq"][:, step, :self.ref_t_dim]
                data_inputs = batch["input_seq"][:, step, :]  # ref_t + obs_t + u_t
                obs = data_inputs[:, self.ref_t_dim:-self.dynamics.ctl_dim]  # obs_t
                initial_history = batch["initial_history"][:, :, -self.dynamics.input_dim:]  # obs_t + u_t
                history_mask = batch["history_mask"]
                self.dynamics.set_initial_h(initial_history, history_mask)
                if self.hidden_dynamics is not None:
                    self.hidden_dynamics.set_initial_h(initial_history, history_mask)
                if not self.obs_ref:
                    delta_ref_rot, delta_ref_pos, ref_t = self.compute_delta(obs, ref_t_origin)
                else:
                    ref_t, noise_info = self.init_noise(ref_t_origin)
                if "u" in self.policy.input_type:
                    last_ctl = batch["target_seq"][:, step, :self.policy.ctl_dim]

                if record:
                    if self.ref_pos_only:
                        sampled_pred.append(obs[0, :6].detach().cpu().numpy())
                        sampled_tgt.append(obs[0, 3:6].detach().cpu().numpy())
                    else:
                        sampled_pred.append(obs[0, :self.policy.ref_dim].detach().cpu().numpy())
                        sampled_tgt.append(obs[0, :self.policy.ref_dim].detach().cpu().numpy())
                    sampled_history = torch.cat([
                        batch["initial_history"][0, :, :self.dynamics.obs_dim][history_mask[0]], 
                        obs[0].unsqueeze(0)], dim=0).detach().cpu().numpy()
            else:
                ref_origin = batch["input_seq"][
                    :, step, self.ref_t_dim-self.policy.ref_dim:self.ref_t_dim]
                if not self.obs_ref:
                    ref_t = torch.cat([ref_t[:, self.policy.ref_dim:], *self.interp_ref(
                        ref_origin, delta_ref_rot, delta_ref_pos)], dim=-1)
                else:
                    new_ref = self.update_noise(ref_origin, noise_info)
                    ref_t = torch.cat(
                        [ref_t[:, self.policy.ref_dim:], new_ref], dim=-1)

            data_ctl = batch["input_seq"][:, step, -self.dynamics.ctl_dim:]
            if "ro" in self.policy.input_type:
                inputs = torch.cat([ref_t, obs], dim=-1)  # ref_t + obs_t
            else:
                inputs = ref_t
            if "u" in self.policy.input_type:
                inputs = torch.cat([inputs, last_ctl], dim=-1)  # ref_t + obs_t + u_t_minus_1
            if step % self.policy.output_step == 0:
                if self.policy.net_type == "mlp":
                    multi_step_ctl = self.policy.recur_forward(inputs, initial_history)
                else:
                    if self.hidden_dynamics is not None:
                        if isinstance(self.hidden_dynamics.h, tuple):
                            multi_step_ctl = self.policy.recur_forward(inputs, self.hidden_dynamics.h[0])
                        else:
                            multi_step_ctl = self.policy.recur_forward(inputs, self.hidden_dynamics.h)
                    else:
                        if isinstance(self.dynamics.h, tuple):
                            multi_step_ctl = self.policy.recur_forward(inputs, self.dynamics.h[0])
                        else:
                            multi_step_ctl = self.policy.recur_forward(inputs, self.dynamics.h)
            ctl = multi_step_ctl[:, (step % self.policy.output_step) * self.policy.ctl_dim:
                                  (step % self.policy.output_step + 1) * self.policy.ctl_dim]  # u_t, h of obs_net
            
            dynamics_inputs = torch.cat([obs, ctl], dim=-1)  # obs_t + u_t
            if self.policy.net_type == "mlp":
                initial_history = torch.cat(
                    [initial_history[:, 1:, :], dynamics_inputs.unsqueeze(1)], dim=1)
            obs = self.dynamics.recur_forward(dynamics_inputs, None, None)  # obs_t_plus_1_pred
            if self.hidden_dynamics is not None:
                self.hidden_dynamics.recur_forward(dynamics_inputs, None, None)  # update hidden dynamics h

            if record:
                if self.ref_pos_only:
                    sampled_pred.append(obs[0, :6].detach().cpu().numpy())
                else:
                    sampled_pred.append(obs[0, :self.policy.ref_dim].detach().cpu().numpy())
                sampled_tgt.append(inputs[0, :self.policy.ref_dim].detach().cpu().numpy())
                sampled_origin_obs.append(batch["input_seq"][
                    0, step, self.ref_t_dim:-self.dynamics.ctl_dim].detach().cpu().numpy())
                sampled_ctl.append(ctl[0].detach().cpu().numpy())
                sampled_data_ctl.append(data_ctl[0].detach().cpu().numpy())
            step_loss = self.compute_loss(ref_t, obs, ctl, last_ctl, data_ctl)
            step_loss = [s * (self.discount ** step) for s in step_loss]
            total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
            last_ctl = ctl

        total_loss = [t_l / self.bptt_steps for t_l in total_loss]
        loss = sum(total_loss)

        batch_info = {f"{type}/loss": loss.detach().cpu().item()}
        for i, l in enumerate(total_loss):
            batch_info[f"{type}/loss_{self.loss_k_name_list[i]}"] = l.detach().cpu().item()
            if self.loss_k_name_list[i] in ["contraction_rot", "contraction_pos"]:
                batch_info[f"{type}/me_{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / self.loss_k_list[i] * self.scale)
            elif self.loss_k_name_list[i] == "kl_divergence":
                batch_info[f"{type}/{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / (self.loss_k_list[i] + 1e-5)) * self.scale
            elif l > 0:
                batch_info[f"{type}/rmse_{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / self.loss_k_list[i] * self.scale) ** 0.5

        return (batch_info, sampled_pred, sampled_tgt, sampled_origin_obs,
                sampled_ctl, sampled_data_ctl, sampled_history)
    