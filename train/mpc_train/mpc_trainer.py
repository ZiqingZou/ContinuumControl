import os
import time
import h5py
import numpy as np
import random
import pandas as pd
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

import train.tools.transforms3d as t3d
from config.parser import Config
from models.dynamics import Dynamics
from train.data_loader import get_dataloader
from train.tools.pred_plot import pred_plot


class MpcPolicyTrainer:
    def __init__(self, cfg_model: dict, cfg_training: dict):
        self.cfg_model = cfg_model
        self.cfg_training = cfg_training
        self.obs_type = cfg_model.dynamics.get("obs_type")
        self.device = torch.device(self.cfg_training.mpc.get("device"))
        self.obs_ref = self.cfg_training.mpc.get("obs_ref")
        self.min_regressive_steps = self.cfg_training.mpc.get("min_regressive_steps")

        # Getdata loader and update loss weights
        if self.obs_ref:
            self.sine_freq_max = self.cfg_training.mpc.get("sine_freq_max")
            self.step_rate = self.cfg_training.mpc.get("step_rate")
            self.bias_max_pos = self.cfg_training.mpc.get("bias_max_pos")
            self.sine_amp_max_pos = self.cfg_training.mpc.get("sine_amp_max_pos")
            self.step_max_pos = self.cfg_training.mpc.get("step_max_pos")
            self.bias_max_rot = self.cfg_training.mpc.get("bias_max_rot")
            self.sine_amp_max_rot = self.cfg_training.mpc.get("sine_amp_max_rot")
            self.step_max_rot = self.cfg_training.mpc.get("step_max_rot")
        else:
            self.closer_scale = self.cfg_training.mpc.get("closer_scale")

        self.bptt_steps = cfg_training.mpc.get("bptt_steps")  # number of steps to unroll
        self.truncation_steps = self.cfg_training.mpc.get("truncation_steps")
        self.max_grad_norm = cfg_training.mpc.get("max_grad_norm")
        self.optimize_iters = cfg_training.mpc.get("optimize_iters")  # number of iterations per batch
        self.discount = cfg_training.mpc.get("discount")

        # Loss computation
        self.loss_k_smooth = cfg_training.mpc.loss_weight.get("k_smooth")
        self.loss_k_rotvec = cfg_training.mpc.loss_weight.get("k_rotvec")
        self.loss_k_pos = cfg_training.mpc.loss_weight.get("k_pos")
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

        self.loss_fn = nn.MSELoss(reduction='none')  #  nn.SmoothL1Loss(beta=1.0)
        if self.discount == 1.0:
            self.scale = 1.0
        else:
            scale = (1 - self.discount ** self.bptt_steps) / (1 - self.discount)
            self.scale = self.bptt_steps / scale
        
        # Getdata loader and update loss weights
        script_path = Path(__file__).resolve()
        train_root = script_path.parent.parent
        self.save_folder = train_root / "checkpoints" / cfg_training.mpc.get("name")
        os.makedirs(self.save_folder, exist_ok=True)

        self.batch_size = self.cfg_training.mpc.get("batch_size")
        self.freq_downsample = self.cfg_training.mpc.get("freq_downsample")
        self.loader = self.get_loader()
        self.dynamics = self.get_model()
        self.ref_pos_only = False
        self.ref_dim = 6
        self.ref_horizon = 50
        self.ref_t_dim = self.ref_dim * self.ref_horizon
        self.u_clip = cfg_training.mpc.get("u_clip")

        # Training setup
        self.lr = cfg_training.mpc.get("learning_rate")
        # self.u = nn.Parameter(torch.zeros(
        #     self.batch_size, 
        #     self.bptt_steps // self.freq_downsample + 1,
        #     self.dynamics.ctl_dim, 
        #     device=self.device))
        # self.optimizer = torch.optim.Adam([self.u], lr=self.lr)
        self.initial_noise_info = None
        self.initial_ref_t = None
        self.delta_ref_rot = None
        self.delta_ref_pos = None
            
        self.iteration = 0

    def set_u(self, batch_size, bptt_step):
        self.u = nn.Parameter(torch.zeros(
            batch_size, 
            bptt_step // self.freq_downsample + 1,
            self.dynamics.ctl_dim, 
            device=self.device))
        self.optimizer = torch.optim.Adam([self.u], lr=self.lr)
    
    def get_loader(self):
        data_source = self.cfg_training.mpc.get("data_source")
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent

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

        test_list = self.cfg_training.mpc.get("test_list")
        test_data_folder = []
        if test_list is not None:
            for testset_name in test_list:
                test_data_folder.append(
                    project_root / "data" / data_source / f"processed_logs_{testset_name}")
        loaders = []
        for data_folder in test_data_folder:
            loader = get_dataloader(
                paths=data_folder,
                input_key_list=list(input_key_list),
                output_key_list=["u_t_minus_1"],
                history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
                bptt_steps=self.bptt_steps,
                min_regressive_steps=self.min_regressive_steps,
                same_len_history=self.cfg_model.dynamics.get("net_type") == "mlp",
                batch_size=self.batch_size,
                num_workers=self.cfg_training.mpc.get("num_workers"),
                history_max_len=self.cfg_model.dynamics.get(
                    "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
                shuffle=False,
            )
            print(f"Set length: {len(loader.dataset)}")
            loaders.append(loader)
        return loaders
    
    def get_model(self):
        dynamics_name = self.cfg_training.mpc.get("dynamics_name")
        dynamics_pth = self.save_folder.parent / dynamics_name / \
            f"dynamics_{self.cfg_training.mpc.get('dynamics_load_type')}.pth"
        dynamics = Dynamics(self.cfg_model, self.device, dynamics_pth)
        return dynamics
    
    def compute_loss(self, ref, obs, ctl, last_ctl):
        loss = []
        if self.loss_k_smooth >= 0.0001:
            if last_ctl is not None:
                loss.append(self.loss_k_smooth * self.loss_fn(
                    ctl, last_ctl).mean(dim=1))  # no detach to ensure both a2b and b2a
            else:
                loss.append(torch.zeros((obs.size(0),), device=self.device))

        loss.append(self.loss_k_rotvec * self.loss_fn(
            obs[:, :3], ref[:, :3].detach()).mean(dim=1))  # rotvec only compute mean loss for stability
        loss.append(self.loss_k_pos * self.loss_fn(
            obs[:, 3:self.ref_dim], ref[:, 3:self.ref_dim].detach()).mean(dim=1))

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

            pos_delta = obs[:, 3:self.ref_dim] - ref_t_origin[:, 3:self.ref_dim]
        else:
            delta_ref_rot = None
            pos_delta = obs[:, :self.ref_dim] - ref_t_origin[:, :self.ref_dim]

        delta_ref_pos = pos_delta * k_dis
        
        new_ref_t = self.interp_ref_t(ref_t_origin, delta_ref_rot, delta_ref_pos)

        return delta_ref_rot, delta_ref_pos, new_ref_t
    
    @ torch.no_grad()
    def interp_ref_t(self, ref_t, delta_ref_rot, delta_ref_pos):
        """ interpolate ref towards obs by delta_ref """
        if delta_ref_rot is not None:
            def split_ref(ref_t):
                ret_t_view = ref_t.view(
                    ref_t.shape[0], self.ref_horizon, self.ref_dim)
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
                ref_t.shape[0], self.ref_horizon, self.ref_dim) + delta_ref_pos.unsqueeze(1)
            new_ref = new_ref_pos.view(ref_t.shape[0], self.ref_t_dim)

        return new_ref

    @ torch.no_grad()
    def interp_ref(self, ref, delta_ref_rot, delta_ref_pos):
        """ interpolate ref towards obs by delta_ref """
        if delta_ref_rot is not None:
            q_ref = t3d.axis_angle_to_quaternion(ref[:, :3])
            q_new_ref = t3d.quaternion_multiply(delta_ref_rot, q_ref)
            new_ref_rot = t3d.quaternion_to_axis_angle(q_new_ref)
            new_ref_pos = ref[:, 3:self.ref_dim] + delta_ref_pos
            return new_ref_rot, new_ref_pos
        else:
            new_ref_pos = ref[:, 3:self.ref_dim] + delta_ref_pos
            return new_ref_pos, 

    @ torch.no_grad()
    def init_noise(self, ref_t_origin):
        bs = ref_t_origin.size(0)
        horizon = self.ref_horizon
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
    
    def update(self, batch, max_step, initial_h=None, record=False):
        # update model parameters based on a single batch
        self.optimizer.zero_grad(set_to_none=True)
        self.dynamics.eval()
        for param in self.dynamics.parameters():
            param.requires_grad = False
        
        # forward pass
        all_pred = []
        all_tgt = []
        all_origin_obs = []
        all_ctl = []
        all_data_ctl = []
        total_loss = [torch.zeros((), device=self.device) for _ in range(len(self.loss_k_list))]
        last_ctl = None
        for step in range(max_step):
            if step == 0:
                data_inputs = batch["input_seq"][:, step, :]  # ref_t + obs_t + u_t
                obs = data_inputs[:, self.ref_t_dim:-self.dynamics.ctl_dim]  # obs_t
                last_ctl = batch["target_seq"][:, step, :self.dynamics.ctl_dim]
                if initial_h is None:
                    initial_history = batch["initial_history"][:, :, -self.dynamics.input_dim:]  # obs_t + u_t
                    self.history_mask = batch["history_mask"]
                    self.dynamics.set_initial_h(initial_history, self.history_mask)
                    initial_h = self.dynamics.h
                    ref_t_origin = batch["input_seq"][:, step, :self.ref_t_dim]
                    if not self.obs_ref:
                        self.delta_ref_rot, self.delta_ref_pos, self.initial_ref_t = self.compute_delta(obs, ref_t_origin)
                    else:
                        self.initial_ref_t, self.initial_noise_info = self.init_noise(ref_t_origin)
                else:
                    self.dynamics.h = initial_h

                if self.obs_ref:
                    self.noise_info = deepcopy(self.initial_noise_info)
                self.ref_t = self.initial_ref_t.clone()

                if record:
                    all_pred.append(obs[:, :self.ref_dim])  # ee only
                    all_tgt.append(obs[:, :self.ref_dim])  # ee only
            else:
                ref_origin = batch["input_seq"][
                    :, step, self.ref_t_dim-self.ref_dim:self.ref_t_dim]
                if not self.obs_ref:
                    self.ref_t = torch.cat([self.ref_t[:, self.ref_dim:], *self.interp_ref(
                        ref_origin, self.delta_ref_rot, self.delta_ref_pos)], dim=-1)
                else:
                    new_ref = self.update_noise(ref_origin, self.noise_info)
                    self.ref_t = torch.cat(
                        [self.ref_t[:, self.ref_dim:], new_ref], dim=-1)
                if step % self.truncation_steps == 0:
                    if isinstance(self.dynamics.h, tuple):
                        self.dynamics.h = tuple(t.detach() for t in self.dynamics.h)
                    else:
                        self.dynamics.h = self.dynamics.h.detach()
                    obs = obs.detach()
                    last_ctl = last_ctl.detach()
                    
            ref = self.ref_t[:, :self.ref_dim]  # one step ref
            data_ctl = batch["input_seq"][:, step, -self.dynamics.ctl_dim:]
            ctl_a = self.u[:obs.size(0), step // self.freq_downsample, :]
            ctl_b = self.u[:obs.size(0), step // self.freq_downsample + 1, :]
            alpha = (step % self.freq_downsample) / self.freq_downsample
            ctl = (1 - alpha) * ctl_a + alpha * ctl_b
            
            # zeros constraint
            ctl = ctl.view(-1, 3, 3)
            mean = ctl.mean(dim=2, keepdim=True) 
            ctl = ctl - mean 
            ctl = ctl.view(-1, 9)
            # norm constraint
            max_val = ctl.abs().max(dim=1, keepdim=True).values
            scale = torch.clamp(self.u_clip / (max_val + 1e-5), max=1.0)
            ctl = ctl * scale 

            dynamics_inputs = torch.cat([obs, ctl], dim=-1)  # obs_t + u_t
            obs = self.dynamics.recur_forward(dynamics_inputs, None, None)  # obs_t_plus_1_pred
            if record:
                all_pred.append(obs[:, :self.ref_dim])
                all_tgt.append(ref[:, :self.ref_dim])
                all_origin_obs.append(batch["input_seq"][
                    :, step, self.ref_t_dim:-self.dynamics.ctl_dim])
                all_ctl.append(ctl[:, :self.dynamics.ctl_dim])
                all_data_ctl.append(data_ctl[:, :self.dynamics.ctl_dim])

            step_loss = self.compute_loss(ref, obs, ctl, last_ctl)  # list of (B,) loss for each loss type
            step_loss = [(s * batched_data["mask"][:, step]) * (self.discount ** step) for s in step_loss]
            total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
            last_ctl = ctl

        total_loss = [t_l / torch.sum(batched_data["mask"], dim=1) for t_l in total_loss]
        loss = sum(total_loss).mean()
        
        # update model parameters
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_([self.u], self.max_grad_norm)
        self.optimizer.step()

        iter_info = {"train/iteration": self.iteration % self.optimize_iters,
                     "train/loss": loss.detach().cpu().item()}
        for i, l in enumerate(total_loss):
            iter_info[f"train/loss_{self.loss_k_name_list[i]}"] = l.mean().detach().cpu().item()
            if l.mean().detach().cpu().item() > 0:
                iter_info[f"train/rmse_{self.loss_k_name_list[i]}"] = (
                    l.mean().detach().cpu().item() / self.loss_k_list[i] * self.scale) ** 0.5

        self.iteration += 1

        return iter_info, all_pred, all_tgt, all_origin_obs, all_ctl, all_data_ctl, initial_h, batched_data["mask"]
    
    @torch.inference_mode()
    def evaluate(self, batch, b, initial_h):
        if self.obs_ref:
            assert self.initial_noise_info is not None
        else:
            assert self.delta_ref_pos is not None and self.delta_ref_rot is not None

        self.dynamics.eval()
        for param in self.dynamics.parameters():
            param.requires_grad = False

        # forward pass
        sampled_pred = []
        sampled_tgt = []
        sampled_origin_obs = []
        sampled_ctl = []
        sampled_data_ctl = []
        record = True
        total_loss = [torch.zeros((), device=self.device) for _ in range(len(self.loss_k_list))]
        sampled_history = None
        last_ctl = None
        for step in range(self.bptt_steps):
            if step == 0:
                obs = batch["input_seq"][b:b+1, step, self.ref_t_dim:-self.dynamics.ctl_dim]  # obs_t
                last_ctl = batch["target_seq"][b:b+1, step, :self.dynamics.ctl_dim]
                self.dynamics.h = initial_h[b:b+1, ...]

                if self.obs_ref:
                    self.noise_info = deepcopy(self.initial_noise_info)
                self.ref_t = self.initial_ref_t.clone()
                
                if record:
                    sampled_pred.append(obs[0, :self.ref_dim].detach().cpu().numpy())  # ee only
                    sampled_tgt.append(obs[0, :self.ref_dim].detach().cpu().numpy())  # ee only
                    sampled_history = torch.cat([
                        batch["initial_history"][0, :, :self.dynamics.obs_dim][self.history_mask[b]], 
                        obs[0].unsqueeze(0)], dim=0).detach().cpu().numpy()
            else:
                ref_origin = batch["input_seq"][
                    :, step, self.ref_t_dim-self.ref_dim:self.ref_t_dim]
                if not self.obs_ref:
                    self.ref_t = torch.cat([self.ref_t[:, self.ref_dim:], *self.interp_ref(
                        ref_origin, self.delta_ref_rot, self.delta_ref_pos)], dim=-1)
                else:
                    new_ref = self.update_noise(ref_origin, self.noise_info)
                    self.ref_t = torch.cat(
                        [self.ref_t[:, self.ref_dim:], new_ref], dim=-1)

            ref = self.ref_t[b:b+1, :self.ref_dim]  # one step ref
            data_ctl = batch["input_seq"][b:b+1, step, -self.dynamics.ctl_dim:]
            ctl_a = self.u[b:b+1, step // self.freq_downsample, :]
            ctl_b = self.u[b:b+1, step // self.freq_downsample + 1, :]
            alpha = (step % self.freq_downsample) / self.freq_downsample
            ctl = (1 - alpha) * ctl_a + alpha * ctl_b
            # ctl = self.u[b:b+1, step, :]
            
            # zeros constraint
            ctl = ctl.view(-1, 3, 3)
            mean = ctl.mean(dim=2, keepdim=True) 
            ctl = ctl - mean 
            ctl = ctl.view(-1, 9)
            # norm constraint
            max_val = ctl.abs().max(dim=1, keepdim=True).values
            scale = torch.clamp(self.u_clip / (max_val + 1e-5), max=1.0)
            ctl = ctl * scale 

            dynamics_inputs = torch.cat([obs, ctl], dim=-1)  # obs_t + u_t
            obs = self.dynamics.recur_forward(dynamics_inputs, None, None)  # obs_t_plus_1_pred

            if record:
                sampled_pred.append(obs[0, :self.ref_dim].detach().cpu().numpy())
                sampled_tgt.append(ref[0, :self.ref_dim].detach().cpu().numpy())
                sampled_origin_obs.append(batch["input_seq"][
                    0, step, self.ref_t_dim:-self.dynamics.ctl_dim].detach().cpu().numpy())
                sampled_ctl.append(ctl[0].detach().cpu().numpy())
                sampled_data_ctl.append(data_ctl[0].detach().cpu().numpy())

            step_loss = self.compute_loss(ref, obs, ctl, last_ctl)
            step_loss = [s * (self.discount ** step) for s in step_loss]
            total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
            last_ctl = ctl

        total_loss = [t_l / self.bptt_steps for t_l in total_loss]
        loss = sum(total_loss)

        batch_info = {"eval/loss": loss.detach().cpu().item()}
        for i, l in enumerate(total_loss):
            batch_info[f"eval/loss_{self.loss_k_name_list[i]}"] = l.detach().cpu().item()
            if l > 0:
                batch_info[f"eval/rmse_{self.loss_k_name_list[i]}"] = (
                    l.detach().cpu().item() / self.loss_k_list[i] * self.scale) ** 0.5

        return batch_info, sampled_pred, sampled_tgt, sampled_origin_obs, sampled_ctl, sampled_data_ctl, sampled_history
    

def collect_batch(loader_list, device):
    trajectories = []
    max_len = 0
    for loader in loader_list:
        total_len = len(loader.dataset)
        bptt_step = total_len + loader.dataset.min_regressive_steps
        max_len = max(max_len, bptt_step)
        for data in loader:
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            bptt_mask = torch.zeros((1, data["input_seq"].size(1)), device=device)
            bptt_mask[:, :bptt_step] = 1.0
            data["bptt_mask"] = bptt_mask
            trajectories.append(data)
            break
    
    # input_seq  (1, steps, dim)
    input_dim = trajectories[0]["input_seq"].size(-1)
    target_dim = trajectories[0]["target_seq"].size(-1)

    traj_num = len(trajectories)
    batched_input = torch.zeros((traj_num, max_len, input_dim), device=device)
    batched_target = torch.zeros((traj_num, max_len, target_dim), device=device)
    batched_mask = torch.zeros((traj_num, max_len), device=device)
    
    for i, t in enumerate(trajectories):
        batched_input[i, :, :] = t["input_seq"][0, :max_len, :]
        batched_target[i, :, :] = t["target_seq"][0, :max_len, :]
        batched_mask[i, :] = t["bptt_mask"][0, :max_len]
        
    return {
        "input_seq": batched_input,
        "target_seq": batched_target,
        "mask": batched_mask,
        "initial_history": torch.cat([t["initial_history"] for t in trajectories], dim=0),
        "history_mask": torch.cat([t["history_mask"] for t in trajectories], dim=0)
    }


def log_data(obs_predict, ref, ctl, save_folder, traj_id):  # numpy arrays
    # Geodesic Loss calculation for rotation error
    # For two rotation vectors p and t, the geodesic distance is the magnitude 
    # of the relative rotation vector (p_mat.T @ t_mat)
    rot_error = []
    for p, t in zip(obs_predict, ref):
        # Using transforms3d (t3d) logic to compute the angular difference
        # Magnitude of axis-angle difference is an approximation of geodesic distance
        # For more precision, convert to matrices and find the angle of R_rel
        p_rotvec = p[:3]
        t_rotvec = t[:3]
        
        # Approximate geodesic error via the norm of the difference in rotvec space
        # Or use: angle = norm(so3_log(R_p.T @ R_t))
        error_vec = p_rotvec - t_rotvec
        rot_error.append(np.linalg.norm(error_vec))

    # Position error (Euclidean distance)
    pos_error = [np.linalg.norm(p[3:6] - t[3:6]) for p, t in zip(obs_predict, ref)]

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
        # Map u_1 through u_9 efficiently
        **{f'u_{i+1}': [u[i] for u in ctl] for i in range(9)}
    }

    df = pd.DataFrame(save_data)
    csv_path = save_folder / f"mpc_results_traj_{traj_id}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Final trajectory data saved to {csv_path}")


def train_mpc(trainer, batched_data):
    print(f"Start training MPC targets !")

    batch_size = batched_data["input_seq"].size(0)
    max_len = batched_data["input_seq"].size(1)

    initial_h = None
    # trainer.u.data.fill_(0.0)
    trainer.set_u(batch_size, max_len)
    # Training iterations
    start_time = time.time()
    for iter in range(trainer.optimize_iters):
        (iter_info, all_pred, all_tgt, all_origin_obs, all_ctl, all_data_ctl, initial_h,
         mask) = trainer.update(batched_data, max_len, initial_h, iter==trainer.optimize_iters-1)  # iter % 10 == 0)  # True)  # 
        iter_info["train/iter"] = iter
        for k, v in iter_info.items():
            print(f"{k}: {v}")
        if all_pred and all_tgt:
            for traj_id in range(all_pred[0].shape[0]):
                pred = torch.stack([pred[traj_id] for pred in all_pred[1:]], dim=0)[mask[traj_id].bool()]
                pred = torch.cat([all_pred[0][traj_id].unsqueeze(0), pred], dim=0).detach().cpu().numpy()
                tgt = torch.stack([tgt[traj_id] for tgt in all_tgt[1:]], dim=0)[mask[traj_id].bool()]
                tgt = torch.cat([all_tgt[0][traj_id].unsqueeze(0), tgt], dim=0).detach().cpu().numpy()
                origin_obs = torch.stack([obs[traj_id] for obs in all_origin_obs])[mask[traj_id].bool()].detach().cpu().numpy()
                ctl = torch.stack([u[traj_id] for u in all_ctl])[mask[traj_id].bool()].detach().cpu().numpy()
                data_ctl = torch.stack([d[traj_id] for d in all_data_ctl])[mask[traj_id].bool()].detach().cpu().numpy()
                os.makedirs(trainer.save_folder / f"trajectory_{traj_id + 1}", exist_ok=True)
                pred_plot(pred, tgt, trainer.save_folder / f"trajectory_{traj_id + 1}",
                          trainer.obs_type, origin_obs, ctl, data_ctl)
    print(f"Iterations time: {time.time() - start_time} s.")
    print(f"Training MPC targets completed.")
    
    return all_pred, all_tgt, all_ctl, mask


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent.parent / "config"
    config_model = Config.load(config_path / "models_config.yaml")
    config_training = Config.load(config_path / "training_config.yaml")

    trainer = MpcPolicyTrainer(config_model, config_training)
    batched_data = collect_batch(trainer.loader, trainer.device)

    # Training loop
    all_pred, all_tgt, all_ctl, mask = train_mpc(trainer, batched_data)
    for traj_id in range(all_pred[0].shape[0]):
        pred = torch.stack([p[traj_id] for p in all_pred[1:]], dim=0)[mask[traj_id].bool()].detach().cpu().numpy()
        tgt = torch.stack([t[traj_id] for t in all_tgt[1:]], dim=0)[mask[traj_id].bool()].detach().cpu().numpy()
        ctl = torch.stack([u[traj_id] for u in all_ctl], dim=0)[mask[traj_id].bool()].detach().cpu().numpy()
        log_data(pred, tgt, ctl, trainer.save_folder, traj_id + 1)