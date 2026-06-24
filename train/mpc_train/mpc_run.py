import os
import time
import h5py
import numpy as np
import random
import array
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

        self.bptt_steps = cfg_training.mpc.get("bptt_steps")  # number of steps to unroll
        self.max_grad_norm = cfg_training.mpc.get("max_grad_norm")
        self.optimize_iters = cfg_training.mpc.get("optimize_iters")  # number of iterations per batch
        self.discount = cfg_training.mpc.get("discount")

        # Loss computation
        self.loss_k_smooth = cfg_training.mpc.loss_weight.get("k_smooth")
        self.loss_k_rotvec = cfg_training.mpc.loss_weight.get("k_rotvec")
        self.loss_k_pos = cfg_training.mpc.loss_weight.get("k_pos")
        self.loss_k_list = []
        self.loss_name_list = []
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

        self.loss_fn = nn.MSELoss()  #  nn.SmoothL1Loss(beta=1.0)
        if self.discount == 1.0:
            self.scale = 1.0
        else:
            scale = (1 - self.discount ** self.bptt_steps) / (1 - self.discount)
            self.scale = self.bptt_steps / scale
        
        # Getdata loader and update loss weights
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        self.data_folder = project_root / "data" / cfg_training.mpc.get("data_folder")
        train_root = script_path.parent.parent
        self.save_folder = train_root / "checkpoints" / cfg_training.mpc.get("name")
        os.makedirs(self.save_folder, exist_ok=True)

        self.dynamics = self.get_model()
        self.loader = self.get_loader()
        self.u = nn.Parameter(torch.zeros(
            1, self.bptt_steps, self.dynamics.ctl_dim, device=self.device))
        self.ref_dim = 6
        self.ref_t_dim = 300
        self.u_clip = cfg_training.mpc.get("u_clip")

        # Training setup
        self.optimizer = torch.optim.Adam([self.u], lr=cfg_training.mpc.get("learning_rate"))
            
        self.iteration = 0

    def get_loader(self):
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

        loader = get_dataloader(
            paths=self.data_folder,
            input_key_list=list(input_key_list),
            output_key_list=["u_t_minus_1"],
            history_window=self.cfg_model.dynamics.get("net_params.mlp.history_window"),
            same_len_history=self.cfg_model.dynamics.get("net_type") == "mlp",
            batch_size=1,
            num_workers=1,
            history_max_len=self.cfg_model.dynamics.get(
                "net_params." +  self.cfg_model.dynamics.get("net_type") + ".initial_h_len"),
            shuffle=False,
        )
        print(f"Set length: {len(loader.dataset)}")
        return loader
    
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
                    ctl, last_ctl))  # no detach to ensure both a2b and b2a
            else:
                loss.append(torch.zeros((), device=self.device))

        if self.loss_k_rotvec >= 0.0001:
            loss.append(self.loss_k_rotvec * self.loss_fn(
            obs[:, :3], ref[:, :3].detach()))
        if self.loss_k_pos >= 0.0001:
            loss.append(self.loss_k_pos * self.loss_fn(
                obs[:, 3:self.ref_dim], ref[:, 3:self.ref_dim].detach()))

        return loss
    
    def update(self, fpath, initial_obs):
        # update model parameters based on a single batch
        self.optimizer.zero_grad(set_to_none=True)
        self.dynamics.eval()
        for param in self.dynamics.parameters():
            param.requires_grad = False

        h = torch.zeros(1, self.dynamics.rnn_depth, self.dynamics.hidden_dim).to(self.device)

        sampled_pred = []
        sampled_tgt = []
        sampled_origin_obs = []
        sampled_ctl = []
        sampled_data_ctl = []

        prediction = initial_obs
        last_ctl = None
        total_loss = [torch.zeros((), device=self.device) for _ in range(len(self.loss_k_list))]
        sampled_pred.append(initial_obs[0, :self.ref_dim].detach().cpu().numpy())  # ee only
        sampled_tgt.append(initial_obs[0, :self.ref_dim].detach().cpu().numpy())  # ee only
        for step in range(self.bptt_steps):
            data = self.loader.dataset.get_data(fpath, step, history_max_len=None)
            data = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}
            ref = data["input_seq"][:, :self.ref_dim]

            ctl = self.u[:, step, :]
                # zeros constraint
            ctl = ctl.view(-1, 3, 3)
            mean = ctl.mean(dim=2, keepdim=True) 
            ctl = ctl - mean 
            ctl = ctl.view(-1, 9)
            # norm constraint
            max_val = ctl.abs().max(dim=1, keepdim=True).values
            scale = torch.clamp(self.u_clip / (max_val + 1e-5), max=1.0)
            ctl = ctl * scale 

            prediction, h = self.dynamics.forward(torch.cat([prediction, ctl], dim=-1), h)

            sampled_pred.append(prediction[0, :self.ref_dim].detach().cpu().numpy())
            sampled_tgt.append(ref[0, :self.ref_dim].detach().cpu().numpy())
            sampled_origin_obs.append(data["input_seq"][
                0, self.ref_t_dim:-self.dynamics.ctl_dim].detach().cpu().numpy())
            sampled_ctl.append(ctl[0].detach().cpu().numpy())
            sampled_data_ctl.append(data["input_seq"][
                0, -self.dynamics.ctl_dim:].detach().cpu().numpy())

            step_loss = self.compute_loss(ref, prediction, ctl, last_ctl)
            step_loss = [s * (self.discount ** step) for s in step_loss]
            total_loss = [t_l + s_l for t_l, s_l in zip(total_loss, step_loss)]
            last_ctl = ctl

        total_loss = [t_l / self.bptt_steps for t_l in total_loss]
        loss = sum(total_loss)
    
        # update model parameters
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_([self.u], self.max_grad_norm)
        self.optimizer.step()

        iter_info = {"train/iteration": self.iteration % self.optimize_iters,
                     "train/loss": loss.detach().cpu().item()}
        for i, l in enumerate(total_loss):
            iter_info[f"train/loss_{self.loss_name_list[i]}"] = l.detach().cpu().item()
            iter_info[f"train/rmse_{self.loss_name_list[i]}"] = (
                l.detach().cpu().item() / self.loss_k_list[i] * self.scale) ** 0.5

        self.iteration += 1

        return (iter_info, sampled_pred, sampled_tgt, sampled_origin_obs, 
                sampled_ctl, sampled_data_ctl)
    

if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent.parent / "config"
    config_model = Config.load(config_path / "models_config.yaml")
    config_training = Config.load(config_path / "training_config.yaml")

    trainer = MpcPolicyTrainer(config_model, config_training)

    # Training loop
    print(f"Start training MPC targets !")

    p = Path(trainer.data_folder)
    files = [f for f in p.iterdir() if f.suffix in {'.h5', '.hdf5'}]
    # for fpath in files:
    fpath = files[2]
    data = trainer.loader.dataset.get_data(fpath, 0, history_max_len=None)
    data = {k: v.to(trainer.device, non_blocking=True) for k, v in data.items()}
    # initial_obs = data["input_seq"][
    #     :, trainer.ref_t_dim:-trainer.dynamics.ctl_dim].detach().clone()  #

    initial_obs = torch.tensor([[
        0.215924, -0.0362972, 0.108334, -160.418, -109.81, 663.98,
        52.0637, -28.6312, -23.4271, 15.3008, -7.64287, -7.65418, -0.0779115, 32.0472, -31.9634,
        0, 0, 0, 0, 0, 0, 0, 0, 0]],  
        dtype=torch.float32).to(trainer.device)

    # Training iterations
    start_time = time.time()
    for iter in range(trainer.optimize_iters):
        (iter_info, sampled_pred, sampled_tgt, sampled_origin_obs, 
            sampled_ctl, sampled_data_ctl) = trainer.update(fpath, initial_obs)
        iter_info["train/iter"] = iter
        for k, v in iter_info.items():
            print(f"{k}: {v}")
        if sampled_pred and sampled_tgt:
            pred_plot(sampled_pred, sampled_tgt, trainer.save_folder, trainer.obs_type, 
                        sampled_origin_obs, sampled_ctl, sampled_data_ctl)
            
    # sampled_ctl [array(9), array(9), ...]
    flat_ctl = np.concatenate(sampled_ctl).ravel().astype(np.float32)
    float_array = array.array('f', flat_ctl)
    with open('u.bin', 'wb') as f:
        float_array.tofile(f)
    print(f"Iterations time: {time.time() - start_time} s.")
