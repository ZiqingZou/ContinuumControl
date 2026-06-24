import torch
import torch.nn as nn

from models.tools.get_norm import get_policy_norm
from models.tools.get_net import get_policy_net
from models.tools.weight_init import apply_with_control, weight_init


class Policy(nn.Module):
    """
    Policy model to recurrently predict the control at each step.
    control = f (obs, ref, h)
    """
    def __init__(self, cfg, device, load_path=None):
        super().__init__()
        self.device = device
        self.obs_type = cfg.policy.obs_type
        self.net_type = cfg.policy.net_type
        self.ref_horizon = cfg.policy.ref_horizon
        self.input_type = cfg.policy.input_type
        self.u_clip = cfg.policy.u_clip
        self.output_step = cfg.policy.output_step
        self.stochastic = cfg.policy.stochastic
        
        (self.obs_dim, self.ref_dim, self.ctl_dim, input_norm_mean, input_norm_std,
         output_norm_mean, output_norm_std) = get_policy_norm(
             cfg.norm, self.input_type, self.obs_type, self.output_step, self.ref_horizon)
        self.register_buffer("input_norm_mean", input_norm_mean.unsqueeze(0))
        self.register_buffer("input_norm_std", input_norm_std.unsqueeze(0))
        self.register_buffer("output_norm_mean", output_norm_mean.unsqueeze(0))
        self.register_buffer("output_norm_std", output_norm_std.unsqueeze(0))

        if "r" in self.input_type:
            self.input_dim = self.ref_dim * self.ref_horizon
            if "o" in self.input_type:
                self.input_dim = self.input_dim + self.obs_dim
            if "u" in self.input_type:
                self.input_dim = self.input_dim + self.ctl_dim
        else:
            raise ValueError(f"Unexpected input_type: {self.input_type}")
        self.output_dim = self.ctl_dim * self.output_step  # concatenate ctl for output_step steps
        
        self.execute_ctl_dim = 9
        self._policy = get_policy_net(
            net_type=self.net_type, input_dim=self.input_dim,
            output_dim=[self.output_dim] if not self.stochastic else [self.output_dim, self.output_dim],
            ctl_plus_obs_dim=self.obs_dim + self.execute_ctl_dim,
            net_params=cfg.policy.net_params.get(self.net_type).to_dict())
    
        if load_path is not None:
            self.load(load_path)
        else:
            apply_with_control(self, weight_init)
        self.to(self.device)
        self.eval()

    @property
    def total_params(self):
        """
        Total params number.
        """
        return sum(p.numel() for p in self.parameters())
    
    @property
    def trainable_params(self):
        """
        Total params number of params requires grad.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def set_initial_h(self, initial_history, history_mask):
        """
        Get initial hidden state from initial history input.

        Args:
            initial_history: [bs, length, input_dim], initial history input to record grad
            history_mask: [bs, length] bool, mask for initial history input 
            (True for real data, False for padded data)

        Returns:
            h: initial hidden states for each sub-dynamics
        """
        normed_history_input = (initial_history[
            :, :, :self.input_dim] - self.input_norm_mean.unsqueeze(1)) / self.input_norm_std.unsqueeze(1)
        self.h = self._dynamics.get_initial_h(normed_history_input, history_mask)
    
    def forward(self, input, h):
        """
        Adapted to output control for this step.

        Args:
            input: Tensor [bs, ref_dim * ref_horizon + obs_dim], 
                last ctl(if "u" in input_type), this obs and future ref_horizon ref_ee concatenated
            h: Tensor [bs, hidden_layers, hidden_dim], history embedding at current step
                or [bs, history_window, his_dim_per_step] for mlp

        Returns:
            control: Tensor [bs, output_dim], control of this step, unclamped
        """
        normed_input = (
            input[:, :self.input_dim] - self.input_norm_mean) / self.input_norm_std  # auto broadcast
    
        if self.net_type == "mlp_ignore_h":
            origin_control = self._policy(normed_input)  # [bs, output_dim]
        elif self.net_type == "mlp_with_hout":
            origin_control = self._policy(torch.cat((h[:, -1, :], normed_input), dim=1))  # [bs, output_dim]
        else:
            origin_control, _ = self._policy(normed_input, h)  # [bs, output_dim]
        
        if self.stochastic:
            mu_out, logvar_out = torch.chunk(origin_control, 2, dim=1)  # [bs, output_dim] each
            if self.training:
                std = torch.exp(0.5 * logvar_out)
                eps = torch.randn_like(std)
                origin_control = mu_out + eps * std  # reparameterization trick
            else:
                origin_control = mu_out  # use mean value during evaluation
        
        denormed_control = origin_control * self.output_norm_std + self.output_norm_mean  # auto broadcast

        if "u" in self.input_type:
            denormed_control = denormed_control + input[:, -self.ctl_dim:].repeat(1, self.output_step)

        return denormed_control

    def constrain_control(self, u):
        """
        Constrain the control output.

        Args:
            u: Tensor [bs, output_dim=len*9], control of this step, unclamped

        Returns:
            control: Tensor [bs, output_dim=len*9], control of this step, clamped
        """
        original_shape = u.shape # [bs, len*9]
        bs = u.shape[0]

        # zeros constraint
        u = u.view(bs, -1, 3, 3)
        mean = u.mean(dim=3, keepdim=True) 
        u = u - mean 
        u = u.view(bs, -1, 9)

        # norm constraint
        max_val = u.abs().max(dim=2, keepdim=True).values
        scale = torch.clamp(self.u_clip / (max_val + 1e-5), max=1.0)
        u = u * scale 

        return u.view(original_shape)

    def recur_forward(self, input, h):
        """
        Adapted to output control for this step.

        Args:
            input: Tensor [bs, ref_dim * ref_horizon + obs_dim], 
                last ctl(if "u" in input_type), this obs and future ref_horizon ref_ee concatenated
            h: Tensor [bs, hidden_layers, hidden_dim], history embedding at current step
                or [bs, history_window, his_dim_per_step] for mlp

        Returns:
            control: Tensor [bs, output_dim], control of this step, clamped
        """
        denormed_control = self.forward(input, h)  # [bs, output_dim]
        denormed_control = self.constrain_control(denormed_control)
        return denormed_control
    
    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"policy": self.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        if isinstance(fp, dict):
            state_dict = fp
        else:
            state_dict = torch.load(fp, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict["policy"])
