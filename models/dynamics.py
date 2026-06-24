import torch
import torch.nn as nn

from models.tools.get_norm import get_dynamics_norm
from models.tools.get_net import get_dynamics_net
from models.tools.weight_init import apply_with_control, weight_init


class Dynamics(nn.Module):
    """
    Dynamics model to recurrently predict the next observation.
    next_obs_predict, next_h = f (obs, u, h)
    """
    def __init__(self, cfg, device, load_path=None):
        super().__init__()
        self.device = device
        self.obs_type = cfg.dynamics.obs_type
        self.net_type = cfg.dynamics.net_type
        self.residual_predict = cfg.dynamics.residual_predict
        
        (self.obs_dim, self.ctl_dim, self.abs_predict_dim, input_norm_mean, input_norm_std,
         output_norm_mean, output_norm_std) = get_dynamics_norm(cfg.norm, self.obs_type, self.residual_predict)
        self.register_buffer("input_norm_mean", input_norm_mean.unsqueeze(0))
        self.register_buffer("input_norm_std", input_norm_std.unsqueeze(0))
        self.register_buffer("output_norm_mean", output_norm_mean.unsqueeze(0))
        self.register_buffer("output_norm_std", output_norm_std.unsqueeze(0))

        self.input_dim = self.obs_dim + self.ctl_dim
        self.output_dim = self.obs_dim if "T" in self.obs_type else self.abs_predict_dim + self.obs_dim
        
        self._dynamics = get_dynamics_net(
            net_type=self.net_type, obs_dim=self.obs_dim, ctl_dim=self.ctl_dim, 
            output_dim=[self.obs_dim] if "T" in self.obs_type else [self.abs_predict_dim, self.obs_dim],
            net_params=cfg.dynamics.net_params.get(self.net_type).to_dict())

        # if net_type is "mlp", h is history inputs of tensor [bs, history_window, input_dim]
        # if net_type is "rnn" or "gru", h is hidden state of tensor [bs, num_layers, hidden_dim]
        # if net_type is "lstm", h is tuple of hidden and cell state of tenor [bs, num_layers, hidden_dim]
        if self.net_type != "mlp":
            self.hidden_dim = cfg.dynamics.net_params.get(self.net_type).get("hidden_dim")
            self.rnn_depth = cfg.dynamics.net_params.get(self.net_type).get("rnn_depth")
        self.h = None
    
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

    def set_zero_h(self, batch_size):
        """
        Set zero hidden state.

        Args:
            batch_size: int, batch size
        """
        self.h = self._dynamics.h0.expand(batch_size, -1, -1).clone()  # [bs, num_layers, hidden_dim]
            
    def forward(self, input, h):
        """
        Adapted to predict the next observation in implementation.

        Args:
            input: Tensor [bs, input_dim], obs and ctl concatenated
            h: hidden states for _dynamics
        Returns:
            prediction: Tensor [bs, output_dim]
                observation prediction of the next step ("T" in obs_type) /
                EE position end rotation and observation prediction ("T" not in obs_type) 
            new_h: new hidden states for _dynamics
        """
        normed_input = (input[:, :self.input_dim] - self.input_norm_mean) / self.input_norm_std  # auto broadcast
        origin_prediction, new_h = self._dynamics(normed_input, h)

        denormed_prediction = origin_prediction * self.output_norm_std + self.output_norm_mean  # auto broadcast
        if self.residual_predict:
            prediction = denormed_prediction + input[:, :self.obs_dim]
        else:
            prediction = denormed_prediction

        return prediction, new_h

    def recur_forward(self, input, initial_history=None, history_mask=None):
        """
        Adapted to recurrently predict the next observation, for one step predict.

        Args:
            input: Tensor [bs, input_dim], obs and ctl concatenated
            initial_history: [bs, length, input_dim], initial history input to record grad
            history_mask: [bs, length] bool, mask for initial history input 
            (True for real data, False for padded data)

        Returns:
            prediction: Tensor [bs, output_dim]
            observation prediction of the next step ("T" in obs_type) /
            EE position end rotation and observation prediction ("T" not in obs_type) 
        """
        # bs = input.size(0)

        if initial_history is not None:
            self.set_initial_h(initial_history, history_mask)

        prediction, new_h = self.forward(input, self.h)
        self.h = new_h

        return prediction
    
    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"dynamics": self.state_dict()}, fp)

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
        self.load_state_dict(state_dict["dynamics"])
        
        # new_state_dict = {}
        # for k, v in state_dict["dynamics"].items():
        #     if k.startswith("_dynamics.0"):
        #         new_k = k.replace("_dynamics.0", "_dynamics")
        #     else:
        #         new_k = k
        #     if "_dynamics.output_net" in new_k:
        #         new_k = new_k.replace("_dynamics.output_net", "_dynamics.output_net.0")
        #     new_state_dict[new_k] = v
        # self.load_state_dict(new_state_dict)
        # torch.save(new_state_dict, "dynamics_best_latest.pth")
