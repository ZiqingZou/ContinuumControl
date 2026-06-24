import torch
import torch.nn as nn
from typing import Type, List, Union


_ACT_MAP = {
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class NormedLinear(nn.Linear):
    """
    Linear layer with optionally LayerNorm, activation and dropout.
    """
    def __init__(
        self,
        *args,
        ln: bool = True,
        act: Type[nn.Module] = nn.ELU,
        dropout: float = 0.0,
    ):
        self.ln = None
        super().__init__(*args)
        self.ln = nn.LayerNorm(self.out_features) if ln else None
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.ln is not None:
            nn.init.ones_(self.ln.weight)
            nn.init.zeros_(self.ln.bias)

    def forward(self, x):
        y = super().forward(x)

        if self.ln is not None:
            y = self.ln(y)

        y = self.act(y)

        if self.dropout is not None:
            y = self.dropout(y)

        return y

    def extra_repr(self):
        repr_ln = f"layernorm={self.ln is not None}"
        repr_act = f"act={self.act.__class__.__name__}"
        repr_dropout = f"dropout={self.dropout.p if self.dropout is not None else 'False'}"
        return f"{repr_ln}, {repr_act}, {repr_dropout}"


class NormedMlp(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: Union[Type[nn.Module], str] = nn.ELU,
        hidden_dim: int = 512,
        depth: int = 5,
        ln: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.ln = ln
        self.act = _ACT_MAP[act] if isinstance(act, str) else act
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (depth - 1) + [out_dim]
        layers = []
        for i in range(depth - 1):
            layers.append(NormedLinear(dims[i], dims[i + 1], ln=ln, act=self.act, dropout=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))  # last layer no norm/activation/dropout

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def extra_repr(self):
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim}, depth={self.depth}, "
            f"act={self.act.__name__}, "
            f"ln={self.ln}, "
            f"dropout={self.dropout if self.dropout > 0 else 'False'}")


class HistoryMlp(nn.Module):
    def __init__(
        self, 
        in_dim_per_step: int,   # input dim for one step
        out_dim_per_step: List[int],  # output dim for one step
        his_dim_per_step: int,  # history input dim for one step
        history_window: int,    # concatenate history inputs of this many steps
        act: str,               # activation function str; options: elu / leaky_relu / relu / swish / gelu / tanh
        **kwargs                # hidden_dim, depth, dropout, ln for NormedMlp
    ):
        super().__init__()
        self.his_dim_per_step = his_dim_per_step
        assert his_dim_per_step <= in_dim_per_step, "his_dim_per_step should not be larger than in_dim_per_step"
        self.history_window = history_window
        self.act = _ACT_MAP[act]

        self.net = nn.ModuleList([
            NormedMlp(his_dim_per_step * history_window + in_dim_per_step, out_dim, act=self.act, **kwargs)
            for out_dim in out_dim_per_step])

    @torch.no_grad()
    def get_initial_h(self, initial_history, history_mask):
        """
        Get initial history for mlp.
        Args:
            initial_history: tensor of shape [bs, length, in_dim_per_step]
            history_mask: tensor of shape [bs, length] bool, not used in mlp because of the fixed history window
        Returns:
            initial_h: tensor of shape [bs, history_window, his_dim_per_step]
        """
        assert initial_history.size(1) >= self.history_window
        initial_h = initial_history[:, -self.history_window:, :self.his_dim_per_step]
        # sampled_h = [initial_h[0]]
        return initial_h  # , sampled_h

    def forward(self, input, h):
        """
        Predict the output given the current input and history.
        Args:
            input: tensor of shape [bs, in_dim_per_step]
            h: tensor of shape [bs, history_window, his_dim_per_step]
        Returns:
            predict: tensor of shape [bs, out_dim_per_step]
            new_h: tensor of shape [bs, history_window, his_dim_per_step]
        """
        reshaped_input = torch.cat(
            [h.reshape(h.shape[0], -1), input], dim=-1)  # [bs, his_dim_per_step * history_window + in_dim_per_step]
        predict = torch.cat([net(reshaped_input) for net in self.net], dim=-1)
        new_h = torch.cat([h[:, 1:, :], input[:, :self.his_dim_per_step].unsqueeze(1)], dim=1)

        return predict, new_h
    