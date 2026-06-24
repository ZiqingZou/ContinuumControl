import torch
import torch.nn as nn
from typing import Type, List

from models.networks.mlp import _ACT_MAP, NormedMlp
from models.networks.rnn_cell import NormedLstmCell


class NormedLstm(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        h_bias: bool = True,
        act: Type[nn.Module] = nn.Tanh,
        num_layers: int = 3,
        ln: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.h_bias = h_bias
        self.act = act
        self.num_layers = num_layers
        self.ln = ln
        self.dropout = dropout

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        layers = []
        for i in range(num_layers):
            layers.append(NormedLstmCell(dims[i], dims[i + 1], h_bias=h_bias, ln=ln, act=act, dropout=dropout))
        self.net = nn.ModuleList(layers)

    def forward(self, x, h):
        """
        Predict the next hidden state given the current input and previous hidden state.
        Args:
            x: tensor of shape [bs, in_dim]
            h: tuple of tensors (h, c) each of shape [bs, num_layers, hidden_dim]
        Returns:
            (next_h, next_c): tuple of tensors (h, c) each of shape [bs, num_layers, hidden_dim]
        """
        next_h_ = []
        next_c_ = []
        _x = x
        for i in range(self.num_layers):
            _h, _c = h[0][:, i, :], h[1][:, i, :]  # hidden state and cell state for first layer
            _x, _c = self.net[i](_x, (_h, _c))  # [bs, hidden_dim]
            next_h_.append(_x.unsqueeze(1))  # [bs, 1, hidden_dim]
            next_c_.append(_c.unsqueeze(1))  # [bs, 1, hidden_dim]
        next_h = torch.cat(next_h_, dim=1)  # [bs, num_layers, hidden_dim]
        next_c = torch.cat(next_c_, dim=1)  # [bs, num_layers, hidden_dim]
        return next_h, next_c

    def extra_repr(self):
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"h_bias={self.h_bias}, "
            f"act={self.act.__name__}, "
            f"num_layers={self.num_layers}, "
            f"ln={self.ln}, "
            f"dropout={self.dropout if self.dropout > 0 else 'False'}")


class HistoryLstm(nn.Module):
    def __init__(
        self, 
        in_dim_per_step: int,   # input dim for one step
        out_dim_per_step: List[int],  # output dim for one step
        hidden_dim: int,        # hidden state dim and hidden layer width (for both rnn and output mlp)
        rnn_act: str,           # activation function for rnn net; options: tanh / relu
        rnn_depth: int,         # number of layers of rnn net
        rnn_h_bias: bool,       # use bias for rnn hidden state
        out_act: str,           # activation function for output mlp; options: elu / leaky_relu / relu / swish / gelu / tanh
        out_depth: int,         # number of layers for output mlp, including output layer
        truncation_len: int,    # length for truncated backpropagation through time
        initial_h_len: int=50,  # length of initial history to use for hidden state initialization
        **kwargs                # additional args for NormedRnn and NormedMlp (dropout and ln)
    ):
        super().__init__()
        self.rnn_act = _ACT_MAP[rnn_act]
        self.out_act = _ACT_MAP[out_act]
        self.truncation_len = truncation_len
        self.initial_h_len = initial_h_len

        h0 = torch.zeros(1, rnn_depth, hidden_dim)  # (batch, num_layers, hidden_size)
        self.register_buffer("h0", h0)
        c0 = torch.zeros(1, rnn_depth, hidden_dim)  # (batch, num_layers, hidden_size)
        self.register_buffer("c0", c0)

        self.rnn_net = NormedLstm(
            in_dim_per_step, hidden_dim, h_bias=rnn_h_bias, act=self.rnn_act, num_layers=rnn_depth, **kwargs)
        self.output_net = nn.ModuleList([
            NormedMlp(hidden_dim, out_dim, act=self.out_act, hidden_dim=hidden_dim, depth=out_depth, **kwargs)
            for out_dim in out_dim_per_step])

    def get_initial_h(self, initial_history, history_mask):
        """
        Get initial hidden state for rnn.
        Args:
            initial_history: tensor of shape [bs, length, in_dim_per_step]
            history_mask: tensor of shape [bs, length] bool, True for real data, False for padded data
        Returns:
            initial_h: tuple of tensors (h, c) each of shape [bs, num_layers, hidden_dim]
        """
        bs, length, _ = initial_history.shape

        initial_h = self.h0.expand(bs, -1, -1).clone()  # [bs, num_layers, hidden_dim]
        initial_c = self.c0.expand(bs, -1, -1).clone()  # [bs, num_layers, hidden_dim]
        # sampled_h = [initial_h[0]]  # list of [num_layers, hidden_dim]
        # sampled_c = [initial_c[0]]  # list of [num_layers, hidden_dim]
        for i in range(max(0, length - self.initial_h_len), length):
            input = initial_history[:, i, :]  # [bs, in_dim_per_step]
            new_h, new_c = self.rnn_net(input, (initial_h, initial_c))  # [bs, num_layers, hidden_dim]
            mask = history_mask[:, i].view(bs, 1, 1)  # [bs, 1, 1]
            initial_h = torch.where(mask, new_h, initial_h)  # update only for real data
            initial_c = torch.where(mask, new_c, initial_c)  # update only for real data
            # sampled_h.append(initial_h[0])
            # sampled_c.append(initial_c[0])
            if i % self.truncation_len == 0:
                initial_h = initial_h.detach()  # truncate gradient
                initial_c = initial_c.detach()  # truncate gradient

        return (initial_h, initial_c)  # , (sampled_h, sampled_c)

    def forward(self, input, h):
        """
        Predict the output given the current input and h.
        Args:
            input: tensor of shape [bs, in_dim_per_step]
            h: tuple of tensors (h, c) each of shape [bs, num_layers, hidden_dim]
        Returns:
            predict: tensor of shape [bs, out_dim_per_step]
            new_h: tuple of tensors (h, c) each of shape [bs, num_layers, hidden_dim]
        """
        new_h, new_c = self.rnn_net(input, h)  # [bs, num_layers, hidden_dim]
        predict = torch.cat([net(new_h[:, -1, :]) for net in self.output_net], dim=-1)  # [bs, out_dim_per_step]
        return predict, (new_h, new_c)
