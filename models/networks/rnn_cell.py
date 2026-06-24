import torch
import torch.nn as nn
from typing import Type


class NormedRnnCell(nn.Module):
    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        h_bias: bool = True,
        ln: bool = True,
        act: Type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.h_bias = h_bias

        self.ln = nn.LayerNorm(h_dim) if ln else None
        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.x_net = nn.Linear(x_dim, h_dim, bias=True)
        self.h_net = nn.Linear(h_dim, h_dim, bias=h_bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_net.weight)
        nn.init.orthogonal_(self.h_net.weight)
        nn.init.zeros_(self.x_net.bias)
        if self.h_net.bias is not None:
            nn.init.zeros_(self.h_net.bias)

    def forward(self, x, h):
        """
        Predict the next hidden state given the current input and previous hidden state.
        Args:
            x: tensor of shape [bs, x_dim]
            h: tensor of shape [bs, h_dim]
        Returns:
            h_next: tensor of shape [bs, h_dim]
        """
        h_next = self.x_net(x) + self.h_net(h)

        if self.ln is not None:
            h_next = self.ln(h_next)

        h_next = self.act(h_next)

        if self.dropout is not None:
            h_next = self.dropout(h_next)

        return h_next

    def extra_repr(self):
        repr_ln = f"layernorm={self.ln is not None}"
        repr_act = f"act={self.act.__class__.__name__}"
        repr_dropout = f"dropout={self.dropout.p if self.dropout is not None else 'False'}"
        return (f"in_dim={self.x_dim}, out_dim={self.h_dim}, h_bias={self.h_bias}, "
                f"{repr_ln}, {repr_act}, {repr_dropout}")


class NormedGruCell(nn.Module):
    """
    GRU cell with optional LayerNorm, activation for candidate, and dropout.
    """
    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        h_bias: bool = True,
        ln: bool = True,
        act: Type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.h_bias = h_bias

        # LayerNorm per gate (z, r, n) if requested
        if ln:
            self.ln_z = nn.LayerNorm(h_dim)
            self.ln_r = nn.LayerNorm(h_dim)
            self.ln_n = nn.LayerNorm(h_dim)
        else:
            self.ln_z = self.ln_r = self.ln_n = None

        self.act = act()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # combine linear transforms for efficiency: produce 3*h_dim outputs
        self.x_net = nn.Linear(x_dim, 3 * h_dim, bias=True)
        self.h_net = nn.Linear(h_dim, 3 * h_dim, bias=h_bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_net.weight)
        nn.init.orthogonal_(self.h_net.weight)
        nn.init.zeros_(self.x_net.bias)
        if self.h_net.bias is not None:
            nn.init.zeros_(self.h_net.bias)

    def forward(self, x, h):
        """
        x: [bs, x_dim]
        h: [bs, h_dim]
        returns h_next: [bs, h_dim]
        """
        # compute combined linear outputs and split into gates
        x_comb = self.x_net(x)          # [bs, 3*h_dim]
        h_comb = self.h_net(h)          # [bs, 3*h_dim]
        x_z, x_r, x_n = x_comb.chunk(3, dim=-1)
        h_z, h_r, h_n = h_comb.chunk(3, dim=-1)

        # update gate z
        z = x_z + h_z
        if self.ln_z is not None:
            z = self.ln_z(z)
        z = torch.sigmoid(z)

        # reset gate r
        r = x_r + h_r
        if self.ln_r is not None:
            r = self.ln_r(r)
        r = torch.sigmoid(r)

        # candidate n: note h_n is multiplied by r
        n = x_n + r * h_n
        
        if self.ln_n is not None:
            n = self.ln_n(n)

        n = self.act(n)

        if self.dropout is not None:
            n = self.dropout(n)

        # GRU update: h_next = (1 - z) * n + z * h
        h_next = (1.0 - z) * n + z * h
        return h_next

    def extra_repr(self):
        repr_ln = f"layernorm={self.ln_z is not None}"
        repr_act = f"act={self.act.__class__.__name__}"
        repr_dropout = f"dropout={self.dropout.p if self.dropout is not None else 'False'}"
        return (f"in_dim={self.x_dim}, out_dim={self.h_dim}, h_bias={self.h_bias}, "
                f"{repr_ln}, {repr_act}, {repr_dropout}")


class NormedLstmCell(nn.Module):
    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        h_bias: bool = True,
        ln: bool = True,
        act: Type[nn.Module] = nn.Tanh,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.h_bias = h_bias

        # LayerNorm per gate if requested
        if ln:
            self.ln_i = nn.LayerNorm(h_dim)
            self.ln_f = nn.LayerNorm(h_dim)
            self.ln_o = nn.LayerNorm(h_dim)
            self.ln_g = nn.LayerNorm(h_dim)
        else:
            self.ln_i = self.ln_f = self.ln_o = self.ln_g = None

        # candidate activation (usually tanh)
        self.act = act()

        # dropout applied to candidate g (can be moved to h_next if preferred)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # combined linear transforms: x_net has bias, h_net no bias
        self.x_net = nn.Linear(x_dim, 4 * h_dim, bias=True)
        self.h_net = nn.Linear(h_dim, 4 * h_dim, bias=h_bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_net.weight)
        nn.init.orthogonal_(self.h_net.weight)
        with torch.no_grad():
            self.x_net.bias[self.h_dim:2*self.h_dim] += 1.0
        if self.h_net.bias is not None:
            nn.init.zeros_(self.h_net.bias)

    def forward(self, x, h):
        """
        x: [bs, x_dim]
        h: ([bs, h_dim], [bs, h_dim])  # (hidden state, cell state)
        returns (h_next, c_next)
        """
        # combined linear outputs
        x_comb = self.x_net(x)         # [bs, 4*h_dim]
        h_comb = self.h_net(h[0])      # [bs, 4*h_dim]
        x_i, x_f, x_o, x_g = x_comb.chunk(4, dim=-1)
        h_i, h_f, h_o, h_g = h_comb.chunk(4, dim=-1)

        # input gate
        i = x_i + h_i
        if self.ln_i is not None:
            i = self.ln_i(i)
        i = torch.sigmoid(i)

        # forget gate (add forget_bias via buffer)
        f = x_f + h_f
        if self.ln_f is not None:
            f = self.ln_f(f)
        f = torch.sigmoid(f)

        # output gate
        o = x_o + h_o
        if self.ln_o is not None:
            o = self.ln_o(o)
        o = torch.sigmoid(o)

        # candidate (g)
        g = x_g + h_g
        if self.ln_g is not None:
            g = self.ln_g(g)
        g = self.act(g)  # usually tanh

        if self.dropout is not None:
            g = self.dropout(g)

        # cell and hidden update
        c_next = f * h[1] + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def extra_repr(self):
        repr_ln = f"layernorm={self.ln_i is not None}"
        repr_act = f"act={self.act.__class__.__name__}"
        repr_dropout = f"dropout={self.dropout.p if self.dropout is not None else 'False'}"
        return (f"in_dim={self.x_dim}, out_dim={self.h_dim}, h_bias={self.h_bias}, "
                f"{repr_ln}, {repr_act}, {repr_dropout}")
