import torch.nn as nn
from typing import List

from models.networks.mlp import HistoryMlp, NormedMlp
from models.networks.rnn import HistoryRnn
from models.networks.lstm import HistoryLstm


def get_dynamics_net(
    net_type: str,
    obs_dim: int,
    ctl_dim: int,
    output_dim: List[int],
    net_params: dict,
) -> nn.Module:
    """
    Factory that constructs dynamics networks.

    Args
      net_type: one of 'mlp', 'rnn', 'gru', 'lstm' selecting the network class
      obs_dim: observation feature dimension
      ctl_dim: control feature dimension
      output_dim: output dimension
      net_params: configuration dict passed to network constructors

    Returns
      delta_net or abs_net (nn.Module)
    """
    # Choose the network class. Assumes these are classes with constructor:

    if net_type == 'mlp':
        net = HistoryMlp(obs_dim + ctl_dim, output_dim, obs_dim + ctl_dim, **net_params)
    elif net_type == 'rnn' or net_type == 'gru':
        net = HistoryRnn(obs_dim + ctl_dim, output_dim, net_type, **net_params)
    elif net_type == 'lstm':
        net = HistoryLstm(obs_dim + ctl_dim, output_dim, **net_params)
    else:
        raise ValueError(f'Unknown dynamics type: {net_type}')

    return net


def get_policy_net(
    net_type: str,
    input_dim: int,
    output_dim: List[int],
    ctl_plus_obs_dim: int,
    net_params: dict,
) -> nn.Module:
    """
    Factory that constructs policy networks.

    Args
      net_type: one of 'mlp', 'rnn', 'gru', 'lstm' selecting the network class
      input_dim: input feature dimension
      output_dim: output feature dimension
      ctl_plus_obs_dim: combined control and observation input dimension for history of mlp
      net_params: configuration dict passed to network constructors

    Returns
      control net (nn.Module)
    """
    # Choose the network class. Assumes these are classes with constructor:

    if net_type == 'mlp':
        net = HistoryMlp(
            input_dim, output_dim, ctl_plus_obs_dim, **net_params)  # history_dim_pre_step = ctl_plus_obs_dim
    elif net_type == 'rnn' or net_type == 'gru':
        net = HistoryRnn(input_dim, output_dim, net_type, **net_params)
    elif net_type == 'lstm':
        net = HistoryLstm(input_dim, output_dim, **net_params)
    elif net_type == 'mlp_ignore_h':
        net = NormedMlp(input_dim, output_dim[0], **net_params)
    elif net_type == 'mlp_with_hout':
        net = NormedMlp(net_params["hidden_dim"] + input_dim, output_dim[0], **net_params)
    else:
        raise ValueError(f'Unknown policy type: {net_type}')

    return net

