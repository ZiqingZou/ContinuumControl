import torch
from typing import Tuple, List

from config.parser import Config


@torch.no_grad()
def get_dynamics_norm(
    norm: Config,
    obs_type: str,
    residual_predict: bool,
    ctl_dim: int = 9,
    abs_predict_dim: int = 6,
    dtype: torch.dtype = torch.float32
) -> Tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Build input/output dimensions and normalization tensors for dynamics model.

    Args:
        norm: config object with fields like norm.u.mean, norm.pos.x.mean, ...
        obs_type: observation type string (e.g., "Tlvq" or subset)
        residual_predict: whether the model predicts residuals to current observation
        mode: dynamics mode string, one of "open_loop" or "closed_loop"
        abs_predict_dim: absolute end-effector transformation prediction dimension (default 6)
        dtype: tensor dtype (default torch.float32)

    Returns:
        obs_dim, ctl_dim, abs_predict_dim, input_norm_mean, input_norm_std,  (sequence: T, l, v, q, u)
        output_norm_mean, output_norm_std  (sequence: T, l, v, q)
    """
    # start with observation part
    obs_dim = 0
    input_mean = []
    input_std  = []

    # T : end-effector pose (rotvec: x,y,z; pos: x,y,z)
    if "T" in obs_type:
        obs_dim += 6
        input_mean.append(torch.tensor(norm.rotvec.mean, dtype=dtype))
        input_mean.append(torch.tensor(norm.pos.mean, dtype=dtype))
        input_std.append(torch.tensor(norm.rotvec.std, dtype=dtype))
        input_std.append(torch.tensor(norm.pos.std, dtype=dtype))
    # l : motor lengths (9)
    if "l" in obs_type:
        obs_dim += 9
        input_mean.append(torch.tensor(norm.l.mean, dtype=dtype))
        input_std.append(torch.tensor(norm.l.std, dtype=dtype))

    # v : motor velocities (9)
    if "v" in obs_type:
        obs_dim += 9
        input_mean.append(torch.tensor(norm.v.mean, dtype=dtype))
        input_std.append(torch.tensor(norm.v.std, dtype=dtype))

    # q : motor torques (9)
    if "q" in obs_type:
        obs_dim += 9
        input_mean.append(torch.tensor(norm.torque.mean, dtype=dtype))
        input_std.append(torch.tensor(norm.torque.std, dtype=dtype))

    # end with control (desired motor velocities) part
    input_mean.append(torch.tensor(norm.v.mean, dtype=dtype))
    input_std.append(torch.tensor(norm.v.std, dtype=dtype))

    if len(input_mean) > 1:
        input_norm_mean = torch.cat(input_mean)
        input_norm_std  = torch.cat(input_std)
    else:
        raise ValueError ("Unexpected obs_type: %s", obs_type)
    
    # input_dim = obs_dim + ctl_dim

    # build output (predicted deltas) parts
    output_mean = []
    output_std  = []

    if residual_predict:
        if "T" in obs_type:
            # output_dim = obs_dim  # outputs predict deltas for the observation part
            output_mean.append(torch.tensor(norm.delta_rotvec.mean, dtype=dtype))
            output_mean.append(torch.tensor(norm.delta_pos.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.delta_rotvec.std, dtype=dtype))
            output_std.append(torch.tensor(norm.delta_pos.std, dtype=dtype))
        else:
            # output_dim = abs_predict_dim + obs_dim
            output_mean.append(torch.tensor(norm.rotvec.mean, dtype=dtype))
            output_mean.append(torch.tensor(norm.pos.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.rotvec.std, dtype=dtype))
            output_std.append(torch.tensor(norm.pos.std, dtype=dtype))

        if "l" in obs_type:
            output_mean.append(torch.tensor(norm.delta_l.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.delta_l.std, dtype=dtype))
        if "v" in obs_type:
            output_mean.append(torch.tensor(norm.delta_v.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.delta_v.std, dtype=dtype))

        if "q" in obs_type:
            output_mean.append(torch.tensor(norm.delta_torque.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.delta_torque.std, dtype=dtype))
    else:
        output_mean.append(torch.tensor(norm.rotvec.mean, dtype=dtype))
        output_mean.append(torch.tensor(norm.pos.mean, dtype=dtype))
        output_std.append(torch.tensor(norm.rotvec.std, dtype=dtype))
        output_std.append(torch.tensor(norm.pos.std, dtype=dtype))

        if "l" in obs_type:
            output_mean.append(torch.tensor(norm.l.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.l.std, dtype=dtype))
        if "v" in obs_type:
            output_mean.append(torch.tensor(norm.v.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.v.std, dtype=dtype))

        if "q" in obs_type:
            output_mean.append(torch.tensor(norm.torque.mean, dtype=dtype))
            output_std.append(torch.tensor(norm.torque.std, dtype=dtype))
    output_norm_mean = torch.cat(output_mean)
    output_norm_std  = torch.cat(output_std)

    return obs_dim, ctl_dim, abs_predict_dim, input_norm_mean, input_norm_std, output_norm_mean, output_norm_std


@torch.no_grad()
def get_policy_norm(
    norm: Config,
    input_type: str,
    obs_type: str,
    output_step: int,
    ref_horizon: int,
    dtype: torch.dtype = torch.float32
) -> Tuple[int, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """
    Build input/output dimensions and normalization tensors for dynamics model.

    Args:
        cfg: config object with fields like cfg.norm.u.mean, cfg.norm.pos.x.mean, ...
        input_type: input type string, one of "hor" (history embedding + observation + ref) or "hr" (observation + ref)
        obs_type: observation type string (e.g., "Tlvq" or subset)
        output_step: number of steps the policy outputs controls for at each step
        ref_horizon: number of future steps of reference trajectory provided to policy per step
        ctl_slice: indices of joint controls to use; e.g., [0,1,3,4,6,7] for skipping every 3rd joint
        dtype: tensor dtype (default torch.float32)

    Returns:
        obs_dim, ref_dim, ctl_dim, input_norm_mean, input_norm_std,  (sequence: T, l, v, q, T_ref_seq)
        output_norm_mean, output_norm_std  (sequence: u)
    """
    # start with observation part
    ctl_dim = 9
    obs_dim = 0
    input_mean = []
    input_std  = []

    # reference part
    if "pos_only" in input_type:
        ref_dim = 3
        input_mean.append(torch.cat(
            [torch.tensor(norm.pos.mean, dtype=dtype)] * ref_horizon))
        input_std.append(torch.cat(
            [torch.tensor(norm.pos.std, dtype=dtype)] * ref_horizon))
    else:
        ref_dim = 6
        input_mean.append(torch.cat(
            [torch.tensor(norm.rotvec.mean, dtype=dtype), torch.tensor(norm.pos.mean, dtype=dtype)] * ref_horizon))
        input_std.append(torch.cat(
            [torch.tensor(norm.rotvec.std, dtype=dtype), torch.tensor(norm.pos.std, dtype=dtype)] * ref_horizon))
    
    if "T" in obs_type:
        obs_dim += 6
    # l : motor lengths (9)
    if "l" in obs_type:
        obs_dim += 9
    # v : motor velocities (9)
    if "v" in obs_type:
        obs_dim += 9
    # q : motor torques (9)
    if "q" in obs_type:
        obs_dim += 9

    if "o" in input_type:
        # T : end-effector pose (rotvec: x,y,z; pos: x,y,z)
        if "T" in obs_type:
            input_mean.append(torch.tensor(norm.rotvec.mean, dtype=dtype))
            input_mean.append(torch.tensor(norm.pos.mean, dtype=dtype))
            input_std.append(torch.tensor(norm.rotvec.std, dtype=dtype))
            input_std.append(torch.tensor(norm.pos.std, dtype=dtype))

        # l : motor lengths (9)
        if "l" in obs_type:
            input_mean.append(torch.tensor(norm.l.mean, dtype=dtype))
            input_std.append(torch.tensor(norm.l.std, dtype=dtype))

        # v : motor velocities (9)
        if "v" in obs_type:
            input_mean.append(torch.tensor(norm.v.mean, dtype=dtype))
            input_std.append(torch.tensor(norm.v.std, dtype=dtype))

        # q : motor torques (9)
        if "q" in obs_type:
            input_mean.append(torch.tensor(norm.torque.mean, dtype=dtype))
            input_std.append(torch.tensor(norm.torque.std, dtype=dtype))

    # end with (the last) control part
    if "u" in input_type:
        # start with (the last) control part
        input_mean.append(torch.tensor(norm.v.mean, dtype=dtype))
        input_std.append(torch.tensor(norm.v.std, dtype=dtype))

    input_norm_mean = torch.cat(input_mean)
    input_norm_std  = torch.cat(input_std)
    
    # input_dim = (ctl_dim +) ref_dim * ref_horizon (+ obs_dim)

    # build output (control at this step) parts
    if "u" in input_type:
        # delta control output
        output_norm_mean = torch.cat(
            [torch.tensor(norm.delta_v.mean, dtype=dtype)] * output_step)
        output_norm_std  = torch.cat(
            [torch.tensor(norm.delta_v.std, dtype=dtype)] * output_step)
    else:
        output_norm_mean = torch.cat(
            [torch.tensor(norm.v.mean, dtype=dtype)] * output_step)
        output_norm_std  = torch.cat(
            [torch.tensor(norm.v.std, dtype=dtype)] * output_step)

    # output_dim = ctl_dim

    return obs_dim, ref_dim, ctl_dim, input_norm_mean, input_norm_std, output_norm_mean, output_norm_std
