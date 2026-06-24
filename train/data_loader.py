from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    """
    PyTorch Dataset for one or more h5 files.
    Sample range in each file is [history_window, len_file - bptt_step].

    Args:
        paths (str or Path): a directory of h5 files.
        input_key_list (List[str]): list of input keys inside each h5 file that contains inputs (e.g. ["u_t"]).
        output_key_list (List[str]): list of output keys inside each h5 file that contains targets (e.g. ["pos_t_plus_1"]).
        history_window (int): History window size used in training.
        bptt_steps (int): Number of steps in each training sequence for truncated BPTT.
        min_regressive_steps (int): Minimum number of steps of load trajectory.
        same_len_history (bool): If True, the initial history returned will always have length equal to history_window (padded if necessary). If False, the initial history will have variable length up to history_window.
        ref_horizon (int): Reference horizon for each step, only used when "ref_t" is in input_key_list.
        ref_pos_only (bool): If True, only the position component of the reference is used.
        history_max_len (int): Maximum length of history to consider.
    """
    def __init__(
        self,
        paths: Union[str, Path],
        input_key_list: List[str],
        output_key_list: List[str],
        history_window: int,
        bptt_steps: int,
        min_regressive_steps: int,
        same_len_history: bool,
        ref_horizon: int = 50,
        ref_pos_only: bool = False,
        history_max_len: int = 50,
    ):
        self.paths = paths
        self.input_key_list = input_key_list
        self.output_key_list = output_key_list
        self.history_window = history_window
        self.bptt_steps = bptt_steps
        self.min_regressive_steps = min_regressive_steps
        self.same_len_history = same_len_history
        self.ref_horizon = ref_horizon
        self.ref_pos_only = ref_pos_only
        self.history_max_len = history_max_len
        self.index: List[Tuple[Path, int]] = []

        self.reload()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        fpath, inner_idx = self.index[idx]
        # open file for this access (safe for DataLoader workers)
        with h5py.File(fpath, "r") as h5f:
            length = h5f["pos_t"].shape[0]

            input_seq = []
            initial_history = []
            for k in self.input_key_list:
                if k not in h5f:
                    raise KeyError(f"Key '{k}' not found in file {fpath}")
                
                if "ref_t" in k:
                    if inner_idx + self.bptt_steps <= length:
                        h5f_k = h5f[k][inner_idx: inner_idx + self.bptt_steps, :6 * self.ref_horizon]
                    else:
                        h5f_k = h5f[k][-1, :6 * self.ref_horizon].reshape(1, -1).repeat(self.bptt_steps, axis=0)
                        h5f_k[:length - inner_idx] = h5f[k][inner_idx:, :6 * self.ref_horizon]

                    if self.ref_pos_only:
                        h5f_k = h5f_k.reshape(-1, 6)[:, 3:6].reshape(self.bptt_steps, 3 * self.ref_horizon)
                else:
                    if inner_idx + self.bptt_steps <= length:
                        h5f_k = h5f[k][inner_idx: inner_idx + self.bptt_steps, :]
                    else:
                        if "u_t" in k:
                            h5f_k = np.zeros((self.bptt_steps, h5f[k].shape[1]), dtype=np.float32)
                        else:
                            h5f_k = h5f[k][-1, :].reshape(1, -1).repeat(self.bptt_steps, axis=0)
                        h5f_k[:length - inner_idx] = h5f[k][inner_idx:, :]

                    if self.same_len_history:
                        history_arr = np.asarray(h5f[k][inner_idx - self.history_window:inner_idx, :], dtype=np.float32)
                        initial_history.append(torch.from_numpy(history_arr))  # shape: (history_window, feature_dim)
                    else:
                        start_idx = max(self.history_window, inner_idx - self.history_max_len)
                        history_arr = np.asarray(h5f[k][start_idx:inner_idx, :], dtype=np.float32)
                        initial_history.append(torch.from_numpy(history_arr))  # shape: (len, feature_dim)
                
                # multi-step forward data
                forward_arr = np.asarray(h5f_k, dtype=np.float32)
                input_seq.append(torch.from_numpy(forward_arr))  # shape: (bptt_steps, feature_dim)

            input_seq = torch.cat(input_seq, dim=-1)  # shape: (bptt_steps, input_dim_per_step)
            initial_history = torch.cat(initial_history, dim=-1)  # shape: (len, input_dim_per_step)

            target_seq = []
            for k in self.output_key_list:
                if k not in h5f:
                    raise KeyError(f"Key '{k}' not found in file {fpath}")
                else:
                    if inner_idx + self.bptt_steps <= length:
                        h5f_k = h5f[k][inner_idx: inner_idx + self.bptt_steps, :]
                    else:
                        h5f_k = h5f[k][-1, :].reshape(1, -1).repeat(self.bptt_steps, axis=0)
                        h5f_k[:length - inner_idx] = h5f[k][inner_idx:, :]
                
                forward_arr = np.asarray(h5f_k, dtype=np.float32)
                target_seq.append(torch.from_numpy(forward_arr))  # shape: (bptt_steps, feature_dim)

            target_seq = torch.cat(target_seq, dim=-1)  # shape: (bptt_steps, target_dim)pos

            out = {"input_seq": input_seq,
                   "initial_history": initial_history,
                   "target_seq": target_seq}
        return out
    
    def reload(self):
        p = Path(self.paths)
        files = [f for f in p.iterdir() if f.suffix in {'.h5', '.hdf5'}]
        if not files:
            raise FileNotFoundError(f"No h5 files found in {self.paths}")

        # build index: list of (file_path, index_within_file)
        self.index.clear()
        for fpath in files:
            # print(f"DEBUG: Attempting to open H5 file: {fpath}")
            with h5py.File(fpath, "r") as h5f:
                length = h5f["pos_t"].shape[0]
                if length < self.history_window + self.min_regressive_steps:
                    print(f"Warning: file {fpath} is too short ({length} entries) and will be skipped.")
                    continue

                # valid indices
                for i in range(self.history_window, length - self.min_regressive_steps + 1):
                    self.index.append((fpath, i))
        if not self.index:
            raise FileNotFoundError(f"No valid h5 sequences found in {self.paths}")
        
    def get_data(self, fpath, step_idx, history_max_len=None):
        # no history, only one step data for closed-loop testing
        inner_idx = step_idx + self.history_window
        with h5py.File(fpath, "r") as h5f:
            length = h5f["pos_t"].shape[0]
            if inner_idx >= length:
                return None

            input_seq = []
            initial_history = []
            for k in self.input_key_list:
                if k not in h5f:
                    raise KeyError(f"Key '{k}' not found in file {fpath}")
                
                if "ref_t" in k:
                    h5f_k = h5f[k][:, :6 * self.ref_horizon]
                else:
                    h5f_k = h5f[k]
                    if history_max_len is not None:
                        if self.same_len_history:
                            history_arr = np.asarray(
                                h5f_k[inner_idx - self.history_window:inner_idx, :], dtype=np.float32)
                            initial_history.append(
                                torch.from_numpy(history_arr))  # shape: (history_window, feature_dim)
                        else:
                            start_idx = max(self.history_window, inner_idx - history_max_len)
                            history_arr = np.asarray(h5f_k[start_idx:inner_idx, :], dtype=np.float32)
                            initial_history.append(torch.from_numpy(history_arr))  # shape: (len, feature_dim)
                
                # multi-step forward data
                forward_arr = np.asarray(h5f_k[inner_idx: inner_idx + 1, :], dtype=np.float32)
                input_seq.append(torch.from_numpy(forward_arr))  # shape: (1, feature_dim)

            input_seq = torch.cat(input_seq, dim=-1)  # shape: (1, input_dim_per_step)
            if history_max_len is not None:
                initial_history = torch.cat(initial_history, dim=-1)  # shape: (len, input_dim_per_step)

            target_seq = []
            for k in self.output_key_list:
                if k not in h5f:
                    raise KeyError(f"Key '{k}' not found in file {fpath}")
                else:
                    forward_arr = np.asarray(h5f[k][inner_idx: inner_idx + 1, :], dtype=np.float32)
                target_seq.append(torch.from_numpy(forward_arr))  # shape: (1, feature_dim)
            target_seq = torch.cat(target_seq, dim=-1)  # shape: (1, target_dim)

            out = {"input_seq": input_seq,
                   "initial_history": initial_history.unsqueeze(
                       0) if history_max_len is not None else torch.empty(0),
                   "history_mask": torch.ones(1, initial_history.size(
                       0), dtype=torch.bool) if history_max_len is not None else torch.empty(0),
                   "target_seq": target_seq}
        return out
    

def collate_fn(batch):
    # batch: list of dicts from __getitem__
    # 1) stack fixed-length sequences
    input_seq = torch.stack([b['input_seq'] for b in batch], dim=0)   # (B, bptt_steps, input_dim)
    target_seq = torch.stack([b['target_seq'] for b in batch], dim=0) # (B, bptt_steps, target_dim)

    # 2) collect histories (variable length)
    histories = [b['initial_history'] for b in batch]                 # list of (len_i, feat)
    lengths = [h.size(0) for h in histories]                          # list of ints
    max_len = max(lengths)

    # 3) reverse time, pad (pad_sequence pads policy at end), then reverse back
    reversed_hist = [torch.flip(h, dims=[0]) for h in histories]      # each (len_i, feat)
    padded_reversed = pad_sequence(reversed_hist, batch_first=True)   # (B, max_len, feat), pad at end
    initial_history = torch.flip(padded_reversed, dims=[1])           # (B, max_len, feat), pad at front

    # 4) mask: True for real data, False for padded front
    device = initial_history.device
    idxs = torch.arange(max_len, device=device).unsqueeze(0)          # (1, max_len)
    lengths_tensor = torch.tensor(lengths, device=device).unsqueeze(1) # (B, 1)
    mask = idxs >= (max_len - lengths_tensor)                         # (B, max_len), bool

    return {
        "input_seq": input_seq,               # (B, bptt_steps, input_dim)
        "initial_history": initial_history,   # (B, max_len, input_dim)
        "history_mask": mask,                 # (B, max_len) bool
        "target_seq": target_seq              # (B, bptt_steps, target_dim)
    }


def get_dataloader(
    paths: Union[str, Path],
    input_key_list: List[str],
    output_key_list: List[str],
    history_window: int,
    same_len_history: bool,
    batch_size: int,
    bptt_steps: int,
    min_regressive_steps: int = 50,
    ref_horizon: int = 50,
    ref_pos_only: bool = False,
    history_max_len: int = 50,
    shuffle: bool = True,  # False, # 
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    ds = H5Dataset(
        paths,
        input_key_list,
        output_key_list,
        history_window,
        bptt_steps,
        min_regressive_steps,
        same_len_history,
        ref_horizon,
        ref_pos_only,
        history_max_len,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,  
    )
