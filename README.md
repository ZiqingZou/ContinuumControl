# ContinuumControl

Learning-based dynamics modeling and tracking control for tendon-driven continuum robots.

## Overview

ContinuumControl provides a PyTorch framework for data-driven modeling and control of tendon-driven continuum manipulators. It consists of two core components:

- **Dynamics Model**: Recurrently predicts the next observation given the current observation and control input: `next_obs, h = f(obs, u, h)`. Supports MLP, RNN, GRU, and LSTM architectures with layer normalization.
- **Policy Model**: Generates control commands to track reference end-effector trajectories: `u = π(obs, ref, h)`. Supports deterministic and stochastic policies with multi-step output.

Both models are trained with truncated backpropagation through time (BPTT) and support online fine-tuning on real robots via socket communication.

## Project Structure

```
ContinuumControl/
├── config/                  # Model and training configurations (YAML)
│   ├── models_config.yaml
│   ├── training_config.yaml
│   └── parser.py           # Hierarchical YAML config parser
├── data/
│   ├── description.txt     # Dataset column description
│   ├── analysis/           # Data preprocessing, visualization, and statistics
│   └── samples_dataset/    # Sample dataset (raw CSV logs)
├── models/
│   ├── dynamics.py         # Dynamics model
│   ├── policy.py           # Policy model
│   ├── networks/           # Network architectures (MLP, RNN, GRU, LSTM)
│   └── tools/              # Get network, normalization, weight initialization
├── train/
│   ├── trainer.py          # DynamicsTrainer and PolicyTrainer
│   ├── data_loader.py      # H5 Dataset and DataLoader
│   ├── offline_train.py    # Offline training script
│   ├── online_train.py     # Online training with real robot (socket)
│   ├── online_train_control.py  # Online training with control loop
│   ├── mpc_train/          # MPC-based policy optimization
│   └── tools/              # Loss functions, plotting, model export, 3D transforms
├── outcome/                # Training output figures (.gitignored)
├── environment.yml
└── LICENSE
```

## Installation

```bash
conda env create -f environment.yml
conda activate continuum
```

Dependencies: Python 3.10, PyTorch 2.4.1 (CUDA 12.1), numpy, pandas, scipy, h5py, matplotlib, wandb.

## Data Format

Raw data is stored as CSV log files under `data/<data_source>/logs/`. Each CSV file contains time-series data recorded from the continuum robot, including:

- End-effector pose (4×4 transformation matrix, flattened)
- Motor lengths, velocities, and torques (9 tendons)
- Motor velocity control commands (9 tendons)
- Marker positions (11 markers, optional)

See `data/description.txt` for the full column specification.

### Preprocessing

```bash
python -m data.analysis.pre_process
```

This converts raw CSV logs into HDF5 files with computed features (rotation vectors, deltas, reference trajectories, etc.).

## Usage

### Offline Training

**Train a dynamics model:**

```bash
python train/offline_train.py --model dynamics
```

**Train a policy model:**

```bash
python train/offline_train.py --model policy
```

### Online Fine-tuning

Fine-tune models on a real robot via socket communication:

```bash
python train/online_train.py
```

### MPC Optimization

Optimize control sequences using a learned dynamics model:

```bash
python train/mpc_train/mpc_run.py
```

### Model Export

Export trained models to TorchScript for deployment:

```bash
python -m train.tools.export_model
```

## Configuration

All hyperparameters are managed via YAML configs:

- `config/models_config.yaml` — Model architecture, observation types, normalization statistics
- `config/training_config.yaml` — Training hyperparameters, loss weights, data sources, online settings

Observation type strings (e.g., `"Tlv"`) define which quantities the model observes:
- `T` — End-effector transform (rotation vector + position)
- `l` — Motor lengths
- `v` — Motor velocities
- `q` — Motor torques

## Citation

If you find this work useful, please cite our papers:

```bibtex
@article{zou2026dynamics,
  title={Learning-Based Dynamics Modeling and Robust Control for Tendon-Driven Continuum Robots},
  author={Ziqing Zou},
  journal={arXiv preprint arXiv:2604.25691},
  year={2026}
}

@article{zou2026reference,
  title={Reference-Augmented Learning for Precise Tracking Policy of Tendon-Driven Continuum Robots},
  author={Ziqing Zou},
  journal={arXiv preprint arXiv:2604.25698},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
