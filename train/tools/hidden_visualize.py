import umap
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA

from config.parser import Config
from offline_train.tools.trainer import DynamicsTrainer


def visualize_rnn_hidden_pca_umap(
        hidden,
        pca_dim=30,
        umap_n_neighbors=15,
        umap_min_dist=0.1,
        interactive=True,
        top_k_neurons=80,
        random_state=42,
):
    """
    Visualize a single-layer RNN hidden state sequence using PCA -> UMAP.

    Parameters
    ----------
    hidden : np.ndarray or torch.Tensor
        Hidden states with shape (T, hidden_dim). If a torch.Tensor is provided,
        it will be converted to numpy via .detach().cpu().numpy().
    pca_dim : int
        Number of PCA components to keep before UMAP. Set to None to skip PCA.
    umap_n_neighbors : int
        UMAP n_neighbors parameter (controls local vs global structure).
    umap_min_dist : float
        UMAP min_dist parameter (controls tightness of clusters).
    interactive : bool
        If True, show an interactive Plotly figure in addition to static plots.
    top_k_neurons : int
        Number of top-variance neurons to show in the heatmap.
    random_state : int
        Random seed for reproducibility.
    """

    # If input is a PyTorch tensor, convert to numpy
    try:
        import torch
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach().cpu().numpy()
    except Exception:
        # torch not installed or hidden not a torch tensor; assume numpy array
        pass

    # Ensure shape is (T, hidden_dim)
    assert hidden.ndim == 2, "hidden must be shape (T, hidden_dim)"

    T = hidden.shape[0]

    # 1) Standardize each feature (neuron) across time:
    #    subtract mean and divide by std to avoid any single dimension dominating.
    X = hidden.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-8)

    # 2) Optional PCA pre-reduction to speed up UMAP and remove noise.
    #    PCA preserves global variance and is fast.
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_pca = pca.fit_transform(X)  # shape (T, pca_dim)
    else:
        X_pca = X  # skip PCA if pca_dim is None or >= original dim

    # 3) UMAP embedding to 2D
    #    n_neighbors controls local vs global; min_dist controls cluster tightness.
    umapper = umap.UMAP(n_neighbors=umap_n_neighbors,
                       min_dist=umap_min_dist,
                       n_components=2,
                       random_state=random_state)
    X_umap = umapper.fit_transform(X_pca)  # shape (T, 2)

    # 4) Static Matplotlib plot: scatter + trajectory lines + sampled arrows
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=np.arange(T), cmap='viridis', s=30)
    # Draw a faint line connecting points to show the trajectory
    plt.plot(X_umap[:, 0], X_umap[:, 1], color='gray', linewidth=0.8, alpha=0.6)

    # Draw arrows at sampled intervals to indicate time direction
    step = max(1, T // 20)  # draw about 20 arrows at most
    for i in range(0, T - step, step):
        dx = X_umap[i + step, 0] - X_umap[i, 0]
        dy = X_umap[i + step, 1] - X_umap[i, 1]
        # Use a small head width so arrows don't clutter the plot
        plt.arrow(X_umap[i, 0], X_umap[i, 1], dx, dy,
                  shape='full', lw=0, length_includes_head=True,
                  head_width=0.02, color='k', alpha=0.6)

    plt.colorbar(sc, label='time step')
    plt.title('RNN hidden states: PCA -> UMAP trajectory')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.tight_layout()
    plt.show()

    # 5) Interactive Plotly visualization (optional)
    if interactive:
        df = {
            'x': X_umap[:, 0],
            'y': X_umap[:, 1],
            'time': np.arange(T)
        }
        # Scatter with color mapped to time and hover showing the time index
        fig = px.scatter(df, x='x', y='y', color='time',
                         color_continuous_scale='viridis',
                         title='Interactive UMAP trajectory (time colored)',
                         labels={'color': 'time'})
        # Add a line trace to show the trajectory path
        fig.add_traces(px.line(df, x='x', y='y').data)
        fig.update_traces(marker=dict(size=6))
        fig.show()

    # 6) Heatmap of top-k neurons by variance across time
    #    This helps identify which neurons change most over the sequence.
    neuron_var = X.var(axis=0)
    top_k = min(top_k_neurons, X.shape[1])
    top_idx = np.argsort(neuron_var)[-top_k:]
    heat_data = X[:, top_idx]  # shape (T, top_k)

    plt.figure(figsize=(10, 6))
    # Transpose so rows correspond to neurons and columns to time steps
    sns.heatmap(heat_data.T, cmap='RdBu_r', center=0, xticklabels=max(1, T // 50), yticklabels=False)
    plt.xlabel('time step')
    plt.ylabel(f'top {top_k} neurons by variance')
    plt.title('time x neuron heatmap (top variance neurons)')
    plt.tight_layout()
    plt.show()

    # Return embeddings and indices for further analysis if needed
    return {
        'umap_2d': X_umap,          # shape (T, 2)
        'pca_repr': X_pca,          # shape (T, pca_dim) or (T, hidden_dim) if PCA skipped
        'top_neuron_indices': top_idx
    }


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parent.parent / "config"
    cfg_model = Config.load(cfg_path / "models_config.yaml")
    cfg_training = Config.load(cfg_path / "training_config.yaml")

    trainer = DynamicsTrainer(cfg_model, cfg_training)
    
    for batch in trainer.train_loader:
        batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
        batch_info, last_sample_print, sampled_pred, sampled_tgt, sampled_ctl = trainer.evaluate(batch)
        sampled_h = trainer.dynamics.sampled_h  # TODO
                                
    for batch in trainer.val_loader:
        batch = {k: v.to(trainer.device, non_blocking=True) for k, v in batch.items()}
        batch_info, last_sample_print, sampled_pred, sampled_tgt, sampled_ctl = trainer.evaluate(batch)

