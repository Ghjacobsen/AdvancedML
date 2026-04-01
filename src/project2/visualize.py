"""Visualization for ensemble VAE geometry experiments.

All plot functions save to PDF and can be re-run from saved results
without retraining or recomputing geodesics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_geodesics(
    all_z: np.ndarray,
    all_labels: np.ndarray,
    curves: np.ndarray,
    title: str = "Geodesics",
    save_path: str = "geodesics.pdf",
    figsize: tuple = (8, 8),
):
    """Plot latent space with geodesic curves overlaid.

    Args:
        all_z: (N, 2) — all latent means from test set.
        all_labels: (N,) — class labels for coloring.
        curves: (num_curves, T, 2) — optimized geodesic curves.
        title: Plot title.
        save_path: Where to save the figure.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Scatter latent points colored by class
    unique_labels = np.unique(all_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 3)))
    for i, lab in enumerate(unique_labels):
        mask = all_labels == lab
        ax.scatter(all_z[mask, 0], all_z[mask, 1], c=[colors[i]], label=f"Digit {int(lab)}", alpha=0.4, s=15)

    # Plot geodesic curves
    for c in range(curves.shape[0]):
        ax.plot(curves[c, :, 0], curves[c, :, 1], "k-", linewidth=1.0, alpha=0.7)
        ax.plot(curves[c, 0, 0], curves[c, 0, 1], "ko", markersize=4)
        ax.plot(curves[c, -1, 0], curves[c, -1, 1], "ks", markersize=4)

    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cov(
    cov_results: dict,
    save_path: str = "cov_plot.pdf",
    figsize: tuple = (7, 5),
):
    """Plot CoV vs number of ensemble decoders.

    Args:
        cov_results: Dict mapping K -> {euclidean_cov_mean, geodesic_cov_mean, ...}.
        save_path: Where to save the figure.
        figsize: Figure size.
    """
    K_values = sorted(cov_results.keys())
    euc_means = [cov_results[K]["euclidean_cov_mean"] for K in K_values]
    euc_stds = [cov_results[K]["euclidean_cov_std"] for K in K_values]
    geo_means = [cov_results[K]["geodesic_cov_mean"] for K in K_values]
    geo_stds = [cov_results[K]["geodesic_cov_std"] for K in K_values]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.errorbar(K_values, euc_means, yerr=euc_stds, marker="o", capsize=4, label="Euclidean", linewidth=2)
    ax.errorbar(K_values, geo_means, yerr=geo_stds, marker="s", capsize=4, label="Geodesic", linewidth=2)

    ax.set_xlabel("Number of ensemble decoders ($K$)")
    ax.set_ylabel("Coefficient of Variation (CoV)")
    ax.set_title("Distance reliability: CoV vs ensemble size")
    ax.set_xticks(K_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_reconstructions(
    originals: np.ndarray,
    reconstructions: np.ndarray,
    n: int = 8,
    save_path: str = "reconstructions.pdf",
):
    """Plot original vs reconstructed images side by side."""
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(originals[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructions[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_geodesic_images(
    curves: np.ndarray,
    decoder,
    device: str = "cpu",
    num_curves: int = 3,
    save_path: str = "geodesic_images.pdf",
):
    """Decode points along geodesic curves to show image transitions.

    Args:
        curves: (num_curves, T, 2) curve points.
        decoder: Decoder module to decode latent points.
        device: Torch device.
        num_curves: How many curves to show.
        save_path: Where to save.
    """
    import torch

    n_show = min(num_curves, curves.shape[0])
    T = curves.shape[1]
    n_steps = min(T, 10)  # Show at most 10 steps per curve
    step_idx = np.linspace(0, T - 1, n_steps, dtype=int)

    fig, axes = plt.subplots(n_show, n_steps, figsize=(n_steps * 1.2, n_show * 1.2))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for c in range(n_show):
            z_curve = torch.tensor(curves[c, step_idx], dtype=torch.float32).to(device)
            decoded = decoder(z_curve).mean.cpu().numpy()
            for s in range(n_steps):
                axes[c, s].imshow(decoded[s, 0], cmap="gray")
                axes[c, s].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
