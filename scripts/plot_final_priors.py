#!/usr/bin/env python
"""
Generate a publication-ready 2x3 grid of prior and aggregate posterior plots
for Flow, MoG, and Gaussian VAE models.

Top row: Prior p(z)
Bottom row: Aggregate Posterior q(z|x)

Usage:
    cd AdvancedML && uv run python scripts/plot_final_priors.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from project.evaluate import load_model
from project.data import get_mnist_loaders

# ── Configuration ──────────────────────────────────────────────────────
PRIORS = ["flow", "mog", "gaussian"]
TITLES = ["Flow", "MoG", "Gaussian"]
BEST_CHECKPOINTS = {
    "flow": "models/flow/run_2/vae_flow_best.pt",
    "mog": "models/mog/run_2/vae_mog_best.pt",
    "gaussian": "models/gaussian/run_2/vae_gaussian_best.pt",
}
N_SAMPLES = 5000
OUTPUT_PATH = "reports/figures/final_priors.png"

FONTSIZE = 18
TITLE_FONTSIZE = 22


def collect_data(model, data_loader, device, n_samples):
    """Collect posterior and prior samples."""
    model.eval()
    posterior_samples = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Posterior", leave=False):
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            posterior_samples.append(z.cpu())
            labels.append(y)
            if sum(s.shape[0] for s in posterior_samples) >= n_samples:
                break

    posterior_samples = torch.cat(posterior_samples, 0)[:n_samples].numpy()
    labels = torch.cat(labels, 0)[:n_samples].numpy()

    with torch.no_grad():
        prior_samples = model.prior().sample((n_samples,)).cpu().numpy()

    return posterior_samples, prior_samples, labels


def project_pca(posterior, prior):
    """Project to 2-D via PCA fitted on joint data."""
    combined = np.vstack([posterior, prior])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    n = len(posterior)
    var_exp = pca.explained_variance_ratio_
    return combined_2d[:n], combined_2d[n:], var_exp


def main():
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader (only need training set for posterior)
    train_loader, _ = get_mnist_loaders(batch_size=256, data_dir=str(root / "data"))

    # ── Collect data per prior ─────────────────────────────────────────
    data = {}
    for prior_name in PRIORS:
        ckpt = str(root / BEST_CHECKPOINTS[prior_name])
        print(f"Loading {prior_name} from {ckpt}")
        model, _ = load_model(ckpt, device)
        post, pri, lab = collect_data(model, train_loader, device, N_SAMPLES)
        post_2d, pri_2d, var_exp = project_pca(post, pri)
        data[prior_name] = {
            "posterior_2d": post_2d,
            "prior_2d": pri_2d,
            "labels": lab,
            "var_exp": var_exp,
        }

    # ── Create figure (independent axes) ─────────────────────────────
    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 8),
    )

    sc = None  # will hold the last scatter for colorbar

    for col, (prior_name, title) in enumerate(zip(PRIORS, TITLES)):
        d = data[prior_name]
        ax_prior = axes[0, col]
        ax_post = axes[1, col]

        # ── Top row: Prior ─────────────────────────────────────────────
        ax_prior.scatter(
            d["prior_2d"][:, 0],
            d["prior_2d"][:, 1],
            c="steelblue",
            alpha=0.3,
            s=3,
            rasterized=True,
        )
        ax_prior.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")

        # ── Bottom row: Aggregate Posterior ────────────────────────────
        sc = ax_post.scatter(
            d["posterior_2d"][:, 0],
            d["posterior_2d"][:, 1],
            c=d["labels"],
            cmap="tab10",
            alpha=0.5,
            s=3,
            vmin=0,
            vmax=9,
            rasterized=True,
        )

    # ── Per-model axis limits (same for prior & posterior of each model) ─
    AXIS_LIMITS = {
        "flow": (-15, 15),
        "mog": (-8, 8),
        "gaussian": (-3, 3),
    }
    for col, prior_name in enumerate(PRIORS):
        lo, hi = AXIS_LIMITS[prior_name]
        for row in range(2):
            axes[row, col].set_xlim(lo, hi)
            axes[row, col].set_ylim(lo, hi)

    # ── Tick & label formatting ────────────────────────────────────────
    for ax in axes.flat:
        ax.tick_params(labelsize=FONTSIZE)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))

    # PC1 on bottom row only, PC2 on left column only (not bold)
    for col in range(3):
        axes[1, col].set_xlabel("PC1", fontsize=FONTSIZE)
        axes[0, col].set_ylabel("PC2", fontsize=FONTSIZE)
    for row in range(2):
        for col in range(3):
            axes[row, col].set_ylabel("")

    # ── Row labels: horizontal text rotated 0° (read like x-axis) ──────
    axes[0, 0].set_ylabel("Prior", fontsize=FONTSIZE,
                           labelpad=30, fontweight="bold", rotation=0)
    axes[1, 0].set_ylabel("Aggregate\nPosterior", fontsize=FONTSIZE,
                           labelpad=30, fontweight="bold", rotation=0)

    # ── Horizontal colorbar below the figure ───────────────────────────
    cbar = fig.colorbar(
        sc,
        ax=axes[1, :].tolist(),
        orientation="horizontal",
        location="bottom",
        shrink=1.0,
        pad=0.15,
        aspect=40,
    )
    cbar.set_ticks(np.arange(0, 10))
    cbar.set_label("Digit", fontsize=FONTSIZE, fontweight = "bold")
    cbar.ax.tick_params(labelsize=FONTSIZE)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    out = root / OUTPUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
