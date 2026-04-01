"""Generate publication-quality figures for the Mini-project 2 report.

Loads saved results from models/ensemble/results/ and produces
clean PDF plots suitable for LaTeX inclusion.

Usage:
    PYTHONPATH=src python src/project2/report_figures.py [--results-dir models/ensemble/results]
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Global RC params for publication quality ──────────────────────────
mpl.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

COLORS = {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"}


def plot_geodesics_sidebyside(results_dir: str, save_path: str):
    """Figure 1: K=1 (left) and K=3 (right) geodesics, shared axes."""
    # Figure 1 is 7" wide but rendered at full \linewidth (~17cm),
    # while Figure 2 is 3.5" at 0.75\linewidth (~12.75cm).
    # Scale fonts up ~1.5x so they appear the same size on the page.
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

    for ax_idx, K in enumerate([1, 3]):
        ax = axes[ax_idx]
        path = os.path.join(results_dir, f"geodesics_K{K}_retrain0.pt")
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"K={K}\ndata missing", transform=ax.transAxes, ha="center")
            continue

        import torch
        data = torch.load(path, weights_only=True)
        all_z = data["all_z"].numpy()
        all_labels = data["all_labels"].numpy()
        curves = data["curves"].numpy()

        # Scatter
        for lab in np.unique(all_labels):
            mask = all_labels == lab
            c = COLORS[str(int(lab))]
            ax.scatter(all_z[mask, 0], all_z[mask, 1], c=c, s=8, alpha=0.35,
                       label=f"Digit {int(lab)}", edgecolors="none", rasterized=True)

        # Geodesic curves
        for ci in range(curves.shape[0]):
            ax.plot(curves[ci, :, 0], curves[ci, :, 1], "k-", linewidth=0.8, alpha=0.65)
            ax.plot(curves[ci, 0, 0], curves[ci, 0, 1], "ko", markersize=2.5, zorder=5)
            ax.plot(curves[ci, -1, 0], curves[ci, -1, 1], "ko", markersize=2.5, zorder=5)

        subtitle = "$K=1$ (single decoder)" if K == 1 else f"$K={K}$ (ensemble)"
        ax.set_xlabel("$z_1$", fontsize=22)
        ax.set_title(subtitle, fontsize=20)
        ax.tick_params(direction="in", labelsize=18)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=5)

    axes[0].set_ylabel("$z_2$", fontsize=22)
    # Legend above the plot, horizontal
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cov_report(results_dir: str, save_path: str):
    """Figure 2: CoV vs K for Euclidean and geodesic distances."""
    cov_path = os.path.join(results_dir, "cov_results.json")
    with open(cov_path) as f:
        cov_raw = json.load(f)
    cov_results = {int(k): v for k, v in cov_raw.items()}

    K_values = sorted(cov_results.keys())
    euc_means = [cov_results[K]["euclidean_cov_mean"] for K in K_values]
    euc_stds = [cov_results[K]["euclidean_cov_std"] for K in K_values]
    geo_means = [cov_results[K]["geodesic_cov_mean"] for K in K_values]
    geo_stds = [cov_results[K]["geodesic_cov_std"] for K in K_values]

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

    ax.errorbar(K_values, euc_means, yerr=euc_stds, marker="o", capsize=4,
                label="Euclidean", linewidth=1.8, markersize=6, color="#1f77b4")
    ax.errorbar(K_values, geo_means, yerr=geo_stds, marker="s", capsize=4,
                label="Geodesic", linewidth=1.8, markersize=6, color="#d62728")

    ax.set_xlabel("Number of decoders ($K$)")
    ax.set_ylabel("CoV")
    ax.set_xticks(K_values)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(direction="in")
    ax.set_xlim(0.5, 3.5)

    # Legend above the plot, horizontal, outside the axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models/ensemble/results")
    parser.add_argument("--output-dir", default="docs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_geodesics_sidebyside(
        args.results_dir,
        os.path.join(args.output_dir, "geodesics.pdf"),
    )
    plot_cov_report(
        args.results_dir,
        os.path.join(args.output_dir, "cov.pdf"),
    )

    print(f"\nAll report figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
