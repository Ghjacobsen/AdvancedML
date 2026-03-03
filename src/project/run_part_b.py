"""
Part B experiment runner: Sampling quality of generative models.

Trains and evaluates three generative models on MNIST:
  1. DDPM (image-space diffusion) on standard MNIST
  2. Latent DDPM (Gaussian VAE + latent-space diffusion) on standard MNIST
  3. Best VAE from Part A (chosen prior, trained on binarised MNIST)

Produces:
  - 4-sample grids for each model
  - FID scores for all models
  - FID vs T (sampling steps) for Latent DDPM
  - Wall-clock sampling times
  - Distribution comparison plot (prior / DDPM latent / aggregate posterior)
  - JSON results summary

Usage:
    uv run python src/project/run_part_b.py                 # full run
    uv run python src/project/run_part_b.py --quick         # 5-epoch test
    uv run python src/project/run_part_b.py --skip-training # eval only
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from project.data import get_standard_mnist_loaders, get_full_test_loader
from project.ddpm import DDPM, UNet, train_ddpm, load_ddpm
from project.latent_ddpm import (
    GaussianVAE,
    LatentDDPM,
    train_gaussian_vae,
    train_latent_ddpm,
    load_gaussian_vae,
    load_latent_ddpm,
)
from project.evaluate import load_model as load_part_a_model
from project.fid import compute_fid


# =============================================================================
# Helpers
# =============================================================================


def get_real_images(n_samples: int, data_dir: str, device: torch.device) -> torch.Tensor:
    """
    Load real standard MNIST images (normalised to [-1, 1]) for FID evaluation.

    Args:
        n_samples: Number of images to return.
        data_dir: MNIST data directory.
        device: Target device.

    Returns:
        Real images, shape (n_samples, 1, 28, 28), in [-1, 1].
    """
    # squeeze_channel=False keeps (1, 28, 28) shape needed for FID
    _, test_loader = get_standard_mnist_loaders(
        batch_size=256, data_dir=data_dir, squeeze_channel=False
    )
    images = []
    for x, _ in test_loader:
        images.append(x)
        if sum(img.shape[0] for img in images) >= n_samples:
            break
    return torch.cat(images, dim=0)[:n_samples].to(device)


def measure_sampling_time(
    sample_fn,
    n_samples: int = 500,
    n_trials: int = 3,
) -> float:
    """
    Measure wall-clock sampling throughput in samples/second.

    Args:
        sample_fn: Callable that generates n_samples images/latents.
        n_samples: Number of samples per trial.
        n_trials: Number of timing trials.

    Returns:
        Mean samples per second across trials.
    """
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        sample_fn(n_samples)
        elapsed = time.perf_counter() - t0
        times.append(n_samples / elapsed)
    return float(np.mean(times))


def save_sample_grid(
    images: torch.Tensor,
    save_path: str,
    title: str,
    n_show: int = 4,
):
    """
    Save a 1×n_show grid of images to disk.

    Args:
        images: Image tensor, shape (N, 28, 28) or (N, 1, 28, 28), in [-1, 1] or [0, 1].
        save_path: Output file path.
        title: Figure title.
        n_show: Number of images to show.
    """
    if images.dim() == 4:
        images = images.squeeze(1)
    # Scale from [-1,1] or [0,1] to [0,1] for display
    images = images.cpu().float()
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_fid_vs_T(
    fid_by_steps: Dict[int, float],
    save_path: str,
):
    """
    Plot FID score vs number of DDIM sampling steps for the Latent DDPM.

    Args:
        fid_by_steps: Mapping from n_steps to FID score.
        save_path: Output file path.
    """
    steps = sorted(fid_by_steps.keys())
    fids = [fid_by_steps[s] for s in steps]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, fids, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.set_xlabel("Sampling steps T", fontsize=12)
    ax.set_ylabel("FID", fontsize=12)
    ax.set_title("Latent DDPM: FID vs Sampling Steps", fontsize=13)
    ax.grid(True, alpha=0.3)
    # Annotate T=0 specially
    if 0 in fid_by_steps:
        ax.annotate(
            "T=0 (prior only)",
            xy=(0, fid_by_steps[0]),
            xytext=(10, fid_by_steps[0] * 0.95),
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_distributions(
    vae: GaussianVAE,
    latent_ddpm: LatentDDPM,
    train_loader,
    device: torch.device,
    save_path: str,
    n_samples: int = 3000,
):
    """
    Compare three latent distributions in 2-D (PCA if latent_dim > 2):
      1. VAE prior: z ~ N(0, I)
      2. Latent DDPM: z ~ p_DDPM(z) (full T steps)
      3. Aggregate posterior: z ~ q(z|x) averaged over training data

    Args:
        vae: Trained GaussianVAE.
        latent_ddpm: Trained LatentDDPM.
        train_loader: Training data loader.
        device: Compute device.
        save_path: Output file path.
        n_samples: Number of samples per distribution.
    """
    vae.eval()
    latent_ddpm.eval()

    # 1. VAE prior
    with torch.no_grad():
        prior_z = torch.randn(n_samples, vae.latent_dim, device=device).cpu().numpy()

    # 2. Latent DDPM samples (full T steps)
    with torch.no_grad():
        ddpm_z = latent_ddpm.sample(n_samples, device).cpu().numpy()

    # 3. Aggregate posterior
    post_z, labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            post_z.append(vae.encode_mean(x).cpu())
            labels.append(y)
            if sum(p.shape[0] for p in post_z) >= n_samples:
                break
    post_z = torch.cat(post_z, dim=0)[:n_samples].numpy()
    labels = torch.cat(labels, dim=0)[:n_samples].numpy()

    # PCA projection to 2-D
    latent_dim = prior_z.shape[1]
    if latent_dim > 2:
        combined = np.vstack([prior_z, ddpm_z, post_z])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        prior_2d = combined_2d[:n_samples]
        ddpm_2d = combined_2d[n_samples : 2 * n_samples]
        post_2d = combined_2d[2 * n_samples :]
        var_exp = pca.explained_variance_ratio_.sum()
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        suffix = f"\n(PCA: {var_exp:.1%} variance explained)"
    else:
        prior_2d, ddpm_2d, post_2d = prior_z, ddpm_z, post_z
        xlabel, ylabel, suffix = "$z_1$", "$z_2$", ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(prior_2d[:, 0], prior_2d[:, 1], c="steelblue", alpha=0.3, s=3)
    axes[0].set_title(f"VAE Prior N(0,I){suffix}", fontsize=11)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(ddpm_2d[:, 0], ddpm_2d[:, 1], c="coral", alpha=0.3, s=3)
    axes[1].set_title(f"Latent DDPM Distribution{suffix}", fontsize=11)
    axes[1].set_xlabel(xlabel)
    axes[1].grid(True, alpha=0.3)

    sc = axes[2].scatter(post_2d[:, 0], post_2d[:, 1], c=labels, cmap="tab10", alpha=0.4, s=3)
    axes[2].set_title(f"Aggregate Posterior q(z|x){suffix}", fontsize=11)
    axes[2].set_xlabel(xlabel)
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[2], label="Digit")

    plt.suptitle("Latent Space Comparison: Prior vs DDPM vs Posterior", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main experiment runner
# =============================================================================


def run_part_b_experiments(
    ddpm_epochs: int = 100,
    vae_epochs: int = 50,
    latent_ddpm_epochs: int = 100,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    ddpm_base_channels: int = 64,
    T: int = 1000,
    batch_size: int = 128,
    n_fid_samples: int = 5000,
    part_a_prior: str = "mog",
    output_dir: str = "./reports",
    models_dir: str = "./models",
    data_dir: str = "./data",
    skip_training: bool = False,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run all Part B experiments.

    Args:
        ddpm_epochs: Training epochs for image-space DDPM.
        vae_epochs: Training epochs for Gaussian VAE.
        latent_ddpm_epochs: Training epochs for Latent DDPM.
        latent_dim: VAE and Latent DDPM latent dimension.
        hidden_dim: Hidden layer width for VAE encoder/decoder.
        ddpm_base_channels: U-Net base channel count.
        T: Number of diffusion timesteps.
        batch_size: Mini-batch size.
        n_fid_samples: Number of samples for FID computation.
        part_a_prior: Which Part A prior to compare ('gaussian', 'mog', or 'flow').
        output_dir: Directory for report figures and JSON.
        models_dir: Directory for model checkpoints.
        data_dir: MNIST data directory.
        skip_training: If True, load existing checkpoints and skip training.
        device: Compute device (auto-detected if None).

    Returns:
        results: Dict with FID scores and sampling times.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ddpm_dir = str(Path(models_dir) / "ddpm")
    vae_dir = str(Path(models_dir) / "gaussian_vae")
    latent_ddpm_dir = str(Path(models_dir) / "latent_ddpm")

    print("=" * 70)
    print("PART B: SAMPLING QUALITY OF GENERATIVE MODELS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"FID samples: {n_fid_samples}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Train / load DDPM
    # ------------------------------------------------------------------
    ddpm_ckpt = Path(ddpm_dir) / "ddpm_best.pt"
    if skip_training and ddpm_ckpt.exists():
        print("\n[1/3] Loading existing DDPM checkpoint...")
        ddpm = load_ddpm(str(ddpm_ckpt), device)
    else:
        print("\n[1/3] Training image-space DDPM...")
        ddpm, _ = train_ddpm(
            epochs=ddpm_epochs,
            batch_size=batch_size,
            base_channels=ddpm_base_channels,
            T=T,
            save_dir=ddpm_dir,
            data_dir=data_dir,
            device=device,
        )

    # ------------------------------------------------------------------
    # 2. Train / load Gaussian VAE + Latent DDPM
    # ------------------------------------------------------------------
    vae_ckpt = Path(vae_dir) / "gaussian_vae_best.pt"
    latent_ckpt = Path(latent_ddpm_dir) / "latent_ddpm_best.pt"

    if skip_training and vae_ckpt.exists():
        print("\n[2/3] Loading existing Gaussian VAE checkpoint...")
        gaussian_vae = load_gaussian_vae(str(vae_ckpt), device)
    else:
        print("\n[2/3] Training Gaussian VAE...")
        gaussian_vae, _ = train_gaussian_vae(
            epochs=vae_epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            save_dir=vae_dir,
            data_dir=data_dir,
            device=device,
        )

    if skip_training and latent_ckpt.exists():
        print("      Loading existing Latent DDPM checkpoint...")
        latent_ddpm = load_latent_ddpm(str(latent_ckpt), device)
    else:
        print("      Training Latent DDPM...")
        latent_ddpm, _ = train_latent_ddpm(
            vae=gaussian_vae,
            latent_dim=latent_dim,
            epochs=latent_ddpm_epochs,
            batch_size=batch_size * 2,
            T=T,
            save_dir=latent_ddpm_dir,
            data_dir=data_dir,
            device=device,
        )

    # ------------------------------------------------------------------
    # 3. Load best Part A VAE
    # ------------------------------------------------------------------
    print(f"\n[3/3] Loading Part A VAE ({part_a_prior} prior)...")
    part_a_ckpt = Path(models_dir) / part_a_prior / "run_0" / f"vae_{part_a_prior}_best.pt"
    if not part_a_ckpt.exists():
        # Fallback to run_0 final
        part_a_ckpt = Path(models_dir) / part_a_prior / "run_0" / f"vae_{part_a_prior}_final.pt"
    if part_a_ckpt.exists():
        part_a_vae, _ = load_part_a_model(str(part_a_ckpt), device)
        has_part_a = True
        print(f"      Loaded: {part_a_ckpt}")
    else:
        print(f"      WARNING: Part A checkpoint not found at {part_a_ckpt}. Skipping Part A VAE.")
        has_part_a = False

    # ------------------------------------------------------------------
    # Real images for FID
    # ------------------------------------------------------------------
    print(f"\nLoading {n_fid_samples} real images for FID...")
    x_real = get_real_images(n_fid_samples, data_dir, device)
    print(f"Real images shape: {x_real.shape}")

    # ------------------------------------------------------------------
    # Generate samples
    # ------------------------------------------------------------------
    print("\nGenerating samples...")

    with torch.no_grad():
        ddpm_samples = ddpm.sample(n_fid_samples, device)  # (N, 1, 28, 28), [-1,1]

    with torch.no_grad():
        z_latent = latent_ddpm.sample(n_fid_samples, device)
        latent_ddpm_imgs = gaussian_vae.decode(z_latent)  # (N, 28, 28)
        latent_ddpm_imgs = latent_ddpm_imgs.unsqueeze(1).clamp(-1.0, 1.0)  # (N, 1, 28, 28)

    if has_part_a:
        with torch.no_grad():
            part_a_samples = part_a_vae.sample(n_fid_samples)  # (N, 28, 28), {0,1}
            part_a_imgs = (part_a_samples.unsqueeze(1).float() * 2.0 - 1.0).to(device)  # [-1,1]

    # ------------------------------------------------------------------
    # Sample grids (4 images each)
    # ------------------------------------------------------------------
    save_sample_grid(
        ddpm_samples[:4],
        str(figures_dir / "ddpm_samples.png"),
        "DDPM Samples (Standard MNIST)",
    )
    save_sample_grid(
        latent_ddpm_imgs[:4],
        str(figures_dir / "latent_ddpm_samples.png"),
        "Latent DDPM Samples (Gaussian VAE + Latent Diffusion)",
    )
    if has_part_a:
        save_sample_grid(
            part_a_imgs[:4],
            str(figures_dir / f"part_a_{part_a_prior}_samples_b.png"),
            f"Part A VAE Samples ({part_a_prior.upper()} prior)",
        )

    # ------------------------------------------------------------------
    # FID scores
    # ------------------------------------------------------------------
    classifier_ckpt = str(Path(models_dir) / "mnist_classifier.pth")
    print(f"\nComputing FID scores (classifier: {classifier_ckpt})...")

    fid_ddpm = compute_fid(x_real, ddpm_samples, device=str(device), classifier_ckpt=classifier_ckpt)
    print(f"FID — DDPM:        {fid_ddpm:.2f}")

    fid_latent = compute_fid(x_real, latent_ddpm_imgs, device=str(device), classifier_ckpt=classifier_ckpt)
    print(f"FID — Latent DDPM: {fid_latent:.2f}")

    fid_part_a = None
    if has_part_a:
        fid_part_a = compute_fid(x_real, part_a_imgs, device=str(device), classifier_ckpt=classifier_ckpt)
        print(f"FID — Part A VAE:  {fid_part_a:.2f}")

    # ------------------------------------------------------------------
    # Latent DDPM FID vs T (sampling steps)
    # ------------------------------------------------------------------
    T_values = [0, 10, 50, 200, T]
    fid_by_steps: Dict[int, float] = {}

    print("\nLatent DDPM FID vs sampling steps T:")
    for n_steps in T_values:
        with torch.no_grad():
            z = latent_ddpm.sample(n_fid_samples, device, n_steps=n_steps)
            imgs = gaussian_vae.decode(z).unsqueeze(1).clamp(-1.0, 1.0)
        fid_val = compute_fid(x_real, imgs, device=str(device), classifier_ckpt=classifier_ckpt)
        fid_by_steps[n_steps] = float(fid_val)
        label = f"T={n_steps}" + (" (prior only)" if n_steps == 0 else "")
        print(f"  {label:<25} FID: {fid_val:.2f}")

    plot_fid_vs_T(fid_by_steps, str(figures_dir / "latent_ddpm_fid_vs_T.png"))

    # ------------------------------------------------------------------
    # Sampling times
    # ------------------------------------------------------------------
    print("\nMeasuring sampling times (samples/second)...")

    ddpm_speed = measure_sampling_time(
        lambda n: ddpm.sample(n, device), n_samples=100, n_trials=2
    )
    print(f"  DDPM:        {ddpm_speed:.2f} samples/s")

    def sample_latent(n):
        z = latent_ddpm.sample(n, device)
        return gaussian_vae.decode(z)

    latent_speed = measure_sampling_time(sample_latent, n_samples=200, n_trials=2)
    print(f"  Latent DDPM: {latent_speed:.2f} samples/s")

    part_a_speed = None
    if has_part_a:
        part_a_speed = measure_sampling_time(
            lambda n: part_a_vae.sample(n), n_samples=500, n_trials=2
        )
        print(f"  Part A VAE:  {part_a_speed:.2f} samples/s")

    # ------------------------------------------------------------------
    # Distribution comparison plot
    # ------------------------------------------------------------------
    print("\nGenerating latent distribution comparison plot...")
    train_loader, _ = get_standard_mnist_loaders(
        batch_size=256, data_dir=data_dir, squeeze_channel=True
    )
    plot_latent_distributions(
        gaussian_vae,
        latent_ddpm,
        train_loader,
        device,
        save_path=str(figures_dir / "latent_distribution_comparison.png"),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART B — RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'FID':>10} {'Speed (samples/s)':>20}")
    print("-" * 57)
    print(f"{'DDPM':<25} {fid_ddpm:>10.2f} {ddpm_speed:>20.2f}")
    print(f"{'Latent DDPM':<25} {fid_latent:>10.2f} {latent_speed:>20.2f}")
    if has_part_a and fid_part_a is not None and part_a_speed is not None:
        print(f"{'Part A VAE (' + part_a_prior + ')':<25} {fid_part_a:>10.2f} {part_a_speed:>20.2f}")
    print("=" * 70)

    print("\nLatent DDPM FID vs T:")
    for n_steps, fid_val in sorted(fid_by_steps.items()):
        print(f"  T={n_steps:<6}  FID: {fid_val:.2f}")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ddpm_epochs": ddpm_epochs,
            "vae_epochs": vae_epochs,
            "latent_ddpm_epochs": latent_ddpm_epochs,
            "latent_dim": latent_dim,
            "T": T,
            "n_fid_samples": n_fid_samples,
            "part_a_prior": part_a_prior,
        },
        "fid": {
            "ddpm": float(fid_ddpm),
            "latent_ddpm": float(fid_latent),
            "part_a_vae": float(fid_part_a) if fid_part_a is not None else None,
        },
        "fid_vs_T": {str(k): v for k, v in fid_by_steps.items()},
        "sampling_speed": {
            "ddpm_samples_per_sec": ddpm_speed,
            "latent_ddpm_samples_per_sec": latent_speed,
            "part_a_vae_samples_per_sec": part_a_speed,
        },
    }

    results_path = Path(output_dir) / "part_b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


# =============================================================================
# CLI entry-point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run Part B experiments: sampling quality of generative models"
    )
    parser.add_argument("--ddpm-epochs", type=int, default=100)
    parser.add_argument("--vae-epochs", type=int, default=50)
    parser.add_argument("--latent-ddpm-epochs", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ddpm-base-channels", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-fid-samples", type=int, default=5000)
    parser.add_argument("--part-a-prior", type=str, default="mog",
                        choices=["gaussian", "mog", "flow"])
    parser.add_argument("--output-dir", type=str, default="./reports")
    parser.add_argument("--models-dir", type=str, default="./models")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load existing checkpoints instead of retraining")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 epochs, 100 FID samples")
    args = parser.parse_args()

    if args.quick:
        args.ddpm_epochs = 5
        args.vae_epochs = 5
        args.latent_ddpm_epochs = 5
        args.n_fid_samples = 100
        args.T = 100
        print("*** QUICK TEST MODE: 5 epochs, T=100, 100 FID samples ***\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_part_b_experiments(
        ddpm_epochs=args.ddpm_epochs,
        vae_epochs=args.vae_epochs,
        latent_ddpm_epochs=args.latent_ddpm_epochs,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        ddpm_base_channels=args.ddpm_base_channels,
        T=args.T,
        batch_size=args.batch_size,
        n_fid_samples=args.n_fid_samples,
        part_a_prior=args.part_a_prior,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        skip_training=args.skip_training,
        device=device,
    )


if __name__ == "__main__":
    main()
