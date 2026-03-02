"""
Visualization module for VAE analysis.

This module provides:
- plot_prior_posterior(): Plot prior vs aggregate posterior comparison
- plot_reconstructions(): Show original and reconstructed images
- plot_samples(): Show samples from the model
- plot_training_curves(): Plot training loss and test ELBO
- create_all_plots(): Generate all plots for Part A report
"""

from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def plot_prior_posterior(
    model,
    data_loader: DataLoader,
    device: torch.device,
    save_path: str = "prior_posterior.png",
    n_samples: int = 5000,
    prior_name: str = "Prior",
):
    """
    Plot prior vs aggregate posterior distributions.
    
    Creates a 3-panel figure:
    1. Aggregate posterior q(z|x) colored by digit class
    2. Prior p(z)
    3. Overlay comparison
    
    Parameters:
    model: [VAE] The VAE model
    data_loader: [DataLoader] Data loader for posterior samples
    device: [torch.device] Device
    save_path: [str] Path to save figure
    n_samples: [int] Number of samples to use
    prior_name: [str] Name of the prior for title
    """
    from tqdm import tqdm
    from sklearn.decomposition import PCA
    
    model.eval()
    
    # Collect posterior samples
    posterior_samples = []
    labels = []
    
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Collecting posterior samples", leave=False):
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            posterior_samples.append(z.cpu())
            labels.append(y)
            
            if sum(s.shape[0] for s in posterior_samples) >= n_samples:
                break
    
    posterior_samples = torch.cat(posterior_samples, dim=0)[:n_samples].numpy()
    labels = torch.cat(labels, dim=0)[:n_samples].numpy()
    
    # Sample from prior
    with torch.no_grad():
        prior_samples = model.prior().sample((n_samples,)).cpu().numpy()
    
    # Project to 2D if needed
    if posterior_samples.shape[1] > 2:
        combined = np.vstack([posterior_samples, prior_samples])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        
        posterior_2d = combined_2d[:n_samples]
        prior_2d = combined_2d[n_samples:]
        
        var_explained = pca.explained_variance_ratio_.sum()
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
        title_suffix = f"\n(PCA: {var_explained:.1%} variance explained)"
    else:
        posterior_2d = posterior_samples
        prior_2d = prior_samples
        xlabel = r'$z_1$'
        ylabel = r'$z_2$'
        title_suffix = ""
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Posterior
    scatter = axes[0].scatter(posterior_2d[:, 0], posterior_2d[:, 1],
                               c=labels, cmap='tab10', alpha=0.5, s=3)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f'Aggregate Posterior $q(z|x)${title_suffix}')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Digit')
    
    # Prior
    axes[1].scatter(prior_2d[:, 0], prior_2d[:, 1],
                   c='steelblue', alpha=0.3, s=3)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'{prior_name} $p(z)${title_suffix}')
    axes[1].grid(True, alpha=0.3)
    
    # Overlay
    axes[2].scatter(prior_2d[:, 0], prior_2d[:, 1],
                   c='steelblue', alpha=0.2, s=3, label='Prior')
    axes[2].scatter(posterior_2d[:, 0], posterior_2d[:, 1],
                   c='red', alpha=0.2, s=3, label='Posterior')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Prior vs Posterior{title_suffix}')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(markerscale=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_samples(
    model,
    device: torch.device,
    n_samples: int = 64,
    save_path: str = "samples.png",
    title: str = "Generated Samples",
):
    """
    Plot samples generated from the model.
    
    Parameters:
    model: [VAE] The VAE model
    device: [torch.device] Device
    n_samples: [int] Number of samples to generate
    save_path: [str] Path to save figure
    title: [str] Figure title
    """
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(n_samples).cpu()
    
    # Determine grid size
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_reconstructions(
    model,
    data_loader: DataLoader,
    device: torch.device,
    n_samples: int = 10,
    save_path: str = "reconstructions.png",
    title: str = "Reconstructions",
):
    """
    Plot original images and their reconstructions.
    
    Parameters:
    model: [VAE] The VAE model
    data_loader: [DataLoader] Data loader
    device: [torch.device] Device
    n_samples: [int] Number of samples to show
    save_path: [str] Path to save figure
    title: [str] Figure title
    """
    model.eval()
    
    # Get a batch
    x, _ = next(iter(data_loader))
    x = x[:n_samples].to(device)
    
    # Reconstruct
    with torch.no_grad():
        q = model.encoder(x)
        z = q.mean  # Use mean for cleaner reconstructions
        recon = model.decoder(z).probs  # Get Bernoulli probabilities
    
    x = x.cpu()
    recon = recon.cpu()
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.2, 2.5))
    
    for i in range(n_samples):
        axes[0, i].imshow(x[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        axes[1, i].imshow(recon[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=10)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "training_curves.png",
    title: str = "Training Curves",
):
    """
    Plot training loss and test ELBO curves.
    
    Parameters:
    history: [dict] Dictionary with 'train_loss' and 'test_elbo' lists
    save_path: [str] Path to save figure
    title: [str] Figure title
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss (neg ELBO)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Test ELBO
    axes[1].plot(epochs, history['test_elbo'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test ELBO')
    axes[1].set_title('Test ELBO')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_elbo_comparison(
    results: Dict[str, Dict],
    save_path: str = "elbo_comparison.png",
):
    """
    Create bar plot comparing test ELBO across priors.
    
    Parameters:
    results: [dict] Results from evaluate.compare_all_priors()
    save_path: [str] Path to save figure
    """
    priors = list(results.keys())
    means = [results[p]['mean'] for p in priors]
    stds = [results[p]['std'] for p in priors]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    bars = ax.bar(priors, means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Prior Type', fontsize=12)
    ax.set_ylabel('Test ELBO', fontsize=12)
    ax.set_title('Test Set Log-Likelihood Comparison\n(Mean ± Std over multiple runs)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_all_plots_for_prior(
    model,
    data_loader: DataLoader,
    device: torch.device,
    prior_name: str,
    history: Dict[str, List[float]],
    output_dir: str = "./reports/figures",
):
    """
    Generate all plots for a single prior type.
    
    Parameters:
    model: [VAE] Trained VAE model
    data_loader: [DataLoader] Data loader
    device: [torch.device] Device
    prior_name: [str] Name of the prior (gaussian, mog, flow)
    history: [dict] Training history
    output_dir: [str] Output directory for figures
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    prior_display = {
        'gaussian': 'Standard Gaussian',
        'mog': 'Mixture of Gaussians',
        'flow': 'Normalizing Flow'
    }.get(prior_name, prior_name)
    
    # 1. Prior vs Posterior
    plot_prior_posterior(
        model, data_loader, device,
        save_path=f"{output_dir}/{prior_name}_prior_posterior.png",
        prior_name=prior_display
    )
    
    # 2. Generated samples
    plot_samples(
        model, device, n_samples=64,
        save_path=f"{output_dir}/{prior_name}_samples.png",
        title=f"Generated Samples ({prior_display} Prior)"
    )
    
    # 3. Reconstructions
    plot_reconstructions(
        model, data_loader, device,
        save_path=f"{output_dir}/{prior_name}_reconstructions.png",
        title=f"Reconstructions ({prior_display} Prior)"
    )
    
    # 4. Training curves
    plot_training_curves(
        history,
        save_path=f"{output_dir}/{prior_name}_training.png",
        title=f"Training Curves ({prior_display} Prior)"
    )
    
    print(f"\nAll plots generated for {prior_display} prior in {output_dir}/")
