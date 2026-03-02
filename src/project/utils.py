"""
Utility functions for VAE analysis and comparison.

This module provides:
- compare_prior_posterior(): Plot prior vs aggregate posterior distributions
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def compare_prior_posterior(
    model,
    data_loader: DataLoader,
    device: torch.device,
    save_path: str = "prior_posterior_comparison.png",
    n_samples: int = 5000,
    prior_name: str = "Prior"
):
    """
    Compare samples from the prior and aggregate posterior.
    
    Creates a 3-panel plot:
    1. Posterior q(z|x) colored by digit class
    2. Prior p(z)
    3. Overlay comparison
    
    If latent_dim > 2, uses PCA to project to 2D.
    
    Parameters:
    model: [VAE] 
       The VAE model.
    data_loader: [torch.utils.data.DataLoader] 
       The data loader.
    device: [torch.device] 
       The device to use.
    save_path: [str] 
       Path to save the comparison plot.
    n_samples: [int]
       Number of samples to collect (default 5000).
    prior_name: [str]
       Name of the prior for plot title.
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    
    # Collect samples from approximate posterior
    posterior_samples = []
    labels = []
    
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Sampling from posterior", leave=False):
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            posterior_samples.append(z.cpu())
            labels.append(y)
            
            # Collect enough samples
            if len(posterior_samples) * x.shape[0] >= n_samples:
                break
    
    posterior_samples = torch.cat(posterior_samples, dim=0)[:n_samples].numpy()
    labels = torch.cat(labels, dim=0)[:n_samples].numpy()
    
    # Sample from prior
    with torch.no_grad():
        prior_samples = model.prior().sample((n_samples,)).cpu().numpy()
    
    # If latent dimension > 2, apply PCA
    if posterior_samples.shape[1] > 2:
        from sklearn.decomposition import PCA
        # Fit PCA on combined data for consistent projection
        combined = np.vstack([posterior_samples, prior_samples])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        
        posterior_2d = combined_2d[:len(posterior_samples)]
        prior_2d = combined_2d[len(posterior_samples):]
        
        var_explained = pca.explained_variance_ratio_.sum()
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
        title_suffix = f"\n(PCA: {var_explained:.1%} variance explained)"
    else:
        posterior_2d = posterior_samples
        prior_2d = prior_samples
        xlabel = 'Latent dimension 1'
        ylabel = 'Latent dimension 2'
        title_suffix = ""
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Posterior colored by class
    scatter1 = axes[0].scatter(posterior_2d[:, 0], posterior_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.5, s=5)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f'Aggregate Posterior q(z|x){title_suffix}')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Digit class')
    
    # Plot 2: Prior
    axes[1].scatter(prior_2d[:, 0], prior_2d[:, 1], 
                   c='blue', alpha=0.3, s=5)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'{prior_name} p(z){title_suffix}')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Overlay
    axes[2].scatter(prior_2d[:, 0], prior_2d[:, 1], 
                   c='blue', alpha=0.2, s=5, label='Prior')
    axes[2].scatter(posterior_2d[:, 0], posterior_2d[:, 1], 
                   c='red', alpha=0.2, s=5, label='Posterior')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Overlay Comparison{title_suffix}')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prior-posterior comparison saved to {save_path}")
    
    return posterior_2d, prior_2d, labels


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


