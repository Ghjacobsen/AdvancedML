"""
Evaluation module for VAE models.

This module provides:
- compute_test_elbo(): Compute test set ELBO for a model
- evaluate_multiple_runs(): Run multiple training runs and compute mean/std
- load_model(): Load trained VAE from checkpoint
"""

from pathlib import Path
from typing import List, Tuple, Optional, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.model import VAE, GaussianEncoder, BernoulliDecoder, create_encoder_net, create_decoder_net
from project.priors import GaussianPrior, MoGPrior, FlowPrior
from project.data import get_mnist_loaders, get_full_test_loader
from project.train import train_model, create_vae, PriorType


def compute_test_elbo(
    model: VAE,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 1,
) -> float:
    """
    Compute test set ELBO (log-likelihood lower bound).
    
    Parameters:
    model: [VAE] Trained VAE model
    test_loader: [DataLoader] Test data loader
    device: [torch.device] Device
    n_samples: [int] Number of importance samples for tighter bound
    
    Returns:
    avg_elbo: [float] Average ELBO over test set
    """
    model.eval()
    total_elbo = 0.0
    n_datapoints = 0
    
    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc="Computing test ELBO", leave=False):
            x = x.to(device)
            batch_size = x.shape[0]
            
            if n_samples == 1:
                # Standard single-sample ELBO
                elbo = model.elbo(x)
                total_elbo += elbo.item() * batch_size
            else:
                # Importance weighted ELBO (tighter bound)
                elbo = compute_iwae_bound(model, x, n_samples)
                total_elbo += elbo.item() * batch_size
            
            n_datapoints += batch_size
    
    return total_elbo / n_datapoints


def compute_iwae_bound(
    model: VAE,
    x: torch.Tensor,
    n_samples: int = 100,
) -> torch.Tensor:
    """
    Compute Importance Weighted Autoencoder (IWAE) bound.
    This provides a tighter lower bound on log p(x) than standard ELBO.
    
    Parameters:
    model: [VAE] The VAE model
    x: [torch.Tensor] Input batch
    n_samples: [int] Number of importance samples
    
    Returns:
    iwae: [torch.Tensor] IWAE bound (scalar)
    """
    batch_size = x.shape[0]
    
    # Expand x for multiple samples
    x_expanded = x.unsqueeze(1).expand(-1, n_samples, *x.shape[1:])
    x_flat = x_expanded.reshape(batch_size * n_samples, *x.shape[1:])
    
    # Get encoder distribution
    q = model.encoder(x)
    
    # Sample multiple z's for each x
    z = q.rsample((n_samples,))  # (n_samples, batch_size, latent_dim)
    z = z.permute(1, 0, 2)  # (batch_size, n_samples, latent_dim)
    z_flat = z.reshape(batch_size * n_samples, -1)
    
    # Compute log weights
    log_p_x_given_z = model.decoder(z_flat).log_prob(x_flat)
    log_p_x_given_z = log_p_x_given_z.reshape(batch_size, n_samples)
    
    log_p_z = model.prior().log_prob(z_flat).reshape(batch_size, n_samples)
    
    # For q(z|x), we need to evaluate at sampled z
    mean, log_std = torch.chunk(model.encoder.encoder_net(x), 2, dim=-1)
    std = torch.exp(log_std)
    
    z_normalized = (z - mean.unsqueeze(1)) / std.unsqueeze(1)
    log_q_z = -0.5 * (z_normalized ** 2 + 2 * log_std.unsqueeze(1) + np.log(2 * np.pi)).sum(-1)
    
    # IWAE bound
    log_weights = log_p_x_given_z + log_p_z - log_q_z
    iwae = torch.logsumexp(log_weights, dim=1) - np.log(n_samples)
    
    return iwae.mean()


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[VAE, dict]:
    """
    Load trained VAE from checkpoint.
    
    Parameters:
    checkpoint_path: [str] Path to checkpoint file
    device: [torch.device] Device to load model to
    
    Returns:
    model: [VAE] Loaded VAE model
    config: [dict] Model configuration
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    # Create model with same config
    model = create_vae(
        prior_type=config["prior_type"],
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
        device=device,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, config


def evaluate_multiple_runs(
    prior_type: str,
    n_runs: int = 3,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    save_dir: str = "./models",
    data_dir: str = "./data",
    device: Optional[torch.device] = None,
):
    """
    Train multiple runs and compute mean/std of test ELBO.
    
    Parameters:
    prior_type: [str] Prior type ("gaussian", "mog", "flow")
    n_runs: [int] Number of training runs
    epochs: [int] Epochs per run
    batch_size: [int] Batch size
    lr: [float] Learning rate
    latent_dim: [int] Latent dimension
    hidden_dim: [int] Hidden dimension
    save_dir: [str] Save directory
    data_dir: [str] Data directory
    device: [torch.device] Device
    
    Returns:
    mean_elbo: [float] Mean test ELBO
    std_elbo: [float] Standard deviation of test ELBO
    all_elbos: [List[float]] All individual ELBO values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_elbos = []
    
    print(f"\n{'='*60}")
    print(f"Running {n_runs} experiments with {prior_type.upper()} prior")
    print(f"{'='*60}")
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        # Train model with different seed
        seed = 42 + run
        model, history = train_model(
            prior_type=prior_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            seed=seed,
            save_dir=f"{save_dir}/run_{run}",
            data_dir=data_dir,
            device=device,
        )
        
        # Evaluate on test set
        test_loader = get_full_test_loader(data_dir=data_dir)
        test_elbo = compute_test_elbo(model, test_loader, device)
        all_elbos.append(test_elbo)
        
        print(f"Run {run + 1} test ELBO: {test_elbo:.2f}")
    
    mean_elbo = np.mean(all_elbos)
    std_elbo = np.std(all_elbos)
    
    print(f"\n{'='*60}")
    print(f"Results for {prior_type.upper()} prior:")
    print(f"Test ELBO: {mean_elbo:.2f} ± {std_elbo:.2f}")
    print(f"{'='*60}")
    
    return mean_elbo, std_elbo, all_elbos


def compare_all_priors(
    n_runs: int = 3,
    epochs: int = 50,
    **kwargs
) -> dict:
    """
    Train and compare all three priors.
    
    Parameters:
    n_runs: [int] Number of runs per prior
    epochs: [int] Training epochs
    **kwargs: Additional arguments for training
    
    Returns:
    results: [dict] Results for each prior type
    """
    results = {}
    
    for pt in ["gaussian", "mog", "flow"]:
        mean_elbo, std_elbo, all_elbos = evaluate_multiple_runs(
            prior_type=pt,
            n_runs=n_runs,
            epochs=epochs,
            **kwargs
        )
        results[pt] = {
            "mean": mean_elbo,
            "std": std_elbo,
            "all_runs": all_elbos,
        }
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON - Test Set Log-Likelihood (ELBO)")
    print("="*60)
    print(f"{'Prior':<15} {'Mean ELBO':<15} {'Std':<10}")
    print("-"*40)
    for prior_type, res in results.items():
        print(f"{prior_type.upper():<15} {res['mean']:<15.2f} {res['std']:<10.2f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VAE models")
    parser.add_argument("--prior", type=str, default=None,
                        choices=["gaussian", "mog", "flow", "all"],
                        help="Prior to evaluate (or 'all' for comparison)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs for mean/std computation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load and evaluate existing checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./models")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.checkpoint:
        # Evaluate existing checkpoint
        model, config = load_model(args.checkpoint, device)
        test_loader = get_full_test_loader(data_dir=args.data_dir)
        test_elbo = compute_test_elbo(model, test_loader, device)
        print(f"Test ELBO: {test_elbo:.2f}")
    
    elif args.prior == "all":
        # Compare all priors
        compare_all_priors(
            n_runs=args.n_runs,
            epochs=args.epochs,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
        )
    
    elif args.prior:
        # Evaluate specific prior
        evaluate_multiple_runs(
            prior_type=args.prior,
            n_runs=args.n_runs,
            epochs=args.epochs,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
        )
