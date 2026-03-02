"""
Training module for VAE models with different priors.

This module provides:
- train(): Train a VAE model for one epoch
- train_model(): Full training loop with checkpointing
- create_vae(): Factory function to create VAE with specified prior type
"""

import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.model import VAE, GaussianEncoder, BernoulliDecoder, create_encoder_net, create_decoder_net
from project.priors import GaussianPrior, MoGPrior, FlowPrior
from project.data import get_mnist_loaders


PriorType = Literal["gaussian", "mog", "flow"]


def train_epoch(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train VAE model for one epoch.
    
    Parameters:
    model: [VAE] The VAE model
    optimizer: [Optimizer] The optimizer
    train_loader: [DataLoader] Training data loader
    device: [torch.device] Device to train on
    epoch: [int] Current epoch number (for progress bar)
    
    Returns:
    avg_loss: [float] Average negative ELBO over the epoch
    """
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for x, _ in pbar:
        x = x.to(device)
        
        optimizer.zero_grad()
        loss = model(x)  # Forward returns negative ELBO
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.2f}"})
    
    return total_loss / len(train_loader)


def evaluate(
    model: VAE,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate VAE model on a dataset.
    
    Parameters:
    model: [VAE] The VAE model
    data_loader: [DataLoader] Data loader to evaluate on
    device: [torch.device] Device
    
    Returns:
    avg_elbo: [float] Average ELBO (higher is better)
    """
    model.eval()
    total_elbo = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            elbo = model.elbo(x)
            total_elbo += elbo.item()
            n_batches += 1
    
    return total_elbo / n_batches


def create_vae(
    prior_type: PriorType,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    n_components: int = 10,  # For MoG
    n_transforms: int = 4,   # For Flow
    n_hidden_flow: int = 64, # For Flow
    device: torch.device = None,
) -> VAE:
    """
    Factory function to create VAE with specified prior type.
    
    Parameters:
    prior_type: [str] One of "gaussian", "mog", "flow"
    latent_dim: [int] Dimension of latent space
    hidden_dim: [int] Hidden dimension for encoder/decoder
    n_components: [int] Number of mixture components (for MoG)
    n_transforms: [int] Number of flow transforms (for Flow)
    n_hidden_flow: [int] Hidden units in flow networks (for Flow)
    device: [torch.device] Device to create model on
    
    Returns:
    model: [VAE] VAE model ready for training
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create encoder and decoder networks
    encoder_net = create_encoder_net(latent_dim, hidden_dim)
    decoder_net = create_decoder_net(latent_dim, hidden_dim)
    
    # Create encoder and decoder
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)
    
    # Create prior based on type
    if prior_type == "gaussian":
        prior = GaussianPrior(latent_dim)
    elif prior_type == "mog":
        prior = MoGPrior(latent_dim, K=n_components)
    elif prior_type == "flow":
        prior = FlowPrior(latent_dim, n_transforms=n_transforms, n_hidden=n_hidden_flow)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")
    
    # Create VAE
    model = VAE(prior, decoder, encoder)
    model = model.to(device)
    
    return model


def train_model(
    prior_type: PriorType,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    seed: Optional[int] = None,
    save_dir: str = "./models",
    data_dir: str = "./data",
    device: Optional[torch.device] = None,
) -> Tuple[VAE, dict]:
    """
    Full training pipeline for VAE with specified prior.
    
    Parameters:
    prior_type: [str] One of "gaussian", "mog", "flow"
    epochs: [int] Number of training epochs
    batch_size: [int] Batch size
    lr: [float] Learning rate
    latent_dim: [int] Latent space dimension
    hidden_dim: [int] Hidden dimension for networks
    seed: [int] Random seed for reproducibility
    save_dir: [str] Directory to save model checkpoints
    data_dir: [str] Directory for MNIST data
    device: [torch.device] Device to train on
    
    Returns:
    model: [VAE] Trained VAE model
    history: [dict] Training history with losses
    """
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
    )
    
    # Create model
    model = create_vae(
        prior_type=prior_type,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        device=device,
    )
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {
        "train_loss": [],
        "test_elbo": [],
    }
    
    best_elbo = float("-inf")
    
    print(f"\n{'='*60}")
    print(f"Training VAE with {prior_type.upper()} prior")
    print(f"Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, optimizer, train_loader, device, epoch)
        history["train_loss"].append(train_loss)
        
        # Evaluate
        test_elbo = evaluate(model, test_loader, device)
        history["test_elbo"].append(test_elbo)
        
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.2f} | Test ELBO: {test_elbo:.2f}")
        
        # Save best model
        if test_elbo > best_elbo:
            best_elbo = test_elbo
            save_path = Path(save_dir) / f"vae_{prior_type}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_elbo": test_elbo,
                "config": {
                    "prior_type": prior_type,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                }
            }, save_path)
    
    # Save final model
    final_path = Path(save_dir) / f"vae_{prior_type}_final.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_elbo": history["test_elbo"][-1],
        "history": history,
        "config": {
            "prior_type": prior_type,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        }
    }, final_path)
    
    print(f"\nTraining complete. Best test ELBO: {best_elbo:.2f}")
    print(f"Model saved to: {final_path}")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VAE with different priors")
    parser.add_argument("--prior", type=str, default="gaussian", 
                        choices=["gaussian", "mog", "flow"],
                        help="Prior type: gaussian, mog, or flow")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./models")
    parser.add_argument("--data-dir", type=str, default="./data")
    
    args = parser.parse_args()
    
    train_model(
        prior_type=args.prior,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
    )
