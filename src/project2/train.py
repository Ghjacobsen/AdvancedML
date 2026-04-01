"""Training pipeline for ensemble VAE experiments.

Handles:
- Training a single VAE
- Training an ensemble of K decoders (each with its own encoder)
- Saving/loading model checkpoints
"""

import os
import torch
from tqdm import tqdm

from project2.model import VAE, GaussianPrior, GaussianEncoder, GaussianDecoder, new_encoder, new_decoder


def noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise to data (as in course handout)."""
    eps = std * torch.randn_like(x)
    return torch.clamp(x + eps, min=0.0, max=1.0)


def train_single_vae(
    model: VAE,
    data_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> VAE:
    """Train a single VAE model (mirrors course handout training loop)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model = model.to(device)

    num_steps = len(data_loader) * epochs
    epoch = 0

    iterator = tqdm(range(num_steps), disable=not verbose)
    for step in iterator:
        x = next(iter(data_loader))[0]
        x = noise(x.to(device))

        optimizer.zero_grad()
        loss = model(x)
        loss.backward()
        optimizer.step()

        if verbose and step % 5 == 0:
            iterator.set_description(f"epoch={epoch}, loss={loss.item():.1f}")

        if (step + 1) % len(data_loader) == 0:
            epoch += 1

    return model


def train_ensemble(
    num_decoders: int,
    data_loader: torch.utils.data.DataLoader,
    M: int = 2,
    epochs_per_decoder: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    save_dir: str | None = None,
    verbose: bool = True,
) -> tuple[list[VAE], list[GaussianDecoder], GaussianEncoder]:
    """Train an ensemble of independently-initialized VAEs.

    Each VAE gets its own encoder and decoder trained from scratch.
    The first encoder is used as the 'reference' for encoding test data.

    Args:
        num_decoders: K — number of VAEs to train.
        data_loader: Training data.
        M: Latent dimension.
        epochs_per_decoder: Epochs per VAE.
        lr: Learning rate.
        device: Torch device.
        save_dir: Directory for checkpoints.
        verbose: Print progress.

    Returns:
        (vaes, decoders, reference_encoder)
    """
    vaes = []
    decoders = []
    reference_encoder = None

    for k in range(num_decoders):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training decoder {k+1}/{num_decoders}")
            print(f"{'='*60}")

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)

        model = train_single_vae(model, data_loader, epochs=epochs_per_decoder, lr=lr, device=device, verbose=verbose)
        model.eval()

        vaes.append(model)
        decoders.append(model.decoder)

        if k == 0:
            reference_encoder = model.encoder

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"vae_decoder_{k}.pt")
            torch.save(model.state_dict(), path)
            if verbose:
                print(f"  Saved: {path}")

    return vaes, decoders, reference_encoder


def load_ensemble(
    save_dir: str,
    num_decoders: int,
    M: int = 2,
    device: str = "cpu",
) -> tuple[list[VAE], list[GaussianDecoder], GaussianEncoder]:
    """Load a previously trained ensemble from checkpoints."""
    vaes = []
    decoders = []
    reference_encoder = None

    for k in range(num_decoders):
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)

        path = os.path.join(save_dir, f"vae_decoder_{k}.pt")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()

        vaes.append(model)
        decoders.append(model.decoder)

        if k == 0:
            reference_encoder = model.encoder

    return vaes, decoders, reference_encoder
