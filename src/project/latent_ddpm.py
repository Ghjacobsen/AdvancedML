"""
Latent DDPM for Part B: Gaussian VAE + DDPM in latent space.

This module provides:
- GaussianDecoder: decoder with Gaussian likelihood for continuous images
- GaussianVAE: VAE using GaussianEncoder + GaussianDecoder, trained on standard MNIST
- LatentNoisePredictor: MLP noise predictor for latent-space diffusion
- LatentDDPM: DDPM operating in the VAE latent space with DDIM sampling
- train_gaussian_vae(): train the Gaussian VAE
- collect_latents(): encode training images to latent means
- train_latent_ddpm(): train the LatentDDPM on collected latents
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from project.data import get_standard_mnist_loaders
from project.model import GaussianEncoder, create_encoder_net, create_decoder_net


# =============================================================================
# Gaussian VAE
# =============================================================================


class GaussianDecoder(nn.Module):
    """
    Decoder with Gaussian likelihood: p(x | z) = N(f_θ(z), I).

    The decoder network maps latent codes to image means; the variance
    is fixed to 1 so the reconstruction objective is MSE.
    """

    def __init__(self, decoder_net: nn.Module):
        """
        Args:
            decoder_net: Network mapping (B, latent_dim) → (B, 28, 28).
        """
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes, shape (B, latent_dim).

        Returns:
            Image means, shape (B, 28, 28).
        """
        return self.decoder_net(z)


class GaussianVAE(nn.Module):
    """
    Variational Autoencoder with Gaussian encoder and Gaussian decoder.

    Trained on standard (non-binarised) MNIST normalised to [-1, 1].
    ELBO = E_q[log p(x|z)] − KL(q(z|x) || N(0,I))
         = −½ ||x − μ_dec(z)||² − KL   (with fixed decoder σ=1).
    """

    def __init__(
        self,
        encoder: GaussianEncoder,
        decoder: GaussianDecoder,
        latent_dim: int,
    ):
        """
        Args:
            encoder: Gaussian encoder producing q(z|x).
            decoder: Gaussian decoder producing image means.
            latent_dim: Dimension of the latent space.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the mean of q(z|x).

        Args:
            x: Images, shape (B, 28, 28).

        Returns:
            Encoder means, shape (B, latent_dim).
        """
        enc_out = self.encoder.encoder_net(x)
        mean, _ = torch.chunk(enc_out, 2, dim=-1)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to image means.

        Args:
            z: Latent codes, shape (B, latent_dim).

        Returns:
            Image means, shape (B, 28, 28).
        """
        return self.decoder(z)

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the ELBO for a batch of images.

        ELBO = −½ · Σ_pixels (x − μ_dec)² / B − KL(q || p)

        Args:
            x: Images, shape (B, 28, 28), in [-1, 1].

        Returns:
            ELBO scalar (higher is better).
        """
        enc_out = self.encoder.encoder_net(x)
        mean, log_std = torch.chunk(enc_out, 2, dim=-1)
        std = torch.exp(log_std)

        # Reparameterisation trick
        z = mean + std * torch.randn_like(std)

        x_recon = self.decoder(z)

        # Gaussian reconstruction loss (sum over pixels, mean over batch)
        recon_loss = 0.5 * ((x_recon - x) ** 2).sum(dim=[1, 2]).mean()

        # Analytical KL: KL(N(μ,σ²) || N(0,1)) = ½(μ²+σ²−2logσ−1)
        kl = 0.5 * (mean ** 2 + std ** 2 - 2.0 * log_std - 1.0).sum(dim=1).mean()

        return -(recon_loss + kl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.elbo(x)

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample images by decoding from the Gaussian prior.

        Args:
            n_samples: Number of images to generate.
            device: Target device.

        Returns:
            Decoded images, shape (n_samples, 28, 28).
        """
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)


# =============================================================================
# Latent DDPM: MLP noise predictor + diffusion in latent space
# =============================================================================


class LatentNoisePredictor(nn.Module):
    """
    MLP noise predictor for diffusion in a low-dimensional latent space.

    Architecture: cat(z_t, t_emb) → 4-layer MLP with SiLU → predicted ε.
    """

    def __init__(
        self,
        latent_dim: int,
        time_dim: int = 128,
        hidden_dim: int = 512,
    ):
        """
        Args:
            latent_dim: Dimension of the latent space.
            time_dim: Dimension of the sinusoidal time embedding.
            hidden_dim: Width of the hidden layers.
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Sinusoidal embedding + MLP projection (mirrors ddpm.TimeEmbedding)
        from project.ddpm import TimeEmbedding
        self.time_embed = TimeEmbedding(time_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noisy latent codes, shape (B, latent_dim).
            t: Timestep indices, shape (B,).

        Returns:
            Predicted noise, shape (B, latent_dim).
        """
        t_emb = self.time_embed(t)
        return self.net(torch.cat([z, t_emb], dim=-1))


class LatentDDPM(nn.Module):
    """
    DDPM operating in the VAE latent space.

    Uses the same linear-beta noise schedule and L_simple objective as the
    image-space DDPM, but the data are latent codes z ∈ ℝ^M rather than images.
    Sampling uses DDIM (deterministic), which allows any number of steps without
    retraining.
    """

    def __init__(
        self,
        noise_predictor: LatentNoisePredictor,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Args:
            noise_predictor: MLP noise predictor.
            T: Total number of forward-diffusion timesteps.
            beta_start: β at t=0.
            beta_end: β at t=T-1.
        """
        super().__init__()
        self.noise_predictor = noise_predictor
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def _diffuse(
        self, z0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample z_t ~ q(z_t | z_0).

        Args:
            z0: Clean latent codes, shape (B, M).
            t: Timestep indices, shape (B,).

        Returns:
            z_t: Noisy latents, shape (B, M).
            eps: Added noise, shape (B, M).
        """
        eps = torch.randn_like(z0)
        sqrt_abar = self.sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_1m_abar = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        z_t = sqrt_abar * z0 + sqrt_1m_abar * eps
        return z_t, eps

    def elbo(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Compute the simplified DDPM ELBO on latent codes.

        L_simple = E_{t, ε} [ ||ε − ε_θ(z_t, t)||² ]

        Args:
            z0: Clean latent codes, shape (B, M).

        Returns:
            MSE loss scalar.
        """
        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=z0.device)
        z_t, eps = self._diffuse(z0, t)
        eps_pred = self.noise_predictor(z_t, t)
        return F.mse_loss(eps_pred, eps)

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        return self.elbo(z0)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        device: torch.device,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate latent codes via DDIM reverse process.

        DDIM update (deterministic, η=0):
            pred_z0  = (z_t − √(1−ᾱ_t) · ε_θ) / √ᾱ_t
            z_{t-1}  = √ᾱ_{t-1} · pred_z0 + √(1−ᾱ_{t-1}) · ε_θ

        Args:
            n_samples: Number of latent codes to generate.
            device: Target device.
            n_steps: Number of denoising steps.
                     0  → pure prior N(0, I) (no diffusion).
                     k  → k DDIM steps from T−1 down to 0.
                     None → full T steps.

        Returns:
            Latent codes, shape (n_samples, latent_dim).
        """
        self.eval()
        latent_dim = self.noise_predictor.latent_dim
        n_steps = n_steps if n_steps is not None else self.T

        z = torch.randn(n_samples, latent_dim, device=device)

        if n_steps == 0:
            return z

        # Select n_steps+1 uniformly spaced timesteps in [0, T-1], then reverse
        tau = torch.linspace(0, self.T - 1, n_steps + 1).long().tolist()
        tau = list(reversed(tau))  # tau[0]=T-1 (most noisy), tau[-1]=0

        for i in range(n_steps):
            t_curr = tau[i]
            t_prev = tau[i + 1] if i + 1 < len(tau) else -1

            t = torch.full((n_samples,), t_curr, device=device, dtype=torch.long)
            eps_pred = self.noise_predictor(z, t)

            abar_curr = self.alpha_bars[t_curr]
            pred_z0 = (z - (1.0 - abar_curr).sqrt() * eps_pred) / abar_curr.sqrt()

            if t_prev >= 0:
                abar_prev = self.alpha_bars[t_prev]
                z = abar_prev.sqrt() * pred_z0 + (1.0 - abar_prev).sqrt() * eps_pred
            else:
                z = pred_z0  # Final step: output the denoised estimate

        return z


# =============================================================================
# Training helpers
# =============================================================================


def train_gaussian_vae(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    seed: Optional[int] = None,
    save_dir: str = "./models/gaussian_vae",
    data_dir: str = "./data",
    device: Optional[torch.device] = None,
) -> Tuple[GaussianVAE, Dict]:
    """
    Train a Gaussian VAE on standard MNIST.

    Args:
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        latent_dim: Dimension of the latent space.
        hidden_dim: Hidden layer width for encoder/decoder MLPs.
        seed: Optional random seed.
        save_dir: Checkpoint directory.
        data_dir: MNIST data directory.
        device: Compute device (auto-detected if None).

    Returns:
        model: Trained GaussianVAE.
        history: Dict with 'train_loss' and 'test_elbo' lists.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Gaussian VAE on: {device}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_standard_mnist_loaders(
        batch_size=batch_size, data_dir=data_dir, squeeze_channel=True
    )

    encoder_net = create_encoder_net(latent_dim, hidden_dim)
    decoder_net = create_decoder_net(latent_dim, hidden_dim)
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    model = GaussianVAE(encoder, decoder, latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    history: Dict = {"train_loss": [], "test_elbo": []}
    best_elbo = float("-inf")

    print(f"\n{'='*60}")
    print(f"Training Gaussian VAE | latent_dim={latent_dim} | epochs={epochs}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.2f}"})

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        model.eval()
        total_elbo = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                total_elbo += model.elbo(x).item()
        test_elbo = total_elbo / len(test_loader)
        history["test_elbo"].append(test_elbo)

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.2f} | Test ELBO: {test_elbo:.2f}")

        if test_elbo > best_elbo:
            best_elbo = test_elbo
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "config": {"latent_dim": latent_dim, "hidden_dim": hidden_dim}},
                Path(save_dir) / "gaussian_vae_best.pt",
            )

    torch.save(
        {"epoch": epochs, "model_state_dict": model.state_dict(), "history": history,
         "config": {"latent_dim": latent_dim, "hidden_dim": hidden_dim}},
        Path(save_dir) / "gaussian_vae_final.pt",
    )
    print(f"\nGaussian VAE training complete. Best test ELBO: {best_elbo:.2f}")
    return model, history


def collect_latents(
    vae: GaussianVAE,
    train_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode all training images to latent means.

    Args:
        vae: Trained GaussianVAE.
        train_loader: Training data loader (squeeze_channel=True).
        device: Compute device.

    Returns:
        Latent means, shape (N, latent_dim).
    """
    vae.eval()
    latents = []
    with torch.no_grad():
        for x, _ in tqdm(train_loader, desc="Collecting latents", leave=False):
            x = x.to(device)
            latents.append(vae.encode_mean(x).cpu())
    return torch.cat(latents, dim=0)


def train_latent_ddpm(
    vae: GaussianVAE,
    latent_dim: int = 20,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    T: int = 1000,
    hidden_dim: int = 512,
    seed: Optional[int] = None,
    save_dir: str = "./models/latent_ddpm",
    data_dir: str = "./data",
    device: Optional[torch.device] = None,
) -> Tuple[LatentDDPM, Dict]:
    """
    Train a LatentDDPM on the latent codes of a pre-trained Gaussian VAE.

    Args:
        vae: Trained GaussianVAE (encoder used to extract latents).
        latent_dim: Dimension of the latent space.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        T: Number of diffusion timesteps.
        hidden_dim: Hidden layer width of the MLP noise predictor.
        seed: Optional random seed.
        save_dir: Checkpoint directory.
        data_dir: MNIST data directory.
        device: Compute device (auto-detected if None).

    Returns:
        model: Trained LatentDDPM.
        history: Dict with 'train_loss' list.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Latent DDPM on: {device}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Collect latent codes from training data
    train_loader, _ = get_standard_mnist_loaders(
        batch_size=batch_size, data_dir=data_dir, squeeze_channel=True
    )
    latents = collect_latents(vae, train_loader, device)  # (N, M)

    latent_dataset = TensorDataset(latents)
    latent_loader = DataLoader(latent_dataset, batch_size=batch_size, shuffle=True)

    predictor = LatentNoisePredictor(latent_dim, hidden_dim=hidden_dim)
    model = LatentDDPM(predictor, T=T).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    history: Dict = {"train_loss": []}
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training Latent DDPM | T={T} | latent_dim={latent_dim} | epochs={epochs}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(latent_loader, desc=f"Epoch {epoch}", leave=False)
        for (z0,) in pbar:
            z0 = z0.to(device)
            optimizer.zero_grad()
            loss = model(z0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(latent_loader)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "config": {"latent_dim": latent_dim, "T": T, "hidden_dim": hidden_dim}},
                Path(save_dir) / "latent_ddpm_best.pt",
            )

    torch.save(
        {"epoch": epochs, "model_state_dict": model.state_dict(), "history": history,
         "config": {"latent_dim": latent_dim, "T": T, "hidden_dim": hidden_dim}},
        Path(save_dir) / "latent_ddpm_final.pt",
    )
    print(f"\nLatent DDPM training complete. Best loss: {best_loss:.4f}")
    return model, history


def load_gaussian_vae(
    checkpoint_path: str, device: torch.device
) -> GaussianVAE:
    """
    Load a trained GaussianVAE from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Target device.

    Returns:
        Loaded GaussianVAE in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    encoder_net = create_encoder_net(cfg["latent_dim"], cfg["hidden_dim"])
    decoder_net = create_decoder_net(cfg["latent_dim"], cfg["hidden_dim"])
    encoder = GaussianEncoder(encoder_net)
    decoder = GaussianDecoder(decoder_net)
    model = GaussianVAE(encoder, decoder, cfg["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_latent_ddpm(
    checkpoint_path: str, device: torch.device
) -> LatentDDPM:
    """
    Load a trained LatentDDPM from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Target device.

    Returns:
        Loaded LatentDDPM in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    predictor = LatentNoisePredictor(
        cfg["latent_dim"], hidden_dim=cfg.get("hidden_dim", 512)
    )
    model = LatentDDPM(predictor, T=cfg["T"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
