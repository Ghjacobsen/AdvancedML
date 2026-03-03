"""
DDPM (Denoising Diffusion Probabilistic Model) for Part B.

This module provides:
- SinusoidalEmbedding / TimeEmbedding: timestep conditioning
- ResBlock / Downsample / Upsample: U-Net building blocks
- UNet: noise prediction network for 28×28 MNIST images
- DDPM: main model with ELBO and sampling
- train_ddpm(): full training pipeline
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.data import get_standard_mnist_loaders


# =============================================================================
# Time Embeddings
# =============================================================================


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for timestep conditioning."""

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension (must be even).
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices, shape (B,).

        Returns:
            Sinusoidal embedding, shape (B, dim).
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by a 2-layer MLP projection."""

    def __init__(self, dim: int):
        """
        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep indices, shape (B,).

        Returns:
            Time embedding, shape (B, dim).
        """
        return self.mlp(self.sinusoidal(t))


# =============================================================================
# U-Net Building Blocks
# =============================================================================


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and additive time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_dim: Dimension of time embedding.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map, shape (B, in_channels, H, W).
            t_emb: Time embedding, shape (B, time_dim).

        Returns:
            Output feature map, shape (B, out_channels, H, W).
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    """2× spatial downsampling via stride-2 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2× spatial upsampling via nearest-neighbour interpolation followed by convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# =============================================================================
# U-Net
# =============================================================================


class UNet(nn.Module):
    """
    U-Net noise predictor for DDPM on 28×28 MNIST images.

    Spatial resolution path: 28 → 14 → 7 (encoder) → 7 → 14 → 28 (decoder).
    Channel path with base_channels C=64: C → 2C → 4C (bottleneck) → 2C → C.
    Skip connections concatenate encoder features into the decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        """
        Args:
            in_channels: Number of input image channels (1 for greyscale MNIST).
            base_channels: Base channel count C; bottleneck uses 4C.
            time_dim: Dimension of the time embedding.
        """
        super().__init__()
        C = base_channels
        self.time_embed = TimeEmbedding(time_dim)

        self.init_conv = nn.Conv2d(in_channels, C, 3, padding=1)

        # Encoder
        self.down_res1 = ResBlock(C, C, time_dim)
        self.down_sample1 = Downsample(C)
        self.down_res2 = ResBlock(C, 2 * C, time_dim)
        self.down_sample2 = Downsample(2 * C)

        # Bottleneck
        self.mid_res1 = ResBlock(2 * C, 4 * C, time_dim)
        self.mid_res2 = ResBlock(4 * C, 4 * C, time_dim)

        # Decoder
        self.up_sample1 = Upsample(4 * C)
        self.up_res1 = ResBlock(4 * C + 2 * C, 2 * C, time_dim)
        self.up_sample2 = Upsample(2 * C)
        self.up_res2 = ResBlock(2 * C + C, C, time_dim)

        self.norm_out = nn.GroupNorm(8, C)
        self.conv_out = nn.Conv2d(C, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given a noisy image and timestep.

        Args:
            x: Noisy image, shape (B, 1, 28, 28).
            t: Timestep indices, shape (B,).

        Returns:
            Predicted noise, shape (B, 1, 28, 28).
        """
        t_emb = self.time_embed(t)

        h = self.init_conv(x)
        h1 = self.down_res1(h, t_emb)
        h2 = self.down_sample1(h1)
        h3 = self.down_res2(h2, t_emb)
        h4 = self.down_sample2(h3)

        h = self.mid_res1(h4, t_emb)
        h = self.mid_res2(h, t_emb)

        h = self.up_sample1(h)
        h = self.up_res1(torch.cat([h, h3], dim=1), t_emb)
        h = self.up_sample2(h)
        h = self.up_res2(torch.cat([h, h1], dim=1), t_emb)

        return self.conv_out(F.silu(self.norm_out(h)))


# =============================================================================
# DDPM
# =============================================================================


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Uses the simplified noise-prediction objective (L_simple).
    Forward process: q(x_t | x_0) = N(√ᾱ_t · x_0, (1 − ᾱ_t) · I).
    """

    def __init__(
        self,
        unet: UNet,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """
        Args:
            unet: U-Net noise prediction network.
            T: Total number of forward-diffusion timesteps.
            beta_start: β at t=0.
            beta_end: β at t=T-1.
        """
        super().__init__()
        self.unet = unet
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
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from the forward process q(x_t | x_0).

        Args:
            x0: Clean images, shape (B, C, H, W).
            t: Timestep indices, shape (B,).

        Returns:
            x_t: Noisy images, shape (B, C, H, W).
            eps: The Gaussian noise that was added.
        """
        eps = torch.randn_like(x0)
        sqrt_abar = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_1m_abar = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        x_t = sqrt_abar * x0 + sqrt_1m_abar * eps
        return x_t, eps

    def elbo(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the simplified DDPM ELBO (MSE noise-prediction objective).

        L_simple = E_{t ~ U(0,T-1), ε ~ N(0,I)} [ ||ε − ε_θ(x_t, t)||² ]

        Args:
            x0: Clean images, shape (B, C, H, W), normalised to [-1, 1].

        Returns:
            loss: Scalar MSE loss (to be minimised).
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self._diffuse(x0, t)
        eps_pred = self.unet(x_t, t)
        return F.mse_loss(eps_pred, eps)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self.elbo(x0)

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples via the DDPM reverse process.

        x_{t-1} = (1/√α_t) · (x_t − (1−α_t)/√(1−ᾱ_t) · ε_θ(x_t, t)) + √β_t · z,
        where z ~ N(0, I) for t > 0, else z = 0.

        Args:
            n_samples: Number of images to generate.
            device: Target device.

        Returns:
            Generated images, shape (n_samples, 1, 28, 28), clamped to [-1, 1].
        """
        self.eval()
        x = torch.randn(n_samples, 1, 28, 28, device=device)

        for t_idx in tqdm(reversed(range(self.T)), desc="DDPM sampling", leave=False, total=self.T):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            eps_pred = self.unet(x, t)

            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bars[t_idx]
            beta = self.betas[t_idx]

            coeff = (1.0 - alpha) / (1.0 - alpha_bar).sqrt()
            x_mean = (x - coeff * eps_pred) / alpha.sqrt()

            if t_idx > 0:
                x = x_mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = x_mean

        return x.clamp(-1.0, 1.0)


# =============================================================================
# Training
# =============================================================================


def train_ddpm(
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 2e-4,
    base_channels: int = 64,
    T: int = 1000,
    seed: Optional[int] = None,
    save_dir: str = "./models/ddpm",
    data_dir: str = "./data",
    device: Optional[torch.device] = None,
) -> Tuple[DDPM, Dict]:
    """
    Full training pipeline for the image-space DDPM on standard MNIST.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        base_channels: U-Net base channel count.
        T: Number of diffusion timesteps.
        seed: Optional random seed.
        save_dir: Directory for model checkpoints.
        data_dir: Directory for MNIST data.
        device: Compute device (auto-detected if None).

    Returns:
        model: Trained DDPM.
        history: Dict with 'train_loss' list.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DDPM on: {device}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    train_loader, _ = get_standard_mnist_loaders(
        batch_size=batch_size, data_dir=data_dir, squeeze_channel=False
    )

    unet = UNet(in_channels=1, base_channels=base_channels)
    model = DDPM(unet, T=T).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    history: Dict = {"train_loss": []}
    best_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training DDPM  |  T={T}  |  C={base_channels}  |  epochs={epochs}")
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": avg_loss,
                 "config": {"base_channels": base_channels, "T": T}},
                Path(save_dir) / "ddpm_best.pt",
            )

    torch.save(
        {"epoch": epochs, "model_state_dict": model.state_dict(), "history": history,
         "config": {"base_channels": base_channels, "T": T}},
        Path(save_dir) / "ddpm_final.pt",
    )
    print(f"\nDDPM training complete. Best loss: {best_loss:.4f}")
    return model, history


def load_ddpm(checkpoint_path: str, device: torch.device) -> DDPM:
    """
    Load a trained DDPM from a checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Target device.

    Returns:
        Loaded DDPM model in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    unet = UNet(in_channels=1, base_channels=cfg["base_channels"])
    model = DDPM(unet, T=cfg["T"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DDPM on MNIST")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./models/ddpm")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    train_ddpm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_channels=args.base_channels,
        T=args.T,
        seed=args.seed,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
    )
