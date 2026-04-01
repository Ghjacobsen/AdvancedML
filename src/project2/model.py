"""VAE model components for ensemble geometry experiments.

Based on course code (02460) by Jes Frellsen and Søren Hauberg, 2024.
Extended for ensemble decoder experiments.
"""

import torch
import torch.nn as nn
import torch.distributions as td


class GaussianPrior(nn.Module):
    """Standard Gaussian prior N(0, I)."""

    def __init__(self, M: int):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    """Gaussian encoder q(z|x) parameterized by a neural network."""

    def __init__(self, encoder_net: nn.Module):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x: torch.Tensor):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    """Gaussian decoder p(x|z) with fixed std."""

    def __init__(self, decoder_net: nn.Module, std: float = 1e-1):
        super().__init__()
        self.decoder_net = decoder_net
        self.std = std

    def forward(self, z: torch.Tensor):
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=self.std), 3)


class VAE(nn.Module):
    """Variational Autoencoder with ELBO training."""

    def __init__(self, prior: nn.Module, decoder: nn.Module, encoder: nn.Module):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.elbo(x)


def new_encoder(M: int = 2) -> nn.Module:
    """Create encoder network matching the course handout architecture."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )


def new_decoder(M: int = 2) -> nn.Module:
    """Create decoder network matching the course handout architecture."""
    return nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )


def create_vae(M: int = 2, device: str = "cpu") -> VAE:
    """Create a fresh VAE instance with the standard architecture."""
    model = VAE(
        GaussianPrior(M),
        GaussianDecoder(new_decoder(M)),
        GaussianEncoder(new_encoder(M)),
    )
    return model.to(device)
