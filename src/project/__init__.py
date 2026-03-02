"""
Project package for VAE experiments with different priors.

Part A: Priors for Variational Autoencoders
- GaussianPrior: Standard N(0,I)
- MoGPrior: Mixture of Gaussians
- FlowPrior: Normalizing flow prior
"""

from project.model import VAE, GaussianEncoder, BernoulliDecoder, create_encoder_net, create_decoder_net
from project.priors import GaussianPrior, MoGPrior, FlowPrior
from project.data import get_mnist_loaders, get_full_test_loader
from project.train import train_model, create_vae

__all__ = [
    # Model components
    "VAE",
    "GaussianEncoder", 
    "BernoulliDecoder",
    "create_encoder_net",
    "create_decoder_net",
    # Priors
    "GaussianPrior",
    "MoGPrior", 
    "FlowPrior",
    # Data
    "get_mnist_loaders",
    "get_full_test_loader",
    # Training
    "train_model",
    "create_vae",
]