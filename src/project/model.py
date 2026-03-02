"""
VAE model components for Part A: Priors for variational autoencoders.

This module provides:
- GaussianEncoder: Encodes input to latent space distribution
- BernoulliDecoder: Decodes latent samples to Bernoulli output (for binarized MNIST)
- VAE: Main VAE class that works with any prior (Gaussian, MoG, Flow)
- Encoder/Decoder network factories for MNIST
"""

import torch
import torch.nn as nn
import torch.distributions as td


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        decoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Variational Autoencoder with pluggable priors.
    
    Uses Monte Carlo estimation for ELBO, which works with all prior types:
    - GaussianPrior: Standard Gaussian N(0,I)
    - MoGPrior: Mixture of Gaussians
    - FlowPrior: Normalizing flow prior
    """
    
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
           The decoder distribution over the data space.
        encoder: [torch.nn.Module]
           The encoder distribution over the latent space.
        """
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data using Monte Carlo estimation.
        
        ELBO = E_q[log p(x|z) + log p(z) - log q(z|x)]
        
        This formulation works for all priors (Gaussian, MoG, Flow).

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           
        Returns:
        elbo: [torch.Tensor]
           Scalar ELBO value (mean over batch).
        """
        q = self.encoder(x)
        z = q.rsample()
        
        # Monte Carlo estimate of ELBO components
        log_p_x_given_z = self.decoder(z).log_prob(x)  # Reconstruction
        log_p_z = self.prior().log_prob(z)             # Prior
        log_q_z_given_x = q.log_prob(z)                # Encoder
        
        elbo = torch.mean(log_p_x_given_z + log_p_z - log_q_z_given_x)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model by sampling z from prior, then x from decoder.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
           
        Returns:
        samples: [torch.Tensor]
           Generated samples.
        """
        z = self.prior().sample((n_samples,))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO (loss) for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
           
        Returns:
        loss: [torch.Tensor]
           Negative ELBO (to minimize).
        """
        return -self.elbo(x)


def create_encoder_net(latent_dim, hidden_dim=256):
    """
    Create encoder network for MNIST (28x28 -> 2*latent_dim).
    
    Parameters:
    latent_dim: [int] Dimension of latent space (M)
    hidden_dim: [int] Hidden layer size
    
    Returns:
    encoder_net: [nn.Sequential] Network that outputs (batch, 2*latent_dim)
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2 * latent_dim)  # mean and log_std
    )


def create_decoder_net(latent_dim, hidden_dim=256):
    """
    Create decoder network for MNIST (latent_dim -> 28x28).
    
    Parameters:
    latent_dim: [int] Dimension of latent space (M)
    hidden_dim: [int] Hidden layer size
    
    Returns:
    decoder_net: [nn.Sequential] Network that outputs (batch, 28, 28)
    """
    return nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 28*28),
        nn.Unflatten(-1, (28, 28))
    )
