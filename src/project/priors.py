# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

"""
Prior distributions for VAE models.

This module provides three types of priors:
1. GaussianPrior: Standard N(0,I) prior
2. MoGPrior: Mixture of Gaussians prior with K components  
3. FlowPrior: Normalizing flow prior using RealNVP-style coupling layers

All priors implement:
- forward(): Returns a distribution object with .sample() and .log_prob() methods
"""

import torch
import torch.nn as nn
import torch.distributions as td


# =============================================================================
# Flow Components (needed for FlowPrior)
# =============================================================================

class GaussianBase(nn.Module):
    """Standard Gaussian base distribution for normalizing flows."""
    
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

        Parameters:
        D: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """Return the base distribution."""
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MaskedCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows (RealNVP style).
    
    Transforms: x = z * exp(s(z_masked)) + t(z_masked) for unmasked dimensions
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            Network that outputs scale factors.
        translation_net: [torch.nn.Module]
            Network that outputs translations.
        mask: [torch.Tensor]
            Binary mask (1 = pass through, 0 = transform).
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """Transform from base to data space."""
        s = self.scale_net(z * self.mask)
        t = self.translation_net(z * self.mask)
        
        x = self.mask * z + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det_J = torch.sum((1 - self.mask) * s, dim=1)
        
        return x, log_det_J
    
    def inverse(self, x):
        """Transform from data to base space."""
        s = self.scale_net(x * self.mask)
        t = self.translation_net(x * self.mask)
        
        z = self.mask * x + (1 - self.mask) * ((x - t) * torch.exp(-s))
        log_det_J = -torch.sum((1 - self.mask) * s, dim=1)
        
        return z, log_det_J


class Flow(nn.Module):
    """
    Normalizing flow model composed of a base distribution and transformations.
    
    Provides .sample() and .log_prob() methods to act as a distribution.
    """
    
    def __init__(self, base, transformations):
        """
        Parameters:
        base: [nn.Module]
            Base distribution module with forward() returning a distribution.
        transformations: [list of nn.Module]
            List of invertible transformation layers.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """Transform from base to data space."""
        sum_log_det_J = 0
        x = z
        for T in self.transformations:
            x, log_det_J = T(x)
            sum_log_det_J += log_det_J
        return x, sum_log_det_J
    
    def inverse(self, x):
        """Transform from data to base space."""
        sum_log_det_J = 0
        z = x
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(z)
            sum_log_det_J += log_det_J
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """Compute log probability under the flow."""
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """Sample from the flow."""
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]


# =============================================================================
# Prior Classes
# =============================================================================

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MoGPrior(nn.Module):
    def __init__(self, M, K=10):
        """
        Define a Mixture of Gaussians prior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        
        # Initialize mixture weights (logits for categorical distribution)
        self.mixture_logits = nn.Parameter(torch.zeros(K), requires_grad=False)
        
        # Initialize means for each component (random initialization)
        self.means = nn.Parameter(torch.randn(K, M), requires_grad=False)
        
        # Initialize standard deviations for each component
        self.stds = nn.Parameter(torch.ones(K, M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution (Mixture of Gaussians).

        Returns:
        prior: [torch.distributions.MixtureSameFamily]
        """
        # Categorical distribution over mixture components
        mix = td.Categorical(logits=self.mixture_logits)
        
        # Component distributions (Gaussians)
        comp = td.Independent(td.Normal(loc=self.means, scale=self.stds), 1)
        
        # Mixture of Gaussians
        return td.MixtureSameFamily(mix, comp)
    
class FlowPrior(nn.Module):
    def __init__(self, M, n_transforms=4, n_hidden=64):
        """
        Define a normalizing flow prior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        n_transforms: [int]
           Number of coupling layers in the flow.
        n_hidden: [int]
           Number of hidden units in coupling networks.
        """
        super(FlowPrior, self).__init__()
        self.M = M
        
        # Base distribution (standard Gaussian)
        self.base = GaussianBase(M)
        
        # Create coupling layers with alternating masks
        transformations = []
        for i in range(n_transforms):
            # Alternate mask: first half vs second half
            mask = torch.zeros(M)
            if i % 2 == 0:
                mask[:M//2] = 1
            else:
                mask[M//2:] = 1
            
            # Scale network with tanh for stability
            scale_net = nn.Sequential(
                nn.Linear(M, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, M),
                nn.Tanh()
            )
            
            # Translation network
            translation_net = nn.Sequential(
                nn.Linear(M, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, M)
            )
            
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        
        # Create flow model
        self.flow = Flow(self.base, transformations)

    def forward(self):
        """
        Return the prior distribution (the flow itself).
        
        Returns:
        prior: [Flow]
            The flow model that can be used as a distribution.
        """
        return self.flow