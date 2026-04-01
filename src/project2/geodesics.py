"""Core geodesic computation for pull-back and ensemble VAE geometry.

Implements:
- Single-decoder pull-back curve energy
- Ensemble (model-average) curve energy (Eq. 1 from project description)
- Geodesic computation via optimization of interior curve points
"""

import torch
import torch.nn as nn


def curve_energy(curve_points: torch.Tensor, decoder: nn.Module) -> torch.Tensor:
    """Compute discrete pull-back curve energy using a single decoder.

    E(c) = sum_i ||f(c(t_i)) - f(c(t_{i+1}))||^2
    where f is the decoder mean function.

    Args:
        curve_points: (N+1, latent_dim) — full curve including endpoints.
        decoder: Decoder whose forward(z) returns a distribution with .mean.

    Returns:
        Scalar — the curve energy.
    """
    decoded = decoder(curve_points).mean
    decoded_flat = decoded.reshape(decoded.shape[0], -1)
    diffs = decoded_flat[1:] - decoded_flat[:-1]
    energy = (diffs**2).sum()
    return energy


def ensemble_curve_energy(
    curve_points: torch.Tensor,
    decoders: list[nn.Module],
    num_mc_samples: int = 50,
) -> torch.Tensor:
    """Compute model-average curve energy using an ensemble of decoders.

    Eq. 1:  E(c) ≈ sum_i E_{l,k} ||f_l(c(t_i)) - f_k(c(t_{i+1}))||^2
    where f_l, f_k drawn uniformly from the decoder ensemble.

    When len(decoders) == 1, reduces to single-decoder energy.

    Args:
        curve_points: (N+1, latent_dim).
        decoders: List of K decoder modules.
        num_mc_samples: Number of (l, k) pairs for MC estimate.

    Returns:
        Scalar — model-average curve energy.
    """
    K = len(decoders)
    device = curve_points.device

    if K == 1:
        return curve_energy(curve_points, decoders[0])

    # Pre-decode all points with all decoders: list of (N+1, D)
    all_decoded = []
    for dec in decoders:
        dec_out = dec(curve_points).mean
        all_decoded.append(dec_out.reshape(dec_out.shape[0], -1))
    # (K, N+1, D)
    all_decoded = torch.stack(all_decoded, dim=0)

    # Sample random (l, k) pairs
    l_indices = torch.randint(0, K, (num_mc_samples,), device=device)
    k_indices = torch.randint(0, K, (num_mc_samples,), device=device)

    # f_l(c(t_i)) for i=0..N-1 and f_k(c(t_{i+1})) for i=0..N-1
    fl_points = all_decoded[l_indices, :-1, :]  # (num_mc, N, D)
    fk_points = all_decoded[k_indices, 1:, :]  # (num_mc, N, D)

    diffs = fl_points - fk_points
    sq_norms = (diffs**2).sum(dim=-1)  # (num_mc, N)

    # Average over MC samples, sum over segments
    energy = sq_norms.mean(dim=0).sum()
    return energy


def compute_geodesic(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    decoders: list[nn.Module],
    num_t: int = 20,
    lr: float = 0.01,
    num_steps: int = 1000,
    num_mc_samples: int = 50,
    lr_decay: float = 0.5,
    decay_every: int = 300,
    verbose: bool = False,
) -> torch.Tensor:
    """Compute a geodesic by optimizing interior curve points.

    Initializes as straight line, optimizes interior points to minimize
    the (ensemble) curve energy via Adam.

    Args:
        z_start: (latent_dim,) starting point.
        z_end: (latent_dim,) ending point.
        decoders: List of decoder modules (length 1 for single decoder).
        num_t: Number of interior points (total curve = num_t + 2).
        lr: Initial Adam learning rate.
        num_steps: Optimization steps.
        num_mc_samples: MC samples for ensemble energy.
        lr_decay: LR decay factor.
        decay_every: Steps between LR decays.
        verbose: Print energy during optimization.

    Returns:
        (num_t + 2, latent_dim) — optimized curve including endpoints.
    """
    device = z_start.device

    # Initialize as linear interpolation
    t_vals = torch.linspace(0, 1, num_t + 2, device=device)[1:-1]
    interior = z_start.unsqueeze(0) + t_vals.unsqueeze(1) * (z_end - z_start).unsqueeze(0)
    interior = nn.Parameter(interior.clone())

    optimizer = torch.optim.Adam([interior], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_every, gamma=lr_decay)

    for step in range(num_steps):
        optimizer.zero_grad()
        full_curve = torch.cat([z_start.unsqueeze(0), interior, z_end.unsqueeze(0)], dim=0)

        if len(decoders) == 1:
            energy = curve_energy(full_curve, decoders[0])
        else:
            energy = ensemble_curve_energy(full_curve, decoders, num_mc_samples)

        energy.backward()
        optimizer.step()
        scheduler.step()

        if verbose and step % 200 == 0:
            print(f"  Step {step:4d}: energy = {energy.item():.4f}")

    with torch.no_grad():
        full_curve = torch.cat([z_start.unsqueeze(0), interior.data, z_end.unsqueeze(0)], dim=0)
    return full_curve


def geodesic_distance(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    decoders: list[nn.Module],
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute geodesic distance = sqrt(energy) of the optimized curve.

    Returns:
        (curve, distance): optimized curve and scalar geodesic distance.
    """
    curve = compute_geodesic(z_start, z_end, decoders, **kwargs)

    with torch.no_grad():
        if len(decoders) == 1:
            energy = curve_energy(curve, decoders[0])
        else:
            energy = ensemble_curve_energy(curve, decoders, num_mc_samples=kwargs.get("num_mc_samples", 100))
        dist = torch.sqrt(energy)

    return curve, dist


def euclidean_distance(z_start: torch.Tensor, z_end: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance in latent space."""
    return torch.norm(z_end - z_start, p=2)
