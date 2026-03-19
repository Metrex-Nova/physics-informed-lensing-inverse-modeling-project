"""Utilities for simulating lensing images from source and convergence maps."""

from __future__ import annotations

from typing import Optional

import torch

from lensing.physics.poisson_solver import solve_potential_fft
from lensing.physics.deflection import deflection_from_potential
from lensing.physics.lens_equation import lens_image


def simulate_lensed_image(
    kappa: torch.Tensor,
    source: torch.Tensor,
    pixel_scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Simulate a lensed image from a convergence map and source image.

    Args:
        kappa: [..., 1, H, W] convergence map (κ)
        source: [..., 1, H, W] source plane brightness
    Returns:
        lensed image: [..., 1, H, W]
    """

    if device is None:
        device = kappa.device

    # Ensure float tensor
    kappa = kappa.to(device=device, dtype=torch.float32)
    source = source.to(device=device, dtype=torch.float32)

    psi = solve_potential_fft(kappa)
    alpha = deflection_from_potential(psi)
    lensed = lens_image(source, alpha)
    return lensed


def add_noise(image: torch.Tensor, sigma: float = 0.01, seed: Optional[int] = None) -> torch.Tensor:
    """Add Gaussian noise to an image."""

    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(image) * sigma
    return image + noise
