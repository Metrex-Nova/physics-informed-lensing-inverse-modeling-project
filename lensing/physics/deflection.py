"""Compute deflection angles from the lensing potential."""

from __future__ import annotations

import torch

from lensing.physics.poisson_solver import _frequency_grid


def deflection_from_potential(psi: torch.Tensor, pixel_scale: float = 1.0) -> torch.Tensor:
    """Compute deflection angles α = ∇ψ using FFT-based derivatives.

    Args:
        psi: Tensor of shape (..., 1, H, W) or (..., H, W)

    Returns:
        alpha: Tensor of shape (..., 2, H, W) (alpha_x, alpha_y)
    """

    squeeze = False
    if psi.ndim == 2:
        psi = psi.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif psi.ndim == 3:
        psi = psi.unsqueeze(1)

    # If psi has an explicit channel dimension, remove it for gradient operations.
    # This ensures the output deflection has shape (B, 2, H, W).
    if psi.shape[1] == 1:
        psi = psi.squeeze(1)

    batch, H, W = psi.shape
    device = psi.device

    kx, ky = _frequency_grid((H, W), pixel_scale=pixel_scale, device=device)

    psi_hat = torch.fft.fft2(psi)
    # derivative in x: i kx psi_hat
    alpha_x_hat = 1j * kx * psi_hat
    alpha_y_hat = 1j * ky * psi_hat

    alpha_x = torch.fft.ifft2(alpha_x_hat).real
    alpha_y = torch.fft.ifft2(alpha_y_hat).real

    alpha = torch.stack([alpha_x, alpha_y], dim=1)

    if squeeze:
        return alpha.squeeze(0)

    return alpha
