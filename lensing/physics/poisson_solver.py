"""FFT-based Poisson solver for lensing potential."""

from __future__ import annotations

import torch


def _frequency_grid(shape: tuple[int, int], pixel_scale: float = 1.0, device=None):
    """Construct kx, ky frequency grids for FFT differentiation."""

    ny, nx = shape
    fy = torch.fft.fftfreq(ny, d=pixel_scale, device=device) * 2.0 * torch.pi
    fx = torch.fft.fftfreq(nx, d=pixel_scale, device=device) * 2.0 * torch.pi
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    return kx, ky


def solve_potential_fft(kappa: torch.Tensor, pixel_scale: float = 1.0) -> torch.Tensor:
    """Solve ∇² ψ = κ using an FFT-based spectral solver.

    Args:
        kappa: Tensor of shape (..., 1, H, W) or (..., H, W)
        pixel_scale: physical size per pixel.

    Returns:
        psi: same shape as kappa.
    """

    orig_shape = kappa.shape
    squeeze = False
    if kappa.ndim == 2:
        kappa = kappa.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif kappa.ndim == 3:
        # assume (B, H, W)
        kappa = kappa.unsqueeze(1)

    batch, _, H, W = kappa.shape
    device = kappa.device

    kx, ky = _frequency_grid((H, W), pixel_scale=pixel_scale, device=device)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid division by zero at zero frequency

    kappa_hat = torch.fft.fft2(kappa)
    psi_hat = -kappa_hat / k2
    psi_hat[..., 0, 0] = 0.0
    psi = torch.fft.ifft2(psi_hat).real

    if squeeze:
        return psi.squeeze(0).squeeze(0)
    if orig_shape == psi.shape:
        return psi
    return psi
