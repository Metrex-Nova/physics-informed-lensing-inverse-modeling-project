"""Lens equation utilities for ray tracing and image reconstruction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_meshgrid(shape: tuple[int, int], device=None) -> torch.Tensor:
    """Create a normalized grid for grid_sample in [-1, 1]."""

    ny, nx = shape
    y = torch.linspace(-1.0, 1.0, ny, device=device)
    x = torch.linspace(-1.0, 1.0, nx, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)
    return grid


def lens_image(source: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Lens a source image using the deflection field.

    Args:
        source: Tensor (B, 1, H, W)
        alpha: Tensor (B, 2, H, W)

    Returns:
        lensed image: Tensor (B, 1, H, W)
    """

    if source.ndim != 4 or alpha.ndim != 4:
        raise ValueError("source must be (B,1,H,W) and alpha must be (B,2,H,W)")

    B, _, H, W = source.shape
    device = source.device

    grid = _normalized_meshgrid((H, W), device=device).unsqueeze(0).expand(B, -1, -1, -1)

    # Alpha is in pixels; convert to normalized coordinates [-1,1]
    # Normalization factor: 2 / (N - 1)
    scale_x = 2.0 / (W - 1)
    scale_y = 2.0 / (H - 1)
    alpha_norm = torch.zeros_like(alpha)
    alpha_norm[:, 0] = alpha[:, 0] * scale_x
    alpha_norm[:, 1] = alpha[:, 1] * scale_y

    # Lens equation: x_s = x_i - alpha(x_i)
    sampling_grid = grid - alpha_norm.permute(0, 2, 3, 1)

    # grid_sample expects (B, C, H, W) input and (B, H, W, 2) grid
    lensed = F.grid_sample(
        source,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return lensed
