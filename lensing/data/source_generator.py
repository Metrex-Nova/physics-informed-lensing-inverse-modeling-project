"""Source-plane image generators for lensing simulations."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def gaussian_blob(shape: Tuple[int, int], center, amplitude=1.0, sigma=5.0):
    """Generate a single 2D gaussian blob."""

    ny, nx = shape
    y = np.arange(ny) - (ny - 1) / 2
    x = np.arange(nx) - (nx - 1) / 2
    xx, yy = np.meshgrid(x, y)
    dx = xx - center[0]
    dy = yy - center[1]
    r2 = dx**2 + dy**2
    return amplitude * np.exp(-0.5 * r2 / (sigma**2))


def random_source(shape: Tuple[int, int], n_blobs: int = 3, seed: Optional[int] = None):
    """Generate a random source image composed of multiple gaussian blobs."""

    if seed is not None:
        np.random.seed(seed)

    ny, nx = shape
    image = np.zeros(shape, dtype=np.float32)

    for _ in range(n_blobs):
        center = (
            np.random.uniform(-nx / 4, nx / 4),
            np.random.uniform(-ny / 4, ny / 4),
        )
        amplitude = np.random.uniform(0.5, 1.0)
        sigma = np.random.uniform(2.0, 10.0)
        image += gaussian_blob(shape, center=center, amplitude=amplitude, sigma=sigma)

    # normalize to [0,1]
    image = image - image.min()
    if image.max() > 0:
        image = image / image.max()
    return image.astype(np.float32)
