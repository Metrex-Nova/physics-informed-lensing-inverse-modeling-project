"""Mass profile generators for synthetic gravitational lensing."""

import numpy as np


def coordinate_grid(shape, pixel_scale: float = 1.0):
    """Produce meshgrid coordinates centered at zero.

    Args:
        shape: (ny, nx)
        pixel_scale: physical size per pixel.

    Returns:
        xx, yy: coordinate arrays of shape (ny, nx)
    """

    ny, nx = shape
    y = (np.arange(ny) - (ny - 1) / 2) * pixel_scale
    x = (np.arange(nx) - (nx - 1) / 2) * pixel_scale
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def sis_kappa(shape, kappa0: float = 1.0, core_radius: float = 0.1, pixel_scale: float = 1.0):
    """Singular isothermal sphere convergence profile (softened).

    κ(r) ∝ 1 / sqrt(r^2 + r_core^2)
    """

    xx, yy = coordinate_grid(shape, pixel_scale=pixel_scale)
    r = np.sqrt(xx**2 + yy**2)
    return kappa0 / np.sqrt(r**2 + core_radius**2)


def nfw_kappa(shape, kappa_s: float = 0.5, r_s: float = 15.0, pixel_scale: float = 1.0):
    """Simplified NFW convergence profile.

    Uses a softened 2D projection of the 3D NFW profile.
    """

    xx, yy = coordinate_grid(shape, pixel_scale=pixel_scale)
    r = np.sqrt(xx**2 + yy**2) + 1e-9
    x = r / r_s
    # simplified convergence (not exact analytic formula), but captures core and outer slope
    kappa = kappa_s / (x * (1 + x) ** 2)
    # clamp to avoid huge values at center
    return np.clip(kappa, a_min=0.0, a_max=np.percentile(kappa, 99.5))
