"""Common metrics for evaluating lensing reconstructions."""

import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse_val = mse(a, b)
    if mse_val == 0:
        return torch.tensor(float("inf"), device=mse_val.device)
    return 10.0 * torch.log10(max_val**2 / mse_val)
