"""Visualization utilities for lensing reconstructions."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_image_grid(
    images: list[torch.Tensor],
    titles: list[str],
    out_path: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Save a 1xN grid of images side-by-side."""

    _ensure_dir(os.path.dirname(out_path))
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        img_np = img.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np[0]
        ax.imshow(img_np, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(losses: dict[str, list[float]], out_path: str) -> None:
    """Plot training/validation loss curves."""

    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in losses.items():
        ax.plot(values, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
