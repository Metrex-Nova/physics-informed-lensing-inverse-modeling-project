"""Experiment script to evaluate noise robustness of trained models."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from lensing.config import get_device, data, experiment, model, training
from lensing.models.baseline_cnn import BaselineCNN
from lensing.models.unet import UNet
from lensing.utils.dataset import LensingDataset
from lensing.utils.metrics import mse


def run():
    device = get_device()
    print(f"Running noise robustness experiment on {device}")

    baseline = BaselineCNN(in_channels=1, out_channels=1, num_filters=model.base_channels).to(device)
    physics = UNet(in_channels=1, out_channels=1, base_channels=model.unet_channels).to(device)

    baseline_ckpt = os.path.join(training.checkpoint_dir, "baseline.pth")
    physics_ckpt = os.path.join(training.checkpoint_dir, "physics_informed.pth")

    if os.path.exists(baseline_ckpt):
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))
    else:
        raise FileNotFoundError("Baseline checkpoint not found; run training first.")

    if os.path.exists(physics_ckpt):
        physics.load_state_dict(torch.load(physics_ckpt, map_location=device))
    else:
        raise FileNotFoundError("Physics-informed checkpoint not found; run training first.")

    baseline.eval()
    physics.eval()

    noise_levels = list(experiment.noise_levels)
    baseline_losses = []
    physics_losses = []

    for noise in noise_levels:
        ds = LensingDataset(
            num_examples=data.num_val,
            image_size=data.image_size,
            profile="sis",
            noise_level=noise,
            seed=12345,
        )
        loader = DataLoader(ds, batch_size=data.batch_size)

        baseline_loss = 0.0
        physics_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                lensed = batch["lensed"].to(device)
                kappa = batch["kappa"].to(device)

                pred_base = baseline(lensed)
                pred_phys = physics(lensed)

                baseline_loss += mse(pred_base, kappa).item() * lensed.shape[0]
                physics_loss += mse(pred_phys, kappa).item() * lensed.shape[0]

        baseline_losses.append(baseline_loss / len(ds))
        physics_losses.append(physics_loss / len(ds))

        print(f"noise={noise:.3f}  baseline_mse={baseline_losses[-1]:.5f}  physics_mse={physics_losses[-1]:.5f}")

    out_dir = os.path.join(training.plot_dir, "noise_robustness")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(noise_levels, baseline_losses, marker="o", label="Baseline")
    plt.plot(noise_levels, physics_losses, marker="o", label="Physics-informed")
    plt.xlabel("Input noise σ")
    plt.ylabel("MSE on κ")
    plt.title("Noise robustness")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "noise_robustness.png"), dpi=150)
    plt.close()

    print(f"Saved noise robustness plot to {out_dir}")
