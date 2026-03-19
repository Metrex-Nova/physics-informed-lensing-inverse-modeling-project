"""Experiment script to compare baseline and physics-informed models."""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader

from lensing.config import get_device, data, model, training
from lensing.models.baseline_cnn import BaselineCNN
from lensing.models.unet import UNet
from lensing.utils.dataset import LensingDataset
from lensing.utils.metrics import mse, psnr
from lensing.utils.visualization import plot_image_grid
from lensing.data.lensing_simulation import simulate_lensed_image


def run():
    device = get_device()
    print(f"Running comparison on {device}")

    val_ds = LensingDataset(
        num_examples=data.num_val,
        image_size=data.image_size,
        profile="sis",
        noise_level=0.0,
        seed=999,
    )
    val_loader = DataLoader(val_ds, batch_size=data.batch_size)

    baseline = BaselineCNN(in_channels=1, out_channels=1, num_filters=model.base_channels).to(device)
    physics = UNet(in_channels=1, out_channels=1, base_channels=model.unet_channels).to(device)

    baseline_ckpt = os.path.join(training.checkpoint_dir, "baseline.pth")
    physics_ckpt = os.path.join(training.checkpoint_dir, "physics_informed.pth")

    if os.path.exists(baseline_ckpt):
        baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))
        print(f"Loaded baseline checkpoint: {baseline_ckpt}")
    else:
        print("Baseline checkpoint not found. Run `python main.py --mode train_baseline` first.")

    if os.path.exists(physics_ckpt):
        physics.load_state_dict(torch.load(physics_ckpt, map_location=device))
        print(f"Loaded physics-informed checkpoint: {physics_ckpt}")
    else:
        print("Physics-informed checkpoint not found. Run `python main.py --mode train_physics` first.")

    baseline.eval()
    physics.eval()

    metrics = {
        "baseline_kappa_mse": [],
        "physics_kappa_mse": [],
        "baseline_lensed_mse": [],
        "physics_lensed_mse": [],
    }

    with torch.no_grad():
        for batch in val_loader:
            lensed = batch["lensed"].to(device)
            kappa = batch["kappa"].to(device)
            source = batch["source"].to(device)

            pred_base = baseline(lensed)
            pred_phys = physics(lensed)

            lensed_base = simulate_lensed_image(pred_base, source)
            lensed_phys = simulate_lensed_image(pred_phys, source)

            metrics["baseline_kappa_mse"].append(mse(pred_base, kappa).item())
            metrics["physics_kappa_mse"].append(mse(pred_phys, kappa).item())
            metrics["baseline_lensed_mse"].append(mse(lensed_base, lensed).item())
            metrics["physics_lensed_mse"].append(mse(lensed_phys, lensed).item())

    report = {k: float(torch.tensor(v).mean()) for k, v in metrics.items()}

    print("\nComparison results (mean over validation set):")
    for name, val in report.items():
        print(f"  {name}: {val:.6f}")

    # save a visualization from the first batch
    sample = val_ds[0]
    with torch.no_grad():
        lensed = sample["lensed"].unsqueeze(0).to(device)
        kappa = sample["kappa"].unsqueeze(0).to(device)
        source = sample["source"].unsqueeze(0).to(device)
        pred_base = baseline(lensed)
        pred_phys = physics(lensed)
        lensed_base = simulate_lensed_image(pred_base, source)
        lensed_phys = simulate_lensed_image(pred_phys, source)

    out_dir = os.path.join(training.plot_dir, "compare")
    os.makedirs(out_dir, exist_ok=True)
    plot_image_grid(
        [lensed[0], kappa[0], pred_base[0], pred_phys[0], lensed[0], lensed_base[0], lensed_phys[0]],
        [
            "Input Lensed",
            "True κ",
            "Baseline κ",
            "Physics κ",
            "Input Lensed",
            "Baseline Re-lensed",
            "Physics Re-lensed",
        ],
        os.path.join(out_dir, "comparison.png"),
    )

    print(f"Saved comparison plot to {out_dir}")
