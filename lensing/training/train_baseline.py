"""Training loop for the baseline (non-physics) model."""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lensing.config import get_device, data, model, training
from lensing.models.baseline_cnn import BaselineCNN
from lensing.utils.dataset import LensingDataset
from lensing.utils.metrics import mse
from lensing.utils.visualization import plot_loss_curve, plot_image_grid


def run():
    device = get_device()
    print(f"Running baseline training on {device}")

    train_ds = LensingDataset(
        num_examples=data.num_train,
        image_size=data.image_size,
        profile="sis",
        noise_level=0.0,
        seed=123,
    )
    val_ds = LensingDataset(
        num_examples=data.num_val,
        image_size=data.image_size,
        profile="sis",
        noise_level=0.0,
        seed=456,
    )

    train_loader = DataLoader(train_ds, batch_size=data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=data.batch_size)

    net = BaselineCNN(in_channels=1, out_channels=1, num_filters=model.base_channels).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, training.epochs + 1):
        net.train()
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{training.epochs} [train]", unit="batch") as pbar:
            for batch in pbar:
                lensed = batch["lensed"].to(device)
                kappa = batch["kappa"].to(device)

                pred = net(lensed)
                loss = loss_fn(pred, kappa)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * lensed.shape[0]
                pbar.set_postfix(train_loss=loss.item())

        epoch_train_loss = running_loss / len(train_ds)
        train_losses.append(epoch_train_loss)

        net.eval()
        running_val_loss = 0.0
        with torch.no_grad(), tqdm(val_loader, desc=f"Epoch {epoch}/{training.epochs} [val]", unit="batch") as pbar:
            for batch in pbar:
                lensed = batch["lensed"].to(device)
                kappa = batch["kappa"].to(device)
                pred = net(lensed)
                running_val_loss += loss_fn(pred, kappa).item() * lensed.shape[0]
                pbar.set_postfix(val_loss=running_val_loss / ((pbar.n + 1) * data.batch_size))

        epoch_val_loss = running_val_loss / len(val_ds)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch:03d}/{training.epochs:03d}  train_loss={epoch_train_loss:.5f}  val_loss={epoch_val_loss:.5f}")

        if epoch % 10 == 0 or epoch == training.epochs:
            _save_visualizations(net, val_ds, device, epoch)

    os.makedirs(training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(training.checkpoint_dir, "baseline.pth")
    torch.save(net.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    plot_loss_curve(
        {"train": train_losses, "val": val_losses},
        os.path.join(training.plot_dir, "baseline_loss.png"),
    )


def _save_visualizations(net: nn.Module, dataset: LensingDataset, device: torch.device, epoch: int):
    net.eval()
    sample = dataset[0]
    lensed = sample["lensed"].unsqueeze(0).to(device)
    kappa_true = sample["kappa"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(lensed)

    out_dir = os.path.join(training.plot_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)

    plot_image_grid(
        [lensed[0], kappa_true[0], pred[0]],
        ["Lensed", "True kappa", "Predicted kappa"],
        os.path.join(out_dir, f"epoch_{epoch:03d}.png"),
    )
