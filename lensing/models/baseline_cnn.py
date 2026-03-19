"""Baseline CNN model for predicting convergence maps from lensed images."""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, num_filters: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
