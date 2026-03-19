"""Configuration and hyperparameters for lensing training and evaluation."""

from dataclasses import dataclass
import torch


@dataclass
class DataConfig:
    image_size: int = 128
    num_train: int = 500
    num_val: int = 100
    batch_size: int = 16


@dataclass
class TrainingConfig:
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_physics: float = 1.0
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"


@dataclass
class ModelConfig:
    base_channels: int = 32
    unet_channels: int = 32


@dataclass
class ExperimentConfig:
    noise_levels: list[float] = (0.0, 0.02, 0.05)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = DataConfig()
training = TrainingConfig()
model = ModelConfig()
experiment = ExperimentConfig()
