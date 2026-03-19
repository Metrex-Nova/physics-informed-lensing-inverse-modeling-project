"""Dataset factory for synthetic gravitational lensing examples."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from lensing.config import DataConfig
from lensing.data.mass_profiles import nfw_kappa, sis_kappa
from lensing.data.source_generator import random_source
from lensing.data.lensing_simulation import simulate_lensed_image, add_noise


class LensingDataset(Dataset):
    """Synthetic dataset for lensing inverse problems."""

    def __init__(
        self,
        num_examples: int,
        image_size: int = DataConfig().image_size,
        profile: str = "sis",
        noise_level: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.num_examples = num_examples
        self.image_size = image_size
        self.profile = profile
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

        self._data = []
        self._generate_examples()

    def _generate_examples(self):
        for idx in range(self.num_examples):
            kappa = self._sample_kappa(idx)
            source = random_source((self.image_size, self.image_size), n_blobs=3, seed=self.rng.randint(0, 1_000_000))

            kappa_t = torch.from_numpy(kappa).unsqueeze(0).unsqueeze(0).float()
            src_t = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float()
            lensed = simulate_lensed_image(kappa_t, src_t)

            if self.noise_level > 0:
                lensed = add_noise(lensed, sigma=self.noise_level, seed=self.rng.randint(0, 1_000_000))

            self._data.append(
                {
                    "lensed": lensed.squeeze(0),
                    "source": src_t.squeeze(0),
                    "kappa": kappa_t.squeeze(0),
                }
            )

    def _sample_kappa(self, idx: int) -> np.ndarray:
        if self.profile == "nfw":
            return nfw_kappa((self.image_size, self.image_size), kappa_s=0.8, r_s=self.image_size / 4)
        # default: SIS
        return sis_kappa((self.image_size, self.image_size), kappa0=1.0, core_radius=self.image_size * 0.02)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._data[idx]
