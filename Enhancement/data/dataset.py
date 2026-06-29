from pathlib import Path
from typing import Union
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """
    Dataset for Thermal Infrared Super-Resolution.

    Input
    -----
    tir_200m.npy

    Target
    ------
    tir_100m_512.npy

    Expected directory structure
    ----------------------------
    output/
        patches/
            LC09_xxxxx/
                sample_001/
                    tir_200m.npy
                    tir_100m_512.npy
                    rgb_100m_512.npy

    Split file format
    -----------------
    LC09_xxxxx/sample_001
    LC09_xxxxx/sample_002
    ...
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split_file: Union[str, Path],
        normalize: bool = True,
        augment: bool = False,
    ) -> None:

        self.root_dir = Path(root_dir)
        self.split_file = Path(split_file)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.root_dir}"
            )

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Split file not found:\n{self.split_file}"
            )

        self.normalize = normalize
        self.augment = augment

        project_root = Path(__file__).resolve().parents[1]
        stats_file = project_root / "configs" / "dataset_stats.yaml"

        with open(stats_file, "r") as f:
            stats = yaml.safe_load(f)

        self.lr_mean = stats["tir_200m"]["mean"][0]
        self.lr_std = stats["tir_200m"]["std"][0]

        self.hr_mean = stats["tir_100m"]["mean"][0]
        self.hr_std = stats["tir_100m"]["std"][0]

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.samples = [line.strip() for line in f if line.strip()]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found inside {self.split_file}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _validate(
        lr: np.ndarray,
        hr: np.ndarray,
        lr_path: Path,
        hr_path: Path,
    ) -> None:

        if lr.shape != (1, 256, 256):
            raise ValueError(
                f"Expected LR shape (256, 256), got {lr.shape}\n{lr_path}"
            )

        if hr.shape != (1, 512, 512):
            raise ValueError(
                f"Expected HR shape (512, 512), got {hr.shape}\n{hr_path}"
            )

        if not np.isfinite(lr).all():
            raise ValueError(f"NaN/Inf detected in {lr_path}")

        if not np.isfinite(hr).all():
            raise ValueError(f"NaN/Inf detected in {hr_path}")

    @staticmethod
    def _augment(
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply identical geometric augmentations
        to LR and HR images.
        """

        # Horizontal Flip
        if torch.rand(1).item() < 0.5:
            lr = np.flip(lr, axis=2).copy()
            hr = np.flip(hr, axis=2).copy()

        # Vertical Flip
        if torch.rand(1).item() < 0.5:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()


        # Random rotation (0°, 90°, 180°, 270°)
        k = int(torch.randint(0, 4, (1,)).item())

        lr = np.rot90(lr, k, axes=(1, 2)).copy()
        hr = np.rot90(hr, k, axes=(1, 2)).copy()

        return lr, hr

    def __getitem__(self, idx: int) -> dict:

        sample = self.samples[idx]

        parts = sample.split("/")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid sample entry '{sample}' in {self.split_file}"
            )

        product_id, sample_id = parts

        sample_dir = self.root_dir / product_id / sample_id

        lr_path = sample_dir / "tir_200m.npy"
        hr_path = sample_dir / "tir_100m_512.npy"

        if not lr_path.exists():
            raise FileNotFoundError(f"Missing file:\n{lr_path}")

        if not hr_path.exists():
            raise FileNotFoundError(f"Missing file:\n{hr_path}")

        lr = np.load(lr_path, allow_pickle=False)
        hr = np.load(hr_path, allow_pickle=False)

        self._validate(lr, hr, lr_path, hr_path)

        if self.augment:
            lr, hr = self._augment(lr, hr)

        lr = torch.as_tensor(lr, dtype=torch.float32)
        hr = torch.as_tensor(hr, dtype=torch.float32)

        if self.normalize:
            lr = (lr - self.lr_mean) / self.lr_std
            hr = (hr - self.hr_mean) / self.hr_std

        return {
            "lr": lr,
            "hr": hr,
            "product_id": product_id,
            "sample_id": sample_id,
        }