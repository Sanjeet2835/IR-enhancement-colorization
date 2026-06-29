from pathlib import Path
from typing import Union
import yaml

import numpy as np
import torch
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    """
    Dataset for Thermal Infrared Super-Resolution.

    Input
    -----
    tir_100m_512.npy

    Target
    ------
    rgb_100m_512.npy

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

        self.tir_mean = stats["tir_100m"]["mean"][0]
        self.tir_std = stats["tir_100m"]["std"][0]

        self.rgb_mean = torch.tensor(
            stats["rgb_100m"]["mean"],
            dtype=torch.float32,
        ).view(3, 1, 1)

        self.rgb_std = torch.tensor(
            stats["rgb_100m"]["std"],
            dtype=torch.float32,
        ).view(3, 1, 1)

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
        tir: np.ndarray,
        rgb: np.ndarray,
        tir_path: Path,
        rgb_path: Path,
    ) -> None:


        if tir.shape != (1, 512, 512):
            raise ValueError(
                f"Expected LR shape (256, 256), got {tir.shape}\n{tir_path}"
            )

        if rgb.shape != (3, 512, 512):
            raise ValueError(
                f"Expected HR shape (512, 512), got {rgb.shape}\n{rgb_path}"
            )

        if not np.isfinite(tir).all():
            raise ValueError(f"NaN/Inf detected in {tir_path}")

        if not np.isfinite(rgb).all():
            raise ValueError(f"NaN/Inf detected in {rgb_path}")

    @staticmethod
    def _augment(
        tir: np.ndarray,
        rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply identical geometric augmentations
        to TIR and RGB image pairs.
        """

        # Horizontal Flip
        if torch.rand(1).item() < 0.5:
            tir = np.flip(tir, axis=2).copy()
            rgb = np.flip(rgb, axis=2).copy()

        # Vertical Flip
        if torch.rand(1).item() < 0.5:
            tir = np.flip(tir, axis=1).copy()
            rgb = np.flip(rgb, axis=1).copy()

        # Random Rotation (0°, 90°, 180°, 270°)
        k = int(torch.randint(0, 4, (1,)).item())

        tir = np.rot90(tir, k, axes=(1, 2)).copy()
        rgb = np.rot90(rgb, k, axes=(1, 2)).copy()

        return tir, rgb

    def __getitem__(self, idx: int) -> dict:

        sample = self.samples[idx]

        parts = sample.split("/")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid sample entry '{sample}' in {self.split_file}"
            )

        product_id, sample_id = parts

        sample_dir = self.root_dir / product_id / sample_id

        tir_path = sample_dir / "tir_100m_512.npy"
        rgb_path = sample_dir / "rgb_100m_512.npy"

        if not tir_path.exists():
            raise FileNotFoundError(f"Missing file:\n{tir_path}")

        if not rgb_path.exists():
            raise FileNotFoundError(f"Missing file:\n{rgb_path}")

        tir = np.load(tir_path, allow_pickle=False)
        rgb = np.load(rgb_path, allow_pickle=False)

        self._validate(tir, rgb, tir_path, rgb_path)

        if self.augment:
            tir, rgb = self._augment(tir, rgb)

        tir = torch.as_tensor(tir, dtype=torch.float32)
        rgb = torch.as_tensor(rgb, dtype=torch.float32)

        if self.normalize:
            tir = (tir - self.tir_mean) / self.tir_std
            rgb = (rgb - self.rgb_mean) / self.rgb_std

        return {
            "tir": tir,
            "rgb": rgb,
            "product_id": product_id,
            "sample_id": sample_id,
        }