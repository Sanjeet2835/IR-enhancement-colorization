from pathlib import Path

import numpy as np
import torch
import yaml

from inference_utils.scripts.unet import UNet


class ColorizationInference:
    """
    Thermal Image Colorization inference wrapper.
    """

    def __init__(
        self,
        device: str = "cuda",
    ) -> None:

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        weights_dir = PROJECT_ROOT / "weights"
        configs_dir = PROJECT_ROOT / "inference_utils" / "configs"

        # ---------------------------------------------------------
        # Load Model Configuration
        # ---------------------------------------------------------

        self.model = UNet()

        # ---------------------------------------------------------
        # Load Dataset Statistics
        # ---------------------------------------------------------

        with open(configs_dir / "dataset_stats.yaml", "r") as f:
            stats = yaml.safe_load(f)

        self.tir_mean = stats["tir_100m"]["mean"][0]
        self.tir_std = stats["tir_100m"]["std"][0]

        self.rgb_mean = torch.tensor(
            stats["rgb_100m"]["mean"],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.rgb_std = torch.tensor(
            stats["rgb_100m"]["std"],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        # ---------------------------------------------------------
        # Load Model Weights
        # ---------------------------------------------------------

        state_dict = torch.load(
            weights_dir / "colorization.pth",
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        tir_patch: np.ndarray,
    ) -> np.ndarray:
        """
        Colorize a single thermal patch.

        Parameters
        ----------
        tir_patch : np.ndarray
            Input thermal patch of shape (1, H, W).

        Returns
        -------
        np.ndarray
            RGB patch of shape (3, H, W).
        """

        x = torch.as_tensor(
            tir_patch,
            dtype=torch.float32,
        ).unsqueeze(0)

        # ---------------------------------------------------------
        # Normalize Input (TIR 100 m)
        # ---------------------------------------------------------

        x = (
            x - self.tir_mean
        ) / self.tir_std

        x = x.to(self.device)

        # ---------------------------------------------------------
        # Inference
        # ---------------------------------------------------------

        prediction = self.model(x)

        # ---------------------------------------------------------
        # De-normalize Output (RGB 100 m)
        # ---------------------------------------------------------

        prediction = (
            prediction * self.rgb_std.to(self.device)
            + self.rgb_mean.to(self.device)
        )

        prediction = prediction.squeeze(0)

        return prediction.cpu().numpy()