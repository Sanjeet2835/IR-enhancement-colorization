from pathlib import Path

import numpy as np
import torch
import yaml

from inference_utils.scripts.edsr import EDSR


class SRInference:
    """
    Super-Resolution inference wrapper.
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

        with open(configs_dir / "sr_yaml.yaml", "r") as f:
            config = yaml.safe_load(f)

        self.model = EDSR(
            num_features=config["num_features"],
            num_blocks=config["num_blocks"],
            scale_factor=config["scale_factor"],
        )

        # ---------------------------------------------------------
        # Load Dataset Statistics
        # ---------------------------------------------------------

        with open(configs_dir / "dataset_stats.yaml", "r") as f:
            stats = yaml.safe_load(f)

        self.lr_mean = stats["tir_200m"]["mean"][0]
        self.lr_std = stats["tir_200m"]["std"][0]

        self.hr_mean = stats["tir_100m"]["mean"][0]
        self.hr_std = stats["tir_100m"]["std"][0]

        # ---------------------------------------------------------
        # Load Model Weights
        # ---------------------------------------------------------

        state_dict = torch.load(
            weights_dir / "sr.pth",
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
        Super-resolve a single thermal patch.

        Parameters
        ----------
        tir_patch : np.ndarray
            Input patch of shape (1, H, W).

        Returns
        -------
        np.ndarray
            Super-resolved patch of shape (1, 2H, 2W).
        """

        x = torch.as_tensor(
            tir_patch,
            dtype=torch.float32,
        ).unsqueeze(0)

        # Normalize input (TIR 200 m)
        x = (
            x - self.lr_mean
        ) / self.lr_std

        x = x.to(self.device)

        # Inference
        prediction = self.model(x)

        # De-normalize output (TIR 100 m)
        prediction = (
            prediction * self.hr_std
            + self.hr_mean
        )

        prediction = prediction.squeeze(0)

        return prediction.cpu().numpy()