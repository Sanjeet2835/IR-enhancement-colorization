import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml


class ColorizationLoss(nn.Module):
    """
    Total loss for thermal image colorization.

    L = L1
      + λg * Gradient Loss
      + λr * Range Loss

    Notes
    -----
    All losses are computed in the normalized space.

    The dataset is z-score normalized before being passed to the network. 
    Physical reflectance bounds [0, 1] are converted once
    into normalized bounds using the dataset statistics.
    """

    def __init__(
        self,
        lambda_gradient: float = 0.1,
        lambda_range: float = 0.01,
    ) -> None:

        super().__init__()

        self.lambda_gradient = lambda_gradient
        self.lambda_range = lambda_range

        self.l1 = nn.L1Loss()

        project_root = Path(__file__).resolve().parents[1]

        stats_file = (
            project_root
            / "configs"
            / "dataset_stats.yaml"
        )

        with open(stats_file, "r") as f:
            stats = yaml.safe_load(f)

        rgb_mean = torch.tensor(
            stats["rgb_100m"]["mean"],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        rgb_std = torch.tensor(
            stats["rgb_100m"]["std"],
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "lower_bound",
            (0.0 - rgb_mean) / rgb_std,
        )

        self.register_buffer(
            "upper_bound",
            (1.0 - rgb_mean) / rgb_std,
        )

        sobel_x = torch.tensor(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer(
            "sobel_x",
            sobel_x,
        )

        self.register_buffer(
            "sobel_y",
            sobel_y,
        )

    def gradient_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        channels = prediction.shape[1]

        sobel_x = self.sobel_x.repeat(
            channels,
            1,
            1,
            1,
        )

        sobel_y = self.sobel_y.repeat(
            channels,
            1,
            1,
            1,
        )

        pred_dx = F.conv2d(
            prediction,
            sobel_x,
            padding=1,
            groups=channels,
        )

        pred_dy = F.conv2d(
            prediction,
            sobel_y,
            padding=1,
            groups=channels,
        )

        target_dx = F.conv2d(
            target,
            sobel_x,
            padding=1,
            groups=channels,
        )

        target_dy = F.conv2d(
            target,
            sobel_y,
            padding=1,
            groups=channels,
        )

        return (
            self.l1(pred_dx, target_dx)
            + self.l1(pred_dy, target_dy)
        )

    def range_loss(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:

        upper = F.relu(
            prediction - self.upper_bound
        )

        lower = F.relu(
            self.lower_bound - prediction
        )

        return (
            upper.mean()
            + lower.mean()
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        l1_loss = self.l1(
            prediction,
            target,
        )

        grad_loss = self.gradient_loss(
            prediction,
            target,
        )

        range_loss = self.range_loss(
            prediction,
        )

        total_loss = (
            l1_loss
            + self.lambda_gradient * grad_loss
            + self.lambda_range * range_loss
        )

        return total_loss