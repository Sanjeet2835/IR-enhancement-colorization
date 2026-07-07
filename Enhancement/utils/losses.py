import torch
import torch.nn as nn


class EnhancementLoss(nn.Module):
    """
    Total enhancement loss.

    L = L1
      + λphysics * Conservation Loss

    Notes
    -----
    The conservation loss enforces energy consistency between
    the predicted 100 m thermal image and the observed 200 m
    thermal image. A 2×2 average pooling operation is applied
    to the prediction before comparison.
    """

    def __init__(
        self,
        lambda_physics: float = 0.1,
    ) -> None:

        super().__init__()

        self.lambda_physics = lambda_physics

        self.l1 = nn.L1Loss()

        self.avg_pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

    def conservation_loss(
        self,
        prediction: torch.Tensor,
        input_200m: torch.Tensor,
    ) -> torch.Tensor:

        pooled_prediction = self.avg_pool(
            prediction
        )

        return self.l1(
            pooled_prediction,
            input_200m,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        input_200m: torch.Tensor,
    ) -> torch.Tensor:

        l1_loss = self.l1(
            prediction,
            target,
        )

        physics_loss = self.conservation_loss(
            prediction,
            input_200m,
        )

        total_loss = (
            l1_loss
            + self.lambda_physics * physics_loss
        )

        return total_loss