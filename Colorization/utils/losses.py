import torch
import torch.nn as nn


class ColorizationLoss(nn.Module):
    """
    L1 reconstruction loss for Thermal Image Colorization.
    """

    def __init__(self) -> None:

        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return self.l1(
            prediction,
            target,
        )