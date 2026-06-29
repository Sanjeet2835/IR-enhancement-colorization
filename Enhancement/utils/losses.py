import torch
import torch.nn as nn
import piq





class CombinedLoss(nn.Module):
    """
    Combined loss for Thermal Infrared Super-Resolution.

    Total Loss =
        alpha * L1 +
        beta * (1 - SSIM)
    """

    def __init__(
        self,
        data_range: float,
        alpha: float = 0.8,
        beta: float = 0.2,
    ) -> None:

        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.l1 = nn.L1Loss()

        # self.ssim = piq.SSIMLoss(
        #     data_range=data_range
        # )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        l1_loss = self.l1(
            prediction,
            target,
        )

        # ssim_loss = self.ssim(
        #     prediction,
        #     target,
        # )

        # total_loss = (
        #     self.alpha * l1_loss
        #     + self.beta * ssim_loss
        # )

        total_loss = l1_loss
        
        return total_loss