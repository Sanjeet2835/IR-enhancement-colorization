from torch import nn

from .blocks import ResidualBlock, UpsampleBlock


class EDSR(nn.Module):
    """
    Lightweight Enhanced Deep Residual Network (EDSR)
    for 2× Thermal Infrared Super-Resolution.

    Architecture

    Input
        ↓
    Head Conv
        ↓
    Residual Blocks × N
        ↓
    Body Conv
        ↓
    Global Skip
        ↓
    PixelShuffle ×2
        ↓
    Output Conv
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        scale_factor: int = 2,
    ) -> None:

        super().__init__()

        # -----------------------
        # Head
        # -----------------------

        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # -----------------------
        # Body
        # -----------------------

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    num_features=num_features,
                    residual_scale=0.1,
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # -----------------------
        # Upsampling
        # -----------------------

        self.upsample = UpsampleBlock(
            num_features=num_features,
            scale_factor=scale_factor,
        )

        # -----------------------
        # Reconstruction
        # -----------------------

        self.reconstruction = nn.Conv2d(
            in_channels=num_features,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

    def forward(self, x):

        x = self.head(x)

        residual = x

        x = self.body(x)

        x = self.body_conv(x)

        x = x + residual

        x = self.upsample(x)

        x = self.reconstruction(x)

        return x