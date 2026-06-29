from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual Block used in EDSR.

    Conv → ReLU → Conv → Residual Scaling → Skip Connection
    """

    def __init__(
        self,
        num_features: int,
        kernel_size: int = 3,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.residual_scale = residual_scale

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            ),
        )

    def forward(self, x):
        residual = self.block(x)
        residual *= self.residual_scale

        return x + residual


class UpsampleBlock(nn.Module):
    """
    PixelShuffle upsampling block.

    Conv → PixelShuffle
    """

    def __init__(
        self,
        num_features: int,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features * (scale_factor ** 2),
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x):
        return self.upsample(x)