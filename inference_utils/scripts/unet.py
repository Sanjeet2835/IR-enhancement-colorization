import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3×3 convolutions with Batch Normalization
    and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)
    
class DownBlock(nn.Module):
    """
    Downsampling block consisting of MaxPooling
    followed by a DoubleConv block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            DoubleConv(
                in_channels,
                out_channels,
            ),
        )

    def forward(self, x):

        return self.block(x)
    
class UpBlock(nn.Module):
    """
    Upsampling block consisting of transposed convolution,
    skip connection concatenation, and DoubleConv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        x = self.conv(x)

        return x
    
class ResidualBlock(nn.Module):
    """
    Residual block used in the bottleneck of the U-Net.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        x = self.block(x)

        x = x + identity

        x = self.relu(x)

        return x
    

class UNet(nn.Module):
    """
    U-Net for Thermal Infrared Colorization.
    """

    def __init__(self):

        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.input_block = DoubleConv(
            1,
            64,
        )

        self.down1 = DownBlock(
            64,
            128,
        )

        self.down2 = DownBlock(
            128,
            256,
        )

        self.down3 = DownBlock(
            256,
            512,
        )

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
        )

        # -------------------------
        # Decoder
        # -------------------------

        self.up3 = UpBlock(
            512,
            256,
        )

        self.up2 = UpBlock(
            256,
            128,
        )

        self.up1 = UpBlock(
            128,
            64,
        )

        # -------------------------
        # Output
        # -------------------------

        self.output = nn.Conv2d(
            64,
            3,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Encoder

        x1 = self.input_block(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        # Bottleneck

        x = self.bottleneck(x4)

        # Decoder

        x = self.up3(x, x3)

        x = self.up2(x, x2)

        x = self.up1(x, x1)

        return self.output(x)