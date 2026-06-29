from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class ColorizationMetrics:
    """
    Image quality metrics for Thermal Image Colorization.

    Metrics
    -------
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index Measure)

    NOTE:
    These metrics are intended to be computed on
    de-normalized RGB images inside the LightningModule.
    """

    def __init__(
        self,
        data_range: float,
    ) -> None:

        self.psnr = PeakSignalNoiseRatio(
            data_range=data_range,
        )

        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=data_range,
        )