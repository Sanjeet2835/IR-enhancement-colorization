from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class SRMetrics:
    """
    Image quality metrics for Thermal Infrared Super-Resolution.

    Metrics
    -------
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index Measure)

    NOTE:
    These metrics should be computed on
    de-normalized images inside the LightningModule.
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