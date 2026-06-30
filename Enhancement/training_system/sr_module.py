from pathlib import Path
import torch
import lightning as L
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Enhancement.model.edsr import EDSR
from Enhancement.utils.losses import CombinedLoss
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class SRLightningModule(L.LightningModule):
    """
    PyTorch Lightning Module for Thermal Infrared Super-Resolution.
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        num_features: int = 64,
        num_blocks: int = 8,
        scale_factor: int = 2,
        alpha: float = 0.8,
        beta: float = 0.2,
        eta_min: float = 1e-6
    ) -> None:

        super().__init__()

        # ---------------------------------------------------------
        # Save Hyperparameters
        # ---------------------------------------------------------

        self.save_hyperparameters()

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------

        self.model = EDSR(
            num_features=num_features,
            num_blocks=num_blocks,
            scale_factor=scale_factor,
        )

        # ---------------------------------------------------------
        # Load Dataset Statistics
        # ---------------------------------------------------------

        project_root = Path(__file__).resolve().parents[1]
        stats_file = project_root / "configs" / "dataset_stats.yaml"

        with open(stats_file, "r") as f:
            stats = yaml.safe_load(f)

        self.hr_mean = stats["tir_100m"]["mean"][0]
        self.hr_std = stats["tir_100m"]["std"][0]
        self.data_range = stats["tir_100m"]["data_range"][0]

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------

        self.loss_fn = CombinedLoss(
            data_range=self.data_range,
            alpha=alpha,
            beta=beta,
        )

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        self.psnr = PeakSignalNoiseRatio(
            data_range=self.data_range,
        )

        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0,
        )
    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(self, x):
        """
        Forward pass through the EDSR model.
        """
        return self.model(x)

    # ---------------------------------------------------------
    # Denormalization
    # ---------------------------------------------------------

    def denormalize(self, x):
        """
        Convert normalized tensors back to the original
        thermal radiance/temperature scale.
        """
        return x * self.hr_std + self.hr_mean
    
        # ---------------------------------------------------------
    # Training Step
    # ---------------------------------------------------------

    def training_step(self, batch, batch_idx):

        lr = batch["lr"]
        hr = batch["hr"]

        pred = self(lr)

        loss = self.loss_fn(
            pred,
            hr,
        )
        
        
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=lr.size(0),
        )

        return loss

    # ---------------------------------------------------------
    # Validation Step
    # ---------------------------------------------------------

    def validation_step(
        self,
        batch,
        batch_idx,
    ):

        lr = batch["lr"]
        hr = batch["hr"]

        pred = self(lr)

        loss = self.loss_fn(
            pred,
            hr,
        )

        pred_denorm = self.denormalize(pred).clamp(
            0,
            self.data_range,
        ).float()

        hr_denorm = self.denormalize(hr).clamp(
            0,
            self.data_range,
        ).float()

        psnr = self.psnr(
            pred_denorm,
            hr_denorm,
        )

        pred_metric = pred_denorm / self.data_range
        hr_metric = hr_denorm / self.data_range

        ssim = self.ssim(
            pred_metric,
            hr_metric,
        )

        self.log_dict(
            {
                "val_loss": loss,
                "val_psnr": psnr,
                "val_ssim": ssim,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=lr.size(0),
        )

        return loss


    # ---------------------------------------------------------
    # Test Step
    # ---------------------------------------------------------

    def test_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:

        lr = batch["lr"]
        hr = batch["hr"]

        pred = self(lr)

        loss = self.loss_fn(
            pred,
            hr,
        )

        pred_denorm = self.denormalize(pred).clamp(
            0,
            self.data_range,
        ).float()

        hr_denorm = self.denormalize(hr).clamp(
            0,
            self.data_range,
        ).float()

        psnr = self.psnr(
            pred_denorm,
            hr_denorm,
        )
        
        pred_metric = pred_denorm / self.data_range
        hr_metric = hr_denorm / self.data_range

        ssim = self.ssim(
            pred_metric,
            hr_metric,
        )

        self.log_dict(
            {
                "test_loss": loss,
                "test_psnr": psnr,
                "test_ssim": ssim,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=lr.size(0),
        )

        return loss
    
    # ---------------------------------------------------------
    # Optimizer & Scheduler
    # ---------------------------------------------------------

    def configure_optimizers(self):

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }