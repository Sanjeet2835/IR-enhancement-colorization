from pathlib import Path

import lightning as L
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from model.unet import UNet
from utils.losses import ColorizationLoss

class ColorizationLightningModule(L.LightningModule):
    """
    PyTorch Lightning Module for Thermal Image Colorization.
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        max_epochs: int,
        eta_min: float = 1e-6,
    ) -> None:

        super().__init__()

        self.save_hyperparameters()

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------

        self.model = UNet()

        # ---------------------------------------------------------
        # Load Dataset Statistics
        # ---------------------------------------------------------

        project_root = Path(__file__).resolve().parents[1]
        stats_file = project_root / "configs" / "dataset_stats.yaml"

        with open(stats_file, "r") as f:
            stats = yaml.safe_load(f)

        self.rgb_mean = torch.tensor(
            stats["rgb_100m"]["mean"],
            dtype=torch.float32,
        ).view(3, 1, 1)

        self.rgb_std = torch.tensor(
            stats["rgb_100m"]["std"],
            dtype=torch.float32,
        ).view(3, 1, 1)

        self.data_range = max(
            stats["rgb_100m"]["data_range"]
        )

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------

        self.loss_fn = ColorizationLoss()

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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        """
        return self.model(x)

    # ---------------------------------------------------------
    # Denormalization
    # ---------------------------------------------------------

    def denormalize(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert normalized RGB tensors back to the
        original RGB value range.
        """

        mean = self.rgb_mean.to(x.device)
        std = self.rgb_std.to(x.device)

        return x * std + mean
    
    # ---------------------------------------------------------
    # Training Step
    # ---------------------------------------------------------

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:

        tir = batch["tir"]
        rgb = batch["rgb"]

        pred = self(tir)

        loss = self.loss_fn(
            pred,
            rgb,
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=tir.size(0),
        )

        return loss

    # ---------------------------------------------------------
    # Validation Step
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Validation Step
    # ---------------------------------------------------------

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:

        tir = batch["tir"]
        rgb = batch["rgb"]

        pred = self(tir)

        loss = self.loss_fn(
            pred,
            rgb,
        )

        # ---------------------------------------------------------
        # De-normalize for evaluation
        # ---------------------------------------------------------

        pred_rgb = self.denormalize(pred).clamp(
            0,
            self.data_range,
        ).float()

        gt_rgb = self.denormalize(rgb).clamp(
            0,
            self.data_range,
        ).float()

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        psnr = self.psnr(
            pred_rgb,
            gt_rgb,
        )

        pred_metric = pred_rgb / self.data_range
        gt_metric = gt_rgb / self.data_range

        ssim = self.ssim(
            pred_metric,
            gt_metric,
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
            batch_size=tir.size(0),
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

        tir = batch["tir"]
        rgb = batch["rgb"]

        pred = self(tir)

        loss = self.loss_fn(
            pred,
            rgb,
        )

        # ---------------------------------------------------------
        # De-normalize for evaluation
        # ---------------------------------------------------------

        pred_rgb = self.denormalize(pred).clamp(
            0,
            self.data_range,
        ).float()

        gt_rgb = self.denormalize(rgb).clamp(
            0,
            self.data_range,
        ).float()

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        psnr = self.psnr(
            pred_rgb,
            gt_rgb,
        )

        pred_metric = pred_rgb / self.data_range
        gt_metric = gt_rgb / self.data_range

        ssim = self.ssim(
            pred_metric,
            gt_metric,
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
            batch_size=tir.size(0),
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