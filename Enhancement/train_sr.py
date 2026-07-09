from pathlib import Path
import torch
import lightning as L
import yaml
from lightning.pytorch.loggers import MLFlowLogger

from Enhancement.data.datamodule import SRDataModule
from Enhancement.training_system.sr_module import SRLightningModule
from Enhancement.utils.callbacks import get_callbacks
from Enhancement.utils.seed import seed_everything


def load_config(config_path: str | Path) -> dict:
    """
    Load experiment configuration.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "sr_config.yaml"
    )

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    seed_everything(config["seed"])

    # ---------------------------------------------------------
    # Data Module
    # ---------------------------------------------------------

    datamodule = SRDataModule(
        data_dir=config["data_root"],
        split_dir=config["split_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    # ---------------------------------------------------------
    # Lightning Module
    # ---------------------------------------------------------
    
    model = SRLightningModule(
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        max_epochs=int(config["max_epochs"]),

        num_features=int(config["num_features"]),
        num_blocks=int(config["num_blocks"]),
        scale_factor=int(config["scale_factor"]),

        lambda_physics=0.1,

        eta_min=float(config["eta_min"]),
    )

    # ---------------------------------------------------------
    # MLflow Logger
    # ---------------------------------------------------------

    logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=config["run_name"],
        tracking_uri="file:./experiments-enhancement/mlruns",
        log_model=False,
    )


    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = L.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],

        max_epochs=config["max_epochs"],

        precision=config["precision"],

        callbacks=get_callbacks(config),

        logger=logger,

        log_every_n_steps=config["log_every_n_steps"],
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

    # ---------------------------------------------------------
    # Testing
    # ---------------------------------------------------------

    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path="best",
    )

    
    # ---------------------------------------------------------
    # Export Best Model as .pth
    # ---------------------------------------------------------

    best_model = SRLightningModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
    )

    output_path = (
        project_root
        / "weights"
        / "edsr.pth"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        best_model.model.state_dict(),
        output_path,
    )

    print(
        f"Saved model weights to:\n{output_path}"
    )

if __name__ == "__main__":
    main()