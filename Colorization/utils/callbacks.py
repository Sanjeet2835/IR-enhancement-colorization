from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def get_callbacks(config: dict):

    checkpoint_callback = ModelCheckpoint(
        monitor=config["monitor"],
        mode=config["mode"],
        dirpath="./experiments-colorization/checkpoints",
        save_top_k=config["save_top_k"],
        save_last=True,

        filename="epoch={epoch:02d}-psnr={val_psnr:.2f}",

        auto_insert_metric_name=False,
    )

    early_stopping = EarlyStopping(
        monitor=config["monitor"],
        mode=config["mode"],

        patience=config["patience"],

        min_delta=config["min_delta"],

        verbose=True,
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch"
    )

    return [
        checkpoint_callback,
        early_stopping,
        lr_monitor,
    ]