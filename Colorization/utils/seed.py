import lightning as L
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for reproducible experiments.
    """

    L.seed_everything(seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False