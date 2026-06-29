from pathlib import Path
from typing import Union

import lightning as L
from torch.utils.data import DataLoader

from .dataset import ColorizationDataset


class ColorizationDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for Thermal Image Colorization.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split_dir: Union[str, Path],
        batch_size: int,
        num_workers: int,
        normalize: bool = True,
        pin_memory: bool = True,
    ) -> None:

        super().__init__()

        self.data_dir = Path(data_dir)
        self.split_dir = Path(split_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.normalize = normalize


    def setup(self, stage: str | None = None) -> None:

        if stage == "fit" or stage is None:

            self.train_dataset = ColorizationDataset(
                root_dir=self.data_dir,
                split_file=self.split_dir / "train.txt",
                normalize=self.normalize,
                augment=True,
            )

            self.val_dataset = ColorizationDataset(
                root_dir=self.data_dir,
                split_file=self.split_dir / "val.txt",
                normalize=self.normalize,
                augment=False,
            )

        if stage == "test" or stage is None:

            self.test_dataset = ColorizationDataset(
                root_dir=self.data_dir,
                split_file=self.split_dir / "test.txt",
                normalize=self.normalize,
                augment=False,
            )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )