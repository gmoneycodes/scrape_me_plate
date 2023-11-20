import lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from typing import Callable

from source.datamodules.datasets import DetectionDataset


class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=8, transforms: Callable = None,
                 fast_dev_run=False, ext: str = ".jpg", data_dir="yolov7", **kwargs):
        """
        :param batch_size: n of samples to feed together to the model
        :param num_workers: n workers to use to use for dataloaders
        :param transforms: transformations to apply
        :param fast_dev_run: gives only one batch
        :param ext: image extension of data
        :param data_dir: folder name in data/images
        """
        super().__init__()
        self.ext = ext
        self.fast_dev_run = fast_dev_run
        self.data_dir = data_dir
        self.batch_size = batch_size if not self.fast_dev_run else 1
        self.num_workers = num_workers
        self.transforms = transforms

        # Class attrs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):

        # Setup datasets for each stage
        if stage == "fit" or stage is None:
            self.train_dataset = DetectionDataset(data_dir=self.data_dir, split="train", ext=".jpg",
                                                  transforms=self.transforms)
            self.val_dataset = DetectionDataset(data_dir=self.data_dir, split="valid", ext=".jpg",
                                                transforms=self.transforms)

        if stage == "test" or stage is None:
            self.test_dataset = DetectionDataset(data_dir=self.data_dir, split="test", ext=".jpg",
                                                 transforms=self.transforms)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
