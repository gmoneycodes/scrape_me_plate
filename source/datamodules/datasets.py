import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import pytorch_lightning as pl
from scrape_me_plate.source import ROOT_DIR

class PlateDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = "yolov7", split: str = "train", ext: str = "jpg",
                 transforms=None):
        """
        :param data_dir (str): directory of your data relative to the project root
        :param transforms (callable, optional): Optional transform to be applied
        """
        self.root_dir = ROOT_DIR
        self.transforms = transforms
        self.image_dir = self.root_dir / "data" / data_dir / split / "images"
        self.label_dir = self.root_dir / "data" / data_dir / split / "label"
        self.images = [file for file in os.listdir(image_dir)]


        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = os.path.join(self.images_dir, self.images[idx])
            image = Image.open(img_name).convert("RGB")

            label_name = os.path.join(self.images_dir.replace("images", "labels"),
                                      self.images[idx].replace(".jpg", ".txt"))

            labels = []
            with open(label_name, "r") as file:
                for line in file:
                    labels.append(float(x) for x in line.split())

            sample = {"image": image, "labels": torch.Tensor(labels)}
            return sample
