import os
from typing import Callable

import torch
from PIL import Image
from source.utils.misc import ROOT_DIR
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, data_dir: str = "yolov7", split: str = "train", ext: str = "jpg",
                 transforms: Callable = None):
        """
        :param data_dir (str): directory of your data relative to the project root
        :param transforms (callable, optional): Optional transform to be applied
        """
        assert split in ["valid", "train", "test"], "Wrong argument for `split` buddy"
        self.split = split
        self.root_dir = ROOT_DIR
        self.transforms = transforms
        self.image_dir = self.root_dir / "data" / data_dir / split / "images"
        self.images = [file for file in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")

        label_name = os.path.join(self.image_dir.replace("images", "labels"),
                                  self.images[idx].replace(".jpg", ".txt"))

        labels = []
        with open(label_name, "r") as file:
            for line in file:
                labels.append(float(x) for x in line.split())

        sample = {"image": image, "labels": torch.Tensor(labels)}
        return sample
