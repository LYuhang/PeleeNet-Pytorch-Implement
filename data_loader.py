# -*- coding: utf-8 -*-

from torch.utils.data import *
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
# from tqdm import tqdm


class dataset(Dataset):
    def __init__(self, path):
        super(dataset, self).__init__()

        # Check train data txt file
        if not os.path.exists(os.path.join(path, "image_label.txt")):
            print("==> The image_label.txt does not exist.")
            exit(0)

        # import all training images
        # transfrom the image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.RandomRotation(degrees=30,
                                      resample=False,
                                      expand=False),
            transforms.ToTensor()
        ])
        # get all images paths and labels
        with open(os.path.join(path, "image_label.txt"), "r") as fp:
            self.lines = fp.read().split("\n")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        fn, lb = self.lines[item].split()
        img = Image.open(fn)
        if hasattr(self, "transform") and self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(np.array(lb, dtype=np.int64))