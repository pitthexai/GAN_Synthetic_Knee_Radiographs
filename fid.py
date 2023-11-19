import numpy as np

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import parse_image_set


class FrechetInceptionDataset(Dataset):
    def __init__(self, real_image_list, fake_image_list):
        self.real_image_list = real_image_list
        self.fake_image_list = fake_image_list

    def __len__(self):
        if len(self.fake_image_list) < len(self.real_image_list):
            return len(self.fake_image_list)

        return len(self.real_image_list)

    def __getitem__(self, idx):
        real = Image.open(self.real_image_list[idx]).convert("RGB")
        fake = Image.open(self.fake_image_list[idx]).convert("RGB")

        return torch.tensor(np.swapaxes(np.array(real), 2, 0), dtype=torch.uint8), torch.tensor(
            np.swapaxes(np.array(fake), 2, 0), dtype=torch.uint8)


def eval_fid(real_image_loc, fake_image_loc, feature_layer=2048):
    if ".txt" not in real_image_loc:
        real_images = [os.path.join(real_image_loc, img) for img in os.listdir(real_image_loc)]
    else:
        real_images = parse_image_set(real_image_loc)

    if ".txt" not in fake_image_loc:
        fake_images = [os.path.join(fake_image_loc, img) for img in os.listdir(fake_image_loc)]
    else:
        fake_images = parse_image_set(fake_image_loc)

    fid_ds = FrechetInceptionDataset(real_images, fake_images)
    dataloader = DataLoader(fid_ds, batch_size=64)

    fid = FrechetInceptionDistance(feature=feature_layer)
    fid_vals = []

    for real, fake in dataloader:
        real, fake = real, fake
        fid.update(real, real=True)
        fid.update(fake, real=False)
        fid_vals.append(fid.compute())
        fid.reset()

    return np.mean(fid_vals)