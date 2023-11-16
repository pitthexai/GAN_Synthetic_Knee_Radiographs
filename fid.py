import numpy as np

import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs
from torchmetrics.image.fid import FrechetInceptionDistance


class FrechetInceptionDataset(Dataset):
    def __init__(self, real_image_list, fake_image_list):
        self.real_image_list = real_image_list
        self.fake_image_list = fake_image_list
        self.transforms = transforms

    def __len__(self):
        if len(self.fake_image_list) < len(self.real_image_list):
            return len(self.fake_image_list)
        
        return len(self.real_image_list)

    def __getitem__(self, idx):
        real = Image.open(self.real_image_list[idx]).convert("RGB")
        fake = Image.open(self.fake_image_list[idx]).convert("RGB")

        return torch.tensor(np.swapaxes(np.array(real), 2, 0), dtype=torch.uint8), torch.tensor(np.swapaxes(np.array(fake), 2,0), dtype=torch.uint8)


def eval_fid(real_image_dir, fake_image_dir, feature_layer=2048):
    real_images = [os.path.join(real_image_dir, img) for img in os.listdir(real_image_dir)]
    fake_images = [os.path.join(fake_image_dir, img) for img in os.listdir(fake_image_dir)]
    fid_ds = FrechetInceptionDataset(real_images, fake_images, transforms=transforms)
    dataloader = DataLoader(fid_ds, batch_size=64)

    fid = FrechetInceptionDistance(feature=feature_layer)
    fid_vals = []

    for real, fake in dataloader:
        fid.update(real, real=True)
        fid.update(fake, real=False)
        fid_vals.append(fid.compute())
        fid.reset()

    return np.mean(fid_vals)