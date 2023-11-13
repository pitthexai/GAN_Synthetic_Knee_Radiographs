import numpy as np
import torch.utils.data as data

from PIL import Image


"""
Infinite Sampler and InfiniteSamplerWrapper taken from:
https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/operation.py
"""

def parse_image_set(image_set):
    files = []
    with open(image_set, "r") as f:
        for image in f.readlines():
            image = image.strip()
            files.append(image)

    return files


def get_dataloader(dataset, batch_size, dataloader_workers=8):
    return iter(data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class ImageFolder(data.Dataset):
    def __init__(self, image_set, transforms=None):
        self.image_files = parse_image_set(image_set)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image