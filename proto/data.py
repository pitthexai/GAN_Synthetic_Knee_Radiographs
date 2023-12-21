import numpy as np
import pandas as pd

import torch
from PIL import Image
from torch.utils import data

import os

import learn2learn as l2l
from learn2learn.data import MetaDataset, TaskDataset
from learn2learn.data.transforms import NWays, KShots,LoadData, RemapLabels

def parse_image_set(image_set):
    files = []
    with open(image_set, "r") as f:
        for image in f.readlines():
            image = image.strip()
            files.append(image)

    return files

def get_proto_dataset(image_list, transforms=None, k=5, nway=5, tasks=500):
    proto_ds = ProtoKLGradeImageFolder(image_list, transforms=transforms)
    proto_meta = MetaDataset(proto_ds)

    task_transforms = [
        NWays(proto_meta, nway),
        KShots(proto_meta, k * 2),
        LoadData(proto_meta),
        RemapLabels(proto_meta),
    ]
    proto_taskset = TaskDataset(proto_meta, task_transforms=task_transforms, num_tasks=tasks)

    return proto_taskset

class ProtoKLGradeImageFolder(data.Dataset):
    def __init__(self, image_list, img_col_id="Image", kl_col_id = "Class", transforms=None):
        self.image_list = pd.read_csv(image_list)
        self.img_col_id = img_col_id
        self.kl_col_id = kl_col_id
        self.total_images = len(self.image_list)
        self.transforms = transforms

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        image1_path = self.image_list.loc[idx][self.img_col_id]
        label = self.image_list.loc[idx][self.kl_col_id]
    
        image1 = Image.open(image1_path).convert("L")

        if self.transforms:
            image1 = self.transforms(image1)

        return image1, torch.tensor(label, dtype=torch.long)

class EmbeddingImageFolder(data.Dataset):
    def __init__(self, image_set, kl_grade="KL", num_samples=750, transforms=None):
        super().__init__()
        self.image_files = [img for img in image_set if kl_grade in img]
        if len(self.image_files) > num_samples:
            self.sample_images(num_samples)
        self.transforms = transforms

    def sample_images(self, num_samples):
        self.image_files = np.random.choice(self.image_files, num_samples)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("L")

        if self.transforms:
            image = self.transforms(image)

        return image