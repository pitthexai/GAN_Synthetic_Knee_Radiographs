import numpy as np
import torch
import torch.utils.data as data

from siamese.utils import filter_files_by_class
from PIL import Image


def get_dataloader(dataset, batch_size, dataloader_workers=8):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=dataloader_workers, pin_memory=True)

def parse_image_set(image_set):
    files = []
    with open(image_set, "r") as f:
        for image in f.readlines():
            image = image.strip()
            files.append(image)

    return files

class SiameseKLGradeImageFolder(data.Dataset):
    def __init__(self, image_list, transforms=None):
        super().__init__()
        self.cats = [f"KL{kl}" for kl in range(0, 5)]
        self.total_images = len(image_list)
        self.image_lists = filter_files_by_class(image_list, self.cats)
        self.transforms = transforms

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        # If the index is even, generate a sample that have two images that are the similar
        if idx % 2 == 0:
            # Randomly select a category
            cat = np.random.choice(self.cats, 1)[0]
            # Randomly select two images
            image1_path, image2_path = np.random.choice(self.image_lists[cat], 2)

            label = 1

        # If the index is odd, generate a pair of two images that are not similar
        else:
            cat1 = np.random.choice(self.cats, 1)[0]

            # Get second category, don't select the same category used for cat1
            cat2 = np.random.choice([cat for cat in self.cats if cat != cat1], 1)[0]

            # Get image samples
            image1_path = np.random.choice(self.image_lists[cat1], 1)[0]
            image2_path = np.random.choice(self.image_lists[cat2], 1)[0]

            label = 0

        image1 = Image.open(image1_path).convert("L")
        image2 = Image.open(image2_path).convert("L")

        if self.transforms:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float)


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