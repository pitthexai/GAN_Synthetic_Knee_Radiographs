import numpy as np
import pandas as pd

import os
from tqdm import tqdm

def filter_kl_files(data_root, kl_cats):
    kl_files = {}
    images = os.listdir(data_root)
    for kl in kl_cats:
        kl_files[kl] = list(filter(lambda x: kl in x, images))

    return kl_files

def generate_pairs(cats, data_root, n_pairs):
    image1_files, image2_files, labels  = [], [], []
    files = filter_kl_files(data_root, cats)

    for i in tqdm(range(n_pairs)):
        # If the index is even, generate a sample that have two images that are the similar
        if i % 2 == 0:
            # Randomly select a category
            cat = np.random.choice(cats, 1)[0]

            # Randomly select two images
            image1, image2 = np.random.choice(files[cat], 2)

            labels.append(1.0)
            image1_files.append(image1)
            image2_files.append(image2)

        # If the index is odd, generate a pair of two images that are not similar
        else:
            cat1 = np.random.choice(cats, 1)[0]

            # Get second category, don't select the same category used for cat1
            cat2 = np.random.choice([cat for cat in cats if cat != cat1], 1)[0]

            # Get image samples
            image1 = np.random.choice(files[cat1],1)[0]
            image2 = np.random.choice(files[cat2], 1)[0]

            labels.append(0.0)
            image1_files.append(image1)
            image2_files.append(image2)

    return image1_files, image2_files, labels
