import argparse

import numpy as np
import regex as re

import os
import shutil

from tqdm import tqdm

from siamese.utils import generate_pairs

def setup_argparse():
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument('--image_root', type=str, default="./", help='Image directory. Image file names should contain KL class.')
    parser.add_argument('--pairs', type=int, default=30000, help="Number of image pairs to create")
    parser.add_argument('--ds', type=str, default='train',
                        help='Dataset to create: train, valid, or test.')
    parser.add_argument('--outdir', type=str, default='./', help='Location to save dataset csv')

    args = parser.parse_args()

    return args


def extract_kl_grades(image_dir):
    kl_re = re.compile("KL[0-4]{1}")
    return np.unique([kl_re.findall(img)[0] for img in os.listdir(image_dir)])

if __name__ == '__main__':
    args = setup_argparse()
    kl_cats = extract_kl_grades(args.image_root)
    image_set1, image_set2, labels = generate_pairs(kl_cats, args.image_root, args.pairs)

    with open(f"{args.outdir}/siamese_kl_pairs_{args.ds}.txt", "w") as f:
        for image1, image2, label in zip(image_set1, image_set2, labels):
            f.write(f"{image1},{image2},{label}\n")
