import argparse

import numpy as np
import regex as re

import os
import shutil

from sklearn.model_selection import train_test_split

def setup_argparse():
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument('--k', type=int, default=10, help='Number of images per class for K-Shot learning')

    parser.add_argument('--path', type=str, default='./',
                        help='Path to directory of knee X-rays containing subfolders for each grade')
    parser.add_argument('--outdir', type=str, default='./', help='Output directory')

    args = parser.parse_args()

    return args


def get_unique_patient_ids(grade0, grade1, grade2, grade3, grade4, pid_regex):
    grade0_patients = [pid_regex.findall(img)[0] for img in grade0]
    grade1_patients = [pid_regex.findall(img)[0] for img in grade1 if ".ipynb_checkpoints" not in img]
    grade2_patients = [pid_regex.findall(img)[0] for img in grade2]
    grade3_patients = [pid_regex.findall(img)[0] for img in grade3]
    grade4_patients = [pid_regex.findall(img)[0] for img in grade4]

    patient_set = (set(grade0_patients).
                   union(grade1_patients).
                   union(grade2_patients).
                   union(grade3_patients).
                   union(grade4_patients))

    return list(patient_set)


def load_kl_grades(root_directory):
    kl_grade0 = os.path.join(root_directory, "0")
    kl_grade1 = os.path.join(root_directory, "1")
    kl_grade2 = os.path.join(root_directory, "2")
    kl_grade3 = os.path.join(root_directory, "3")
    kl_grade4 = os.path.join(root_directory, "4")

    grade0_files = [os.path.join(root_directory, "0", img) for img in os.listdir(kl_grade0)]
    grade1_files = [os.path.join(root_directory, "1", img) for img in os.listdir(kl_grade1)]
    grade2_files = [os.path.join(root_directory, "2", img) for img in os.listdir(kl_grade2)]
    grade3_files = [os.path.join(root_directory, "3", img) for img in os.listdir(kl_grade3)]
    grade4_files = [os.path.join(root_directory, "4", img) for img in os.listdir(kl_grade4)]

    return grade0_files, grade1_files, grade2_files, grade3_files, grade4_files


def copy_dataset_files(image_set, out_dir, kl_grade):
    for img in image_set:
        img_name = img.split("/")[-1].split(".")[0]
        shutil.copy(img, os.path.join(out_dir, f"{img_name}_KL{kl_grade}.jpg"))

    print(f"Image set for KL Grade {kl_grade} copied to {out_dir}")


def generate_datasets(args):
    patient_regex = re.compile("9[0-9]{6}")

    grade0_files, grade1_files, grade2_files, grade3_files, grade4_files = load_kl_grades(args.path)
    unique_patient_ids = get_unique_patient_ids(grade0_files, grade1_files, grade2_files, grade3_files, grade4_files,
                                                patient_regex)

    train, test = train_test_split(unique_patient_ids, test_size=0.5, random_state=42)

    train, valid = train_test_split(train, test_size=0.5, random_state=42)

    grade0_patients_train = [img for img in grade0_files if patient_regex.findall(img)[0] in train]
    grade1_patients_train = [img for img in grade1_files if
                             ".ipynb_checkpoints" not in img and patient_regex.findall(img)[0] in train]
    grade2_patients_train = [img for img in grade2_files if patient_regex.findall(img)[0] in train]
    grade3_patients_train = [img for img in grade3_files if patient_regex.findall(img)[0] in train]
    grade4_patients_train = [img for img in grade4_files if patient_regex.findall(img)[0] in train]

    grade0_patients_valid = [img for img in grade0_files if patient_regex.findall(img)[0] in valid]
    grade1_patients_valid = [img for img in grade1_files if
                             ".ipynb_checkpoints" not in img and patient_regex.findall(img)[0] in valid]
    grade2_patients_valid= [img for img in grade2_files if patient_regex.findall(img)[0] in valid]
    grade3_patients_valid = [img for img in grade3_files if patient_regex.findall(img)[0] in valid]
    grade4_patients_valid = [img for img in grade4_files if patient_regex.findall(img)[0] in valid]

    grade0_patients_test = [img for img in grade0_files if patient_regex.findall(img)[0] in test]
    grade1_patients_test = [img for img in grade1_files if
                             ".ipynb_checkpoints" not in img and patient_regex.findall(img)[0] in test]
    grade2_patients_test = [img for img in grade2_files if patient_regex.findall(img)[0] in test]
    grade3_patients_test = [img for img in grade3_files if patient_regex.findall(img)[0] in test]
    grade4_patients_test = [img for img in grade4_files if patient_regex.findall(img)[0] in test]
    
    train_pth = os.path.join(args.outdir, "train")
    valid_pth = os.path.join(args.outdir, "valid")
    test_pth = os.path.join(args.outdir, "test")
    
    if not os.path.exists(train_pth):
        os.makedirs(train_pth)
        
        copy_dataset_files(grade0_patients_train, os.path.join(args.outdir, "train"), 0)
        copy_dataset_files(grade1_patients_train, os.path.join(args.outdir, "train"), 1)
        copy_dataset_files(grade2_patients_train, os.path.join(args.outdir, "train"), 2)
        copy_dataset_files(grade3_patients_train, os.path.join(args.outdir, "train"), 3)
        copy_dataset_files(grade4_patients_train, os.path.join(args.outdir, "train"), 4)
        
    if not os.path.exists(valid_pth):
        os.makedirs(valid_pth)
        copy_dataset_files(grade0_patients_valid, os.path.join(args.outdir, "valid"), 0)
        copy_dataset_files(grade1_patients_valid, os.path.join(args.outdir, "valid"), 1)
        copy_dataset_files(grade2_patients_valid, os.path.join(args.outdir, "valid"), 2)
        copy_dataset_files(grade3_patients_valid, os.path.join(args.outdir, "valid"), 3)
        copy_dataset_files(grade4_patients_valid, os.path.join(args.outdir, "valid"), 4)

    if not os.path.exists(test_pth):
        os.makedirs(test_pth)
        copy_dataset_files(grade0_patients_test, os.path.join(args.outdir, "test"), 0)
        copy_dataset_files(grade1_patients_test, os.path.join(args.outdir, "test"), 1)
        copy_dataset_files(grade2_patients_test, os.path.join(args.outdir, "test"), 2)
        copy_dataset_files(grade3_patients_test, os.path.join(args.outdir, "test"), 3)
        copy_dataset_files(grade4_patients_test, os.path.join(args.outdir, "test"), 4)

    sample_few_shot(args)
    
def filter_kl_grade(images, kl):
    return [img for img in images if f"KL{kl}" in img]

def sample_few_shot(args):
    train = [f"{args.outdir}/{img}" for img in os.listdir(os.path.join(args.outdir, "train"))]
    valid = [f"{args.outdir}/{img}" for img in os.listdir(os.path.join(args.outdir, "valid"))]

    np.random.seed(42)
    
    train_samples = []
    valid_samples = []
    for kl in range(0, 5):
        train_filtered = filter_kl_grade(train, kl)
        valid_filtered = filter_kl_grade(valid, kl)
        train_samples.extend(np.random.choice(train_filtered, args.k, replace=False))
        valid_samples.extend(np.random.choice(valid_filtered, args.k, replace=False))
    
    with open(f"{args.outdir}/{args.k}-shot_train.txt", "w") as f:
        for samp in train_samples:
            print(samp)
            f.write(f"{samp}\n")

    with open(f"{args.outdir}/{args.k}-shot_valid.txt", "w") as f:
        for samp in valid_samples:
            f.write(f"{samp}\n")

if __name__ == '__main__':
    args = setup_argparse()
    generate_datasets(args)
    



