import json
import os

import torch

def filter_files_by_class(image_list, cats):
    files = {}
    for k in cats:
        files[k] = list(filter(lambda x: k in x, image_list))

    return files

def get_dir(args):
    task_name = 'siamese_results/' + args.name
    saved_model_folder = os.path.join(task_name, 'models')

    os.makedirs(saved_model_folder, exist_ok=True)

    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder


def save_best_model(saved_model_folder, net):
    torch.save(net.state_dict(), os.path.join(saved_model_folder, 'siamese_best.pt'))
