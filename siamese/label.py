import numpy as np

import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.pairwise import euclidean
from torchvision import transforms

from siamese.data import EmbeddingImageFolder
from siamese.utils import filter_files_by_class


def get_mean_embeddings(model, train_data, out_path):
    files_by_class = filter_files_by_class(train_data, cats=[f"KL{kl}" for kl in range(0, 5)])
    trfms = transforms.Compose([
        transforms.Resize((int(224), int(224))),
        transforms.ToTensor()])

    embedding_arr = []
    for cat in files_by_class:
        data = EmbeddingImageFolder(files_by_class[cat], kl_grade=cat, transforms=trfms)
        dataloader = DataLoader(data)

        with torch.no_grad():
            embeddings = torch.tensor([])
            for img in dataloader:
                img = img.cuda()
                embeddings = torch.cat([embeddings, model.get_embedding(img).cpu()])
            print(embeddings.mean(dim=1).cpu().numpy())
            embedding_arr.append(embeddings.mean(dim=0).numpy())

    torch.save(torch.tensor(np.array(embedding_arr)), str(os.path.join(out_path, "train_embeddings.pt")))
