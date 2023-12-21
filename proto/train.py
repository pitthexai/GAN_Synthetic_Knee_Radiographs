import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torchvision import transforms
from torchmetrics import Accuracy
from torchmetrics.functional.pairwise import pairwise_euclidean_distance

from proto.model import ProtoNet
from proto.data import get_proto_dataset
from proto.utils import get_dir, save_best_model
import random


class Train:
    def __init__(self, args):
        self.__parse_args(args)

        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device("cuda:0")

        self.__init_model()

        self.metrics = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": []
        }

        self.saved_model_folder = get_dir(args)
        self.n_task_samples = 300
        self.transforms = transforms.Compose([
            transforms.Resize((int(self.im_size), int(self.im_size))),
            transforms.ToTensor()
        ])

    def __parse_args(self, args):
        self.train_set = args.train_path
        self.valid_set = args.valid_path
        self.k_shot = args.k
        self.num_classes = args.num_classes
        self.tasks = args.num_tasks
        self.im_size = args.im_size
        self.batch_size = args.batch_size
        self.use_cuda = args.cuda
        self.epochs = args.epochs
        self.checkpoint = args.ckpt
        self.lr = args.lr

    def __init_model(self):
        self.net = ProtoNet()
        self.net.to(self.device)
        
        if torch.cuda.device_count() >=1:
            self.net = nn.DataParallel(self.net, device_ids=[0, 1])
        self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.acc = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

        if self.checkpoint is not None:
            self.__load_checkpoint()

    def __load_checkpoint(self):
        self.net = torch.load(self.checkpoint)

    def one_iter(self, batch, mode="train"):
        
        if mode == "train":
            self.net.train()
        else:
            self.net.eval()

        self.optimizer.zero_grad()

        data, labels = batch
        data = data.to(self.device)
        labels = labels.to(self.device)

        # Sort data samples by labels
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        # Compute support and query embeddings
        embeddings = self.net(data.float())
        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(self.num_classes) * (self.k_shot*2)
        for offset in range(self.k_shot):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)
        support = embeddings[support_indices].float()
        support = support.reshape(self.num_classes, self.k_shot, -1).mean(dim=1)
        query = embeddings[query_indices].float()
        labels = labels[query_indices].long()

        logits = -pairwise_euclidean_distance(query, support)
        acc = self.acc(logits.argmax(dim=1).view(labels.shape), labels)

        loss = self.criterion(logits, labels)
        loss.backward()

        self.optimizer.step()

        if mode == "train":
            self.metrics["train_loss"].append(loss.item())
            self.metrics["train_acc"].append(acc.item())
        else:
            self.metrics["valid_loss"].append(loss.item())
            self.metrics["valid_acc"].append(acc.item())

    def train_model(self):
        train_dataset = get_proto_dataset(self.train_set, transforms=self.transforms, k=self.k_shot, nway=self.num_classes, tasks=self.tasks)
        valid_dataset = get_proto_dataset(self.valid_set, transforms=self.transforms, k=self.k_shot, nway=self.num_classes, tasks=self.tasks)

#         train_dataloader = get_dataloader(train_dataset, batch_size=self.batch_size)
#         valid_dataloader = get_dataloader(valid_dataset, batch_size=self.batch_size)

        best_model = None
        best_acc = 0

        for epoch in range(self.epochs + 1):

            # Sample from taskset
            for i in tqdm(range(self.n_task_samples)):
                self.one_iter(train_dataset.sample())

            print(f'Train Epoch [{epoch + 1}/{self.epochs + 1}]\tLoss: {np.mean(self.metrics["train_loss"][-self.n_task_samples:])} ' +
                  f'\tAccuracy: {np.mean(self.metrics["train_acc"][-self.n_task_samples:])}')

            if epoch % 1 == 0:
                # v_loss = self.eval_iter(valid_dataloader)
                for i in tqdm(range(self.n_task_samples)):
                    self.one_iter(valid_dataset.sample(), mode="eval")
                    
                print(f'Valid Epoch [{epoch + 1}/{self.epochs + 1}]\tLoss: {np.mean(self.metrics["valid_loss"][-self.n_task_samples:])} ' +
                  f'\tAccuracy: {np.mean(self.metrics["valid_acc"][-self.n_task_samples:])}')
               
                if np.mean(self.metrics["valid_acc"][-self.n_task_samples:]) > best_acc:
                    best_acc = np.mean(self.metrics["valid_acc"][-self.n_task_samples:])
                    best_model = copy.deepcopy(self.net)

                    save_best_model(self.saved_model_folder, best_model)