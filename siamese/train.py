import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torchvision import transforms
from torchmetrics import Accuracy
from torchmetrics.functional.pairwise import pairwise_euclidean_distance


from siamese.model import SiameseNetwork
from siamese.data import SiameseKLGradeImageFolder, get_dataloader
from siamese.loss import ContrastiveLoss
from siamese.utils import get_dir, save_best_model
import random


class Train:
    def __init__(self, args):
        self.__parse_args(args)

        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device("cuda:0")

        print(self.device)
        self.__init_model()

        self.metrics = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": []
        }

        self.saved_model_folder = get_dir(args)

        self.transforms = transforms.Compose([
            transforms.Resize((int(self.im_size), int(self.im_size))),
            transforms.ToTensor()
        ])

    def __parse_args(self, args):
        self.train_set = args.train_path
        self.valid_set = args.valid_path
        self.im_size = args.im_size
        self.batch_size = args.batch_size
        self.use_cuda = args.cuda
        self.epochs = args.epochs
        self.checkpoint = args.ckpt
        self.lr = args.lr

    def __init_model(self):
        self.net = SiameseNetwork()
        self.net.to(self.device)
        self.criterion = ContrastiveLoss(margin=2)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.acc = Accuracy(task="binary").to(self.device)

        if self.checkpoint is not None:
            self.__load_checkpoint()

    def __load_checkpoint(self):
        self.net = torch.load(self.checkpoint)

    def compute_labels(self, out1, out2, threshold = 0.5):
        out = pairwise_euclidean_distance(out1, out1, reduction="mean")
        return out

    def one_iter(self, dataloader):
        self.net.train()
        accuracies = []
        losses = []

        for im1, im2, labels in dataloader:
            self.optimizer.zero_grad()

            im1, im2, labels = im1.cuda(), im2.cuda(), labels.cuda()
            out1, out2 = self.net(im1, im2)

            loss = self.criterion(out1, out2, labels)
            loss.backward()
            losses.append(loss.item())
            # accuracies.append(self.acc(self.compute_labels(out1, out2), labels).item())

            self.optimizer.step()

        avg_loss = np.mean(losses)
       # avg_acc = np.mean(accuracies)
        self.metrics["train_loss"].append(avg_loss)
        #self.metrics["train_acc"].append(avg_acc)

        return avg_loss, #avg_acc

    def eval_iter(self, dataloader):
        self.net.eval()
        accuracies = []
        losses = []

        with torch.no_grad():
            for im1, im2, labels in dataloader:
                im1, im2, labels = im1.cuda(), im2.cuda(), labels.cuda()
                out1, out2 = self.net(im1, im2)
                loss = self.criterion(out1, out2, labels)

                losses.append(loss.item())
                # accuracies.append(self.acc(self.compute_labels(out1, out2), labels).item())

        avg_loss = np.mean(losses)
        #avg_acc = np.mean(accuracies)
        self.metrics["valid_loss"].append(avg_loss)
        # elf.metrics["valid_acc"].append(avg_acc)

        return avg_loss, #avg_acc

    def train_model(self):
        train_dataset = SiameseKLGradeImageFolder(self.train_set, transforms=self.transforms)
        valid_dataset = SiameseKLGradeImageFolder(self.valid_set, transforms=self.transforms)

        train_dataloader = get_dataloader(train_dataset, batch_size=self.batch_size)
        valid_dataloader = get_dataloader(valid_dataset, batch_size=self.batch_size)

        best_model = None
        best_acc = 0

        for epoch in range(self.epochs + 1):
            t_loss = self.one_iter(train_dataloader)
            print(f'Train Epoch [{epoch + 1}/{self.epochs + 1}]\tLoss: {t_loss}') #'\tAccuracy: {t_acc}')

            if epoch % 10 == 0:
                v_loss = self.eval_iter(valid_dataloader)
                print(f'Validation Epoch [{epoch + 1}/{self.epochs + 1}]\tLoss: {v_loss}') # '\tAccuracy: {v_acc}')

#                 if v_acc > best_acc:
#                     best_acc = v_acc
#                     best_model = copy.deepcopy(self.net)

        save_best_model(self.saved_model_folder, self.net)