import torch.optim as optim

from tqdm import tqdm

from torchvision import transforms

from fastgan import lpips
import random

from data import ImageFolder, IndividualKLGradeImageFolder, get_dataloader
from diffaug import DiffAugment
from models import Generator, Discriminator, weights_init
from utils import *


class Train:
    def __init__(self, args):
        self.__parse_args(args)

        self.device = torch.device("cpu")
        if self.use_cuda:
            self.device = torch.device("cuda:0")

        self.__init_models()
        self.metrics = {}

        self.diff_aug_policy = 'color,translation'
        self.transforms = transforms.Compose([
            transforms.Resize((int(self.im_size), int(self.im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __parse_args(self, args):
        self.image_set = args.path
        self.kl_grade = args.kl_grade
        self.total_iterations = args.iter
        self.checkpoint = args.ckpt
        self.batch_size = args.batch_size
        self.im_size = args.im_size
        self.ndf = 64
        self.ngf = 64
        self.nz = 256
        self.nlr = 0.0002
        self.nbeta1 = 0.5
        self.use_cuda = True
        self.multi_gpu = True
        self.dataloader_workers = 8
        self.current_iteration = args.start_iter
        self.save_interval = 100
        self.saved_model_folder, self.saved_image_folder = get_dir(args)

    def __init_models(self):
        self.netG = Generator(ngf=self.ngf, nz=self.nz, im_size=self.im_size)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ndf=self.ndf, im_size=self.im_size)
        self.netD.apply(weights_init)

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.avg_param_G = copy_G_params(self.netG)

        self.percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.nlr, betas=(self.nbeta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.nlr, betas=(self.nbeta1, 0.999))

        if self.checkpoint is not None:
            self.__load_checkpoint()

    def __load_checkpoint(self):
        ckpt = torch.load(self.checkpoint)
        print(ckpt.keys())
        self.netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        self.netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        self.avg_param_G = ckpt['g_ema']
        self.optimizerG.load_state_dict(ckpt['opt_g'])
        self.optimizerD.load_state_dict(ckpt['opt_d'])
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.current_iteration = int(self.checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    def train_d(self, data, label="real"):
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = self.netD(data, label, part=part)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
                  self.percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
                  self.percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
                  self.percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
            err.backward()
            return pred.mean().item(), rec_all, rec_small, rec_part
        else:
            pred = self.netD(data, label)
            err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
            err.backward()
            return pred.mean().item()

    def one_iter(self, dataloader):
        real_images = next(dataloader)
        real_images = real_images.to(self.device)

        current_batch_size = real_images.size(0)
        noise = torch.Tensor(current_batch_size, self.nz).normal_(0, 1).to(self.device)

        fake_images = self.netG(noise)

        real_image = DiffAugment(real_images, policy=self.diff_aug_policy)
        fake_images = [DiffAugment(fake, policy=self.diff_aug_policy) for fake in fake_images]

        ## 2. train Discriminator
        self.netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = self.train_d(real_image, label="real")
        self.train_d([fi.detach() for fi in fake_images], label="fake")
        self.optimizerD.step()

        ## 3. train Generator
        self.netG.zero_grad()
        pred_g = self.netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        self.optimizerG.step()

        for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        return real_image, rec_img_all, rec_img_small, rec_img_part

    def train_model(self):
        if not self.kl_grade:
            dataset = ImageFolder(self.image_set, transforms=self.transforms)
        else:
            dataset = IndividualKLGradeImageFolder(self.image_set, self.kl_grade, transforms=self.transforms)

        dataloader = get_dataloader(dataset, batch_size=self.batch_size)

        fixed_noise = torch.FloatTensor(8, self.nz).normal_(0, 1).to(self.device)

        for iteration in tqdm(range(self.current_iteration, self.total_iterations + 1)):
            real_image, rec_img_all, rec_img_small, rec_img_part = self.one_iter(dataloader)

            if iteration % (self.save_interval * 10) == 0:
                save_iter_image(iteration, self.saved_image_folder, self.netG, self.avg_param_G,
                                fixed_noise, real_image, rec_img_all, rec_img_small, rec_img_part)

            if iteration % (self.save_interval * 50) == 0 or iteration == self.total_iterations:
                save_model(iteration, self.saved_model_folder, self.netG, self.netD, self.avg_param_G,
                           self.optimizerG, self.optimizerD)

