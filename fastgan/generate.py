import torch
import torch.nn.functional as F
from torchvision import utils as vutils

import os
from tqdm import tqdm

def generate_samples(generator, n_sample, out_dir, device="cuda", noise_dim=256, batch_size=16):
    with torch.no_grad():
        for i in tqdm(range(n_sample // batch_size)):
            noise = torch.randn(batch_size, noise_dim).to(device)
            imgs = generator(noise)[0]
            imgs = F.interpolate(imgs, 512)
            for j, img in enumerate(imgs):
                vutils.save_image(img.add(1).mul(0.5),
                                  os.path.join(out_dir, '%d.png' % (i * batch_size + j)))
