import argparse

import os
import torch

from fastgan.models import Generator
from fastgan.generate import generate_samples

def setup_argparse():
    parser = argparse.ArgumentParser(description='Generate synthetic knee images from trained GAN')
    parser.add_argument('--artifacts', type=str, default=".", help='path to artifacts.')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--end_iter', type=int, default=5)
    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=1024)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = setup_argparse()
    noise_dim = 256
    device = torch.device('cuda:%d' % (args.cuda))

    net_ig = Generator(ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)  # , big=args.big )
    net_ig.to(device)

    for epoch in [10000 * i for i in range(args.start_iter, args.end_iter + 1)]:
        ckpt = f"{args.artifacts}/models/{epoch}.pth"
        checkpoint = torch.load(ckpt, map_location=lambda a, b: a)

        # Remove prefix `module`.
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])
        # net_ig.eval()
        print('load checkpoint success, epoch %d' % epoch)
        net_ig.to(device)

        del checkpoint

        dist = f'{args.dist}/eval_%d' % (epoch)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        generate_samples(net_ig, args.n_sample, dist, batch_size=args.batch_size)
