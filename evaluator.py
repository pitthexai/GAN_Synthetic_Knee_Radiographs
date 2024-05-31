import argparse
from models import Generator

import torch
from fid import eval_fid

def setup_argparse():
    parser = argparse.ArgumentParser(description='Knee KL Grade GAN')

    parser.add_argument('--real', type=str, default='./',
                        help='Path to real images')
    parser.add_argument('--fake', type=str, default='./',
                        help='Path to fake images')
    parser.add_argument('--incpt_layer', type=int, default=2048, help='Feature layer of InceptionV3 for FID')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = setup_argparse()
    final_fid = eval_fid(args.real, args.fake, feature_layer=args.incpt_layer)
    print("Average FID score:", final_fid)
